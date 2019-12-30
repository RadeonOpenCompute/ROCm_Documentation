
.. _ROCm-Tools:

=====================
ROCm Tools
=====================

HCC: Heterogeneous Compute Compiler
=====================================

**HCC : An open source C++ compiler for heterogeneous devices**

This repository hosts the HCC compiler implementation project. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as HC, C++ 17 ParallelSTL and transforms it into the AMD GCN ISA.

Deprecation Notice
*******************

AMD is deprecating HCC to put more focus on HIP development and on other languages supporting heterogeneous compute. We will no longer develop any new feature in HCC and we will stop maintaining HCC after its final release, which is planned for June 2019. If your application was developed with the hc C++ API, we would encourage you to transition it to other languages supported by AMD, such as HIP or OpenCL. HIP and hc language share the same compiler technology, so many hc kernel language features (including inline assembly) are also available through the HIP compilation path.

The project is based on LLVM+CLANG. For more information, please visit :ref:`HCCguide`

GCN Assembler and Disassembler
==============================

The Art of AMDGCN Assembly: How to Bend the Machine to Your Will
*****************************************************************
The ability to write code in assembly is essential to achieving the best performance for a GPU program. In a previous blog we described how to combine several languages in a single program using ROCm and Hsaco. This article explains how to produce Hsaco from assembly code and also takes a closer look at some new features of the GCN architecture. I'd like to thank Ilya Perminov of Luxsoft for co-authoring this blog post. Programs written for GPUs should achieve the highest performance possible. Even carefully written ones, however, won’t always employ 100% of the GPU’s capabilities. Some reasons are the following:

 * The program may be written in a high level language that does not expose all of the features available on the hardware.
 * The compiler is unable to produce optimal ISA code, either because the compiler needs to ‘play it safe’ while adhering to the     	semantics of a language or because the compiler itself is generating un-optimized code.

Consider a program that uses one of GCN’s new features (source code is available on `GitHub <https://github.com/RadeonOpenCompute/LLVM-AMDGPU-Assembler-Extra>`_). Recent hardware architecture updates—DPP and DS Permute instructions—enable efficient data sharing between wavefront lanes. To become more familiar with the instruction set, review the `GCN ISA Reference Guide <https://github.com/olvaffe/gpu-docs/blob/master/amd-open-gpu-docs/AMD_GCN3_Instruction_Set_Architecture.pdf>`_. Note: the assembler is currently experimental; some of syntax we describe may change.

DS Permute Instructions
**************************
Two new instructions, ds_permute_b32 and ds_bpermute_b32, allow VGPR data to move between lanes on the basis of an index from another VGPR. These instructions use LDS hardware to route data between the 64 lanes, but they don’t write to LDS memory. The difference between them is what to index: the source-lane ID or the destination-lane ID. In other words, ds_permute_b32 says “put my lane data in lane i,” and ds_bpermute_b32 says “read data from lane i.” The GCN ISA Reference Guide provides a more formal description. The test kernel is simple: read the initial data and indices from memory into GPRs, do the permutation in the GPRs and write the data back to memory. An analogous OpenCL kernel would have this form:

.. code:: cpp

  __kernel void hello_world(__global const uint * in, __global const uint * index, __global uint * out)
  {
      size_t i = get_global_id(0);
      out[i] = in[ index[i] ];
  }

Passing Parameters to a Kernel
*******************************
Formal HSA arguments are passed to a kernel using a special read-only memory segment called kernarg. Before a wavefront starts, the base address of the kernarg segment is written to an SGPR pair. The memory layout of variables in kernarg must employ the same order as the list of kernel formal arguments, starting at offset 0, with no padding between variables—except to honor the requirements of natural alignment and any align qualifier. The example host program must create the kernarg segment and fill it with the buffer base addresses. The HSA host code might look like the following:

.. code:: cpp

  /*
  * This is the host-side representation of the kernel arguments that the simplePermute kernel expects.
  */
  struct simplePermute_args_t {
	uint32_t * in;
	uint32_t * index;
	uint32_t * out;
  };
  /*
   * Allocate the kernel-argument buffer from the correct region.
  */
  hsa_status_t status;
  simplePermute_args_t * args = NULL;
  status = hsa_memory_allocate(kernarg_region, sizeof(simplePermute_args_t), (void**)(&args));
  assert(HSA_STATUS_SUCCESS == status);
  aql->kernarg_address = args;
  /*
  * Write the args directly to the kernargs buffer;
  * the code assumes that memory is already allocated for the
  * buffers that in_ptr, index_ptr and out_ptr point to
  */
  args->in = in_ptr;
  args->index = index_ptr;
  args->out = out_ptr;

The host program should also allocate memory for the in, index and out buffers. In the GitHub repository, all the run-time-related  stuff is hidden in the Dispatch and Buffer classes, so the sample code looks much cleaner:

.. code:: cpp

  // Create Kernarg segment
  if (!AllocateKernarg(3 * sizeof(void*))) { return false; }

  // Create buffers
  Buffer *in, *index, *out;
  in = AllocateBuffer(size);
  index = AllocateBuffer(size);
  out = AllocateBuffer(size);

  // Fill Kernarg memory
  Kernarg(in); // Add base pointer to “in” buffer
  Kernarg(index); // Append base pointer to “index” buffer
  Kernarg(out); // Append base pointer to “out” buffer

Initial Wavefront and Register State To launch a kernel in real hardware, the run time needs information about the kernel, such as

   * The LDS size
   * The number of GPRs
   * Which registers need initialization before the kernel starts

  All this data resides in the amd_kernel_code_t structure. A full description of the structure is available in the `AMDGPU-ABI <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Codeobj-format.html?highlight=finalizer>`_       	specification. This is what it looks like in source code:

::

   .hsa_code_object_version 2,0
   .hsa_code_object_isa 8, 0, 3, "AMD", "AMDGPU"

   .text
   .p2align 8
   .amdgpu_hsa_kernel hello_world

   hello_world:

   .amd_kernel_code_t
   enable_sgpr_kernarg_segment_ptr = 1
   is_ptr64 = 1
   compute_pgm_rsrc1_vgprs = 1
   compute_pgm_rsrc1_sgprs = 0
   compute_pgm_rsrc2_user_sgpr = 2
   kernarg_segment_byte_size = 24
   wavefront_sgpr_count = 8
   workitem_vgpr_count = 5
   .end_amd_kernel_code_t

   s_load_dwordx2  s[4:5], s[0:1], 0x10
   s_load_dwordx4  s[0:3], s[0:1], 0x00
   v_lshlrev_b32  v0, 2, v0
   s_waitcnt     lgkmcnt(0)
   v_add_u32     v1, vcc, s2, v0
   v_mov_b32     v2, s3
   v_addc_u32    v2, vcc, v2, 0, vcc
   v_add_u32     v3, vcc, s0, v0
   v_mov_b32     v4, s1
   v_addc_u32    v4, vcc, v4, 0, vcc
   flat_load_dword  v1, v[1:2]
   flat_load_dword  v2, v[3:4]
   s_waitcnt     vmcnt(0) & lgkmcnt(0)
   v_lshlrev_b32  v1, 2, v1
   ds_bpermute_b32  v1, v1, v2
   v_add_u32     v3, vcc, s4, v0
   v_mov_b32     v2, s5
   v_addc_u32    v4, vcc, v2, 0, vcc
   s_waitcnt     lgkmcnt(0)
   flat_store_dword  v[3:4], v1
   s_endpgm

Currently, a programmer must manually set all non-default values to provide the necessary information. Hopefully, this situation will change with new updates that bring automatic register counting and possibly a new syntax to fill that structure. Before the start of every wavefront execution, the GPU sets up the register state on the basis of the enable_sgpr_* and enable_vgpr_* flags. VGPR v0 is always initialized with a work-item ID in the x dimension. Registers v1 and v2 can be initialized with work-item IDs in the y and z dimensions, respectively. Scalar GPRs can be initialized with a work-group ID and work-group count in each dimension, a dispatch ID, and pointers to kernarg, the aql packet, the aql queue, and so on. Again, the AMDGPU-ABI specification contains a full list in in the section on initial register state. For this example, a 64-bit base kernarg address will be stored in the s[0:1] registers (enable_sgpr_kernarg_segment_ptr = 1), and the work-item thread ID will occupy v0 (by default). Below is the scheme showing initial state for our kernel. 


.. image:: initial_state-768x387.png


The GPR Counting
******************
The next amd_kernel_code_t fields are obvious: is_ptr64 = 1 says we are in 64-bit mode, and kernarg_segment_byte_size = 24 describes the kernarg segment size. The GPR counting is less straightforward, however. The workitem_vgpr_count holds the number of vector registers that each work item uses, and wavefront_sgpr_count holds the number of scalar registers that a wavefront uses. The code above employs v0–v4, so workitem_vgpr_count = 5. But wavefront_sgpr_count = 8 even though the code only shows s0–s5, since the special registers VCC, FLAT_SCRATCH and XNACK are physically stored as part of the wavefront’s SGPRs in the highest-numbered SGPRs. In this example, FLAT_SCRATCH and XNACK are disabled, so VCC has only two additional registers. In current GCN3 hardware, VGPRs are allocated in groups of 4 registers and SGPRs in groups of 16. Previous generations (GCN1 and GCN2) have a VGPR granularity of 4 registers and an SGPR granularity of 8 registers. The fields compute_pgm_rsrc1_*gprs contain a device-specific number for each register-block type to allocate for a wavefront. As we said previously, future updates may enable automatic counting, but for now you can use following formulas for all three GCN GPU generations:

::

  compute_pgm_rsrc1_vgprs = (workitem_vgpr_count-1)/4

  compute_pgm_rsrc1_sgprs = (wavefront_sgpr_count-1)/8

Now consider the corresponding assembly:

::

  // initial state:
  //   s[0:1] - kernarg base address
  //   v0 - workitem id

  s_load_dwordx2  s[4:5], s[0:1], 0x10  // load out_ptr into s[4:5] from kernarg
  s_load_dwordx4  s[0:3], s[0:1], 0x00  // load in_ptr into s[0:1] and index_ptr into s[2:3] from kernarg
  v_lshlrev_b32  v0, 2, v0              // v0 *= 4;
  s_waitcnt     lgkmcnt(0)              // wait for memory reads to finish

  // compute address of corresponding element of index buffer
  // i.e. v[1:2] = &index[workitem_id]
  v_add_u32     v1, vcc, s2, v0
  v_mov_b32     v2, s3
  v_addc_u32    v2, vcc, v2, 0, vcc

  // compute address of corresponding element of in buffer
  // i.e. v[3:4] = &in[workitem_id]
  v_add_u32     v3, vcc, s0, v0
  v_mov_b32     v4, s1
  v_addc_u32    v4, vcc, v4, 0, vcc

  flat_load_dword  v1, v[1:2] // load index[workitem_id] into v1
  flat_load_dword  v2, v[3:4] // load in[workitem_id] into v2
  s_waitcnt     vmcnt(0) & lgkmcnt(0) // wait for memory reads to finish

  // v1 *= 4; ds_bpermute_b32 uses byte offset and registers are dwords
  v_lshlrev_b32  v1, 2, v1

  // perform permutation
  // temp[thread_id] = v2
  // v1 = temp[v1]
  // effectively we got v1 = in[index[thread_id]]
  ds_bpermute_b32  v1, v1, v2

  // compute address of corresponding element of out buffer
  // i.e. v[3:4] = &out[workitem_id]
  v_add_u32     v3, vcc, s4, v0
  v_mov_b32     v2, s5
  v_addc_u32    v4, vcc, v2, 0, vcc

  s_waitcnt     lgkmcnt(0) // wait for permutation to finish

  // store final value in out buffer, i.e. out[workitem_id] = v1
  flat_store_dword  v[3:4], v1

  s_endpgm

Compiling GCN ASM Kernel Into Hsaco
**************************************
The next step is to produce a Hsaco from the ASM source. LLVM has added support for the AMDGCN assembler, so you can use Clang to do all the necessary magic:

.. code:: sh

  clang -x assembler -target amdgcn--amdhsa -mcpu=fiji -c -o test.o asm_source.s

  clang -target amdgcn--amdhsa test.o -o test.co

The first command assembles an object file from the assembly source, and the second one links everything (you could have multiple source files) into a Hsaco. Now, you can load and run kernels from that Hsaco in a program. The `GitHub examples <https://github.com/RadeonOpenCompute/LLVM-AMDGPU-Assembler-Extra>`_ use Cmake to automatically compile ASM sources. In a future post we will cover DPP, another GCN cross-lane feature that allows vector instructions to grab operands from a neighboring lane.



GCN Assembler Tools
====================

Overview
********
This repository contains the following useful items related to AMDGPU ISA assembler:

   * amdphdrs: utility to convert ELF produced by llvm-mc into AMD Code Object (v1)
   * examples/asm-kernel: example of AMDGPU kernel code
   * examples/gfx8/ds_bpermute: transfer data between lanes in a wavefront with ds_bpermute_b32
   * examples/gfx8/dpp_reduce: calculate prefix sum in a wavefront with DPP instructions
   * examples/gfx8/s_memrealtime: use s_memrealtime instruction to create a delay
   * examples/gfx8/s_memrealtime_inline: inline assembly in OpenCL kernel version of s_memrealtime
   * examples/api/assemble: use LLVM API to assemble a kernel
   * examples/api/disassemble: use LLVM API to disassemble a stream of instructions
   * bin/sp3_to_mc.pl: script to convert some AMD sp3 legacy assembler syntax into LLVM MC
   * examples/sp3: examples of sp3 convertable code

At the time of this writing (February 2016), LLVM trunk build and latest ROCR runtime is needed.

LLVM trunk (May or later) now uses lld as linker and produces AMD Code Object (v2).

Building
*********
Top-level CMakeLists.txt is provided to build everything included. The following CMake variables should be set:

   * HSA_DIR (default /opt/hsa/bin): path to ROCR Runtime
   * LLVM_DIR: path to LLVM build directory

To build everything, create build directory and run cmake and make:

.. code:: sh

  mkdir build
  cd build
  cmake -DLLVM_DIR=/srv/git/llvm.git/build ..
  make

Examples that require clang will only be built if clang is built as part of llvm.

Use cases
**********
**Assembling to code object with llvm-mc from command line**

The following llvm-mc command line produces ELF object asm.o from assembly source asm.s:

.. code:: sh

  llvm-mc -arch=amdgcn -mcpu=fiji -filetype=obj -o asm.o asm.s

**Assembling to raw instruction stream with llvm-mc from command line**

It is possible to extract contents of .text section after assembling to code object:

.. code:: sh

  llvm-mc -arch=amdgcn -mcpu=fiji -filetype=obj -o asm.o asm.s
  objdump -h asm.o | grep .text | awk '{print "dd if='asm.o' of='asm' bs=1 count=$[0x" $3 "] skip=$[0x" $6 "]"}' | bash

**Disassembling code object from command line**

The following command line may be used to dump contents of code object:

.. code:: sh

  llvm-objdump -disassemble -mcpu=fiji asm.o

This includes text disassembly of .text section.

**Disassembling raw instruction stream from command line**

The following command line may be used to disassemble raw instruction stream (without ELF structure):

.. code:: sh

  hexdump -v -e '/1 "0x%02X "' asm | llvm-mc -arch=amdgcn -mcpu=fiji -disassemble

Here, hexdump is used to display contents of file in hexadecimal (0x.. form) which is then consumed by llvm-mc.

Assembling source into code object using LLVM API
**************************************************
Refer to examples/api/assemble.

Disassembling instruction stream using LLVM API
**************************************************
Refer to examples/api/disassemble.

**Using amdphdrs**

Note that normally standard lld and Code Object version 2 should be used which is closer to standard ELF format.

amdphdrs (now obsolete) is complimentary utility that can be used to produce AMDGPU Code Object version 1.
For example, given assembly source in asm.s, the following will assemble it and link using amdphdrs:

.. code:: sh

  llvm-mc -arch=amdgcn -mcpu=fiji -filetype=obj -o asm.o asm.s
  andphdrs asm.o asm.co

Differences between LLVM AMDGPU Assembler and AMD SP3 assembler
****************************************************************
**Macro support**

SP3 supports proprietary set of macros/tools. sp3_to_mc.pl script attempts to translate them into GAS syntax understood by llvm-mc.
flat_atomic_cmpswap instruction has 32-bit destination

LLVM AMDGPU:

::

  flat_atomic_cmpswap v7, v[9:10], v[7:8]

SP3:

::

  flat_atomic_cmpswap v[7:8], v[9:10], v[7:8]

Atomic instructions that return value should have glc flag explicitly

LLVM AMDGPU:

::

  flat_atomic_swap_x2 v[0:1], v[0:1], v[2:3] glc

SP3:

::

  flat_atomic_swap_x2 v[0:1], v[0:1], v[2:3]

References
***********
   *  `LLVM Use Guide for AMDGPU Back-End <http://llvm.org/docs/AMDGPUUsage.html>`_
   *  AMD ISA Documents
       *  `AMD GCN3 Instruction Set Architecture (2016) <http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf>`_
       *  `AMD_Southern_Islands_Instruction_Set_Architecture <https://developer.amd.com/wordpress/media/2012/12/AMD_Southern_Islands_Instruction_Set_Architecture.pdf>`_

rocprof
=======

1. Overview
***********

| The rocProf is a command line tool implemented on the top of
  rocProfiler and rocTracer APIs. Source code for rocProf may be found
  here: GitHub:
  `https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/bin/rocprof <https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/bin/rocprof>`__
| This command line tool is implemented as a script which is setting up
  the environment for attaching the profiler and then run the provided
  application command line. The tool uses two profiling plugins loaded
  by ROC runtime and based on rocProfiler and rocTracer for collecting
  metrics/counters, HW traces and runtime API/activity traces. The tool
  consumes an input XML or text file with counters list or trace
  parameters and provides output profiling data and statistics in
  various formats as text, CSV and JSON traces. Google Chrome tracing
  can be used to visualize the JSON traces with runtime API/activity
  timelines and per kernel counters data.

2. Profiling Modes
******************

‘rocprof’ can be used for GPU profiling using HW counters and
application tracing

2.1. GPU profiling
******************

GPU profiling is controlled with input file which defines a list of
metrics/counters and a profiling scope. An input file is provided using
option ‘-i ’. Output CSV file with a line per submitted kernel is
generated. Each line has kernel name, kernel parameters and counter
values. By option ‘—stats’ the kernel execution stats can be generated
in CSV format. Currently profiling has limitation of serializing
submitted kernels. An example of input file:

::

      # Perf counters group 1
      pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
      # Perf counters group 2
      pmc : TCC_HIT[0], TCC_MISS[0]
      # Filter by dispatches range, GPU index and kernel names
      # supported range formats: "3:9", "3:", "3"
      range: 1 : 4
      gpu: 0 1 2 3
      kernel: simple Pass1 simpleConvolutionPass2

An example of profiling command line for ‘MatrixTranspose’ application

::

   $ rocprof -i input.txt MatrixTranspose
   RPL: on '191018_011134' from '/…./rocprofiler_pkg' in '/…./MatrixTranspose'
   RPL: profiling '"./MatrixTranspose"'
   RPL: input file 'input.txt'
   RPL: output dir '/tmp/rpl_data_191018_011134_9695'
   RPL: result dir '/tmp/rpl_data_191018_011134_9695/input0_results_191018_011134'
   ROCProfiler: rc-file '/…./rpl_rc.xml'
   ROCProfiler: input from "/tmp/rpl_data_191018_011134_9695/input0.xml"
     gpu_index =
     kernel =
     range =
     4 metrics
       L2CacheHit, VFetchInsts, VWriteInsts, MemUnitStalled
     0 traces
   Device name Ellesmere [Radeon RX 470/480/570/570X/580/580X]
   PASSED!

   ROCPRofiler: 1 contexts collected, output directory /tmp/rpl_data_191018_011134_9695/input0_results_191018_011134
   RPL: '/…./MatrixTranspose/input.csv' is generated

**2.1.1. Counters and metrics**

There are two profiling features, metrics and traces. Hardware
performance counters are treated as the basic metrics and the formulas
can be defined for derived metrics. Counters and metrics can be
dynamically configured using XML configuration files with counters and
metrics tables:

 - Counters table entry, basic metric: counter name, block name, event
   id
 - Derived metrics table entry: metric name, an expression for
   calculation the metric from the counters

Metrics XML File Example:

::

   <gfx8>
       <metric name=L1_CYCLES_COUNTER block=L1 event=0 descr=”L1 cache cycles”></metric>
       <metric name=L1_MISS_COUNTER block=L1 event=33 descr=”L1 cache misses”></metric>
       . . .
   </gfx8>

   <gfx9>
       . . .
   </gfx9>

   <global>
     <metric
       name=L1_MISS_RATIO
       expr=L1_CYCLES_COUNT/L1_MISS_COUNTER
       descry=”L1 miss rate metric”
     ></metric>
   </global>

**2.1.1.1. Metrics query**

Available counters and metrics can be queried by options ‘—list-basic’
for counters and ‘—list-derived’ for derived metrics. The output for
counters indicates number of block instances and number of block counter
registers. The output for derived metrics prints the metrics
expressions. Examples:

::

   $ rocprof --list-basic
   RPL: on '191018_014450' from '/opt/rocm/rocprofiler' in '/…./MatrixTranspose'
   ROCProfiler: rc-file '/…./rpl_rc.xml'
   Basic HW counters:
     gpu-agent0 : GRBM_COUNT : Tie High - Count Number of Clocks
         block GRBM has 2 counters
     gpu-agent0 : GRBM_GUI_ACTIVE : The GUI is Active
         block GRBM has 2 counters
         . . .
     gpu-agent0 : TCC_HIT[0-15] : Number of cache hits.
         block TCC has 4 counters
     gpu-agent0 : TCC_MISS[0-15] : Number of cache misses. UC reads count as misses.
         block TCC has 4 counters
         . . .

   $ rocprof --list-derived
   RPL: on '191018_015911' from '/opt/rocm/rocprofiler' in '/home/evgeny/work/BUILD/0_MatrixTranspose'
   ROCProfiler: rc-file '/home/evgeny/rpl_rc.xml'
   Derived metrics:
     gpu-agent0 : TCC_HIT_sum : Number of cache hits. Sum over TCC instances.
         TCC_HIT_sum = sum(TCC_HIT,16)
     gpu-agent0 : TCC_MISS_sum : Number of cache misses. Sum over TCC instances.
         TCC_MISS_sum = sum(TCC_MISS,16)
     gpu-agent0 : TCC_MC_RDREQ_sum : Number of 32-byte reads. Sum over TCC instaces.
         TCC_MC_RDREQ_sum = sum(TCC_MC_RDREQ,16)
       . . .

**2.1.1.2. Metrics collecting**

Counters and metrics accumulated per kernel can be collected using input
file with a list of metrics, see an example in 2.1. Currently profiling
has limitation of serializing submitted kernels. The number of counters
which can be dumped by one run is limited by GPU HW by number of counter
registers per block. The number of counters can be different for
different blocks and can be queried, see 2.1.1.1.

**2.1.1.2.1. Blocks instancing**

GPU blocks are implemented as several identical instances. To dump
counters of specific instance square brackets can be used, see an
example in 2.1. The number of block instances can be queried, see
2.1.1.1.

**2.1.1.2.2. HW limitations**

The number of counters which can be dumped by one run is limited by GPU
HW by number of counter registers per block. The number of counters can
be different for different blocks and can be queried, see 2.1.1.1.

 - Metrics groups

To dump a list of metrics exceeding HW limitations the metrics list can
be split on groups. The tool supports automatic splitting on optimal
metric groups:

::

   $ rocprof -i input.txt ./MatrixTranspose
   RPL: on '191018_032645' from '/opt/rocm/rocprofiler' in '/…./MatrixTranspose'
   RPL: profiling './MatrixTranspose'
   RPL: input file 'input.txt'
   RPL: output dir '/tmp/rpl_data_191018_032645_12106'
   RPL: result dir '/tmp/rpl_data_191018_032645_12106/input0_results_191018_032645'
   ROCProfiler: rc-file '/…./rpl_rc.xml'
   ROCProfiler: input from "/tmp/rpl_data_191018_032645_12106/input0.xml"
     gpu_index =
     kernel =
     range =
     20 metrics
       Wavefronts, VALUInsts, SALUInsts, SFetchInsts, FlatVMemInsts, LDSInsts, FlatLDSInsts, GDSInsts, VALUUtilization, FetchSize, WriteSize, L2CacheHit, VWriteInsts, GPUBusy, VALUBusy, SALUBusy, MemUnitStalled, WriteUnitStalled, LDSBankConflict, MemUnitBusy
     0 traces
   Device name Ellesmere [Radeon RX 470/480/570/570X/580/580X]

   Input metrics out of HW limit. Proposed metrics group set:
    group1: L2CacheHit VWriteInsts MemUnitStalled WriteUnitStalled MemUnitBusy FetchSize FlatVMemInsts LDSInsts VALUInsts SALUInsts SFetchInsts FlatLDSInsts GPUBusy Wavefronts
    group2: WriteSize GDSInsts VALUUtilization VALUBusy SALUBusy LDSBankConflict

   ERROR: rocprofiler_open(), Construct(), Metrics list exceeds HW limits

   Aborted (core dumped)
   Error found, profiling aborted.

________________________________

 - Collecting with multiple runs

To collect several metric groups a full application replay is used by
defining several ‘pmc:’ lines in the input file, see 2.1.


2.2. Application tracing
************************

Supported application tracing includes runtime API and GPU activity
tracing’ Supported runtimes are: ROCr (HSA API) and HIP Supported GPU
activity: kernel execution, async memory copy, barrier packets. The
trace is generated in JSON format compatible with Chrome tracing. The
trace consists of several sections with timelines for API trace per
thread and GPU activity. The timelines events show event name and
parameters. Supported options: ‘—hsa-trace’, ‘—hip-trace’, ‘—sys-trace’,
where ‘sys trace’ is for HIP and HSA combined trace.

**2.2.1. HIP runtime trace**

The trace is generated by option ‘—hip-trace’ and includes HIP API
timelines and GPU activity at the runtime level.

**2.2.2. ROCr runtime trace**

The trace is generated by option ‘—hsa-trace’ and includes ROCr API
timelines and GPU activity at AQL queue level. Also, can provide
counters per kernel.

**2.2.3. KFD driver trace**

Is planned to include Thunk API trace and memory allocations/migration
tracing.

**2.2.4. Code annotation**

Support for application code annotation. Start/stop API is supported to
programmatically control the profiling. A ‘roctx’ library provides
annotation API. Annotation is visualized in JSON trace as a separate
"Markers and Ranges" timeline section.

**2.2.4.1. Start/stop API**

::

   // Tracing start API
   void roctracer_start();

   // Tracing stop API
   void roctracer_stop();

**2.2.4.2. rocTX basic markers API**

::

   // A marker created by given ASCII message
   void roctxMark(const char* message);

   // Returns the 0 based level of a nested range being started by given message associated to this range.
   // A negative value is returned on the error.
   int roctxRangePush(const char* message);

   // Marks the end of a nested range.
   // Returns the 0 based level the range.
   // A negative value is returned on the error.
   int roctxRangePop();

**2.3. Multiple GPUs profiling**

The profiler supports multiple GPU’s profiling and provide GPI id for
counters and kernels data in CSV output file. Also, GPU id is indicating
for respective GPU activity timeline in JSON trace.

3. Profiling control
********************

Profiling can be controlled by specifying a profiling scope, by
filtering trace events and specifying interesting time intervals.

3.1. Profiling scope
********************

Counters profiling scope can be specified by GPU id list, kernel name
substrings list and dispatch range. Supported range formats examples:
"3:9", "3:", "3". You can see an example of input file in 2.1.

3.2. Tracing control
********************

Tracing can be filtered by events names using profiler input file and by
enabling interesting time intervals by command line option.

**3.2.1. Filtering traced APIs**

A list of traced API names can be specified in profiler input file. An
example of input file line for ROCr runtime trace (HAS API): hsa:
hsa_queue_create hsa_amd_memory_pool_allocate

**3.2.2. Tracing time period**

Trace can be dumped periodically with initial delay, dumping period
length and rate:

::

   --trace-period <dealy:length:rate>

3.3. Concurrent kernels
***********************

Currently concurrent kernels profiling is not supported which is a
planned feature. Kernels are serialized.

3.4. Multi-processes profiling
******************************

Multi-processes profiling is not currently supported.

3.5. Errors logging
*******************

Profiler errors are logged to global logs:

::

   /tmp/aql_profile_log.txt
   /tmp/rocprofiler_log.txt
   /tmp/roctracer_log.txt

4. 3rd party visualization tools
********************************

‘rocprof’ is producing JSON trace compatible with Chrome Tracing, which
is an internal trace visualization tool in Google Chrome.

4.1. Chrome tracing
*******************

Good review can be found by the link:
`https://aras-p.info/blog/2017/01/23/Chrome-Tracing-as-Profiler-Frontend/ <https://aras-p.info/blog/2017/01/23/Chrome-Tracing-as-Profiler-Frontend/>`__

5. Command line options
***********************

The command line options can be printed with option ‘-h’:

::

   $ rocprof -h
   RPL: on '191018_023018' from '/opt/rocm/rocprofiler' in '/…./MatrixTranspose'
   ROCm Profiling Library (RPL) run script, a part of ROCprofiler library package.
   Full path: /opt/rocm/rocprofiler/bin/rocprof
   Metrics definition: /opt/rocm/rocprofiler/lib/metrics.xml

   Usage:
     rocprof [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

   Options:
     -h - this help
     --verbose - verbose mode, dumping all base counters used in the input metrics
     --list-basic - to print the list of basic HW counters
     --list-derived - to print the list of derived metrics with formulas

     -i <.txt|.xml file> - input file
         Input file .txt format, automatically rerun application for every pmc line:

           # Perf counters group 1
           pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts VALUUtilization FetchSize
           # Perf counters group 2
           pmc : WriteSize L2CacheHit
           # Filter by dispatches range, GPU index and kernel names
           # supported range formats: "3:9", "3:", "3"
           range: 1 : 4
           gpu: 0 1 2 3
           kernel: simple Pass1 simpleConvolutionPass2

         Input file .xml format, for single profiling run:

           # Metrics list definition, also the form "<block-name>:<event-id>" can be used
           # All defined metrics can be found in the 'metrics.xml'
           # There are basic metrics for raw HW counters and high-level metrics for derived counters
           <metric name=SQ:4,SQ_WAVES,VFetchInsts
           ></metric>

           # Filter by dispatches range, GPU index and kernel names
           <metric
             # range formats: "3:9", "3:", "3"
             range=""
             # list of gpu indexes "0,1,2,3"
             gpu_index=""
             # list of matched sub-strings "Simple1,Conv1,SimpleConvolution"
             kernel=""
           ></metric>

     -o <output file> - output CSV file [<input file base>.csv]
     -d <data directory> - directory where profiler store profiling data including traces [/tmp]
         The data directory is renoving autonatically if the directory is matching the temporary one, which is the default.
     -t <temporary directory> - to change the temporary directory [/tmp]
         By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

     --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
     --timestamp <on|off> - to turn on/off the kernel disoatches timestamps, dispatch/begin/end/complete [off]
     --ctx-wait <on|off> - to wait for outstanding contexts on profiler exit [on]
     --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
     --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]

     --stats - generating kernel execution stats, file <output name>.stats.csv
     --hsa-trace - to trace HSA, generates API execution stats and JSON file chrome-tracing compatible
     --hip-trace - to trace HIP, generates API execution stats and JSON file chrome-tracing compatible
     --sys-trace - to trace HIP/HSA APIs and GPU activity, generates stats and JSON trace chrome-tracing compatible
       Generated files: <output name>.hsa_stats.txt <output name>.json
       Traced API list can be set by input .txt or .xml files.
       Input .txt:
         hsa: hsa_queue_create hsa_amd_memory_pool_allocate
       Input .xml:
         <trace name="HSA">
           <parameters list="hsa_queue_create, hsa_amd_memory_pool_allocate">
           </parameters>
         </trace>

     --trace-period <dealy:length:rate> - to enable trace with initial delay, with periodic sample length and rate
       Supported time formats: <number(m|s|ms|us)>

   Configuration file:
     You can set your parameters defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:/home/evgeny:<package path>
     First the configuration file is looking in the current directory, then in your home, and then in the package directory.
     Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat'.
     An example of 'rpl_rc.xml':
       <defaults
         basenames=off
         timestamp=off
         ctx-limit=0
         heartbeat=0
       ></defaults> 

6. Publicly available counters and metrics
******************************************

The following counters are publicly available for commercially available
VEGA10/20 GPUs.

Counters:

::

   •   GRBM_COUNT : Tie High - Count Number of Clocks
   •   GRBM_GUI_ACTIVE : The GUI is Active
   •   SQ_WAVES : Count number of waves sent to SQs. (per-simd, emulated, global)
   •   SQ_INSTS_VALU : Number of VALU instructions issued. (per-simd, emulated)
   •   SQ_INSTS_VMEM_WR : Number of VMEM write instructions issued (including FLAT). (per-simd, emulated)
   •   SQ_INSTS_VMEM_RD : Number of VMEM read instructions issued (including FLAT). (per-simd, emulated)
   •   SQ_INSTS_SALU : Number of SALU instructions issued. (per-simd, emulated)
   •   SQ_INSTS_SMEM : Number of SMEM instructions issued. (per-simd, emulated)
   •   SQ_INSTS_FLAT : Number of FLAT instructions issued. (per-simd, emulated)
   •   SQ_INSTS_FLAT_LDS_ONLY : Number of FLAT instructions issued that read/wrote only from/to LDS (only works if EARLY_TA_DONE is enabled). (per-simd, emulated)
   •   SQ_INSTS_LDS : Number of LDS instructions issued (including FLAT). (per-simd, emulated)
   •   SQ_INSTS_GDS : Number of GDS instructions issued. (per-simd, emulated)
   •   SQ_WAIT_INST_LDS : Number of wave-cycles spent waiting for LDS instruction issue. In units of 4 cycles. (per-simd, nondeterministic)
   •   SQ_ACTIVE_INST_VALU : regspec 71? Number of cycles the SQ instruction arbiter is working on a VALU instruction. (per-simd, nondeterministic)
   •   SQ_INST_CYCLES_SALU : Number of cycles needed to execute non-memory read scalar operations. (per-simd, emulated)
   •   SQ_THREAD_CYCLES_VALU : Number of thread-cycles used to execute VALU operations (similar to INST_CYCLES_VALU but multiplied by # of active threads). (per-simd)
   •   SQ_LDS_BANK_CONFLICT : Number of cycles LDS is stalled by bank conflicts. (emulated)
   •   TA_TA_BUSY[0-15] : TA block is busy. Perf_Windowing not supported for this counter.
   •   TA_FLAT_READ_WAVEFRONTS[0-15] : Number of flat opcode reads processed by the TA.
   •   TA_FLAT_WRITE_WAVEFRONTS[0-15] : Number of flat opcode writes processed by the TA.
   •   TCC_HIT[0-15] : Number of cache hits.
   •   TCC_MISS[0-15] : Number of cache misses. UC reads count as misses.
   •   TCC_EA_WRREQ[0-15] : Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Atomics may travel over the same interface and are generally classified as write requests. This does not include probe commands.
   •   TCC_EA_WRREQ_64B[0-15] : Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface.
   •   TCC_EA_WRREQ_STALL[0-15] : Number of cycles a write request was stalled.
   •   TCC_EA_RDREQ[0-15] : Number of TCC/EA read requests (either 32-byte or 64-byte)
   •   TCC_EA_RDREQ_32B[0-15] : Number of 32-byte TCC/EA read requests
   •   TCP_TCP_TA_DATA_STALL_CYCLES[0-15] : TCP stalls TA data interface. Now Windowed.

The following derived metrics have been defined and the profiler metrics
XML specification can be found at:
`https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/test/tool/metrics.xml <https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/test/tool/metrics.xml>`__.

Metrics:

::

   •   TA_BUSY_avr : TA block is busy. Average over TA instances.
   •   TA_BUSY_max : TA block is busy. Max over TA instances.
   •   TA_BUSY_min : TA block is busy. Min over TA instances.
   •   TA_FLAT_READ_WAVEFRONTS_sum : Number of flat opcode reads processed by the TA. Sum over TA instances.
   •   TA_FLAT_WRITE_WAVEFRONTS_sum : Number of flat opcode writes processed by the TA. Sum over TA instances.
   •   TCC_HIT_sum : Number of cache hits. Sum over TCC instances.
   •   TCC_MISS_sum : Number of cache misses. Sum over TCC instances.
   •   TCC_EA_RDREQ_32B_sum : Number of 32-byte TCC/EA read requests. Sum over TCC instances.
   •   TCC_EA_RDREQ_sum : Number of TCC/EA read requests (either 32-byte or 64-byte). Sum over TCC instances.
   •   TCC_EA_WRREQ_sum : Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. Sum over TCC instances.
   •   TCC_EA_WRREQ_64B_sum : Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq interface. Sum over TCC instances.
   •   TCC_WRREQ_STALL_max : Number of cycles a write request was stalled. Max over TCC instances.
   •   TCC_MC_WRREQ_sum : Number of 32-byte effective writes. Sum over TCC instaces.
   •   FETCH_SIZE : The total kilobytes fetched from the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
   •   WRITE_SIZE : The total kilobytes written to the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
   •   GPUBusy : The percentage of time GPU was busy.
   •   Wavefronts : Total wavefronts.
   •   VALUInsts : The average number of vector ALU instructions executed per work-item (affected by flow control).
   •   SALUInsts : The average number of scalar ALU instructions executed per work-item (affected by flow control).
   •   VFetchInsts : The average number of vector fetch instructions from the video memory executed per work-item (affected by flow control). Excludes FLAT instructions that fetch from video memory.
   •   SFetchInsts : The average number of scalar fetch instructions from the video memory executed per work-item (affected by flow control).
   •   VWriteInsts : The average number of vector write instructions to the video memory executed per work-item (affected by flow control). Excludes FLAT instructions that write to video memory.
   •   FlatVMemInsts : The average number of FLAT instructions that read from or write to the video memory executed per work item (affected by flow control). Includes FLAT instructions that read from or write to scratch.
   •   LDSInsts : The average number of LDS read or LDS write instructions executed per work item (affected by flow control).  Excludes FLAT instructions that read from or write to LDS.
   •   FlatLDSInsts : The average number of FLAT instructions that read or write to LDS executed per work item (affected by flow control).
   •   GDSInsts : The average number of GDS read or GDS write instructions executed per work item (affected by flow control).
   •   VALUUtilization : The percentage of active vector ALU threads in a wave. A lower number can mean either more thread divergence in a wave or that the work-group size is not a multiple of 64. Value range: 0% (bad), 100% (ideal - no thread divergence).
   •   VALUBusy : The percentage of GPUTime vector ALU instructions are processed. Value range: 0% (bad) to 100% (optimal).
   •   SALUBusy : The percentage of GPUTime scalar ALU instructions are processed. Value range: 0% (bad) to 100% (optimal).
   •   Mem32Bwrites :
   •   FetchSize : The total kilobytes fetched from the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
   •   WriteSize : The total kilobytes written to the video memory. This is measured with all extra fetches and any cache or memory effects taken into account.
   •   L2CacheHit : The percentage of fetch, write, atomic, and other instructions that hit the data in L2 cache. Value range: 0% (no hit) to 100% (optimal).
   •   MemUnitBusy : The percentage of GPUTime the memory unit is active. The result includes the stall time (MemUnitStalled). This is measured with all extra fetches and writes and any cache or memory effects taken into account. Value range: 0% to 100% (fetch-bound).
   •   MemUnitStalled : The percentage of GPUTime the memory unit is stalled. Try reducing the number or size of fetches and writes if possible. Value range: 0% (optimal) to 100% (bad).
   •   WriteUnitStalled : The percentage of GPUTime the Write unit is stalled. Value range: 0% to 100% (bad).
   •   ALUStalledByLDS : The percentage of GPUTime ALU units are stalled by the LDS input queue being full or the output queue being not ready. If there are LDS bank conflicts, reduce them. Otherwise, try reducing the number of LDS accesses if possible. Value range: 0% (optimal) to 100% (bad).
   •   LDSBankConflict : The percentage of GPUTime LDS is stalled by bank conflicts. Value range: 0% (optimal) to 100% (bad).


ROC Profiler
============

ROC profiler library. Profiling with perf-counters and derived metrics. Library supports GFX8/GFX9.

HW specific low-level performance analysis interface for profiling of GPU compute applications. The profiling includes HW performance counters with complex performance metrics.

**Metrics**

The link to profiler default metrics XML `specification <https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/test/tool/metrics.xml>`_.

**Download**

To clone ROC Profiler from GitHub use the folowing command:

.. code:: sh

  git clone https://github.com/ROCm-Developer-Tools/rocprofiler

The library source tree:

   *  bin
       *  rocprof - Profiling tool run script
   *  doc - Documentation
   *  inc/rocprofiler.h - Library public API
   *  src - Library sources
       *  core - Library API sources
       *  util - Library utils sources
       *  xml - XML parser
   *  test - Library test suite
       *  tool - Profiling tool
           *  tool.cpp - tool sources
           *  metrics.xml - metrics config file
       *  ctrl - Test controll
       *  util - Test utils
       *  simple_convolution - Simple convolution test kernel

**Build**


Build environment:

.. code:: sh

  export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
  export CMAKE_BUILD_TYPE=<debug|release> # release by default
  export CMAKE_DEBUG_TRACE=1 # to enable debug tracing


**To Build with the current installed ROCm:**


.. code:: sh

  To build and install to /opt/rocm/rocprofiler
  export CMAKE_PREFIX_PATH=/opt/rocm/include/hsa:/opt/rocm
  cd ../rocprofiler
  mkdir build
  cd build
  cmake ..
  make
  make install

**Internal 'simple_convolution' test run script:**

.. code:: sh

  cd ../rocprofiler/build
  ./run.sh

**To enable error messages logging to '/tmp/rocprofiler_log.txt':**


.. code:: sh

  export ROCPROFILER_LOG=1

**To enable verbose tracing:**

.. code:: sh

  export ROCPROFILER_TRACE=1


**Profiling utility usage**

.. code:: sh

  rocprof [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

  Options:
  -h - this help
  --verbose - verbose mode, dumping all base counters used in the input metrics
  --list-basic - to print the list of basic HW counters
  --list-derived - to print the list of derived metrics with formulas

  -i <.txt|.xml file> - input file
      Input file .txt format, automatically rerun application for every pmc line:

        # Perf counters group 1
        pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts FetchSize
        # Perf counters group 2
        pmc : VALUUtilization,WriteSize L2CacheHit
        # Filter by dispatches range, GPU index and kernel names
        # supported range formats: "3:9", "3:", "3"
        range: 1 : 4
        gpu: 0 1 2 3
        kernel: simple Pass1 simpleConvolutionPass2

      Input file .xml format, for single profiling run:

        # Metrics list definition, also the form "<block-name>:<event-id>" can be used
        # All defined metrics can be found in the 'metrics.xml'
        # There are basic metrics for raw HW counters and high-level metrics for derived counters
        <metric name=SQ:4,SQ_WAVES,VFetchInsts
        ></metric>

        # Filter by dispatches range, GPU index and kernel names
        <metric
          # range formats: "3:9", "3:", "3"
          range=""
          # list of gpu indexes "0,1,2,3"
          gpu_index=""
          # list of matched sub-strings "Simple1,Conv1,SimpleConvolution"
          kernel=""
        ></metric>

  -o <output file> - output CSV file [<input file base>.csv]
    The output CSV file columns meaning in the columns order:
      Index - kernels dispatch order index
      KernelName - the dispatched kernel name
      gpu-id - GPU id the kernel was submitted to
      queue-id - the ROCm queue unique id the kernel was submitted to
      queue-index - The ROCm queue write index for the submitted AQL packet
      tid - system application thread id which submitted the kernel
      grd - the kernel's grid size
      wgr - the kernel's work group size
      lds - the kernel's LDS memory size
      scr - the kernel's scratch memory size
      vgpr - the kernel's VGPR size
      sgpr - the kernel's SGPR size
      fbar - the kernel's barriers limitation
      sig - the kernel's completion signal
      ... - The columns with the counters values per kernel dispatch
      DispatchNs/BeginNs/EndNs/CompleteNs - timestamp columns if time-stamping was enabled
      
  -d <data directory> - directory where profiler store profiling data including thread treaces [/tmp]
      The data directory is renoving autonatically if the directory is matching the temporary one, which is the default.
  -t <temporary directory> - to change the temporary directory [/tmp]
      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
  --timestamp <on|off> - to turn on/off the kernel dispatches timestamps, dispatch/begin/end/complete [off]
    Four kernel timestamps in nanoseconds are reported:
        DispatchNs - the time when the kernel AQL dispatch packet was written to the queue
        BeginNs - the kernel execution begin time
        EndNs - the kernel execution end time
        CompleteNs - the time when the completion signal of the AQL dispatch packet was received

  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]

  --stats - generating kernel executino stats, file <output name>.stats.csv
  --hip-trace - to trace HIP, generates API execution stats/trace and JSON file viewable in chrome tracing
    'HCC_HOME' env var is required to be set to where 'hcc' is installed.
  --hsa-trace - to trace HSA, generates API execution stats/trace and JSON file viewable in chrome tracing
    Generated files: <output name>.hsa_stats.txt <output name>.json
    Traced API list can be set by input .txt or .xml files.
    Input .txt:
      hsa: hsa_queue_create hsa_amd_memory_pool_allocate
    Input .xml:
      <trace name="HSA">
        <parameters api="hsa_queue_create, hsa_amd_memory_pool_allocate">
        </parameters>
      </trace>

  Configuration file:
  You can set your parameters defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:/home/      evgeny:<package path>
  First the configuration file is looking in the current directory, then in your home, and then in the package directory.
  Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat'.
  An example of 'rpl_rc.xml':
    <defaults
      basenames=off
      timestamp=off
      ctx-limit=0
      heartbeat=0
    ></defaults>



ROC Tracer
============

ROC-tracer library, Runtimes Generic Callback/Activity APIs.
The goal of the implementation is to provide a generic independent from
specific runtime profiler to trace API and asyncronous activity.

The API provides functionality for registering the runtimes API callbacks and
asyncronous activity records pool support.

The library source tree:

    *  inc/roctracer.h - Library public API
    *  src - Library sources
        *  core - Library API sources
        *  util - Library utils sources
    *  test - test suit
        *  MatrixTranspose - test based on HIP MatrixTranspose sample

**Documentation**

.. code:: sh

  - API declaration: inc/roctracer.h
  - Code example: test/MatrixTranspose/MatrixTranspose.cpp

**To build and run test**


.. code:: sh
  
  - ROCm-2.3 or higher is required

  - Python2.7 is required.
    The required modules: CppHeaderParser, argparse.
    To install:
    sudo pip install CppHeaderParser argparse

  - CLone development branches of roctracer:
    git clone -b amd-master https://github.com/ROCm-Developer-Tools/roctracer

  - To customize environment, below are defaults:
   export HIP_PATH=/opt/rocm/HIP
   export HCC_HOME=/opt/rocm/hcc/
   export CMAKE_PREFIX_PATH=/opt/rocm

  - Build ROCtracer
   export CMAKE_BUILD_TYPE=<debug|release> # release by default
   cd <your path>/roctracer && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm .. && make -j <nproc>

  - To build and run test
   make mytest
   run.sh
  
  - To install
   make install
   or
   make package && dpkg -i *.deb


AOMP - V 0.7-5
================

Overview
**********

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on AMD GPUs. Since AOMP is a clang/llvm compiler, it also supports GPU offloading with HIP, CUDA, and OpenCL.

Some sources to support OpenMP target offload on AMD GPUs have not yet been merged into the upstream LLVM trunk. However all sources used by AOMP are available in `AOMP repositories <https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md#repositories>`_. One of those repositories is a `mirror of the LLVM monorepo llvm-project <https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md#repositories>`_ with a set of commits applied to a stable LLVM release branch.

The bin directory of this repository contains a README.md and build scripts needed to download, build, and install AOMP from source. In addition to the mirrored `LLVM project repository <https://github.com/ROCm-Developer-Tools/llvm-project>`_, AOMP uses a number of open-source ROCm components. The build scripts will download, build, and install all components needed for AOMP. However, we recommend that you install the latest release of the `debian or rpm package <https://github.com/ROCm-Developer-Tools/aomp/releases>`_ for AOMP described in the install section.

AOMP Install
**************

Platform Install Options:

    * Ubuntu or Debian
    * SUSE SLES-15-SP1
    * RHEL 7
    * Install Without Root
    * Build and Install from release source tarball
    * Development Source Build and Install


AOMP Debian/Ubuntu Install
----------------------------

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.

On Ubuntu 18.04 LTS (bionic beaver), run these commands:

::
  
  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1804_0.7-5_amd64.deb
  sudo dpkg -i aomp_Ubuntu1804_0.7-5_amd64.deb


On Ubuntu 16.04, run these commands:

::

  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1604_0.7-5_amd64.deb
  sudo dpkg -i aomp_Ubuntu1604_0.7-5_amd64.deb

The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

Prerequisites
----------------

**AMD KFD Driver**

These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found `HERE <https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository>`_.

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  sudo apt install rock-dkms

  sudo reboot
  sudo usermod -a -G video $USER



**NVIDIA CUDA Driver**

If you build AOMP with support for nvptx GPUs, you must first install CUDA 10. Note these instructions reference the install for Ubuntu 16.04.

**Download Instructions for CUDA (Ubuntu 16.04)**

    Go to https://developer.nvidia.com/cuda-10.0-download-archive
    For Ubuntu 16.04, select Linux®, x86_64, Ubuntu, 16.04, deb(local) and then click Download. Note you can change these options for your specific distribution type.
    Navigate to the debian in your Linux® directory and run the following commands:

::

   sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
   sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda


Depending on your system the CUDA install could take a very long time.

AOMP SUSE SLES-15-SP1 Install
-------------------------------

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.

::

  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_SLES15_SP1-0.7-5.x86_64.rpm
  sudo rpm -i aomp_SLES15_SP1-0.7-5.x86_64.rpm


Confirm AOMP environment variable is set:

::

  echo $AOMP


**Prerequisites**

The ROCm kernel driver is required for AMD GPU support. Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

**AMD KFD DRIVER**

**Important Note:** There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:

::

  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu


**Set Group Access**

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

**NVIDIA CUDA Driver**

If you build AOMP with support for nvptx GPUs, you must first install CUDA 10.

Download Instructions for CUDA (SLES15)

    Go to https://developer.nvidia.com/cuda-10.0-download-archive
    For SLES-15, select Linux®, x86_64, SLES, 15.0, rpm(local) and then click Download.
    Navigate to the rpm in your Linux® directory and run the following commands:

::

  sudo rpm -i cuda-repo-sles15-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo zypper refresh
  sudo zypper install cuda


If prompted, select the 'always trust key' option. Depending on your system the CUDA install could take a very long time.

**Important Note:** If using a GUI on SLES-15-SP1, such as gnome, the installation of CUDA may cause the GUI to fail to load. This seems to be caused by a symbolic link pointing to nvidia-libglx.so instead of xorg-libglx.so. This can be fixed by updating the symbolic link:

::

  sudo rm /etc/alternatives/libglx.so
  sudo ln -s /usr/lib64/xorg/modules/extensions/xorg/xorg-libglx.so /etc/alternatives/libglx.so


AOMP RHEL 7 Install
---------------------

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.

**The installation may need the following dependency:**

::

  sudo yum install perl-Digest-MD5


**Download and Install**

::

  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_REDHAT_7-0.7-5.x86_64.rpm
  sudo rpm -i aomp_REDHAT_7-0.7-5.x86_64.rpm

If CUDA is not installed the installation may cancel, to bypass this:

::

  sudo rpm -i --nodeps aomp_REDHAT_7-0.7-5.x86_64.rpm

Confirm AOMP environment variable is set:

::

  echo $AOMP


**Prerequisites**

The ROCm kernel driver is required for AMD GPU support. Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

**AMD KFD Driver**

::

  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm


**Install and setup Devtoolset-7**

Devtoolset-7 is recommended, follow instructions 1-3 here:
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD. https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

**Install dkms tool**

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0

**Install rock-dkms**

::

  sudo yum install rock-dkms


**Set Group Access**

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

**NVIDIA CUDA Driver**

To build AOMP with support for nvptx GPUs, you must first install CUDA 10. We recommend CUDA 10.0. CUDA 10.1 will not work until AOMP moves to the trunk development of LLVM 9. The CUDA installation is now optional.

**Download Instructions for CUDA (CentOS/RHEL 7)**

    * Go to https://developer.nvidia.com/cuda-10.0-download-archive
    * For SLES-15, select Linux®, x86_64, RHEL or CentOS, 7, rpm(local) and then click Download.
    * Navigate to the rpm in your Linux® directory and run the following commands:

::

  sudo rpm -i cuda-repo-rhel7-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo yum clean all
  sudo yum install cuda

Install Without Root
----------------------

By default, the packages install their content to the release directory /usr/lib/aomp_0.X-Y and then a symbolic link is created at /usr/lib/aomp to the release directory. This requires root access.

Once installed go to `TESTINSTALL <https://github.com/ROCm-Developer-Tools/aomp/blob/master/docs/TESTINSTALL.md>`_ for instructions on getting started with AOMP examples.

**Debian**

To install the debian package without root access into your home directory, you can run these commands.
On Ubuntu 18.04 LTS (bionic beaver):

::

   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1804_0.7-5_amd64.deb
   dpkg -x aomp_Ubuntu1804_0.7-5_amd64.deb /tmp/temproot


On Ubuntu 16.04:

::

   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1604_0.7-5_amd64.deb
   dpkg -x aomp_Ubuntu1604_0.7-5_amd64.deb /tmp/temproot

::

   mv /tmp/temproot/usr $HOME
   export PATH=$PATH:$HOME/usr/lib/aomp/bin
   export AOMP=$HOME/usr/lib/aomp

The last two commands could be put into your .bash_profile file so you can always access the compiler.

**RPM**

To install the rpm package without root access into your home directory, you can run these commands.

::

   mkdir /tmp/temproot ; cd /tmp/temproot 
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_SLES15_SP1-0.7-5.x86_64.rpm
   rpm2cpio aomp_SLES15_SP1-0.7-5.x86_64.rpm | cpio -idmv
   mv /tmp/temproot/usr/lib $HOME
   export PATH=$PATH:$HOME/rocm/aomp/bin
   export AOMP=$HOME/rocm/aomp

The last two commands could be put into your .bash_profile file so you can always access the compiler.

Build and Install From Release Source Tarball
------------------------------------------------

The AOMP build and install from the release source tarball can be done manually or with spack. Building from source requires a number of platform dependencies. These dependencies are not yet provided with the spack configuration file. So if you are building from source either manually or building with spack, you must install the prerequisites for the platforms listed below.

**Source Build Prerequisites**

To build AOMP from source you must: 1. install certain distribution packages, 2. ensure the KFD kernel module is installed and operating, 3. create the Unix video group, and 4. install spack if required.

**1. Required Distribution Packages**

**Debian or Ubuntu Packages**

::

   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk


**SLES-15-SP1 Packages**

::

  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel


**RHEL 7 Packages**

Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD. https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

**The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.**

::

  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel


The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin

::

  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake'


**2. Verify KFD Driver**

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.

**Debian or Ubuntu Support**

These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found `HERE <https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository>`_.

::

  wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  sudo apt install rock-dkms


**SUSE SLES-15-SP1 Support**

**Important Note:** There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:

::

  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu


**RHEL 7 Support**

::

  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm


**Install dkms tool**

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`


Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0


**Install rock-dkms**

::

  sudo yum install rock-dkms

3. Create the Unix Video Group

Regardless of Linux distribution, you must create a video group to contain the users authorized to use the GPU.

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

4. Install spack

To use spack to build and install from the release source tarball, you must install spack first. Please refer to these `install instructions for instructions <https://spack.readthedocs.io/en/latest/getting_started.html#installation>`_ on installing spack. Remember,the aomp spack configuration file is currently missing dependencies, so be sure to install the packages listed above before proceeding.

**Build AOMP manually from release source tarball**

To build and install aomp from the release source tarball run these commands:

::

   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp-0.7-5.tar.gz
   tar -xzf aomp-0.7-5.tar.gz
   cd aomp
   nohup make &

Depending on your system, the last command could take a very long time. So it is recommended to use nohup and background the process. The simple Makefile that make will use runs build script "build_aomp.sh" and sets some flags to avoid git checks and applying ROCm patches. Here is that Makefile:

::

  AOMP ?= /usr/local/aomp
  AOMP_REPOS = $(shell pwd)
  all:
        AOMP=$(AOMP) AOMP_REPOS=$(AOMP_REPOS) AOMP_CHECK_GIT_BRANCH=0 AOMP_APPLY_ROCM_PATCHES=0 $(AOMP_REPOS)/aomp/bin/build_aomp.sh


If you set the environment variable AOMP, the Makefile will install to that directory. Otherwise, the Makefile will install into /usr/local. So you must have authorization to write into /usr/local if you do not set the environment variable AOMP. Let's assume you set the environment variable AOMP to "$HOME/rocm/aomp" in .bash_profile. The build_aomp.sh script will install into $HOME/rocm/aomp_0.7-5 and create a symbolic link from $HOME/rocm/aomp to $HOME/rocm/aomp_0.7-5. This feature allows multiple versions of AOMP to be installed concurrently. To enable a backlevel version of AOMP, simply set AOMP to $HOME/rocm/aomp_0.7-4.

**Build AOMP with spack**

Assuming your have installed the prerequisites listed above, use these commands to fetch the source and build aomp. Currently the aomp configuration is not yet in the spack git hub so you must create the spack package first.

::

   wget https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/package.py
   spack create -n aomp -t makefile --force https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp-0.7-5.tar.gz
   spack edit aomp
   spack install aomp


The "spack create" command will download and start an editor of a newly created spack config file. With the exception of the sha256 value, copy the contents of the downloaded package.py file into into the spack configuration file. You may restart this editor with the command "spack edit aomp"

Depending on your system, the "spack install aomp" command could take a very long time. Unless you set the AOMP environment variable, AOMP will be installed in /usr/local/aomp_ with a symbolic link from /usr/local/aomp to /usr/local/aomp_. Be sure you have write access to /usr/local or set AOMP to a location where you have write access.


Source Install V 0.7-6 (DEV)
--------------------------------

Build and install from sources is possible. However, the source build for AOMP is complex for several reasons.

    * Many repos are required. The clone_aomp.sh script ensures you have all repos and the correct branch.
    * Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
    * It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
    * Additional package dependencies are required that are not required when installing the AOMP package.

**Source Build Prerequisites**

**1. Required Distribution Packages**

**Debian or Ubuntu Packages**

::

   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk

**SLES-15-SP1 Packages**

::

  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel


**RHEL 7 Packages**

Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD. https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

**The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.**

::

  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel


The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin

::

  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake


**2. Verify KFD Driver**

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.

**Debian or Ubuntu Support**

These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found `HERE <https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository>`_.

::

  wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  sudo apt install rock-dkms


**SUSE SLES-15-SP1 Support**

**Important Note:** There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:

::

  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu

**RHEL 7 Support**

::

  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

**Install dkms tool**

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0

**Install rock-dkms**

::

  sudo yum install rock-dkms

**3. Create the Unix Video Group**

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

**Clone and Build AOMP**

::

   cd $HOME ; mkdir -p git/aomp ; cd git/aomp
   git clone https://github.com/rocm-developer-tools/aomp
   cd $HOME/git/aomp/aomp/bin

**Choose a Build Version (Development or Release)** The development version is the next version to be released. It is possible that the development version is broken due to regressions that often occur during development. If instead, you want to build from the sources of a previous release such as 0.7-5 that is possible as well.

**For the Development Branch:**

   git checkout master
   git pull

For the Release Branch:

   git checkout rel_0.7-5
   git pull

Clone and Build:

   ./clone_aomp.sh
   ./build_aomp.sh

Depending on your system, the last two commands could take a very long time. For more information, please refer :ref:`AOMP developers README`.

You only need to do the checkout/pull in the AOMP repository. The file "bin/aomp_common_vars" lists the branches of each repository for a particular AOMP release. In the master branch of AOMP, aomp_common_vars lists the development branches. It is a good idea to run clone_aomp.sh twice after you checkout a release to be sure you pulled all the checkouts for a particular release.

For more information on Release Packages, click `here <https://github.com/ROCm-Developer-Tools/aomp/releases>`_

Test Install
*************

**Getting Started**

The default install location is /usr/lib/aomp. To run the given examples, for example in /usr/lib/aomp/examples/openmp do the following:

**Copy the example openmp directory somewhere writable**

::

  cd /usr/lib/aomp/examples/
  cp -rp openmp /work/tmp/openmp-examples
  cd /work/tmp/openmp-examples/vmulsum

**Point to the installed AOMP by setting AOMP environment variable**

::

  export AOMP=/usr/lib/aomp

**Make Instructions**

::

  make clean
  make run


Run 'make help' for more details.

View the OpenMP Examples `README <https://github.com/ROCm-Developer-Tools/aomp/blob/master/examples/openmp>`_ for more information.

AOMP Limitations
*****************

See the `release notes <https://github.com/ROCm-Developer-Tools/aomp/releases>`_ in github. Here are some limitations.

 - Dwarf debugging is turned off for GPUs. -g will turn on host level debugging only.
 - Some simd constructs fail to vectorize on both host and GPUs.  

ROCmValidationSuite
=====================

The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

ROCmValidationSuite Modules
******************************

**GPU Properties – GPUP**

The GPU Properties module queries the configuration of a target device and returns the device’s static characteristics. These static values can be used to debug issues such as device support, performance and firmware problems.

**GPU Monitor – GM module**

The GPU monitor tool is capable of running on one, some or all of the GPU(s) installed and will report various information at regular intervals. The module can be configured to halt another RVS modules execution if one of the quantities exceeds a specified boundary value.

**PCI Express State Monitor – PESM module?**

The PCIe State Monitor tool is used to actively monitor the PCIe interconnect between the host platform and the GPU. The module will register a “listener” on a target GPU’s PCIe interconnect, and log a message whenever it detects a state change. The PESM will be able to detect the following state changes:

    * PCIe link speed changes
    * GPU power state changes

**ROCm Configuration Qualification Tool - RCQT module**

The ROCm Configuration Qualification Tool ensures the platform is capable of running ROCm applications and is configured correctly. It checks the installed versions of the ROCm components and the platform configuration of the system. This includes checking that dependencies, corresponding to the associated operating system and runtime environment, are installed correctly. Other qualification steps include checking:

    * The existence of the /dev/kfd device
    * The /dev/kfd device’s permissions
    * The existence of all required users and groups that support ROCm
    * That the user mode components are compatible with the drivers, both the KFD and the amdgpu driver.
    * The configuration of the runtime linker/loader qualifying that all ROCm libraries are in the correct search path.

**PCI Express Qualification Tool – PEQT module**

The PCIe Qualification Tool consists is used to qualify the PCIe bus on which the GPU is connected. The qualification test will be capable of determining the following characteristics of the PCIe bus interconnect to a GPU:

    * Support for Gen 3 atomic completers
    * DMA transfer statistics
    * PCIe link speed
    * PCIe link width

**SBIOS Mapping Qualification Tool – SMQT module**

The GPU SBIOS mapping qualification tool is designed to verify that a platform’s SBIOS has satisfied the BAR mapping requirements for VDI and Radeon Instinct products for ROCm support.

Refer to the “ROCm Use of Advanced PCIe Features and Overview of How BAR Memory is Used In ROCm Enabled System” web page for more information about how BAR memory is initialized by VDI and Radeon products.

**P2P Benchmark and Qualification Tool – PBQT module**

The P2P Benchmark and Qualification Tool is designed to provide the list of all GPUs that support P2P and characterize the P2P links between peers. In addition to testing for P2P compatibility, this test will perform a peer-to-peer throughput test between all P2P pairs for performance evaluation. The P2P Benchmark and Qualification Tool will allow users to pick a collection of two or more GPUs on which to run. The user will also be able to select whether or not they want to run the throughput test on each of the pairs.

Please see the web page “ROCm, a New Era in Open GPU Computing” to find out more about the P2P solutions available in a ROCm environment.

**PCI Express Bandwidth Benchmark – PEBB module**

The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA transfers between system memory and a target GPU card’s memory. The maximum bandwidth obtained is reported to help debug low bandwidth issues. The benchmark should be capable of targeting one, some or all of the GPUs installed in a platform, reporting individual benchmark statistics for each.

**GPU Stress Test - GST module**

The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM (Single/Double-precision General Matrix Multiplication) workload on one, some or all GPUs. The GPUs can be of the same or different types. The duration of the benchmark should be configurable, both in terms of time (how long to run) and iterations (how many times to run).

The test should be capable driving the power level equivalent to the rated TDP of the card, or levels below that. The tool must be capable of driving cards at TDP-50% to TDP-100%, in 10% incremental jumps. This should be controllable by the user.

**Input EDPp Test - IET module**

The Input EDPp Test generates EDP peak power on all input rails. This test is used to verify if the system PSU is capable of handling the worst case power spikes of the board. Peak Current at defined period = 1 minute moving average power.

Examples and about config files `link <https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/blob/roc-3.0.0/doc/ugsrc/ug1main.md>`_.

Prerequisites
***************

Ubuntu :

::

    sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git

CentOS :

::

    sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ 

RHEL :

::

  sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ 
    
  wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm
    
  sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm

SLES :

::

  sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64
   
  sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64
   
  sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ 

Install ROCm stack, rocblas and rocm_smi64
*********************************************

Install ROCm stack for Ubuntu/CentOS, Refer https://github.com/RadeonOpenCompute/ROCm

Install rocBLAS and rocm_smi64 :

Ubuntu :

::

  sudo apt-get install rocblas rocm_smi64

CentOS & RHEL :

::

  sudo yum install rocblas rocm_smi64

SUSE :

::

  sudo zypper install rocblas rocm_smi64


**Note:** If rocm_smi64 is already installed but "/opt/rocm/rocm_smi/ path doesn't exist. Do below:

Ubuntu : sudo dpkg -r rocm_smi64 && sudo apt install rocm_smi64

CentOS & RHEL : sudo rpm -e rocm_smi64 && sudo yum install rocm_smi64

SUSE : sudo rpm -e rocm_smi64 && sudo zypper install rocm_smi64

Building from Source
********************** 

This section explains how to get and compile current development stream of RVS.

**Clone repository**

::

  git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git


**Configure and build RVS:**

::

  cd ROCmValidationSuite


If OS is Ubuntu and SLES, use cmake

::

  cmake ./ -B./build
 
  make -C ./build


If OS is CentOS and RHEL, use cmake3

::

  cmake3 ./ -B./build

  make -C ./build


Build package:
 
::
 
  cd ./build
 
  make package

Note:_ based on your OS, only DEB or RPM package will be built. You may ignore an error for the unrelated configuration

**Install package:**

::

  Ubuntu : sudo dpkg -i rocm-validation-suite*.deb
  CentOS & RHEL & SUSE : sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm


**Running RVS**

Running version built from source code:

::

  cd ./build/bin
  sudo ./rvs -d 3
  sudo ./rvsqa.new.sh  ; It will run complete rvs test suite


Regression
***********

Regression is currently implemented for PQT module only. It comes in the form of a Python script run_regression.py.

The script will first create valid configuration files on $RVS_BUILD/regression folder. It is done by invoking prq_create_conf.py script to generate valid configuration files. If you need different tests, modify the prq_create_conf.py script to generate them.

Then, it will iterate through generated files and invoke RVS to specifying also JSON output and -d 3 logging level.

Finally, it will iterate over generated JSON output files and search for ERROR string. Results are written into $RVS_BUILD/regression/regression_res file.

Results are written into $RVS_BUILD/regression/

**Environment variables**

Before running the run_regression.py you first need to set the following environment variables for location of RVS source tree and build folders (ajdust for your particular clone):

::

  export WB=/work/yourworkfolder
  export RVS=$WB/ROCmValidationSuite
  export RVS_BUILD=$RVS/../build

**Running the script**

Just do:

::

  cd $RVS/regression
  ./run_regression.py



ROCr Debug Agent
================

The ROCr Debug Agent is a library that can be loaded by ROCm Platform
Runtime to provide the following functionality:

-  Print the state of wavefronts that report memory violation or upon
   executing a ``s_trap 2`` instruction.
-  Allows SIGINT (``ctrl c``) or SIGTERM (``kill -15``) to print
   wavefront state of aborted GPU dispatches.
-  It is enabled on Vega10 (since ROCm1.9), Vega20 (since ROCm2.0) GPUs.

Usage
*****

To use the ROCr Debug Agent set the following environment variable:

.. code:: sh

   export HSA_TOOLS_LIB=librocr_debug_agent64.so

This will use the ROCr Debug Agent library installed at
/opt/rocm/lib/librocr_debug_agent64.so by default since the ROCm
installation adds /opt/rocm/lib to the system library path. To use a
different version set the LD_LIBRARY_PATH, for example:

.. code:: sh

   export LD_LIBRARY_PATH=/path_to_directory_containing_librocr_debug_agent64.so

To display the machine code instructions of wavefronts, together with
the source text location, the ROCr Debug Agent uses the llvm-objdump
tool. Ensure that a version that supports AMD GCN GPUs is on your
``$PATH``. For example, for ROCm 3.0:

.. code:: sh

   export PATH=/opt/rocm/opencl/bin/x86_64/:$PATH

Execute your application.

If the application encounters a GPU error it will display the wavefront
state of the GPU to ``stdout``. Possible error states include:

-  The GPU executes a memory instruction that causes a memory violation.
   This is reported as an XNACK error state.
-  Queue error.
-  The GPU executes an ``S_TRAP`` instruction. The ``__builtin_trap()``
   language builtin can be used to generate a ``S_TRAP``.
-  A SIGINT (``ctrl c``) or SIGTERM (``kill -15``) signal is sent to the
   application while executing GPU code. Enabled by the
   ``ROCM_DEBUG_ENABLE_LINUX_SIGNALS`` environment variable.

For example, a sample print out for GPU memory fault is:

::

   Memory access fault by GPU agent: AMD gfx900
   Node: 1
   Address: 0x18DB4xxx (page not present;write access to a read-only page;)

   64 wavefront(s) found in XNACK error state @PC: 0x0000001100E01310
   printing the first one:

      EXEC: 0xFFFFFFFFFFFFFFFF
    STATUS: 0x00412460
   TRAPSTS: 0x30000000
        M0: 0x00001010

        s0: 0x00C00000    s1: 0x80000010    s2: 0x10000000    s3: 0x00EA4FAC
        s4: 0x17D78400    s5: 0x00000000    s6: 0x01039000    s7: 0x00000000
        s8: 0x00000000    s9: 0x00000000   s10: 0x17D78400   s11: 0x04000000
       s12: 0x00000000   s13: 0x00000000   s14: 0x00000000   s15: 0x00000000
       s16: 0x0103C000   s17: 0x00000000   s18: 0x00000000   s19: 0x00000000
       s20: 0x01037060   s21: 0x00000000   s22: 0x00000000   s23: 0x00000011
       s24: 0x00004000   s25: 0x00010000   s26: 0x04C00000   s27: 0x00000010
       s28: 0xFFFFFFFF   s29: 0xFFFFFFFF   s30: 0x00000000   s31: 0x00000000

   Lane 0x0
        v0: 0x00000003    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000
   Lane 0x1
        v0: 0x00000004    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000
   Lane 0x2
        v0: 0x00000005    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000
   Lane 0x3
        v0: 0x00000006    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000

       .
       .
       .

   Lane 0x3C
        v0: 0x0000001F    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000
   Lane 0x3D
        v0: 0x00000020    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000
   Lane 0x3E
        v0: 0x00000021    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000
   Lane 0x3F
        v0: 0x00000022    v1: 0x18DB4400    v2: 0x18DB4400    v3: 0x00000000
        v4: 0x00000000    v5: 0x00000000    v6: 0x00700000    v7: 0x00800000

   Faulty Code Object:

   /tmp/ROCm_Tmp_PID_5764/ROCm_Code_Object_0:      file format ELF64-amdgpu-hsacobj

   Disassembly of section .text:
   the_kernel:
   ; /home/qingchuan/tests/faulty_test/vector_add_kernel.cl:12
   ; d[100000000] = ga[gid & 31];
           v_mov_b32_e32 v1, v2                                       // 0000000012F0: 7E020302
           v_mov_b32_e32 v4, v3                                       // 0000000012F4: 7E080303
           v_add_i32_e32 v1, vcc, s10, v1                             // 0000000012F8: 3202020A
           v_mov_b32_e32 v5, s22                                      // 0000000012FC: 7E0A0216
           v_addc_u32_e32 v4, vcc, v4, v5, vcc                        // 000000001300: 38080B04
           v_mov_b32_e32 v2, v1                                       // 000000001304: 7E040301
           v_mov_b32_e32 v3, v4                                       // 000000001308: 7E060304
           s_waitcnt lgkmcnt(0)                                       // 00000000130C: BF8CC07F
           flat_store_dword v[2:3], v0                                // 000000001310: DC700000 00000002
   ; /home/qingchuan/tests/faulty_test/vector_add_kernel.cl:13
   ; }
           s_endpgm                                                   // 000000001318: BF810000

   Faulty PC offset: 1310

   Aborted (core dumped)

Options
*******

Dump Output
-----------

By default the wavefront dump is sent to ``stdout``.

To save to a file use:

.. code:: sh

   export ROCM_DEBUG_WAVE_STATE_DUMP=file

This will create a file called ``ROCm_Wave_State_Dump`` in code object
directory (see below).

To return to the default ``stdout`` use either of the following:

.. code:: sh

   export ROCM_DEBUG_WAVE_STATE_DUMP=stdout
   unset ROCM_DEBUG_WAVE_STATE_DUMP

Linux Signal Control
--------------------

The following environment variable can be used to enable dumping
wavefront states when SIGINT (``ctrl c``) or SIGTERM (``kill -15``) is
sent to the application:

.. code:: sh

   export ROCM_DEBUG_ENABLE_LINUX_SIGNALS=1

Either of the following will disable this behavior:

.. code:: sh

   export ROCM_DEBUG_ENABLE_LINUX_SIGNALS=0
   unset ROCM_DEBUG_ENABLE_LINUX_SIGNALS

Code Object Saving
------------------

When the ROCr Debug Agent is enabled, each GPU code object loaded by the
ROCm Platform Runtime will be saved in a file in the code object
directory. By default the code object directory is
``/tmp/ROCm_Tmp_PID_XXXX/`` where ``XXXX`` is the application process
ID. The code object directory can be specified using the following
environent variable:

.. code:: sh

   export ROCM_DEBUG_SAVE_CODE_OBJECT=code_object_directory

This will use the path ``/code_object_directory``.

Loaded code objects will be saved in files named ``ROCm_Code_Object_N``
where N is a unique integer starting at 0 of the order in which the code
object was loaded.

If the default code object directory is used, then the saved code object
file will be deleted when it is unloaded with the ROCm Platform Runtime,
and the complete code object directory will be deleted when the
application exits normally. If a code object directory path is specified
then neither the saved code objects, nor the code object directory will
be deleted.

To return to using the default code object directory use:

.. code:: sh

   unset ROCM_DEBUG_SAVE_CODE_OBJECT

Logging
-------

By default ROCr Debug Agent logging is disabled. It can be enabled to
display to ``stdout`` using:

.. code:: sh

   export ROCM_DEBUG_ENABLE_AGENTLOG=stdout

Or to a file using:

.. code:: sh

   export ROCM_DEBUG_ENABLE_AGENTLOG=<filename>

Which will write to the file ``<filename>_AgentLog_PID_XXXX.log``.

To disable logging use:

.. code:: sh

   unset ROCM_DEBUG_ENABLE_AGENTLOG

ROCm-GDB
=========

**This ROCm Debugger is a Deprecated project.**

As of 2018, this is a deprecated software project. The ROCm software team is working on a new GDB-based debugger that works with the ROCr Debug Agent to support debugging GPU kernels.

The ROCm-GDB repository includes the source code for ROCm-GDB. ROCm-GDB is a modified version of GDB 7.11 revised to work with the ROCr Debug Agent to support debugging GPU kernels on Radeon Open Compute platforms (ROCm).

For more information refer `here <https://github.com/rocmarchive/ROCm-GDB>`_


ROCm Binary Utilities
======================
Documentation need to be updated.


MIVisionX
=========

.. image:: https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/images/MIVisionX.png
  :align: center
  :width: 400
  :alt: MIVisionX
  :target: https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/

MIVisionX toolkit is a set of comprehensive computer vision and machine intelligence libraries, utilities, and applications bundled into a single toolkit. AMD MIVisionX delivers highly optimized open source implementation of the `Khronos OpenVX™ <https://www.khronos.org/openvx/>`_ and OpenVX™ Extensions along with Convolution Neural Net Model Compiler & Optimizer supporting `ONNX <https://onnx.ai/>`_, and `Khronos NNEF™ <https://www.khronos.org/nnef>`_ exchange formats. The toolkit allows for rapid prototyping and deployment of optimized workloads on a wide range of computer hardware, including small embedded x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.

* `AMD OpenVX <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#amd-openvx>`_
* `AMD OpenVX Extensions <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#amd-openvx-extensions>`_
    * `Loom 360 Video Stitch Library <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_loomsl/>`_
    * `Neural Net Library <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_nn/#openvx-neural-network-extension-library-vx_nn>`_
    * `OpenCV Extension <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_opencv/#amd-opencv-extension>`_
    * `RPP Extension <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx_extensions/amd_rpp>`_
    * `WinML Extension <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_winml/#amd-winml-extension>`_
* `Applications <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#applications>`_
* `Neural Net Model Compiler and Optimizer <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#neural-net-model-compiler--optimizer>`_
* `RALI <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/rali/>`_
* `Samples <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/samples/#samples>`_
* `Toolkit <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#toolkit>`_
* `Utilities <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#utilities>`_
    * `Inference Generator <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/inference_generator/#inference-generator>`_
    * `Loom Shell <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/loom_shell/#radeon-loomshell>`_
    * `RunCL <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/runcl/#amd-runcl>`_
    * `RunVX <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/runvx/#amd-runvx>`_
* `Prerequisites <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#prerequisites>`_
* `Build and Install MIVisionX <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#build--install-mivisionx>`_
* `Verify the Installation <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#verify-the-installation>`_
* `Docker <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#docker>`_
* `Release Notes <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#release-notes>`_

AMD OpenVX
**********

.. image:: https://www.khronos.org/assets/uploads/ceimg/made/assets/uploads/apis/OpenVX_100px_June16_210_75.png
  :align: center
  :width: 300
  :alt: OpenVX
  :target: https://www.khronos.org/openvx/

AMD OpenVX [`amd_openvx <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#amd-openvx>`__] is a highly optimized open source implementation of the `Khronos OpenVX <https://www.khronos.org/openvx/>`_ computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

AMD OpenVX Extensions
*********************

The OpenVX framework provides a mechanism to add new vision functions to OpenVX by 3rd party vendors. This project has below mentioned OpenVX `modules <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/#amd-openvx-extensions-amd_openvx_extensions>`_ and utilities to extend `amd_openvx <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx/#amd-openvx-amd_openvx>`_ project, which contains the AMD OpenVX Core Engine.

    * `amd_loomsl <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_loomsl/>`_: AMD Radeon Loom stitching library for live 360 degree video applications.

    .. image:: https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/images/loom-2.jpg
      :align: center
      :width: 600
      :alt: Loom Stitch

    * `amd_nn <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_nn/#openvx-neural-network-extension-library-vx_nn>`_: OpenVX neural network module

    .. image:: https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/images/modelCompilerWorkflow.png
      :align: center
      :width: 600
      :alt: AMD OpenVX Neural Net Extension

    * `amd_opencv <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_opencv/#amd-opencv-extension>`_: OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels

    * `amd_winml <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_winml/#amd-winml-extension>`_: WinML extension will allow developers to import a pre-trained ONNX model into an OpenVX graph and add hundreds of different pre & post processing vision/generic/user-defined functions, available in OpenVX and OpenCV interop, to the input and output of the neural net model. This will allow developers to build an end to end application for inference.

    .. image:: https://raw.githubusercontent.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/master/docs/images/winmlFrameWorks.png
      :align: center
      :width: 600
      :alt: AMD WinML

Applications
************

MIVisionX has a number of `applications <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#applications>`_ built on top of OpenVX modules, it uses AMD optimized libraries to build applications which can be used to prototype or used as models to develop a product.

  * `Cloud Inference Application <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/cloud_inference/#cloud-inference-application>`_: This sample application does inference using a client-server system.
  * `Digit Test <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/apps/dg_test#amd-dgtest>`_ This sample application is used to recognize hand written digits.
  * `MIVisionX OpenVX Classsification <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#mivisionx-openvx-classsification>`_: This sample application shows how to run supported pre-trained caffe models with MIVisionX RunTime.
  * `MIVisionX WinML Classification <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#mivisionx-winml-classification>`_: This sample application shows how to run supported ONNX models with MIVisionX RunTime on Windows.
  * `MIVisionX WinML YoloV2 <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#mivisionx-winml-yolov2>`_: This sample application shows how to run tiny yolov2(20 classes) with MIVisionX RunTime on Windows.
  * `External Applications <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#external-application>`_


Neural Net Model Compiler And Optimizer
***************************************

.. image:: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/docs/images/frameworks.png?raw=true
  :align: center
  :width: 800
  :alt: Neural Net Model Compiler And Optimizer

Neural Net Model Compiler & Optimizer `model_compiler <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/model_compiler/#neural-net-model-compiler--optimizer>`_ converts pre-trained neural net models to MIVisionX runtime code for optimized inference.

RALI
****
The Radeon Augmentation Library `RALI <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/rali/>`_ is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user.

Samples
*******

`MIVisionX samples <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/samples/#samples>`_ using OpenVX and OpenVX extension libraries

**GDF - Graph Description Format**

MIVisionX samples using runvx with GDF

**skintonedetect.gdf**

.. image:: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/samples/images/skinToneDetect_image.PNG?raw=true
  :align: center
  :width: 600
  :alt: skintonedetect

usage:

::

  runvx skintonedetect.gdf

**canny.gdf**

.. image:: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/samples/images/canny_image.PNG?raw=true
  :align: center
  :width: 600
  :alt: canny

usage:

::

  runvx canny.gdf

**skintonedetect-LIVE.gdf**

Using live camera

usage:

::

  runvx -frames:live skintonedetect-LIVE.gdf

**canny-LIVE.gdf**

Using live camera

usage:

::
 
  runvx -frames:live canny-LIVE.gdf

**OpenCV_orb-LIVE.gdf**


Using live camera

usage:

::

  runvx -frames:live OpenCV_orb-LIVE.gdf

**Note:** More samples available on `GitHub <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/samples#samples>`_

Toolkit
*******

`MIVisionX Toolkit <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/toolkit/#mivisionx-toolkit>`_, is a comprehensive set of help tools for neural net creation, development, training, and deployment. The Toolkit provides you with helpful tools to design, develop, quantize, prune, retrain, and infer your neural network work in any framework. The Toolkit is designed to help you deploy your work to any AMD or 3rd party hardware, from embedded to servers.

MIVisionX provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle, from creating a model to deploying them for your target platforms.

Utilities
*********

* `inference_generator <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/inference_generator/#inference-generator>`_: generate inference library from pre-trained CAFFE models
* `loom_shell <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/loom_shell/#radeon-loomsh>`_: an interpreter to prototype 360 degree video stitching applications using a script
* `RunVX <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/runvx/#amd-runvx>`_: command-line utility to execute OpenVX graph described in GDF text file
* `RunCL <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/runcl/#amd-runcl>`_: command-line utility to build, execute, and debug OpenCL programs

Prerequisites
*************

    * CPU: SSE4.1 or above CPU, 64-bit
    * GPU: `GFX7 or above <https://rocm.github.io/hardware.html>`_ [optional]
    * APU: Carrizo or above [optional]

**Note:** Some modules in MIVisionX can be built for CPU only. To take advantage of advanced features and modules we recommend using AMD GPUs or AMD APUs.

**Windows**

    * Windows 10
    * Windows SDK
    * Visual Studio 2017
    * Install the latest drivers and `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0`>_
    * `OpenCV <https://github.com/opencv/opencv/releases/tag/3.4.0>`_
          * Set OpenCV_DIR environment variable to OpenCV/build folder
          * Add %OpenCV_DIR%\x64\vc14\bin or %OpenCV_DIR%\x64\vc15\bin to your PATH

**Linux**

    * Install `ROCm <https://rocm.github.io/ROCmInstall.html>`__
    * ROCm CMake, MIOpenGEMM & MIOpen for Neural Net Extensions (vx_nn)
    * CMake 2.8 or newer `download <http://cmake.org/download/>`_
    * Qt Creator for `Cloud Inference Client <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/1.3.0/apps/cloud_inference/client_app/README.md>`_
    * `Protobuf <https://github.com/google/protobuf>`_ for inference generator & model compiler
        * install libprotobuf-dev and protobuf-compiler needed for vx_nn
    * ` OpenCV <https://github.com/opencv/opencv/releases/tag/3.4.0>`_
        * Set OpenCV_DIR environment variable to OpenCV/build folder
    * `FFMPEG <https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4>`_ - Optional
        * FFMPEG is required for amd_media & mv_deploy modules


Pre-requisites setup script - MIVisionX-setup.py
************************************************
 
For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

**MIVisionX-setup.py**- This scipts builds all the prerequisites required by MIVisionX. The setup script creates a deps folder and installs all the prerequisites, this script only needs to be executed once. If -d option for directory is not given the script will install deps folder in ‘~/’ directory by default, else in the user specified folder.

**Prerequisites for running the scripts**


   * ubuntu 16.04/18.04 or CentOS 7.5/7.6
   * `ROCm supported hardware <https://rocm.github.io/hardware.html>`_
   * `ROCm <https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories>`__

usage:

::

  python MIVisionX-setup.py --directory [setup directory - optional]
                            --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                            --miopen    [MIOpen Version - optional (default:2.1.0)]
                            --miopengemm[MIOpenGEMM Version - optional (default:1.1.5)]
                            --ffmpeg    [FFMPEG Installation - optional (default:no) [options:Install ffmpeg - yes]]
                            --rpp       [RPP Installation - optional (default:yes) [options:yes/no]]

**Note:** use --installer **yum** for **CentOS**


Build & Install MIVisionX
*************************

**Windows**

**Using .msi packages**

    * `MIVisionX-installer.msi <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases>`_: MIVisionX
    * `MIVisionX_WinML-installer.msi <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases>`_: MIVisionX for WinML

**Using Visual Studio 2017 on 64-bit Windows 10**

    * Install `OpenCL_SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0>`_
    * Install `OpenCV <https://github.com/opencv/opencv/releases/tag/3.4.0>`_ with/without `contrib <https://github.com/opencv/opencv_contrib>`_ to support camera capture, image display, & opencv extensions
        * Set OpenCV_DIR environment variable to OpenCV/build folder
        * Add %OpenCV_DIR%\x64\vc14\bin or %OpenCV_DIR%\x64\vc15\bin to your PATH
    * Use MIVisionX.sln to build for x64 platform

**NOTE:** vx_nn is not supported on Windows in this release


**Linux**

**Using apt-get/yum**


**Prerequisites**

    * Ubuntu 16.04/18.04 or CentOS 7.5/7.6
    * `ROCm supported hardware <https://rocm.github.io/hardware.html>`_
    * `ROCm <https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories>`__

**Ubuntu**
::

  sudo apt-get install mivisionx


**CentOS**
::

  sudo yum install mivisionx

**Note:**

    * vx_winml is not supported on linux
    * source code will not available with apt-get/yum install
    * executables placed in /opt/rocm/mivisionx/bin and libraries in /opt/rocm/mivisionx/lib
    * OpenVX and module header files into /opt/rocm/mivisionx/include
    * model compiler, toolkit, & samples placed in /opt/rocm/mivisionx
    * Package (.deb & .rpm) install requires OpenCV v3.4.0 to execute AMD OpenCV extensions



**Using MIVisionX-setup.py and CMake on Linux (Ubuntu 16.04/18.04 or CentOS 7.5/7.6) with ROCm**

    * Install `ROCm <https://rocm.github.io/ROCmInstall.html>`__
    * Use the below commands to setup and build MIVisionX


::

  git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
  cd MIVisionX


::

  python MIVisionX-setup.py --directory [setup directory - optional]
                            --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                            --miopen    [MIOpen Version - optional (default:2.1.0)]
                            --miopengemm[MIOpenGEMM Version - optional (default:1.1.5)]
                            --ffmpeg    [FFMPEG Installation - optional (default:no) [options:Install ffmpeg - yes]]    
                            --rpp       [RPP Installation - optional (default:yes) [options:yes/no]]


**Note:** Use --installer yum for **CentOS**


::

  mkdir build
  cd build
  cmake ../
  make -j8
  sudo make install

**Note:**

    * vx_winml is not supported on Linux
    * the installer will copy all executables into /opt/rocm/mivisionx/bin and libraries into /opt/rocm/mivisionx/lib
    * the installer also copies all the OpenVX and module header files into /opt/rocm/mivisionx/include folder


**Using CMake on Linux (Ubuntu 16.04 64-bit or CentOS 7.5 / 7.6 ) with ROCm**

   * Install `ROCm <https://rocm.github.io/ROCmInstall.html>`_
   * git clone, build and install other ROCm projects (using cmake and % make install) in the below order for vx_nn.
       * `rocm-cmake <https://github.com/RadeonOpenCompute/rocm-cmake>`_
       * `MIOpenGEMM <https://github.com/ROCmSoftwarePlatform/MIOpenGEMM>`_
       * `MIOpen <https://github.com/ROCmSoftwarePlatform/MIOpen>`_ – make sure to use -DMIOPEN_BACKEND=OpenCL option with cmake
   * install `protobuf <https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.2>`__
   * install `OpenCV <https://github.com/opencv/opencv/releases/tag/3.3.0>`__
   * install `FFMPEG n4.0.4 <https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4>`_ - Optional
   * build and install (using cmake and % make install)
       * executables will be placed in bin folder
       * libraries will be placed in lib folder
       * the installer will copy all executables into /opt/rocm/mivisionx/bin and libraries into /opt/rocm/lib
       * the installer also copies all the OpenVX and module header files into /opt/rocm/mivisionx/include folder
   * add the installed library path to LD_LIBRARY_PATH environment variable (default /opt/rocm/mivisionx/lib)
   * add the installed executable path to PATH environment variable (default /opt/rocm/mivisionx/bin)


Verify the Installation
***********************

**Linux**

    * The installer will copy all executables into /opt/rocm/mivisionx/bin and libraries into /opt/rocm/mivisionx/lib

    * The installer also copies all the OpenVX and OpenVX module header files into /opt/rocm/mivisionx/include folder

    * Apps, Samples, Documents, Model Compiler and Toolkit are placed into /opt/rocm/mivisionx

    * Run samples to verify the installation
        
        * **Canny Edge Detection**
 
.. image:: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/samples/images/canny_image.PNG?raw=true
  :align: center
  :width: 600
    
::

  export PATH=$PATH:/opt/rocm/mivisionx/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
  runvx /opt/rocm/mivisionx/samples/gdf/canny.gdf 

Note: More samples are available `here <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/1.3.0/samples#samples>`_


Docker
******

MIVisionX provides developers with docker images for Ubuntu 16.04, Ubuntu 18.04, CentOS 7.5, & CentOS 7.6. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

**MIVisionX Docker**

   * `Ubuntu 16.04 <https://hub.docker.com/r/mivisionx/ubuntu-16.04>`_
   * `Ubuntu 18.04 <https://hub.docker.com/r/mivisionx/ubuntu-18.04>`_
   * `CentOS 7.5 <https://hub.docker.com/r/mivisionx/centos-7.5>`_
   * `CentOS 7.6 <https://hub.docker.com/r/mivisionx/centos-7.6>`_

**Docker Workflow Sample on Ubuntu 16.04/18.04**

**Prerequisites**

   * Ubuntu 16.04/18.04
   * `rocm supported hardware <https://rocm.github.io/hardware.html>`_


**Workflow**

**Step 1 - Install rocm-dkms**

::

   sudo apt update
   sudo apt dist-upgrade
   sudo apt install libnuma-dev
   sudo reboot

::

   wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dkms
   sudo reboot


**Step 2 - Setup Docker**

::

   sudo apt-get install curl
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   sudo apt-get update
   apt-cache policy docker-ce
   sudo apt-get install -y docker-ce
   sudo systemctl status docker

**Step 3 - Get Docker Image**

::

   sudo docker pull mivisionx/ubuntu-16.04


**Step 4 - Run the docker image**

::

   sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host           mivisionx/  ubuntu-16.04



* Optional: Map localhost directory on the docker image 
      * option to map the localhost directory with trained caffe models to be accessed on the docker image.
      * usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH}
 
       
::
     
     sudo docker run -it -v /home/:/root/hostDrive/ --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-16.04



**Note: Display option with docker**

    * Using host display
     
::
 
     xhost +local:root
     sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video 
     --network host --env DISPLAY=unix$DISPLAY --privileged --volume $XAUTH:/root/.Xauthority 
     --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-16.04:latest



* Test display with MIVisionX sample

    
::

    export PATH=$PATH:/opt/rocm/mivisionx/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
    runvx /opt/rocm/mivisionx/samples/gdf/canny.gdf 

Release Notes
*************

**Known issues**

   * Package (.deb & .rpm) install requires OpenCV v3.4.0 to execute AMD OpenCV extensions
   

**Tested configurations**

    * Windows 10
    * Linux: Ubuntu - 16.04/18.04 & CentOS - 7.5/7.6
    * ROCm: rocm-dkms - 2.9.6
    * rocm-cmake - `github master:ac45c6e <https://github.com/RadeonOpenCompute/rocm-cmake/tree/master>`_
    * MIOpenGEMM - `1.1.5 <https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5>`_
    * MIOpen - `2.1.0 <https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.1.0>`_
    * Protobuf - `V3.5.2 <https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.2>`_
    * OpenCV - `3.4.0 <https://github.com/opencv/opencv/releases/tag/3.4.0>`_
    * Dependencies for all the above packages
