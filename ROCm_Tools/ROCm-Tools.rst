
.. _ROCm-Tools:

=====================
ROCm Tools
=====================

HCC
====

**HCC is an Open Source, Optimizing C++ Compiler for Heterogeneous Compute**

This repository hosts the HCC compiler implementation project. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP, and transforms it into the AMD GCN ISA.

The project is based on LLVM+CLANG. For more information, please visit the :ref:`HCCwiki`

Download HCC
###############

The project now employs git submodules to manage external components it depends upon. It it advised to add --recursive when you clone the project so all submodules are fetched automatically.

For example:

.. code:: sh

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

For more information about git submodules, please refer to `git documentation <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_.


Build HCC from source
#######################

To configure and build HCC from source, use the following steps:

.. code:: sh

  mkdir -p build; cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make

To install it, use the following steps:

.. code:: sh

  sudo make install

Use HCC
########

For C++AMP source codes:

.. code:: sh

  hcc `clamp-config --cxxflags --ldflags` foo.cpp

**WARNING: From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**

For HC source codes:

.. code:: sh

  hcc `hcc-config --cxxflags --ldflags` foo.cpp

In case you build HCC from source and want to use the compiled binaries directly in the build directory:

For C++AMP source codes:

.. code:: sh

  # notice the --build flag
  bin/hcc `bin/clamp-config --build --cxxflags --ldflags` foo.cpp

**WARNING: From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**

For HC source codes:

.. code:: sh

  # notice the --build flag
  bin/hcc `bin/hcc-config --build --cxxflags --ldflags` foo.cpp


Multiple ISA
###############
HCC now supports having multiple GCN ISAs in one executable file. You can do it in different ways:
**use ``--amdgpu-target=`` command line option**

It's possible to specify multiple `` --amdgpu-target= `` option.

Example:

.. code:: sh

 # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
 # be produced
  hcc `hcc-config --cxxflags --ldflags` \
    --amdgpu-target=gfx701 \
    --amdgpu-target=gfx801 \
    --amdgpu-target=gfx802 \
    --amdgpu-target=gfx803 \
    foo.cpp

**use ``HCC_AMDGPU_TARGET`` env var**

use ``,`` to delimit each AMDGPU target in HCC. Example:

.. code:: sh

  export HCC_AMDGPU_TARGET=gfx701,gfx801,gfx802,gfx803
  # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
  # be produced
  hcc `hcc-config --cxxflags --ldflags` foo.cpp

**configure HCC use CMake ``HSA_AMDGPU_GPU_TARGET`` variable**

If you build HCC from source, it's possible to configure it to automatically produce multiple ISAs via `HSA_AMDGPU_GPU_TARGET` CMake variable.

Use ``;`` to delimit each AMDGPU target.
Example:

.. code:: sh

 # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
 # be produced by default
 cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_DEVICE_LIB_DIR=~hcc/ROCm-Device-Libs/build/dist/lib \
    -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx801;gfx802;gfx803" \
    ../hcc

CodeXL Activity Logger
#########################

To enable the `CodeXL Activity Logger <https://github.com/RadeonOpenCompute/ROCm-Profiler/tree/master/CXLActivityLogger>`_, use the  ``USE_CODEXL_ACTIVITY_LOGGER`` environment variable.

Configure the build in the following way:

.. code:: sh

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=<AMD GPU ISA version string> \
    -DROCM_DEVICE_LIB_DIR=<location of the ROCm-Device-Libs bitcode> \
    -DUSE_CODEXL_ACTIVITY_LOGGER=1 \
    <ToT HCC checkout directory>

In your application compiled using hcc, include the CodeXL Activiy Logger header:

.. code:: sh

  #include <CXLActivityLogger.h>

For information about the usage of the Activity Logger for profiling, please refer to its `documentation <https://github.com/RadeonOpenCompute/ROCm-Profiler/blob/master/CXLActivityLogger/doc/AMDTActivityLogger.pdf>`_.

HCC with ThinLTO Linking
#########################

To enable the ThinLTO link time, use the ``KMTHINLTO`` environment variable.

Set up your environment in the following way:

.. code:: sh

  export KMTHINLTO=1

**ThinLTO Phase 1 - Implemented**

For applications compiled using hcc, ThinLTO could significantly improve link-time performance. This implementation will maintain kernels in their .bc file format, create module-summaries for each, perform llvm-lto's cross-module function importing and then perform clamp-device (which uses opt and llc tools) on each of the kernel files. These files are linked with lld into one .hsaco per target specified.

**ThinLTO Phase 2 - Under development**
This ThinLTO implementation which will use llvm-lto LLVM tool to replace clamp-device bash script. It adds an optllc option into ThinLTOGenerator, which will perform in-program opt and codegen in parallel.



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

Currently, a programmer must manually set all non-default values to provide the necessary information. Hopefully, this situation will change with new updates that bring automatic register counting and possibly a new syntax to fill that structure. Before the start of every wavefront execution, the GPU sets up the register state on the basis of the enable_sgpr_* and enable_vgpr_* flags. VGPR v0 is always initialized with a work-item ID in the x dimension. Registers v1 and v2 can be initialized with work-item IDs in the y and z dimensions, respectively. Scalar GPRs can be initialized with a work-group ID and work-group count in each dimension, a dispatch ID, and pointers to kernarg, the aql packet, the aql queue, and so on. Again, the AMDGPU-ABI specification contains a full list in in the section on initial register state. For this example, a 64-bit base kernarg address will be stored in the s[0:1] registers (enable_sgpr_kernarg_segment_ptr = 1), and the work-item thread ID will occupy v0 (by default). Below is the scheme showing initial state for our kernel. initial_state

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


ROC Profiler
============

ROC profiler library. Profiling with perf-counters and derived metrics. Library supports GFX8/GFX9.

HW specific low-level performance analysis interface for profiling of GPU compute applications. The profiling includes HW performance counters with complex performance metrics and HW traces.

Profiling tool 'rocprof':
   *  Cmd-line tool for dumping public per kernel perf-counters/metrics and kernel timestamps
   *  Input file with counters list and kernels selecting parameters
   *  Multiple counters groups and app runs supported
   *  Kernel execution is serialized
   *  HSA API/activity stats and tracing
   *  Output results in CSV and JSON chrome tracing formats

Download
########

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

Build
#####

Build environment:

.. code:: sh

  export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
  export CMAKE_BUILD_TYPE=<debug|release> # release by default
  export CMAKE_DEBUG_TRACE=1 # to enable debug tracing

To configure, build, install to /opt/rocm/rocprofiler:

.. code:: sh

  mkdir -p build
  cd build
  export CMAKE_PREFIX_PATH=/opt/rocm
  cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
  make
  sudo make install

To test the built library:

.. code:: sh

  cd build
  ./run.sh

To enable error messages logging to '/tmp/rocprofiler_log.txt':

.. code:: sh

  export ROCPROFILER_LOG=1

To enable verbose tracing:

.. code:: sh

  export ROCPROFILER_TRACE=1


Profiling Tool 'rocprof' Usage
##############################

The following shows the command-line usage of the 'rocprof' tool:

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
  -d <data directory> - directory where profiler store profiling data including thread treaces [/tmp]
      The data directory is renoving autonatically if the directory is matching the temporary one, which is the default.
  -t <temporary directory> - to change the temporary directory [/tmp]
      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
  --timestamp <on|off> - to turn on/off the kernel disoatches timestamps, dispatch/begin/end/complete [off]
  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]

  --hsa-trace - to trace HSA, generates API execution stats and JSON file viewable in chrome tracing
    Requires to set three options '--hsa-trace --stats --timestamp on'
    Will be simplified to just one option in the next release
    Generated files: <output name>.stats.csv <output name>.hsa_stats.txt <output name>.json

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


ROCr Debug Agent
================

The ROCr Debug Agent is a library that can be loaded by ROCm Platform
Runtime to provide the following functionality:

-  Print the state of wavefronts that report memory violation or upon
   executing a ``s_trap 2`` instruction.
-  Allows SIGINT (``ctrl c``) or SIGTERM (``kill -15``) to print
   wavefront state of aborted GPU dispatches.
-  It is enabled on Vega10 GPUs on ROCm2.6.

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
``$PATH``. For example, for ROCm 2.6:

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

The ROCm-GDB is being revised to work with the ROCr Debug Agent to
support debugging GPU kernels on Radeon Open Compute platforms (ROCm)
and will be available in an upcoming release.

Radeon Compute Profiler
==============

The Radeon Compute Profiler (RCP) is a performance analysis tool that gathers data from the API run-time and GPU for OpenCL™ and ROCm/HSA applications. This information can be used by developers to discover bottlenecks in the application and to find ways to optimize the application's performance.

Please see the `RCP GitHub repository <https://github.com/GPUOpen-Tools/RCP>`_ for more information.



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

Documentation
#############

.. code:: sh

  - API description: inc/roctracer.h
  - Code example: test/MatrixTranspose/MatrixTranspose.cpp

To build and run test
######################

.. code:: sh

   cd <your path>

  - CLone development branches of roctracer and HIP/HCC:
   git clone -b amd-master https://github.com/ROCmSoftwarePlatform/roctracer.git
   git clone -b master https://github.com/ROCm-Developer-Tools/HIP.git
   git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

  - Set environment:
   export HIP_PATH=<your path>/HIP
   export HCC_HOME=<your path>/hcc/build
   export CMAKE_PREFIX_PATH=/opt/rocm

  - Build HCC:
   cd <your path>/hcc && mkdir build && cd build &&
   cmake -DUSE_PROF_API=1 -DPROF_API_HEADER_PATH=<your path>/roctracer/inc/ext .. && make -j <nproc>
  
  - Build HIP:
   cd <your path>/HIP && mkdir build && cd build &&
   cmake -DUSE_PROF_API=1 -DPROF_API_HEADER_PATH=<your path>/roctracer/inc/ext .. && make -j <nproc>
   ln -s <your path>/HIP/build <your path>/HIP/lib
  
  - Build ROCtracer
   cd <your path>/roctracer && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm .. && make -j <nproc>

  - To build and run test
   make mytest
   run.sh
  
  - To install
   make install
   or
   make package && dpkg -i *.deb



CodeXL
=========
CodeXL is a comprehensive tool suite that enables developers to harness the benefits of GPUs and APUs. It includes powerful GPU debugging, comprehensive GPU profiling, and static OpenCL™, OpenGL®, Vulkan® and DirectX® kernel/shader analysis capabilities, enhancing accessibility for software developers to enter the era of heterogeneous computing. CodeXL is available as a standalone user interface application for Windows® and Linux®.

Please see the `CodeXL GitHub repository <https://github.com/GPUOpen-Tools/CodeXL>`_ for more information.

GPUPerfAPI
==============

The GPU Performance API (GPUPerfAPI, or GPA) is a powerful library, providing access to GPU Performance Counters. It can help analyze the performance and execution characteristics of applications using a Radeon™ GPU. This library is used by Radeon Compute Profiler and CodeXL as well as several third-party tools.

Please see the `GPA GitHub repository <https://github.com/GPUOpen-Tools/GPA>`_ for more information.

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

**MIVisionX toolkit is a set of comprehensive computer vision and machine intelligence libraries, utilities, and applications bundled into a single toolkit. AMD MIVisionX delivers highly optimized open source implementation of the Khronos OpenVX™ and OpenVX™ Extensions along with Convolution Neural Net Model Compiler & Optimizer supporting ONNX, and Khronos NNEF™ exchange formats. The toolkit allows for rapid prototyping and deployment of optimized workloads on a wide range of computer hardware, including small embedded x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.**

* `AMD OpenVX <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#amd-openvx>`_
* `AMD OpenVX Extensions <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#amd-openvx-extensions>`_
    * `Loom 360 Video Stitch Library <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_loomsl/>`_
    * `Neural Net Library <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_nn/#openvx-neural-network-extension-library-vx_nn>`_
    * `OpenCV Extension <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_opencv/#amd-opencv-extension>`_
    * `WinML Extension <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/amd_openvx_extensions/amd_winml/#amd-winml-extension>`_
* `Applications <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#applications>`_
* `Neural Net Model Compiler and Optimizer <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#neural-net-model-compiler--optimizer>`_
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
###########

.. image:: https://www.khronos.org/assets/uploads/ceimg/made/assets/uploads/apis/OpenVX_100px_June16_210_75.png
  :align: center
  :width: 300
  :alt: OpenVX
  :target: https://www.khronos.org/openvx/

AMD OpenVX [`amd_openvx <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#amd-openvx>`__] is a highly optimized open source implementation of the `Khronos OpenVX <https://www.khronos.org/openvx/>`_ computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded x86 CPUs and large workstation discrete GPUs.

AMD OpenVX Extensions
#######################

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
############

MIVisionX has a number of `applications <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/#applications>`_ built on top of OpenVX modules, it uses AMD optimized libraries to build applications which can be used to prototype or used as models to develop a product.

  * `Cloud Inference Application <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/cloud_inference/#cloud-inference-application>`_: This sample application does inference using a client-server system.
  * `Digit Test <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/apps/dg_test#amd-dgtest>`_ This sample application is used to recognize hand written digits.
  * `MIVisionX OpenVX Classsification <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#mivisionx-openvx-classsification>`_: This sample application shows how to run supported pre-trained caffe models with MIVisionX RunTime.
  * `MIVisionX WinML Classification <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#mivisionx-winml-classification>`_: This sample application shows how to run supported ONNX models with MIVisionX RunTime on Windows.
  * `MIVisionX WinML YoloV2 <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#mivisionx-winml-yolov2>`_: This sample application shows how to run tiny yolov2(20 classes) with MIVisionX RunTime on Windows.
  * `External Applications <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/apps/#external-application>`_


Neural Net Model Compiler And Optimizer
#######################################

.. image:: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/docs/images/frameworks.png?raw=true
  :align: center
  :width: 800
  :alt: Neural Net Model Compiler And Optimizer

Neural Net Model Compiler & Optimizer `model_compiler <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/model_compiler/#neural-net-model-compiler--optimizer>`_ converts pre-trained neural net models to MIVisionX runtime code for optimized inference.

Samples
########

`MIVisionX samples <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/samples/#samples>`_ using OpenVX and OpenVX extension libraries

GDF - Graph Description Format
-------------------------------

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
#######

`MIVisionX Toolkit <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/toolkit/#mivisionx-toolkit>`_, is a comprehensive set of help tools for neural net creation, development, training, and deployment. The Toolkit provides you with helpful tools to design, develop, quantize, prune, retrain, and infer your neural network work in any framework. The Toolkit is designed to help you deploy your work to any AMD or 3rd party hardware, from embedded to servers.

MIVisionX provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle, from creating a model to deploying them for your target platforms.

Utilities
#########

* `inference_generator <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/inference_generator/#inference-generator>`_: generate inference library from pre-trained CAFFE models
* `loom_shell <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/loom_shell/#radeon-loomsh>`_: an interpreter to prototype 360 degree video stitching applications using a script
* `RunVX <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/runvx/#amd-runvx>`_: command-line utility to execute OpenVX graph described in GDF text file
* `RunCL <https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/utilities/runcl/#amd-runcl>`_: command-line utility to build, execute, and debug OpenCL programs

Prerequisites
##############

    * CPU: SSE4.1 or above CPU, 64-bit
    * GPU: `GFX7 or above <https://rocm.github.io/hardware.html>`_ [optional]
    * APU: Carrizo or above [optional]

**Note:** Some modules in MIVisionX can be built for CPU only. To take advantage of advanced features and modules we recommend using AMD GPUs or AMD APUs.

Windows
--------
    * Windows 10
    * Windows SDK
    * Visual Studio 2017
    * Install the latest drivers and `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0`>_
    * `OpenCV <https://github.com/opencv/opencv/releases/tag/3.4.0>`_
          * Set OpenCV_DIR environment variable to OpenCV/build folder
          * Add %OpenCV_DIR%\x64\vc14\bin or %OpenCV_DIR%\x64\vc15\bin to your PATH

Linux
------

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
-------------------------------------------------
 
For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

**MIVisionX-setup.py**- This scipts builds all the prerequisites required by MIVisionX. The setup script creates a deps folder and installs all the prerequisites, this script only needs to be executed once. If -d option for directory is not given the script will install deps folder in ‘~/’ directory by default, else in the user specified folder.

Prerequisites for running the scripts
---------------------------------------

   * ubuntu 16.04/18.04 or CentOS 7.5/7.6
   * `ROCm supported hardware <https://rocm.github.io/hardware.html>`_
   * `ROCm <https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories>`__

usage:

::

  python MIVisionX-setup.py --directory [setup directory - optional]
                            --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                            --miopen    [MIOpen Version - optional (default:1.8.1)]
                            --ffmpeg    [FFMPEG Installation - optional (default:no) [options:Install ffmpeg - yes]]


**Note:** use --installer **yum** for **CentOS**


Build & Install MIVisionX
##########################

**Windows**
-----------

**Using .msi packages**
~~~~~~~~~~~~~~~~~~~~~~~

    * `MIVisionX-installer.msi <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases>`_: MIVisionX
    * `MIVisionX_WinML-installer.msi <https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases>`_: MIVisionX for WinML

**Using Visual Studio 2017 on 64-bit Windows 10**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * Install `OpenCL_SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0>`_
    * Install `OpenCV <https://github.com/opencv/opencv/releases/tag/3.4.0>`_ with/without `contrib <https://github.com/opencv/opencv_contrib>`_ to support camera capture, image display, & opencv extensions
        * Set OpenCV_DIR environment variable to OpenCV/build folder
        * Add %OpenCV_DIR%\x64\vc14\bin or %OpenCV_DIR%\x64\vc15\bin to your PATH
    * Use MIVisionX.sln to build for x64 platform

**NOTE:** vx_nn is not supported on Windows in this release


**Linux**
---------

**Using apt-get/yum**
~~~~~~~~~~~~~~~~~~~~~

Prerequisites
`````````````

    * Ubuntu 16.04/18.04 or CentOS 7.5/7.6
    * `ROCm supported hardware <https://rocm.github.io/hardware.html>`_
    * `ROCm <https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories>`__

**Ubuntu**
`````````````
::

  sudo apt-get install mivisionx


**CentOS**
`````````````
::

  sudo yum install mivisionx

**Note:**

    * vx_winml is not supported on linux
    * source code will not available with apt-get/yum install
    * executables placed in /opt/rocm/mivisionx/bin and libraries in /opt/rocm/mivisionx/lib
    * OpenVX and module header files into /opt/rocm/mivisionx/include
    * model compiler, toolkit, & samples placed in /opt/rocm/mivisionx
    * Package (.deb & .rpm) install requires OpenCV v3.4.0 to execute AMD OpenCV extensions



Using MIVisionX-setup.py and CMake on Linux (Ubuntu 16.04/18.04 or CentOS 7.5/7.6) with ROCm
----------------------------------------------------------------------------------------------
    * Install `ROCm <https://rocm.github.io/ROCmInstall.html>`__
    * Use the below commands to setup and build MIVisionX


::

  git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
  cd MIVisionX


::

  python MIVisionX-setup.py --directory [setup directory - optional]
                          --installer [Package management tool - optional (default:apt-get) [options: Ubuntu:apt-get;CentOS:yum]]
                          --miopen    [MIOpen Version - optional (default:1.8.1)]
                          --ffmpeg    [FFMPEG Installation - optional (default:no) [options:Install ffmpeg - yes]]    



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


Using CMake on Linux (Ubuntu 16.04 64-bit or CentOS 7.5 / 7.6 ) with ROCm
-------------------------------------------------------------------------------

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
##########################

Linux
------

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
######

MIVisionX provides developers with docker images for Ubuntu 16.04, Ubuntu 18.04, CentOS 7.5, & CentOS 7.6. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

MIVisionX Docker
----------------

   * `Ubuntu 16.04 <https://hub.docker.com/r/mivisionx/ubuntu-16.04>`_
   * `Ubuntu 18.04 <https://hub.docker.com/r/mivisionx/ubuntu-18.04>`_
   * `CentOS 7.5 <https://hub.docker.com/r/mivisionx/centos-7.5>`_
   * `CentOS 7.6 <https://hub.docker.com/r/mivisionx/centos-7.6>`_

Docker Workflow Sample on Ubuntu 16.04/18.04
--------------------------------------------

**Prerequisites**

   * Ubuntu 16.04/18.04
   * `rocm supported hardware <https://rocm.github.io/hardware.html>`_


Workflow
--------

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
##############

Known issues
-------------

   * Package (.deb & .rpm) install requires OpenCV v3.4.0 to execute AMD OpenCV extensions

Tested configurations
----------------------

    * Windows 10
    * Linux: Ubuntu - 16.04/18.04 & CentOS - 7.5/7.6
    * ROCm: rocm-dkms - 2.6.22
    * rocm-cmake - `github master:ac45c6e <https://github.com/RadeonOpenCompute/rocm-cmake/tree/master>`_
    * MIOpenGEMM - `1.1.5 <https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5>`_
    * MIOpen - `2.0.0 <https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.0.0>`_
    * Protobuf - `V3.5.2 <https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.2>`_
    * OpenCV - `3.4.0 <https://github.com/opencv/opencv/releases/tag/3.4.0>`_
    * Dependencies for all the above packages