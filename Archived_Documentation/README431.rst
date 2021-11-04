AMD ROCm™ v4.3.1 Point Release Notes
====================================

This document describes the features, fixed issues, and information
about downloading and installing the AMD ROCm™ software.

It also covers known issues in this release.

List of Supported Operating Systems
-----------------------------------

The AMD ROCm platform supports the following operating systems:

============== ===================
OS             Kernel
============== ===================
SLES15 SP3     5.3.18-24.49
RHEL 7.9       3.10.0-1160.6.1.el7
CentOS 7.9     3.10.0-1127
RHEL 8.4       4.18.0-193.1.1.el8
CentOS 8.3     4.18.0-193.el8
Ubuntu 18.04.5 5.4.0-71-generic
Ubuntu 20.04.2 5.8.0-48-generic
============== ===================

What's New in This Release
--------------------------

The ROCm v4.3.1 release consists of the following enhancements:

Support for RHEL V8.4
~~~~~~~~~~~~~~~~~~~~~

This release extends support for RHEL v8.4.

Support for SLES V15 Service Pack 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release extends support for SLES v15 SP3.

Pass Manager Update
~~~~~~~~~~~~~~~~~~~

In the AMD ROCm 4.3.1 release, the ROCm compiler uses the legacy pass
manager, by default, to provide a better performance experience with
some workloads.

Previously, in ROCm v4.3, the default choice for the ROCm compiler was
the new pass manager.

For more information about legacy and new pass managers, see
http://llvm.org.

Known Issues in This Release
----------------------------

General Userspace and Application Freeze on MI25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some workloads on MI25, general user space and application freeze
are observed, and the GPU resets intermittently. Note, the freeze may
take hours to reproduce.

This issue is under active investigation, and no workarounds are
available currently.

hipRTC - File Not Found Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

hipRTC may fail, and users may encounter the following error:

::

       <built-in>:1:10: fatal error: '__clang_hip_runtime_wrapper.h' file not found
       #include "__clang_hip_runtime_wrapper.h"

Suggested Workarounds
^^^^^^^^^^^^^^^^^^^^^

-  Set LLVM_PATH in the environment to /llvm. Note, if ROCm is installed
   at the default location, then LLVM_PATH must be set to
   /opt/rocm/llvm.

-  Add “-I /llvm/lib/clang/13.0.0/include/” to compiler options in the
   call to hiprtcCompileProgram (). Note, this workaround requires the
   following changes in the code:

   ::

        // set NUM_OPTIONS to one more than the number of options that was previously required
        const char* options[NUM_OPTIONS];
        // fill other options[] here
        std::string sarg = "-I/opt/rocm/llvm/lib/clang/13.0.0/include/";
        options[NUM_OPTIONS - 1] = sarg.c_str();
        hiprtcResult compileResult{hiprtcCompileProgram(prog, NUM_OPTIONS, options)};"

AMD ROCm™ v4.3 Release Notes
============================

This document describes the features, fixed issues, and information
about downloading and installing the AMD ROCm™ software. It also covers
known issues and deprecations in this release.


ROCm Installation Updates
-------------------------

Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm platform is designed to support the following operating
systems:

============== ===================
OS             Kernel
============== ===================
SLES15 SP3     5.3.18-24.49
RHEL 7.9       3.10.0-1160.6.1.el7
CentOS 7.9     3.10.0-1127
RHEL 8.4       4.18.0-193.1.1.el8
CentOS 8.3     4.18.0-193.el8
Ubuntu 18.04.5 5.4.0-71-generic
Ubuntu 20.04.2 5.8.0-48-generic
============== ===================

Fresh Installation of AMD ROCM V4.3 Recommended
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete uninstallation of previous ROCm versions is required before
installing a new version of ROCm. **An upgrade from previous releases to
AMD ROCm v4.3 is not supported**. For more information, refer to the AMD
ROCm Installation Guide at

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

**Note**: AMD ROCm release v3.3 or prior releases are not fully
compatible with AMD ROCm v3.5 and higher versions. You must perform a
fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or
older to 3.5 or higher versions and vice-versa.

**Note**: *render* group is required only for Ubuntu v20.04. For all
other ROCm supported operating systems, continue to use video group.

-  For ROCm v3.5 and releases thereafter, the clinfo path is changed to
   /opt/rocm/opencl/bin/clinfo.

-  For ROCm v3.3 and older releases, the clinfo path remains
   /opt/rocm/opencl/bin/x86_64/clinfo.   ## ROCm Multi-Version
   Installation Update

With the AMD ROCm v4.3 release, the following ROCm multi-version
installation changes apply:

The meta packages rocm-dkms are now deprecated for multi-version ROCm
installs. For example, rocm-dkms3.7.0, rocm-dkms3.8.0.

-  Multi-version installation of ROCm should be performed by installing
   rocm-dev using each of the desired ROCm versions. For example,
   rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.

-  The rock-dkms loadable kernel modules should be installed using a
   single rock-dkms package.

-  ROCm v3.9 and above will not set any ldconfig entries for ROCm
   libraries for multi-version installation. Users must set
   LD_LIBRARY_PATH to load the ROCm library version of choice.

**NOTE**: The single version installation of the ROCm stack remains the
same. The rocm-dkms package can be used for single version installs and
is not deprecated at this time.

Support for Enviornment Modules
-------------------------------

Environment modules are now supported. This enhancement in the ROCm v4.3
release enables users to switch between ROCm v4.2 and ROCm v4.3 easily
and efficiently.

For more information about installing environment modules, refer to

https://modules.readthedocs.io/en/latest/

AMD ROCm Documentation Updates
-------------------------------

AMD ROCm Installation Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm Installation Guide in this release includes:

-  Supported Environments

-  Installation Instructions

-  HIP Installation Instructions

For more information, refer to the ROCm documentation website at:

https://rocmdocs.amd.com/en/latest/

AMD ROCm - HIP Documentation Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access the following documentation, see

https://github.com/RadeonOpenCompute/ROCm/tree/roc-4.3.x

-  HIP Programming Guide v4.3


-  HIP API Guide v4.3


-  HIP-Supported CUDA API Reference Guide v4.3


-  **NEW** - AMD ROCm Compiler Reference Guide v4.3


-  HIP FAQ

https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq

ROCm Data Center User and API Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access the following documentation, see

https://github.com/RadeonOpenCompute/ROCm/tree/roc-4.3.x

-  ROCm Data Center Tool User Guide

   -  Prometheus (Grafana) Integration with Automatic Node Detection


-  ROCm Data Center Tool API Guide


ROCm SMI API Documentation Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access the following documentation, see

https://github.com/RadeonOpenCompute/ROCm/tree/roc-4.3.x

-  ROCm SMI API Guide


ROC Debugger User and API Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access the following documentation, see

https://github.com/RadeonOpenCompute/ROCm/tree/roc-4.3.x

-  ROC Debugger User Guide


-  Debugger API Guide


General AMD ROCm Documentation Links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access the following links for more information:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supped platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html

-  For AMD ROCm Release History, see

https://rocmdocs.amd.com/en/latest/Current_Release_Notes/ROCm-Version-History.html

.. _whats-new-in-this-release-1:

What's New in This Release
---------------------------

HIP Enhancements
~~~~~~~~~~~~~~~~~~~~

HIP Versioning Update
^^^^^^^^^^^^^^^^^^^^^^^^^^

The HIP version definition is updated from the ROCm v4.2 release as
follows:

::

       HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + 
       HIP_VERSION_PATCH)

The HIP version can be queried from a HIP API call

::

       hipRuntimeGetVersion(&runtimeVersion);  

**Note**: The version returned will be greater than the version in
previous ROCm releases.

Support for Managed Memory Allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HIP now supports and automatically manages Heterogeneous Memory
Management (HMM) allocation. The HIP application performs a capability
check before making the managed memory API call hipMallocManaged.

**Note**: The *managed* keyword is unsupported currently.

::

       int managed_memory = 0;
       HIPCHECK(hipDeviceGetAttribute(&managed_memory,
        hipDeviceAttributeManagedMemory,p_gpuDevice));
       if (!managed_memory ) {
       printf ("info: managed memory access not supported on the device %d\n Skipped\n", p_gpuDevice);
       }
       else {
        HIPCHECK(hipSetDevice(p_gpuDevice));
       HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));
       . . .
       }

Kernel Enqueue Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Developers can control kernel command serialization from the host using
the following environment variable, AMD_SERIALIZE_KERNEL

-  AMD_SERIALIZE_KERNEL = 1, Wait for completion before enqueue,

-  AMD_SERIALIZE_KERNEL = 2, Wait for completion after enqueue,

-  AMD_SERIALIZE_KERNEL = 3, Both.

This environment variable setting enables HIP runtime to wait for GPU
idle before/after any GPU command.

NUMA-aware Host Memory Allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Non-Uniform Memory Architecture (NUMA) policy determines how memory
is allocated and selects a CPU closest to each GPU.

NUMA also measures the distance between the GPU and CPU devices. By
default, each GPU selects a Numa CPU node that has the least NUMA
distance between them; the host memory is automatically allocated
closest to the memory pool of the NUMA node of the current GPU device.

Note, using the *hipSetDevice* API with a different GPU provides access
to the host allocation. However, it may have a longer NUMA distance.

New Atomic System Scope Atomic Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HIP now provides new APIs with \_system as a suffix to support system
scope atomic operations. For example, atomicAnd atomic is dedicated to
the GPU device, and atomicAnd_system allows developers to extend the
atomic operation to system scope from the GPU device to other CPUs and
GPU devices in the system.

For more information, refer to the HIP Programming Guide at,

https://github.com/RadeonOpenCompute/ROCm/tree/roc-4.3.x

Indirect Function Call and C++ Virtual Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the new release of the ROCm compiler supports indirect function
calls and C++ virtual functions on a device, there are some known
limitations and issues.

**Limitations**

-  An address to a function is device specific. Note, a function address
   taken on the host can not be used on a device, and a function address
   taken on a device can not be used on the host. On a system with
   multiple devices, an address taken on one device can not be used on a
   different device.

-  C++ virtual functions only work on the device where the object was
   constructed.

-  Indirect call to a device function with function scope shared memory
   allocation is not supported. For example, LDS.

-  Indirect call to a device function defined in a source file different
   than the calling function/kernel is only supported when compiling the
   entire program with -fgpu-rdc.

**Known Issues in This Release**

-  Programs containing kernels with different launch bounds may crash
   when making an indirect function call. This issue is due to a
   compiler issue miscalculating the register budget for the callee
   function.

-  Programs may not work correctly when making an indirect call to a
   function that uses more resources. For example, scratch memory,
   shared memory, registers made available by the caller.

-  Compiling a program with objects with pure or deleted virtual
   functions on the device will result in a linker error. This issue is
   due to the missing implementation of some C++ runtime functions on
   the device.

-  Constructing an object with virtual functions in private or shared
   memory may crash the program due to a compiler issue when generating
   code for the constructor.

ROCm Data Center Tool
---------------------

Prometheus (Grafana) Integration with Automatic Node Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm Data Center (RDC) tool enables you to use Consul to discover
the rdc_prometheus service automatically. Consul is “a service mesh
solution providing a full-featured control plane with service discovery,
configuration, and segmentation functionality.” For more information,
refer to their website at https://www.consul.io/docs/intro.

The ROCm Data Center Tool uses Consul for health checks of RDC’s
integration with the Prometheus plug-in (rdc_prometheus), and these
checks provide information on its efficiency.

Previously, when a new compute node was added, users had to change
prometheus_targets.json to use Consul manually. Now, with the Consul
agent integration, a new compute node can be discovered automatically.

https://github.com/RadeonOpenCompute/ROCm/tree/roc-4.3.x

Coarse Grain Utilization
~~~~~~~~~~~~~~~~~~~~~~~~

This feature provides a counter that displays the coarse grain GPU usage
information, as shown below.

Sample output

::

           $ rocm_smi.py --showuse
           ============================== % time GPU is busy =============================
                  GPU[0] : GPU use (%): 0
                  GPU[0] : GFX Activity: 3401

Add 64-bit Energy Accumulator In-band
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature provides an average value of energy consumed over time in a
free-flowing RAPL counter, a 64-bit Energy Accumulator.

Sample output

::

       $ rocm_smi.py --showenergycounter
       =============================== Consumed Energy ================================
       GPU[0] : Energy counter: 2424868
       GPU[0] : Accumulated Energy (uJ): 0.0   

Support for Continuous Clocks Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm SMI will support continuous clock values instead of the previous
discrete levels. Moving forward the updated sysfs file will consist of
only MIN and MAX values and the user can set the clock value in the
given range.

Sample output:

::

       $ rocm_smi.py --setsrange 551 1270 
       Do you accept these terms? [y/N] y                                                                                    
       ============================= Set Valid sclk Range=======
       GPU[0]          : Successfully set sclk from 551(MHz) to 1270(MHz)                                                     
       GPU[1]          : Successfully set sclk from 551(MHz) to 1270(MHz)                                                     
       =========================================================================
                          
       $ rocm_smi.py --showsclkrange                                                                                                                                                                    
       ============================ Show Valid sclk Range======                     

       GPU[0]          : Valid sclk range: 551Mhz - 1270Mhz                                                                  
       GPU[1]          : Valid sclk range: 551Mhz - 1270Mhz             

Memory Utilization Counters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature provides a counter display memory utilization information
as shown below.

Sample output

::

          $ rocm_smi.py --showmemuse
       ========================== Current Memory Use ==============================

       GPU[0] : GPU memory use (%): 0
       GPU[0] : Memory Activity: 0

Performance Determinism
~~~~~~~~~~~~~~~~~~~~~~~

ROCm SMI supports performance determinism as a unique mode of operation.
Performance variations are minimal as this enhancement allows users to
control the entry and exit to set a soft maximum (ceiling) for the GFX
clock.

Sample output

::

       $ rocm_smi.py --setperfdeterminism 650
       cat pp_od_clk_voltage
       GFXCLK:                
       0: 500Mhz
       1: 650Mhz *
       2: 1200Mhz
       $ rocm_smi.py --resetperfdeterminism    

**Note**: The idle clock will not take up higher clock values if no
workload is running. After enabling determinism, users can run a GFX
workload to set performance determinism to the desired clock value in
the valid range.

::

   * GFX clock could either be less than or equal to the max value set in this mode. GFX clock will be at the max clock set in this mode only when required by the running     workload.

   * VDDGFX will be higher by an offset (75mv or so based on PPTable) in the determinism mode.

HBM Temperature Metric Per Stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature will enable ROCm SMI to report all HBM temperature values
as shown below.

Sample output

::

       $ rocm_smi.py –showtemp
       ================================= Temperature =================================
       GPU[0] : Temperature (Sensor edge) (C): 29.0
       GPU[0] : Temperature (Sensor junction) (C): 36.0
       GPU[0] : Temperature (Sensor memory) (C): 45.0
       GPU[0] : Temperature (Sensor HBM 0) (C): 43.0
       GPU[0] : Temperature (Sensor HBM 1) (C): 42.0
       GPU[0] : Temperature (Sensor HBM 2) (C): 44.0
       GPU[0] : Temperature (Sensor HBM 3) (C): 45.0

ROCm Math and Communication Libraries
-------------------------------------

rocBLAS
~~~~~~~

**Optimizations**

-  Improved performance of non-batched and batched rocblas_Xgemv for
   gfx908 when m <= 15000 and n <= 15000

-  Improved performance of non-batched and batched rocblas_sgemv and
   rocblas_dgemv for gfx906 when m <= 6000 and n <= 6000

-  Improved the overall performance of non-batched and batched
   rocblas_cgemv for gfx906

-  Improved the overall performance of rocblas_Xtrsv

For more information, refer to

https://rocblas.readthedocs.io/en/master/

rocRAND
~~~~~~~

**Enhancements**

-  gfx90a support added

-  gfx1030 support added

-  gfx803 supported re-enabled

**Fixed**

-  Memory leaks in Poisson tests has been fixed.

-  Memory leaks when generator has been created but setting
   seed/offset/dimensions display an exception has been fixed.

For more information, refer to

https://rocrand.readthedocs.io/en/latest/

rocSOLVER
~~~~~~~~~

**Enhancements**

Linear solvers for general non-square systems:

-  GELS now supports underdetermined and transposed cases

-  Inverse of triangular matrices

-  TRTRI (with batched and strided_batched versions)

-  Out-of-place general matrix inversion

-  GETRI_OUTOFPLACE (with batched and strided_batched versions)

-  Argument names for the benchmark client now match argument names from
   the public API

**Fixed Issues**

-  Known issues with Thin-SVD. The problem was identified in the test
   specification, not in the thin-SVD implementation or the rocBLAS
   gemm_batched routines.

-  Benchmark client no longer crashes as a result of leading dimension
   or stride arguments not being provided on the command line.

**Optimizations**

-  Improved general performance of matrix inversion (GETRI)

For more information, refer to

https://rocsolver.readthedocs.io/en/latest/

rocSPARSE
~~~~~~~~~

**Enhancements**

-  (batched) tridiagonal solver with and without pivoting

-  dense matrix sparse vector multiplication (gemvi)

-  support for gfx90a

-  sampled dense-dense matrix multiplication (sddmm)

**Improvements**

-  client matrix download mechanism

-  boost dependency in clients removed

For more information, refer to

https://rocsparse.readthedocs.io/en/latest/usermanual.html#rocsparse-gebsrmv

hipBLAS
~~~~~~~

**Enhancements**

-  Added *hipblasStatusToString*

**Fixed**

-  Added catch() blocks around API calls to prevent the leak of C++
   exceptions

rocFFT
~~~~~~

**Changes**

-  Re-split device code into single-precision, double-precision, and
   miscellaneous kernels.

**Fixed Issues**

-  double-precision planar->planar transpose.

-  3D transforms with unusual strides, for SBCC-optimized sizes.

-  Improved buffer placement logic.

For more information, refer to

https://rocfft.readthedocs.io/en/rocm-4.3.0/

hipFFT
~~~~~~

**Fixed Issues**

-  CMAKE updates

-  Added callback API in hipfftXt.h header.

rocALUTION
~~~~~~~~~~

**Enhancements**

-  Support for gfx90a target

-  Support for gfx1030 target

**Improvements**

-  Install script

For more information, refer to

rocTHRUST
~~~~~~~~~

**Enhancements**

-  Updated to match upstream Thrust 1.11

-  gfx90a support added

-  gfx803 support re-enabled

hipCUB

Enhancements

-  DiscardOutputIterator to backend header

ROCProfiler Enhancements
------------------------

Tracing Multiple MPI Ranks
~~~~~~~~~~~~~~~~~~~~~~~~~~

When tracing multiple MPI ranks in ROCm v4.3, users must use the form:

::

       mpirun ... <mpi args> ... rocprof ... <rocprof args> ... application ... <application args>
       

**NOTE**: This feature differs from ROCm v4.2 (and lower), which used
“rocprof … mpirun … application”.

This change was made to enable ROCProfiler to handle process forking
better and launching via mpirun (and related) executables.

From a user perspective, this new execution mode requires:

1. Generation of trace data per MPI (or process) rank.

2. Use of a new `“merge_traces.sh” utility
   script <https://github.com/ROCm-Developer-Tools/rocprofiler/blob/rocm-4.3.x/bin/merge_traces.sh>`__
   to combine traces from multiple processes into a unified trace for
   profiling.

For example, to accomplish step #1, ROCm provides a simple bash wrapper
that demonstrates how to generate a unique output directory per process:

::

       $ cat wrapper.sh
       #! /usr/bin/env bash
       if [[ -n ${OMPI_COMM_WORLD_RANK+z} ]]; then
         # mpich
         export MPI_RANK=${OMPI_COMM_WORLD_RANK}
       elif [[ -n ${MV2_COMM_WORLD_RANK+z} ]]; then
         # ompi
         export MPI_RANK=${MV2_COMM_WORLD_RANK}
       fi
       args="$*"
       pid="$$"
       outdir="rank_${pid}_${MPI_RANK}"
       outfile="results_${pid}_${MPI_RANK}.csv"
       eval "rocprof -d ${outdir} -o ${outdir}/${outfile} $*"

This script:

-  Determines the global MPI rank (implemented here for OpenMPI and
   MPICH only)

-  Determines the process id of the MPI rank

-  Generates a unique output directory using the two

To invoke this wrapper, use the following command:

::

       mpirun <mpi args> ./wrapper.sh --hip-trace <application> <args>

This generates an output directory for each used MPI rank. For example,

::

       $ ls -ld rank_* | awk {'print $5" "$9'}
       4096 rank_513555_0
       4096 rank_513556_1

Finally, these traces may be combined using the `merge traces
script <https://github.com/ROCm-Developer-Tools/rocprofiler/blob/rocm-4.3.x/bin/merge_traces.sh>`__.
For example,

::

       $  ./merge_traces.sh -h
       Script for aggregating results from multiple rocprofiler out directries.
       Full path: /opt/rocm/bin/merge_traces.sh
       Usage:
       merge_traces.sh -o <outputdir> [<inputdir>...]

Use the following input arguments to the merge_traces.sh script to
control which traces are merged and where the resulting merged trace is
saved.

-  -o <*outputdir*> - output directory where the results are aggregated.

-  <*inputdir*>… - space-separated list of rocprofiler directories. If
   not specified, CWD is used.

For example, if an output directory named “unified” was supplied to the
``merge_traces.sh`` script, the file ‘unified/results.json’ will be
generated, and the contains trace data from both MPI ranks.

Known issue for ROCProfiler

| Collecting several counter collection passes (multiple “pmc:” lines in
  an counter input file) is not supported in a single run.
| The workaround is to break the multiline counter input file into
  multiple single-line counter input files and execute runs.

.. _known-issues-in-this-release-1:

Known Issues in This Release
------------------------------

The following are the known issues in this release.

Upgrade to AMD ROCm v4.3 Not Supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An upgrade from previous releases to AMD ROCm v4.2 is not supported.
Complete uninstallation of previous ROCm versions is required before
installing a new version of ROCm.

\_LAUNCH BOUNDS_Ignored During Kernel Launch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HIP runtime returns the hipErrorLaunchFailure error code when an
application tries to launch kernel with a block size larger than the
launch bounds mentioned during compile time. If no launch bounds were
specified during the compile time, the default value of 1024 is assumed.
Refer to the HIP trace for more information about the failing kernel. A
sample error in the trace is shown below:

Snippet of the HIP trace

::

       :3:devprogram.cpp           :2504: 2227377746776 us: Using Code Object V4.
       :3:hip_module.cpp           :361 : 2227377768546 us: 7670 : [7f7c6eddd180] ihipModuleLaunchKernel ( 0x0x16fe080, 2048, 1, 1,        1024, 1, 1, 0, stream:<null>, 0x7ffded8ad260, char array:<null>, event:0, event:0, 0, 0 )
       :1:hip_module.cpp           :254 : 2227377768572 us: Launch params (1024, 1, 1) are larger than launch bounds (64) for      kernel _Z8MyKerneliPd
       :3:hip_platform.cpp         :667 : 2227377768577 us: 7670 : [7f7c6eddd180] ihipLaunchKernel: Returned hipErrorLaunchFailure         :
       :3:hip_module.cpp           :493 : 2227377768581 us: 7670 : [7f7c6eddd180] hipLaunchKernel: Returned hipErrorLaunchFailure :

There is no known workaround at this time.

PYCACHE Folder Exists After ROCM SMI Library Uninstallation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users may observe that the /opt/rocm-x/bin/**pycache** folder continues
to exist even after the rocm_smi_lib uninstallation. Workaround: Delete
the /opt/rocm-x/bin/**pycache** folder manually before uninstalling
rocm_smi_lib.



