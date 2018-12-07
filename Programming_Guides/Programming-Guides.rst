.. _Programming-Guides:

=================
Programming Guide
=================

ROCm Languages
================


ROCm, Lingua Franca,  C++, OpenCL and Python
#############################################
The open-source ROCm stack offers multiple programming-language choices. The goal is to give you a range of tools to help solve the
problem at hand. Here, we describe some of the options and how to choose among them.



HCC: Heterogeneous Compute Compiler
####################################
What is the Heterogeneous Compute (HC) API? It’s a C++ dialect with extensions to launch kernels and manage accelerator memory. It closely tracks the evolution of C++ and will incorporate parallelism and concurrency features as the C++ standard does. For example, HC includes early support for the C++17 Parallel STL. At the recent ISO C++ meetings in Kona and Jacksonville, the committee was excited about enabling the language to express all forms of parallelism, including multicore CPU, SIMD and GPU. We’ll be following these developments closely, and you’ll see HC move quickly to include standard C++ capabilities.

The Heterogeneous Compute Compiler (HCC) provides two important benefits:

**Ease of development**

 * A full C++ API for managing devices, queues and events
 * C++ data containers that provide type safety, multidimensional-array indexing and automatic data management
 * C++ kernel-launch syntax using parallel_for_each plus C++11 lambda functions
 * A single-source C++ programming environment---the host and source code can be in the same source file and use the same C++        	language;templates and classes work naturally across the host/device boundary
 * HCC generates both host and device code from the same compiler, so it benefits from a consistent view of the source code using the
   same Clang-based language parser

**Full control over the machine**

 * Access AMD scratchpad memories (“LDS”)
 * Fully control data movement, prefetch and discard
 * Fully control asynchronous kernel launch and completion
 * Get device-side dependency resolution for kernel and data commands (without host involvement)
 * Obtain HSA agents, queues and signals for low-level control of the architecture using the HSA Runtime API
 * Use [direct-to-ISA](https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA) compilation

When to Use HC
###############
Use HC when you're targeting the AMD ROCm platform: it delivers a single-source, easy-to-program C++ environment without compromising
performance or control of the machine.

HIP: Heterogeneous-Computing Interface for Portability
#########################################################
What is Heterogeneous-Computing Interface for Portability (HIP)? It’s a C++ dialect designed to ease conversion of Cuda applications to portable C++ code. It provides a C-style API and a C++ kernel language. The C++ interface can use templates and classes across the
host/kernel boundary.

The Hipify tool automates much of the conversion work by performing a source-to-source transformation from Cuda to HIP. HIP code can run on AMD hardware (through the HCC compiler) or Nvidia hardware (through the NVCC compiler) with no performance loss compared with the original Cuda code.

Programmers familiar with other GPGPU languages will find HIP very easy to learn and use. AMD platforms implement this language using the HC dialect described above, providing similar low-level control over the machine.

When to Use HIP
################
Use HIP when converting Cuda applications to portable C++ and for new projects that require portability between AMD and Nvidia. HIP provides a C++ development language and access to the best development tools on both platforms.

OpenCL™: Open Compute Language
################################
What is OpenCL ?  It’s a framework for developing programs that can execute across a wide variety of heterogeneous platforms. AMD, Intel
and Nvidia GPUs support version 1.2 of the specification, as do x86 CPUs and other devices (including FPGAs and DSPs). OpenCL provides a C run-time API and C99-based kernel language.

When to Use OpenCL
####################
Use OpenCL when you have existing code in that language and when you need portability to multiple platforms and devices. It runs on
Windows, Linux and Mac OS, as well as a wide variety of hardware platforms (described above).

Anaconda Python With Numba
###########################
What is Anaconda ?  It’s a modern open-source analytics platform powered by Python. Continuum Analytics, a ROCm platform partner,  is the driving force behind it. Anaconda delivers high-performance capabilities including acceleration of HSA APUs, as well as
ROCm-enabled discrete GPUs via Numba. It gives superpowers to the people who are changing the world.

Numba
#######
Numba gives you the power to speed up your applications with high-performance functions written directly in Python. Through a few
annotations, you can just-in-time compile array-oriented and math-heavy Python code to native machine instructions---offering
performance similar to that of C, C++ and Fortran---without having to switch languages or Python interpreters.

Numba works by generating optimized machine code using the LLVM compiler infrastructure at import time, run time or statically
(through the included Pycc tool). It supports Python compilation to run on either CPU or GPU hardware and is designed to integrate with Python scientific software stacks, such as NumPy.

  * `Anaconda® with Numba acceleration <http://numba.pydata.org/numba-doc/latest/index.html>`_

When to Use Anaconda
#####################
Use Anaconda when you’re handling large-scale data-analytics,
scientific and engineering problems that require you to manipulate
large data arrays.

Wrap-Up
#######
From a high-level perspective, ROCm delivers a rich set of tools that
allow you to choose the best language for your application.

 * HCC (Heterogeneous Compute Compiler) supports HC dialects
 * HIP is a run-time library that layers on top of HCC (for AMD ROCm platforms; for Nvidia, it uses the NVCC compiler)
 * The following will soon offer native compiler support for the GCN ISA:
    * OpenCL 1.2+
    * Anaconda (Python) with Numba

All are open-source projects, so you can employ a fully open stack from the language down to the metal. AMD is committed to providing an open ecosystem that gives developers the ability to choose; we are excited about innovating quickly using open source and about
interacting closely with our developer community. More to come soon!

Table Comparing Syntax for Different Compute APIs
##################################################



+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Term 	          	|  CUDA 	|       HIP       |       HC 	        |      C++AMP 	         |  OpenCL                   |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Device 	        | int deviceId  | int deviceId 	  | hc::accelerator     |  concurrency::	 |  cl_device                |
|			|		|		  |		        |  accelerator 	         |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Queue 	                | cudaStream_t  |   hipStream_t   | hc:: 	        | concurrency::          | cl_command_queue          |
|			|		|	     	  | accelerator_view    | accelerator_view       |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Event 	                | cudaEvent_t 	|   hipEvent_t 	  | hc::                | concurrency::          |                           |
|                       |               |                 | completion_future   | completion_future      |   cl_event                |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Memory                 |   void * 	|    void * 	  |void *; hc::array;   | concurrency::array;    |   cl_mem                  |
|			|		|                 |hc::array_view       |concurrency::array_view |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |     grid      |    grid         |	extent	        |	      extent	 |	   NDRange	     |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
|                       |    block      |    block	  |      tile           |	       tile 	 |	  work-group	     |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
|                       |    thread     |    thread       |      thread         |	      thread 	 |	work-item            |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
|                       |     warp      |    warp         |    wavefront        |	       N/A	 |  sub-group                |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Thread index           |threadIdx.x    | hipThreadIdx_x  |  t_idx.local[0]     |    t_idx.local[0]      |  get_local_id(0)          |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Block index            | blockIdx.x    | hipBlockIdx_x   |  t_idx.tile[0]      |    t_idx.tile[0]       | get_group_id(0)           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Block  dim             | blockDim.x    | hipBlockDim_x   | t_ext.tile_dim[0]   |  t_idx.tile_dim0       |get_local_size(0)          |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Grid-dim               | gridDim.x     | hipGridDim_x    |   	t_ext[0]        |      t_ext[0]          |get_global_size(0)         |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Device Function        | __device__    | __device__      |[[hc]] (detected     |                        |Implied in device          |
|                       |               |                 |automatically in     |    restrict(amp)       |Compilation                |
|                       |               |                 |many case)           |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Host Function          | __host_       |__host_ (default)|[[cpu]] (default)    |                        |Implied in host            |
|                       |  (default)    |                 |                     |  strict(cpu) (default) |Compilation                |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
| Host +                |  __host__     |  __host_        |[[hc]] [[cpu]]       |                        |No equivalent              |
| Device                |  __device__   | __device__      |                     |  restrict(amp,cpu)     |                           |
| Function              |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Kernel Launch          |               |                 |                     | concurrency::          |                           |
|                       |   <<< >>>     | hipLaunchKernel |hc::                 | parallel_for_each      |clEnqueueND-               |
|                       |               |                 |parallel_for_each    |                        |RangeKernel                |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |                        |                           |
|Global Memory          |  __global__   |   __global__    |Unnecessary/         |  Unnecessary/Implied   |  __global                 |
|                       |               |                 |Implied              |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |                        |                           |
|Group Memory           | __shared__    | __shared__      | tile_static         |   tile_static          |   __local                 |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |Unnecessary/         |                        |                           |
|Constant               | __constant__  |   __constant__  |Implied              |Unnecessary / Implied   |   __constant              |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |__syncthreads  |__syncthreads    |tile_static.barrier()| 	t_idx.barrier()  |barrier(CLK_LOCAL_MEMFENCE)|
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |   concurrency::        |                           |
|Atomic Builtins        | atomicAdd     |  atomicAdd      |hc::atomic_fetch_add |   atomic_fetch_add     |      atomic_add           |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |                        |                           |
|Precise Math           |  cos(f)       |   cos(f)        | hc::                |   concurrency::        |      	cos(f)       |
|                       |               |                 | precise_math::cos(f)|   precise_math::cos(f) |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |hc::fast_math::cos(f)|   concurrency::        |                           |
|Fast Math              | __cos(f)      |  __cos(f)       |                     |   fast_math::cos(f)    |    native_cos(f)          |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |hc::                 |concurrency::           |                           |
|Vector                 |   float4      |   	float4    |short_vector::float4 |graphics::float_4       |         float4            |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+



Notes
#######

1. For HC and C++AMP, assume a captured _tiled_ext_ named "t_ext" and captured _extent_ named "ext".  These languages use captured variables to pass information to the kernel rather than using special built-in functions so the exact variable name may vary.
2. The indexing functions (starting with `thread-index`) show the terminology for a 1D grid.  Some APIs use reverse order of xyz / 012 indexing for 3D grids.
3. HC allows tile dimensions to be specified at runtime while C++AMP requires that tile dimensions be specified at compile-time.  Thus hc syntax for tile dims is ``t_ext.tile_dim[0]``  while C++AMP is ``t_ext.tile_dim0``.
4. **From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**


HC Programming Guide
====================

**What is the Heterogeneous Compute (HC) API ?**

It’s a C++ dialect with extensions to launch kernels and manage accelerator memory. It closely tracks the evolution of C++ and will incorporate parallelism and concurrency features as the C++ standard does. For example, HC includes early support for the C++17 Parallel STL. At the recent ISO C++ meetings in Kona and Jacksonville, the committee was excited about enabling the language to express all forms of parallelism, including multicore CPU, SIMD and GPU. We’ll be following these developments closely, and you’ll see HC move quickly to include standard C++ capabilities.

The Heterogeneous Compute Compiler (HCC) provides two important benefits:

Ease of development


   * A full C++ API for managing devices, queues and events
   * C++ data containers that provide type safety, multidimensional-array indexing and automatic data management
   * C++ kernel-launch syntax using parallel_for_each plus C++11 lambda functions
   * A single-source C++ programming environment---the host and source code can be in the same source file and use the same C++     	 language; templates and classes work naturally across the host/device boundary
   * HCC generates both host and device code from the same compiler, so it benefits from a consistent view of the source code using   	   the same Clang-based language parser

Full control over the machine


    * Access AMD scratchpad memories (“LDS”)
    * Fully control data movement, prefetch and discard
    * Fully control asynchronous kernel launch and completion
    * Get device-side dependency resolution for kernel and data commands (without host involvement)
    * Obtain HSA agents, queues and signals for low-level control of the architecture using the HSA Runtime API
    * Use `direct-to-ISA <https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA>`_ compilation

Platform Requirements
######################
Accelerated applications could be run on Radeon discrete GPUs from the Fiji family (AMD R9 Nano, R9 Fury, R9 Fury X, FirePro S9300 x2, Polaris 10, Polaris 11) paired with an Intel Haswell CPU or newer. HCC would work with AMD HSA APUs (Kaveri, Carrizo); however, they are not our main support platform and some of the more advanced compute capabilities may not be available on the APUs.

HCC currently only works on Linux and with the open source ROCK kernel driver and the ROCR runtime (see Installation for details). It will not work with the closed source AMD graphics driver.

Compiler Backends
###################
This backend compiles GPU kernels into native GCN ISA, which can be directly executed on the GPU hardware. It's being actively developed by the Radeon Technology Group in LLVM.

**When to Use HC**
 Use HC when you're targeting the AMD ROCm platform: it delivers a single-source, easy-to-program C++ environment without compromising performance or control of the machine.

Installation
##################

**Prerequisites**

Before continuing with the installation, please make sure any previously installed hcc compiler has been removed from on your system.
Install `ROCm <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html>`_ and make sure it works correctly.

Ubuntu
########

**Ubuntu 14.04**

Support for 14.04 has been deprecated.

**Ubuntu 16.04**

Follow the instruction `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-ubuntu>`_ to setup the ROCm apt repository and install the rocm or the rocm-dev meta-package.

**Fedora 24**

Follow the instruction `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-fedora>`_ to setup the ROCm apt repository and install the rocm or the rocm-dev meta-package.

**RHEL 7.4/CentOS 7**

Follow the instruction `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-fedora>`_ to setup the ROCm apt repository and install the rocm or the rocm-dev meta-package for RHEL/CentOS. Currently, HCC support for RHEL 7.4 and CentOS 7 is experimental and the compiler has to be built from source. Note: CentOS 7 cmake is outdated, will need to use alternate cmake3.

**openSUSE Leap 42.3**

Currently, HCC support for openSUSE is experimental and the compiler has to be built from source.

Download HCC
################
 The project now employs git submodules to manage external components it depends upon. It it advised to add --recursive when you clone the project so all submodules are fetched automatically.

For example

.. code:: sh

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

Build HCC from source
######################

First, install the build dependencies

.. code:: sh

  # Ubuntu 14.04
  sudo apt-get install git cmake make g++  g++-multilib gcc-multilib libc++-dev libc++1 libc++abi-dev libc++abi1 python findutils libelf1 libpci3 file debianutils libunwind8-dev hsa-rocr-dev hsa-ext-rocr-dev hsakmt-roct-dev pkg-config rocm-utils

.. code:: sh

  # Ubuntu 16.04
  sudo apt-get install git cmake make g++  g++-multilib gcc-multilib python findutils libelf1 libpci3 file debianutils libunwind- dev hsa-rocr-dev hsa-ext-rocr-dev hsakmt-roct-dev pkg-config rocm-utils

.. code:: sh

   # Fedora 23/24
   sudo dnf install git cmake make gcc-c++ python findutils elfutils-libelf pciutils-libs file pth rpm-build libunwind-devel hsa- rocr- dev hsa-ext-rocr-dev hsakmt-roct-dev pkgconfig rocm-utils

Clone the HCC source tree

.. code:: sh

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

Create a build directory and run cmake in that directory to configure the build

.. code:: sh

  mkdir build;
  cd build;
  cmake ../hcc

Compile HCC

.. code:: sh

  make -j

Run the unit tests

.. code:: sh

  make test

Create an installer package (DEB or RPM file)

.. code:: sh

  make package



To configure and build HCC from source, use the following steps

.. code:: sh

  mkdir -p build; cd build
  # NUM_BUILD_THREADS is optional
  # set the number to your CPU core numbers time 2 is recommended
  # in this example we set it to 96
    cmake -DNUM_BUILD_THREADS=96 \
    -DCMAKE_BUILD_TYPE=Release \
  ..
  make

To install it, use the following steps

.. code:: sh

  sudo make install

Use HCC
########

For C++AMP source codes

.. code:: sh

  hcc `clamp-config --cxxflags --ldflags` foo.cpp

**WARNING: From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**

For HC source codes

.. code:: sh

 hcc `hcc-config --cxxflags --ldflags` foo.cpp

In case you build HCC from source and want to use the compiled binaries directly in the build directory:

For C++AMP source codes

.. code:: sh

  # notice the --build flag
  bin/hcc `bin/clamp-config --build --cxxflags --ldflags` foo.cpp

**WARNING: From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**

For HC source codes

.. code:: sh

  # notice the --build flag
  bin/hcc `bin/hcc-config --build --cxxflags --ldflags` foo.cpp

**Compiling for Different GPU Architectures**

By default, HCC will auto-detect all the GPU's local to the compiling machine and set the correct GPU architectures. Users could use the --amdgpu-target=<GCN Version> option to compile for a specific architecture and to disable the auto-detection. The following table shows the different versions currently supported by HCC.

There exists an environment variable HCC_AMDGPU_TARGET to override the default GPU architecture globally for HCC; however, the usage of this environment variable is NOT recommended as it is unsupported and it will be deprecated in a future release.

============ ================== ==============================================================
GCN Version   GPU/APU Family       Examples of Radeon GPU

============ ================== ==============================================================
gfx701        GFX7               FirePro W8100, FirePro W9100, Radeon R9 290, Radeon R9 390

gfx801        Carrizo APU        FX-8800P

gfx803        GFX8               R9 Fury, R9 Fury X, R9 Nano, FirePro S9300 x2, Radeon RX 480,
                                 Radeon RX 470, Radeon RX 460

gfx900        GFX9                 Vega10

============ ================== ==============================================================

Multiple ISA
#############

HCC now supports having multiple GCN ISAs in one executable file. You can do it in different ways: **use :: --amdgpu-target= command line option**
It's possible to specify multiple --amdgpu-target= option.

Example

.. code:: sh

 # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
 # be produced
 hcc `hcc-config --cxxflags --ldflags` \
    --amdgpu-target=gfx701 \
    --amdgpu-target=gfx801 \
    --amdgpu-target=gfx802 \
    --amdgpu-target=gfx803 \
    foo.cpp

**use :: HCC_AMDGPU_TARGET env var**

Use, to delimit each AMDGPU target in HCC. Example

.. code:: sh

  export HCC_AMDGPU_TARGET=gfx701,gfx801,gfx802,gfx803
  # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
  # be produced
  hcc `hcc-config --cxxflags --ldflags` foo.cpp

**configure HCC using the CMake HSA_AMDGPU_GPU_TARGET variable**

If you build HCC from source, it's possible to configure it to automatically produce multiple ISAs via :: HSA_AMDGPU_GPU_TARGET CMake variable.
Use ; to delimit each AMDGPU target. Example

.. code:: sh

 # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
 # be produced by default
 cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_DEVICE_LIB_DIR=~hcc/ROCm-Device-Libs/build/dist/lib \
    -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx801;gfx802;gfx803" \
    ../hcc

CodeXL Activity Logger
#######################

To enable the CodeXL Activity Logger, use the  USE_CODEXL_ACTIVITY_LOGGER environment variable.

Configure the build in the following way

.. code:: sh

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=<AMD GPU ISA version string> \
    -DROCM_DEVICE_LIB_DIR=<location of the ROCm-Device-Libs bitcode> \
    -DUSE_CODEXL_ACTIVITY_LOGGER=1 \
    <ToT HCC checkout directory>

In your application compiled using hcc, include the CodeXL Activiy Logger header

.. code:: cpp

  #include <CXLActivityLogger.h>

For information about the usage of the Activity Logger for profiling, please refer to `documentation <https://documentation.help/CodeXL/amdtactivitylogger-library.htm>`_



HC Best Practices
=================

HC comes with two header files as of now:

    * `hc.hpp <http://scchan.github.io/hcc/hc_8hpp.html>`_ : Main header file for HC
    * `hc_math.hpp <http://scchan.github.io/hcc/hc__math_8hpp_source.html>`_ : Math functions for HC

Most HC APIs are stored under "hc" namespace, and the class name is the same as their counterpart in C++AMP "Concurrency" namespace. Users of C++AMP should find it easy to switch from C++AMP to HC.

================================== =====================
C++AMP 		       			             HC
================================== =====================
``Concurrency::accelerator``	     ``hc::accelerator``
``Concurrency::accelerator_view``  ``hc::accelerator_view``
``Concurrency::extent``		         ``hc::extent``
``Concurrency::index``		         ``hc::index``
``Concurrency::completion_future`` ``hc::completion_future``
``Concurrency::array``		         ``hc::array``
``Concurrency::array_view``     	 ``hc::array_view``
================================== =====================


HCC built-in macros
#######################
Built-in macros:

====================== ===============================================================================
Macro                  Meaning
====================== ===============================================================================
``__HCC__``		         always be 1
``__hcc_major__``	     major version number of HCC
``__hcc_minor__``	     minor version number of HCC
``__hcc_patchlevel__`` patchlevel of HCC
``__hcc_version__``	   combined string of ``__hcc_major__``, ``__hcc_minor__``, ``__hcc_patchlevel__``
====================== ===============================================================================

The rule for ``__hcc_patchlevel__`` is: yyWW-(HCC driver git commit #)-(HCC clang git commit #)

   * yy stands for the last 2 digits of the year
   * WW stands for the week number of the year

Macros for language modes in use:

================== ==========================================================================
 Macro             Meaning
================== ==========================================================================
``__KALMAR_AMP__`` 1 in case in C++ AMP mode (-std=c++amp; **Removed from ROCm 2.0 onwards**)
``__KALMAR_HC__``  1 in case in HC mode (-hc)
================== ==========================================================================

Compilation mode: HCC is a single-source compiler where kernel codes and host codes can reside in the same file. Internally HCC would trigger 2 compilation iterations, and the following macros can be used by user programs to determine which mode the compiler is in.

========================== ===============================================================
Macro           		       Meaning
========================== ===============================================================
``__KALMAR_ACCELERATOR__`` not 0 in case the compiler runs in kernel code compilation mode
``__KALMAR_CPU__``         not 0 in case the compiler runs in host code compilation mode
========================== ===============================================================

HC-specific features
########################

   * relaxed rules in operations allowed in kernels
   * new syntax of tiled_extent and tiled_index
   * dynamic group segment memory allocation
   * true asynchronous kernel launching behavior
   * additional HSA-specific APIs

Differences between HC API and C++ AMP
#########################################
Despite HC and C++ AMP sharing many similar program constructs (e.g. parallel_for_each, array, array_view, etc.), there are several significant differences between the two APIs.

**Support for explicit asynchronous parallel_for_each**
In C++ AMP, the parallel_for_each appears as a synchronous function call in a program (i.e. the host waits for the kernel to complete); howevever, the compiler may optimize it to execute the kernel asynchronously and the host would synchronize with the device on the first access of the data modified by the kernel. For example, if a parallel_for_each writes the an array_view, then the first access to this array_view on the host after the parallel_for_each would block until the parallel_for_each completes.

HC supports the automatic synchronization behavior as in C++ AMP. In addition, HC's parallel_for_each supports explicit asynchronous execution. It returns a completion_future (similar to C++ std::future) object that other asynchronous operations could synchronize with, which provides better flexibility on task graph construction and enables more precise control on optimization.


**Annotation of device functions**

C++ AMP uses the restrict(amp) keyword to annotate functions that runs on the device.

.. code:: cpp

 void foo() restrict(amp) { .. } ... parallel_for_each(...,[=] () restrict(amp) { foo(); });

HC uses a function attribute ([[hc]] or __attribute__((hc)) ) to annotate a device function.

.. code:: cpp

  void foo() [[hc]] { .. } ... parallel_for_each(...,[=] () [[hc]] { foo(); });

The [[hc]] annotation for the kernel function called by parallel_for_each is optional as it is automatically annotated as a device function by the hcc compiler. The compiler also supports partial automatic [[hc]] annotation for functions that are called by other device functions within the same source file:

Since bar is called by foo, which is a device function, the hcc compiler will automatically annotate bar as a device function ``void bar() { ... } void foo() [[hc]] { bar(); }``


**Dynamic tile size**

C++ AMP doesn't support dynamic tile size. The size of each tile dimensions has to be a compile-time constant specified as template arguments to the tile_extent object:

 `extent<2> <http://scchan.github.io/hcc/classConcurrency_1_1extent.html>`_  ex(x, y)

 To create a tile extent of 8x8 from the extent object,note that the tile dimensions have to be constant values:

   tiled_extent<8,8> t_ex(ex)

parallel_for_each(t_ex, [=](tiled_index<8,8> t_id) restrict(amp) { ... });

    HC supports both static and dynamic tile size:

   `extent<2> <http://scchan.github.io/hcc/classConcurrency_1_1extent.html>`_ ex(x,y)

To create a tile extent from dynamically calculated values,note that the the tiled_extent template takes the rank instead of dimensions

         tx = test_x ? tx_a : tx_b;

         ty = test_y ? ty_a : ty_b;

         tiled_extent<2> t_ex(ex, tx, ty);

         parallel_for_each(t_ex, [=](tiled_index<2> t_id) [[hc]] { ... });

**Support for memory pointer**

C++ AMP doesn't support lambda capture of memory pointer into a GPU kernel.

HC supports capturing memory pointer by a GPU kernel.

allocate GPU memory through the HSA API
``int* gpu_pointer; hsa_memory_allocate(..., &gpu_pointer); ... parallel_for_each(ext, [=](index i) [[hc]] { gpu_pointer[i[0]]++; }``

For HSA APUs that supports system wide shared virtual memory, a GPU kernel can directly access system memory allocated by the host:
``int* cpu_memory = (int*) malloc(...); ... parallel_for_each(ext, [=](index i) [[hc]] { cpu_memory[i[0]]++; });``


HCC Profile Mode
##################

HCC supports low-overhead profiler to trace or summarize command timestamp information to stderr for any HCC or HIP program. Tho profiler messages are interleaved with the trace output from the application - which is handy to identify the region-of-interest and can complement deeper analysis with the CodeXL GUI Additionally, the hcc profiler requires only console mode access and can be used on machine where graphics are not available or are hard to access.

Some other useful features:

* Calculates the actual bandwidth for memory transfers
* Identifies PeerToPeer memory copies
* Shows start / stop timestamps for each command (if requested)
* Shows barrier commands and the time they spent waiting to resolve (if requested)

**Enable and configure**


| HCC_PROFILE=1 shows a summary of kernel and data commands when hcc exits (under development).
| HCC_PROFILE=2 enables a profile message after each command (kernel or data movement) completes.

| Additionally, the HCC_PROFILE_VERBOSE variable controls the information shown in the profile log. This is a bit-vector:
| 0x2 : Show start and stop timestamps for each command.
| 0x4 : Show the device.queue.cmdseqnum for each command.
| 0x8 : Show the short CPU TID for each command (not supported).
| 0x10 : Show logs for barrier commands.

**Sample Output**


Kernel Commands
++++++++++++++++

This shows the simplest trace output for kernel commands with no additional verbosity flags

.. code:: sh

 $ HCC_PROFILE=2 ./my-hcc-app ...
 profile:  kernel;            Im2Col;   17.8 us;
 profile:  kernel;  tg_betac_alphaab;   32.6 us;
 profile:  kernel;     MIOpenConvUni;  125.4 us;

.. code:: sh

  PROFILE:  TYPE;    KERNEL_NAME     ;  DURATION;

This example shows profiled kernel commands with full verbose output

.. code:: sh

 $ HCC_PROFILE=2 HCC_PROFILE_VERBOSE=0xf ./my-hcc-app ...
 profile:  kernel;            Im2Col;   17.8 us;  94859076277181; 94859076294941; #0.3.1;
 profile:  kernel;  tg_betac_alphaab;   32.6 us;  94859537593679; 94859537626319; #0.3.2;
 profile:  kernel;     MIOpenConvUni;  125.4 us;  94860077852212; 94860077977651; #0.3.3;

.. code:: sh

  PROFILE:  TYPE;    KERNEL_NAME     ;  DURATION;  START         ; STOP          ; ID

* PROFILE: always "profile:" to distinguish it from other output.
* TYPE: the command type : kernel, copy, copyslo, or barrier. The examples and descriptions in this section are all kernel commands.
* KERNEL_NAME: the (short) kernel name.
* DURATION: command duration measured in us. This is measured using the GPU timestamps and represents the command execution on the accelerator device.
* START: command start time in ns. (if HCC_PROFILE_VERBOSE & 0x2)
* STOP: command stop time in ns. (if HCC_PROFILE_VERBOSE & 0x2)
* ID: command id in device.queue.cmd format. (if HCC_PROFILE_VERBOSE & 0x4). The cmdsequm is a unique monotonically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.

Memory Copy Commands
+++++++++++++++++++++

This example shows memory copy commands with full verbose output:

.. code:: sh

 profile: copyslo; HostToDevice_sync_slow;   909.2 us; 94858703102; 94858704012; #0.0.0; 2359296 bytes;  2.2 MB;   2.5 GB/s;
 profile:    copy; DeviceToHost_sync_fast;   117.0 us; 94858726408; 94858726525; #0.0.0; 1228800 bytes;  1.2 MB;   10.0 GB/s;
 profile:    copy; DeviceToHost_sync_fast;     9.0 us; 94858726668; 94858726677; #0.0.0; 400 bytes;      0.0 MB;   0.0 GB/s;
 profile:    copy; HostToDevice_sync_fast;    15.2 us; 94858727639; 94858727654; #0.0.0; 9600 bytes;     0.0 MB;   0.6 GB/s;
 profile:    copy; HostToDevice_async_fast;  131.5 us; 94858729198; 94858729330; #0.6.1; 1228800 bytes;  1.2 MB;   8.9 GB/s;
 PROFILE:  TYPE;    COPY_NAME             ;  DURATION;       START;       STOP;  ID    ; SIZE_BYTES;     SIZE_MB;  BANDWIDTH;


* PROFILE: always "profile:" to distinguish it from other output.
* TYPE: the command type : kernel, copy, copyslo,or barrier. The examples and descriptions in this section are all copy or copyslo commands.
* COPY_NAME has 3 parts:
	* Copy kind: HostToDevice, HostToHost, DeviceToHost, DeviceToDevice, or PeerToPeer. DeviceToDevice indicates the copy occurs on a single device while PeerToPeer indicates a copy between devices.
	* Sync or Async. Synchronous copies indicate the host waits for the completion for the copy. Asynchronous copies are launched by the host without waiting for the copy to complete.
	* Fast or Slow. Fast copies use the GPUs optimized copy routines from the hsa_amd_memory_copy routine. Slow copies typically involve unpinned host memory and can't take the fast path.
	* For example `HostToDevice_async_fast`.

* DURATION: command duration measured in us. This is measured using the GPU timestamps and represents the command execution on the accelerator device.
* START: command start time in ns. (if HCC_PROFILE_VERBOSE & 0x2)
* STOP: command stop time in ns. (if HCC_PROFILE_VERBOSE & 0x2)
* ID: command id in device.queue.cmd format. (if HCC_PROFILE_VERBOSE & 0x4). The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.
* SIZE_BYTES: the size of the transfer, measured in bytes.
* SIZE_MB: the size of the transfer, measured in megabytes.
* BANDWIDTH: the bandwidth of the transfer, measured in GB/s.

Barrier Commands
+++++++++++++++++

Barrier commands are only enabled if HCC_PROFILE_VERBOSE 0x10

An example barrier command with full vebosity

.. code:: sh

 profile: barrier; deps:0_acq:none_rel:sys;  5.3 us;   94858731419410; 94858731424690; # 0.0.2;
 PROFILE:  TYPE;   BARRIER_NAME           ;  DURATION; START         ; STOP          ; ID    ;

* PROFILE: always "profile:" to distinguish it from other output.
* TYPE: the command type: either kernel, copy, copyslo, or barrier. The examples and descriptions in this section are all copy commands. Copy indicates that the runtime used a call to the fast hsa memory copy routine while copyslo indicates that the copy was implemented with staging buffers or another less optimal path. copy computes the commands using device-side timestamps while copyslo computes the bandwidth based on host timestamps.
* BARRIER_NAME has 3 parts:
	* **deps:#** - the number of input dependencies into the barrier packet.
	* **acq:** - the acquire fence for the barrier. May be none, acc(accelerator or agent), sys(system). See HSA AQL spec for additional information.
	* **rel:** - the release fence for the barrier. May be none, acc(accelerator or agent), sys(system). See HSA AQL spec for additional information.
* DURATION: command duration measured in us. This is measured using the GPU timestamps from the time the barrier reaches the head of the queue to when it executes. Thus this includes the time to wait for all input dependencies, plus the previous command to complete, plus any fence operations performed by the barrier.
* START: command start time in ns. (if HCC_PROFILE_VERBOSE & 0x2)
* STOP: command stop time in ns. (if HCC_PROFILE_VERBOSE & 0x2)
* ID: the command id in device.queue.cmd format. (if HCC_PROFILE_VERBOSE & 0x4). The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.

Overhead
+++++++++

The hcc profiler does not add any additional synchronization between commands or queues. Profile information is recorded when a command is deleted. The profile mode will allocate a signal for each command to record the timestamp information. This can add 1-2 us to the overall program execution for command which do not already use a completion signal. However, the command duration (start-stop) is still accurate. Trace mode will generate strings to stderr which will likely impact the overall application exection time. However, the GPU duration and timestamps are still valid. Summary mode accumulates statistics into an array and should have little impact on application execution time.

Additional Details and tips
++++++++++++++++++++++++++++

* Commands are logged in the order they are removed from the internal HCC command tracker. Typically this is the same order that      	commands are dispatched, though sometimes these may diverge. For example, commands from different devices,queues, or cpu threads    	may be interleaved on the hcc trace display to stderr. If a single view in timeline order is required, enable and sort by the       	profiler START timestamps (HCC_PROFILE_VERBOSE=0x2)
* If the application keeps a reference to a completion_future, then the command timestamp may be reported significantly after it      	occurs.
* HCC_PROFILE has an (untested) feature to write to a log file.


API documentation
####################
`API reference of HCC <https://scchan.github.io/hcc/>`_

HIP Programing Guide
====================

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in compute kernels, including classes, namespaces, operator overloading, templates and more. Additionally, it defines other language features designed specifically to target accelerators, such as the following:

   * A kernel-launch syntax that uses standard C++, resembles a function call and is portable to all HIP targets
   * Short-vector headers that can serve on a host or a device
   * Math functions resembling those in the "math.h" header included with standard C++ compilers
   * Built-in functions for accessing specific GPU hardware capabilities

This section describes the built-in variables and functions accessible from the HIP kernel. It’s intended for readers who are familiar with Cuda kernel syntax and want to understand how HIP is different.

  * :ref:`HIP-GUIDE`


HIP Best Practices
==================

 * :ref:`HIP-porting-guide`
 * :ref:`HIP-terminology`
 * :ref:`hip_profiling`
 * :ref:`HIP_Debugging`
 * :ref:`Kernel_language`
 * :ref:`HIP-Terms`
 * :ref:`HIP-bug`
 * :ref:`hipporting-driver-api`
 * :ref:`CUDAAPIHIP`
 * :ref:`CUDAAPIHIPTEXTURE`
 * :ref:`HIP-FAQ`
 * :ref:`HIP-Term2`



OpenCL Programing Guide
========================

* :ref:`Opencl-Programming-Guide`

OpenCL Best Practices
======================

* :ref:`Optimization-Opencl`