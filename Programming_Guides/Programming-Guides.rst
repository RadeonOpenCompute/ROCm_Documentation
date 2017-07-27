.. _Programming-Guides:

=================
Programing Guide
=================


HC Programing Guide
===================
**What is the Heterogeneous Compute (HC) API?**
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

**When to Use HC**
 Use HC when you're targeting the AMD ROCm platform: it delivers a single-source, easy-to-program C++ environment without compromising performance or control of the machine.

**Download HCC**
 The project now employs git submodules to manage external components it depends upon. It it advised to add --recursive when you clone the project so all submodules are fetched automatically.

For example: ::

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

**Build HCC from source**
*************************

First, install the build dependencies: ::

  # Ubuntu 14.04
  sudo apt-get install git cmake make g++  g++-multilib gcc-multilib libc++-dev libc++1 libc++abi-dev libc++abi1 python findutils     	libelf1 libpci3 file debianutils libunwind8-dev hsa-rocr-dev hsa-ext-rocr-dev hsakmt-roct-dev pkg-config rocm-utils

::  

  # Ubuntu 16.04
  sudo apt-get install git cmake make g++  g++-multilib gcc-multilib python findutils libelf1 libpci3 file debianutils libunwind-     	dev hsa-rocr-dev hsa-ext-rocr-dev hsakmt-roct-dev pkg-config rocm-utils

::

   # Fedora 23/24
   sudo dnf install git cmake make gcc-c++ python findutils elfutils-libelf pciutils-libs file pth rpm-build libunwind-devel   	     	hsa- rocr- dev hsa-ext-rocr-dev hsakmt-roct-dev pkgconfig rocm-utils

Clone the HCC source tree: ::

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

Create a build directory and run cmake to configure the build: ::

  mkdir build; cd build
  cmake ../hcc

Compile HCC: ::

  make -j

Run the unit tests: :: 

  make test

Create an installer package (DEB or RPM file)

::

  make package



To configure and build HCC from source, use the following steps: ::
 
  mkdir -p build; cd build
  # NUM_BUILD_THREADS is optional
  # set the number to your CPU core numbers time 2 is recommended
  # in this example we set it to 96 
  cmake -DNUM_BUILD_THREADS=96 \
      -DCMAKE_BUILD_TYPE=Release \
      ..
  make

To install it, use the following steps: ::
  
  sudo make install

**Use HCC**
***********

For C++AMP source codes: ::

  hcc `clamp-config --cxxflags --ldflags` foo.cpp

For HC source codes: ::
 
 hcc `hcc-config --cxxflags --ldflags` foo.cpp

In case you build HCC from source and want to use the compiled binaries directly in the build directory:

For C++AMP source codes: ::

  # notice the --build flag
  bin/hcc `bin/clamp-config --build --cxxflags --ldflags` foo.cpp

For HC source codes: ::

  # notice the --build flag
  bin/hcc `bin/hcc-config --build --cxxflags --ldflags` foo.cpp

**Compiling for Different GPU Architectures**

By default, HCC would auto-detect all the GPUs available it's running on and set the correct GPU architectures. Users could use the --amdgpu-target=<GCN Version> option to compile for a specific architecture and to disable the auto-detection. The following table shows the different versions currently supported by HCC.

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
*************
HCC now supports having multiple GCN ISAs in one executable file. You can do it in different ways:
**use :: --amdgpu-target= command line option**
It's possible to specify multiple --amdgpu-target= option. Example: ::

 # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would 
 # be produced
 hcc `hcc-config --cxxflags --ldflags` \
    --amdgpu-target=gfx701 \
    --amdgpu-target=gfx801 \
    --amdgpu-target=gfx802 \
    --amdgpu-target=gfx803 \
    foo.cpp

use :: HCC_AMDGPU_TARGET env var

Use , to delimit each AMDGPU target in HCC. Example: ::
  
  export HCC_AMDGPU_TARGET=gfx701,gfx801,gfx802,gfx803
  # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would 
  # be produced
  hcc `hcc-config --cxxflags --ldflags` foo.cpp

**configure HCC use CMake HSA_AMDGPU_GPU_TARGET variable**
If you build HCC from source, it's possible to configure it to automatically produce multiple ISAs via :: HSA_AMDGPU_GPU_TARGET CMake variable.
Use ; to delimit each AMDGPU target. Example: ::



 # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would 
 # be produced by default
 cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_DEVICE_LIB_DIR=~hcc/ROCm-Device-Libs/build/dist/lib \
    -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx801;gfx802;gfx803" \
    ../hcc

**CodeXL Activity Logger**
**************************

To enable the CodeXL Activity Logger, use the  USE_CODEXL_ACTIVITY_LOGGER environment variable.

Configure the build in the following way: ::

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=<AMD GPU ISA version string> \
    -DROCM_DEVICE_LIB_DIR=<location of the ROCm-Device-Libs bitcode> \
    -DUSE_CODEXL_ACTIVITY_LOGGER=1 \
    <ToT HCC checkout directory>

In your application compiled using hcc, include the CodeXL Activiy Logger header: ::
 
  #include <CXLActivityLogger.h>

For information about the usage of the Activity Logger for profiling, please refer to its documentation.



HC Best Practices
=================


HIP Programing Guide
=====================



HIP Best Practices
==================



OpenCL Programing Guide
========================




OpenCL Best Practices
======================





