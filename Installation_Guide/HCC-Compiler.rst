# ROCm Documentation has moved to docs.amd.com

.. meta::
   :http-equiv=Refresh: 0; url='https://docs.amd.com'

.. _HCC-Compiler:

HCC : An open source C++ compiler for heterogeneous devices (Deprecated)
==========================================================================

This repository hosts the HCC compiler implementation project. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP, and transforms it into the AMD GCN ISA.

The project is based on LLVM+CLANG. For more information, please visit the `hcc wiki: <https://github.com/RadeonOpenCompute/hcc/wiki>`_

https://github.com/RadeonOpenCompute/hcc/wiki

Download HCC
################

The project now employs git submodules to manage external components it depends upon. It it advised to add --recursive when you clone the project so all submodules are fetched automatically.

For example:

.. code:: sh

    # automatically fetches all submodules
    git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

For more information about git submodules, please refer to `git documentation <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_.

Build HCC from source
########################

To configure and build HCC from source, use the following steps:

.. code:: sh

    mkdir -p build; cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

To install it, use the following steps:

.. code:: sh

    sudo make install

Use HCC
##########

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
#################
HCC now supports having multiple GCN ISAs in one executable file. You can do it in different ways:

use --amdgpu-target= command line option
*******************************************

It's possible to specify multiple **--amdgpu-target=**  option. Example:

.. code:: sh

    # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
    # be produced
    hcc `hcc-config --cxxflags --ldflags` \
        --amdgpu-target=gfx701 \
        --amdgpu-target=gfx801 \
        --amdgpu-target=gfx802 \
        --amdgpu-target=gfx803 \
        foo.cpp


use HCC_AMDGPU_TARGET env var
********************************

Use , to delimit each AMDGPU target in HCC. Example:

.. code:: sh

    export HCC_AMDGPU_TARGET=gfx701,gfx801,gfx802,gfx803
    # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
    # be produced
    hcc `hcc-config --cxxflags --ldflags` foo.cpp


configure HCC use CMake HSA_AMDGPU_GPU_TARGET variable
************************************************************

If you build HCC from source, it's possible to configure it to automatically produce multiple ISAs via HSA_AMDGPU_GPU_TARGET CMake variable.

Use ; to delimit each AMDGPU target. Example:

.. code:: sh

    # ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would
    # be produced by default
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DROCM_DEVICE_LIB_DIR=~hcc/ROCm-Device-Libs/build/dist/lib \
        -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx801;gfx802;gfx803" \
        ../hcc


CodeXL Activity Logger
###########################

To enable the `CodeXL Activity Logger <https://github.com/RadeonOpenCompute/ROCm-Profiler/tree/master/CXLActivityLogger>`_, use the USE_CODEXL_ACTIVITY_LOGGER environment variable.

Configure the build in the following way:

.. code:: sh

    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DHSA_AMDGPU_GPU_TARGET=<AMD GPU ISA version string> \
        -DROCM_DEVICE_LIB_DIR=<location of the ROCm-Device-Libs bitcode> \
        -DUSE_CODEXL_ACTIVITY_LOGGER=1 \
        <ToT HCC checkout directory>


In your application compiled using hcc, include the CodeXL Activity Logger header:

.. code:: cpp

    #include <CXLActivityLogger.h>


For information about the usage of the Activity Logger for profiling, please refer to its `documentation <https://github.com/RadeonOpenCompute/ROCm-Profiler/blob/master/CXLActivityLogger/doc/AMDTActivityLogger.pdf>`_.

HCC with ThinLTO Linking
###########################

To enable the ThinLTO link time, use the KMTHINLTO environment variable.

Set up your environment in the following way:

.. code:: sh

    export KMTHINLTO=1

ThinLTO Phase 1 - Implemented
********************************

For applications compiled using hcc, ThinLTO could significantly improve link-time performance. This implementation will maintain kernels in their .bc file format, create module-summaries for each, perform llvm-lto's cross-module function importing and then perform clamp-device (which uses opt and llc tools) on each of the kernel files. These files are linked with lld into one .hsaco per target specified.

ThinLTO Phase 2 - Under development
**************************************

This ThinLTO implementation which will use llvm-lto LLVM tool to replace clamp-device bash script. It adds an optllc option into ThinLTOGenerator, which will perform in-program opt and codegen in parallel.
