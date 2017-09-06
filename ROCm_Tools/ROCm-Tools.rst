
.. _ROCm-Tools:

=====================
ROCm Tools
=====================


HCC
=====

HCC is an Open Source, Optimizing C++ Compiler for Heterogeneous Compute
#########################################################################

HCC supports heterogeneous offload to AMD APUs and discrete GPUs via HSA enabled runtimes and drivers. It is an ISO compliant C++ 11/14 compiler. It is based on Clang, the LLVM Compiler Infrastructure and the “libc++” C++ standard library.

Platform Requirements
*********************

Accelerated applications could be run on Radeon discrete GPUs from the Fiji family (AMD R9 Nano, R9 Fury, R9 Fury X, FirePro S9300 x2, Polaris 10, Polaris 11) paired with an Intel Haswell CPU or newer. HCC would work with AMD HSA APUs (Kaveri, Carrizo); however, they are not our main support platform and some of the more advanced compute capabilities may not be available on the APUs.

HCC currently only works on Linux and with the open source ROCK kernel driver and the ROCR runtime (see Installation for details). It will not work with the closed source AMD graphics driver.

Compiler Backends
******************
This backend compiles GPU kernels into native GCN ISA, which could be directly execute on the GPU hardware. It's being actively developed by the Radeon Technology Group in LLVM.


Installation
############
Prerequisites
**************
Before continuing with the installation, please make sure any previously installed hcc compiler has been removed from on your system.
Install ROCm and make sure it works correctly.

Ubuntu
******
Ubuntu 14.04
*************
Follow the instruction here to setup the ROCm apt repository and install the rocm or the rocm-dev meta-package.

Ubuntu 16.04
*************
Ubuntu 16.04 is also supported but currently it has to be built from source.

Fedora
******
HCC compiler has been tested on Fedora 23 but currently it has to be built from source.

**Download HCC**
 The project now employs git submodules to manage external components it depends upon. It it advised to add --recursive when you clone the project so all submodules are fetched automatically.

For example: ::

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git


Building HCC from Source
************************
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

How to use HCC
##############
Here's a simple `saxpy example <https://gist.github.com/scchan/540d410456e3e2682dbf018d3c179008>`_ written with the hc API.

Compiling Your First HCC Program
*********************************
To compile and link in a single step:

# Assume HCC is installed and added to PATH
# Notice the the hcc-config command is between two backticks 
hcc `hcc-config --cxxflags --ldflags` saxpy.cpp -o saxpy

To build with separate compile and link steps:

# Assume HCC is installed and added to PATH
# Notice the the hcc-config command is between two backticks 
hcc `hcc-config --cxxflags` saxpy.cpp -c -o saxpy.cpp.o
hcc `hcc-config --ldflags` saxpy.cpp.o -o saxpy

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



API documentation
*******************
`API reference of HCC <https://scchan.github.io/hcc/>`_



GCN Assembler and Disassembler
==============================

GCN Assembler Tools
====================

ROCm-GDB
=========

The ROCm-GDB repository includes the source code for ROCm-GDB. ROCm-GDB is a modified version of GDB 7.11 that supports debugging GPU kernels on Radeon Open Compute platforms (ROCm).

Package Contents
##################
The ROCm-GDB repository includes

   * A modified version of gdb-7.11 to support GPU debugging. Note the main ROCm specific files are located in gdb-7.11/gdb with the 	  rocm-* prefix.
   * The ROCm debug facilities library located in amd/HwDbgFacilities/. This library provides symbol processing for GPU kernels.

Build Steps
############
 
1. Clone the ROCm-GDB repository

::
   
    git clone https://github.com/RadeonOpenCompute/ROCm-GDB.git

2. The gdb build has been modified with new files and configure settings to enable GPU debugging. The scripts below should be run to 	  compile gdb. The run_configure_rocm.sh script calls the GNU autotools configure with additional parameters. The   	 	    	run_configure_rocm.sh script will create the build directory to build the gdb executable in a out of source manner

::

    ./run_configure_rocm.sh debug

3.    The run_configure_rocm.sh script also generates the run_make_rocm.sh which sets environment variables for the Make step

::
   
   ./run_make_rocm.sh


Running ROCm-GDB
################

The run_make_rocm.sh script builds the gdb executable which will be located in build/gdb/

To run the ROCm debugger, you'd also need to get the ROCm GPU Debug SDK.

Before running the rocm debugger, the LD_LIBRARY_PATH should include paths to

    The ROCm GPU Debug Agent library built in the ROCm GPU Debug SDK (located in gpudebugsdk/lib/x86_64)
    The ROCm GPU Debugging library binary shippped with the ROCm GPU Debug SDK (located in gpudebugsdk/lib/x86_64)
    Before running ROCm-GDB, please update your .gdbinit file with text in gpudebugsdk/src/HSADebugAgent/gdbinit. The rocmConfigure function in the ~/.gdbinit sets up gdb internals for supporting GPU kernel debug.
    The gdb executable should be run from within the rocm-gdb-local script. The ROCm runtime requires certain environment variables to enable kernel debugging and this is set up by the rocm-gdb-local script.

./rocm-gdb-local < sample application>

    A brief tutorial on how to debug GPU applications using ROCm-GDB :ref:`ROCm-Tools/rocm-debug`

ROCm Debugger API
=================

ROCm-Profiler
==============

CodeXL
=========

GPUperfAPI
==============

ROCm Binary Utilities
======================
