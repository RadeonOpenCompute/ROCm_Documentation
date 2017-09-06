
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
CodeXL is a comprehensive tool suite that enables developers to harness the benefits of CPUs, GPUs and APUs. It includes powerful GPU debugging, comprehensive GPU and CPU profiling, DirectX12® Frame Analysis, static OpenCL™, OpenGL®, Vulkan® and DirectX® kernel/shader analysis capabilities, and APU/CPU/GPU power profiling, enhancing accessibility for software developers to enter the era of heterogeneous computing. CodeXL is available both as a Visual Studio® extension and a standalone user interface application for Windows® and Linux®.

Motivation
###########
CodeXL, previously a tool developed as closed-source by Advanced Micro Devices, Inc., is now released as Open Source. AMD believes that adopting the open-source model and sharing the CodeXL source base with the world can help developers make better use of CodeXL and make CodeXL a better tool.

To encourage 3rd party contribution and adoption, CodeXL is no longer branded as an AMD product. AMD will still continue development of this tool and upload new versions and features to GPUOpen.

Installation and Build
************************

Windows: To install CodeXL, use the `provided <https://github.com/GPUOpen-Tools/CodeXL/releases>`_executable file CodeXL_*.exe
Linux: To install CodeXL, use the `provided <https://github.com/GPUOpen-Tools/CodeXL/releases>`_ RPM file, Debian file, or simply extract the compressed archive onto your hard drive.
Refer to BUILD.md for information on building CodeXL from source.

Contributors
############

CodeXL's GitHub repository (http://github.com/GPUOpen-Tools/CodeXL) is moderated by Advanced Micro Devices, Inc. as part of the GPUOpen initiative.

AMD encourages any and all contributors to submit changes, features, and bug fixes via Git pull requests to this repository.

Users are also encouraged to submit issues and feature requests via the repository's issue tracker.

License
########
CodeXL is part of the GPUOpen.com initiative. CodeXL source code and binaries are released under the following MIT license:

Copyright © 2016 Advanced Micro Devices, Inc. All rights reserved.

MIT LICENSE: Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Attribution and Copyrights
##########################
Component licenses can be found under the CodeXL GitHub repository source root, in the /Setup/Legal/ folder.

OpenCL is a trademark of Apple Inc. used by permission by Khronos. OpenGL is a registered trademark of Silicon Graphics, Inc. in the United States and/or other countries worldwide. Microsoft, Windows, DirectX and Visual Studio are registered trademarks of Microsoft Corporation in the United States and/or other jurisdictions. Vulkan is a registered trademark of Khronos Group Inc. in the United States and/or other jurisdictions. Linux is the registered trademark of Linus Torvalds in the United States and/or other jurisdictions.

LGPL (Copyright ©1991, 1999 Free Software Foundation, Inc. 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA). Use of the Qt library is governed by the GNU Lesser General Public License version 2.1 (LGPL v 2.1). CodeXL uses QT 5.5.1. Source code for QT is available here: http://qt-project.org/downloads. The QT source code has not been tempered with and the built binaries are identical to what any user that downloads the source code from the web and builds them will produce.

Boost is Copyright © Beman Dawes, 2003.
[CR]LunarG, Inc. is Copyright © 2015 LunarG, Inc.
jqPlot is copyright © 2009-2011 Chris Leonello.
glew - The OpenGL Extension Wrangler Library is Copyright © 2002-2007, Milan Ikits <milan ikits[]ieee org>, Copyright © 2002-2007, Marcelo E. Magallon <mmagallo[]debian org>, Copyright © 2002, Lev Povalahev, All rights reserved.
lgplib is Copyright © 1994-1998, Thomas G. Lane., Copyright © 1991-2013, Thomas G. Lane, Guido Vollbeding.
LibDwarf (BSD) is Copyright © 2007 John Birrell (jb@freebsd.org), Copyright © 2010 Kai Wang, All rights reserved.
libpng is Copyright © 1998-2014 Glenn Randers-Pehrson, (Version 0.96 Copyright © 1996, 1997 Andreas Dilger) (Version 0.88 Copyright © 1995, 1996 Guy Eric Schalnat, Group 42, Inc.).
QScintilla is Copyright © 2005 by Riverbank Computing Limited info@riverbankcomputing.co.uk.
TinyXML is released under the zlib license © 2000-2007, Lee Thomason, © 2002-2004, Yves Berquin © 2005, Tyge Lovset.
UTF8cpp is Copyright © 2006 Nemanja Trifunovic.
zlib is Copyright © 1995-2010 Jean-loup Gailly and Mark Adler, Copyright © 2003 Chris Anderson christop@charm.net, Copyright © 1998-2010 Gilles Vollant (minizip) ( http://www.winimage.com/zLibDll/minizip.html ), Copyright © 2009-2010 Mathias Svensson ( http://result42.com ), Copyright © 2007-2008 Even Rouault.
QCustomPlot, an easy to use, modern plotting widget for Qt, Copyright (C) 2011-2015 Emanuel Eichhammer

GPUperfAPI
==============

The GPU Performance API (GPUPerfAPI, or GPA) is a powerful library, providing access to GPU Performance Counters. It can help analyze the performance and execution characteristics of applications using a Radeon™ GPU. This library is used by both CodeXL and GPU PerfStudio.

Major Features
###############

   * Provides a standard API for accessing GPU Performance counters for both graphics and compute workloads across multiple GPU APIs.
   * Supports DirectX11, OpenGL, OpenGLES, OpenCL™, and ROCm/HSA
   * Developer Preview for DirectX12 (no hardware-based performance counter support yet)
   * Supports all current GCN-based Radeon graphics cards and APUs.
   * Supports both Windows and Linux
   * Provides derived "public" counters based on raw HW counters
   * "Internal" version provides access to some raw hardware counters. See "Public" vs "Internal" Versions for more information.

What's New
##########
    Version 2.23 (6/27/17)
     * Add support for additional GPUs, including Vega series GPUs
     * Allow unit tests to be built and run on Linux

System Requirements
#####################
    * An AMD Radeon GCN-based GPU or APU
    * Radeon Software Crimson ReLive Edition 17.4.3 or later (Driver Packaging Version 17.10 or later).
       * For Vega support, a driver with Driver Packaging Version 17.20 or later is required
    * Pre-GCN-based GPUs or APUs are no longer supported by GPUPerfAPI. Please use an older version (2.17) with older hardware.
    * Windows 7, 8.1, and 10
    * Ubuntu (16.04 and later) and RHEL (7 and later) distributions

Cloning the Repository
######################
To clone the GPA repository, execute the following git commands

  *  git clone https://github.com/GPUOpen-Tools/GPA.git After cloning the repository, please run the following python script to retrieve the 	  required dependencies (see BUILD.md for more information):
  *  python Scripts/UpdateCommon.py UpdateCommon has replaced the use of git submodules in the GPA repository

Source Code Directory Layout
##############################
    `Build <https://github.com/GPUOpen-Tools/GPA/tree/master/Build>`_ -- contains both Linux and Windows build-related files
    `Common <https://github.com/GPUOpen-Tools/GPA/tree/master/Build>`_ -- Common libs, header and source code not found in other repositories
    `Doc <https://github.com/GPUOpen-Tools/GPA/tree/master/Doc>`_ -- contains User Guide and Doxygen configuration files
    `Src/DeviceInfo <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/DeviceInfo>`_ -- builds a lib containing the Common/Src/DeviceInfo code (Linux only)
    `Src/GPUPerfAPI-Common <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPI-Common>`_-- contains source code for a Common library shared by all versions of GPUPerfAPI
    `Src/GPUPerfAPICL <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPICL>`_ - contains the source for the OpenCL™ version of GPUPerfAPI
    `Src/GPUPerfAPICounterGenerator <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPICounterGenerator>`_ - contains the source code for a Common library providing all counter data
    `Src/GPUPerfAPICounters <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPICounters>`_ - contains the source code for a library that can be used to query counters without an active GPUPerfAPI context
    `Src/GPUPerfAPIDX <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIDX>`_ - contains source code shared by the DirectX versions of GPUPerfAPI
    `Src/GPUPerfAPIDX11 <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIDX11>`_ - contains the source for the DirectX11 version of GPUPerfAPI
    `Src/GPUPerfAPIDX12 <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIDX12>`_ - contains the source for the DirectX12 version of GPUPerfAPI (Developer Preview)
    `Src/GPUPerfAPIGL <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIGL>`_ - contains the source for the OpenGL version of GPUPerfAPI
    `Src/GPUPerfAPIGLES <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIGLES>_ - contains the source for the OpenGLES version of GPUPerfAPI
    `Src/GPUPerfAPIHSA <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIHSA>`_- contains the source for the ROCm/HSA version of GPUPerfAPI
    `Src/GPUPerfAPIUnitTests <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/GPUPerfAPIUnitTests>`_- contains a small set of unit tests for GPUPerfAPI
    `Src/PublicCounterCompiler <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/PublicCounterCompiler>`_ - source code for a tool to generate C++ code for public counters from text files defining the counters.
    `Src/PublicCounterCompilerInputFiles <https://github.com/GPUOpen-Tools/GPA/tree/master/Src/PublicCounterCompilerInputFiles>`_ - input files that can be fed as input to the PublicCounterCompiler tool
    `Scripts <https://github.com/GPUOpen-Tools/GPA/tree/master/Scripts>`_ -- scripts to use to clone/update dependent repositories

Public" vs "Internal" Versions
###############################
This open source release supports building both the "Public" and "Internal" versions of GPUPerfAPI. By default the Visual Studio solution and the Linux build scripts will produce what is referred to as the "Public" version of GPUPerfAPI. This version exposes "Public", or "Derived", counters. These are counters that are computed using a set of hardware counters. Until now, only the Public the version of GPUPerfAPI was available on the AMD Developer website. As part of the open-source effort, we are also providing the ability to build the "Internal" versions of GPUPerfAPI. In addition to exposing the same counters as the Public version, the Internal version also exposes some of the hardware Counters available in the GPU/APU. It's important to note that not all hardware counters receive the same validation as other parts of the hardware on all GPUs, so in some cases accuracy of counter data cannot be guaranteed. The usage of the Internal version is identical to the Public version. The only difference will be in the name of the library an application loads at runtime and the list of counters exposed by the library. See the Build Instructions for more information on how to build and use the Internal version. In the future, we see there being only a single version of GPUPerfAPI, with perhaps a change in the API to allow users of GPA to indicate whether the library exposes just the Derived counters or both the Derived and the Hardware counters. We realize using the term "Internal" for something which is no longer actually Internal-to-AMD can be a bit confusing, and we will aim to change this in the future.

Known Issues
#############
  *  The OpenCL™ version of GPUPerfAPI requires at least Driver Version 17.30.1071 for Vega GPUs on Windows. Earlier driver versions have      	    either missing or incomplete support for collecting OpenCL performance counters


ROCm Binary Utilities
======================
