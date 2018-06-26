
.. _ROCm-Compiler-SDK:

==========================
ROCm Compiler SDK
==========================

GCN Native ISA LLVM Code Generator
###################################

* :ref:`ROCm-Native-ISA`

ROCm Code Object Format
########################

* :ref:`ROCm-Codeobj-format`

ROCm Device Library
##################


OVERVIEW
********

This repository contains the following libraries:

======= ============================================= ==============
Name 	Comments 			               Dependencies
======= ============================================= ==============
irif 	Interface to LLVM IR 	
ocml 	Open Compute Math library(:ref:`ocml`) 		irif
oclc 	Open Compute library controls (documentation) 	
ockl 	Open Compute Kernel library. 			irif
opencl 	OpenCL built-in library 			ocml, ockl
hc 	Heterogeneous Compute built-in library 		ocml, ockl
======= ============================================= ==============

All libraries are compiled to LLVM Bitcode which can be linked. Note that libraries use specific AMDGPU intrinsics.

BUILDING
*********

To build it, use RadeonOpenCompute LLVM/LLD/Clang. Default branch on these repositories is "amd-common", which may contain AMD-specific codes yet upstreamed.

::

   git clone git@github.com:RadeonOpenCompute/llvm.git llvm_amd-common
   cd llvm_amd-common/tools
   git clone git@github.com:RadeonOpenCompute/lld.git lld
   git clone git@github.com:RadeonOpenCompute/clang.git clang
   cd ..
   mkdir -p build
   cd build
   cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
      ..      

Testing also requires amdhsacod utility from ROCm Runtime.

Use out-of-source CMake build and create separate directory to run CMake.

The following build steps are performed:

::

   mkdir -p build
   cd build
   export LLVM_BUILD=... (path to LLVM build)
   CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DAMDHSACOD=$HSA_DIR/bin/x86_64/amdhsacod ..
   make

It is also possible to use compiler that only has AMDGPU target enabled if you build prepare-builtins separately with host compiler and pass explicit target option to CMake:

::

   export LLVM_BUILD=... (path to LLVM build)
   # Build prepare-builtins
   cd utils
   mkdir build
   cd build
   cmake -DLLVM_DIR=$LLVM_BUILD ..
   make
   # Build bitcode libraries
   cd ../..
   mkdir build
   cd build
   CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DAMDHSACOD=$HSA_DIR/bin/x86_64/amdhsacod -DCMAKE_C_FLAGS="-target amdgcn--amdhsa"    	  DCMAKE_CXX_FLAGS="-target amdgcn--amdhsa" -DPREPARE_BUILTINS=`cd ../utils/build/prepare-builtins/; pwd`/prepare-builtins ..

To install artifacts: make install

To run offline tests: make test

To create packages for the library: make package


USING BITCODE LIBRARIES
***************************
The bitcode libraries should be linked to user bitcode (obtained from source) before final code generation with llvm-link or -mlink-bitcode-file option of clang.

For OpenCL, the list of bitcode libraries includes opencl, its dependencies (ocml, ockl, irif) and oclc control libraries selected according to OpenCL compilation mode. Assuming that the build of this repository was done in /srv/git/ROCm-Device-Libs/build, the following command line shows how to compile simple OpenCL source test.cl into code object test.so:

::

   clang -x cl -Xclang -finclude-default-header \
       -target amdgcn--amdhsa -mcpu=fiji \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/opencl/opencl.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ocml/ocml.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ockl/ockl.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_correctly_rounded_sqrt_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_daz_opt_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_finite_only_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_isa_version_803.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_unsafe_math_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/irif/irif.amdgcn.bc \
       test.cl -o test.so

TESTING
********

Currently all tests are offline:

   * OpenCL source is compiled to LLVM bitcode
   * Test bitcode is linked to library bitcode with llvm-link
   * Clang OpenCL compiler is run on resulting bitcode, producing code object.
   * Resulting code object is passed to llvm-objdump and amdhsacod -test.

The output of tests (which includes AMDGPU disassembly) can be displayed by running ctest -VV in build directory.

Tests for OpenCL conformance kernels can be enabled by specifying -DOCL_CONFORMANCE_HOME= to CMake, for example, cmake ... -DOCL_CONFORMANCE_HOME=/srv/hsa/drivers/opencl/tests/extra/hsa/ocl/conformance/1.2

ROCr Runtime
#############

Github link of ROCr Runtime check `Here <https://github.com/RadeonOpenCompute/ROCR-Runtime>`_

HSA Runtime API and runtime for ROCm
*************************************
This repository includes the user-mode API interfaces and libraries necessary for host applications to launch compute kernels to available HSA ROCm kernel agents. Reference source code for the core runtime is also available.
Initial target platform requirements

   * CPU: Intel Haswell or newer, Core i5, Core i7, Xeon E3 v4 & v5; Xeon E5 v3
   * GPU: Fiji ASIC (AMD R9 Nano, R9 Fury and R9 Fury X)
   * GPU: Polaris ASIC (AMD RX480)

Source code
*************
The HSA core runtime source code for the ROCR runtime is located in the src subdirectory. Please consult the associated README.md file for contents and build instructions.

Binaries for Ubuntu & Fedora and installation instructions
************************************************************
Pre-built binaries are available for installation from the ROCm package repository. For ROCR, they include:

Core runtime package:

   * HSA include files to support application development on the HSA runtime for the ROCR runtime
   * A 64-bit version of AMD's HSA core runtime for the ROCR runtime

Runtime extension package:

   * A 64-bit version of AMD's finalizer extension for ROCR runtime
   * A 64-bit version of AMD's runtime tools library
   * A 64-bit version of AMD's runtime image library, which supports the HSAIL image implementation only.

The contents of these packages are installed in /opt/rocm/hsa and /opt/rocm by default. The core runtime package depends on the hsakmt-roct-dev package

Installation instructions can be found in the `ROCm Documentation <https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html>`_


Infrastructure
***************
The HSA runtime is a thin, user-mode API that exposes the necessary interfaces to access and interact with graphics hardware driven by the AMDGPU driver set and the ROCK kernel driver. Together they enable programmers to directly harness the power of AMD discrete graphics devices by allowing host applications to launch compute kernels directly to the graphics hardware.

The capabilities expressed by the HSA Runtime API are:

   * Error handling
   * Runtime initialization and shutdown
   * System and agent information
   * Signals and synchronization
   * Architected dispatch
   * Memory management
   * HSA runtime fits into a typical software architecture stack.

The HSA runtime provides direct access to the graphics hardware to give the programmer more control of the execution. An example of low level hardware access is the support of one or more user mode queues provides programmers with a low-latency kernel dispatch interface, allowing them to develop customized dispatch algorithms specific to their application.

The HSA Architected Queuing Language is an open standard, defined by the HSA Foundation, specifying the packet syntax used to control supported AMD/ATI Radeon (c) graphics devices. The AQL language supports several packet types, including packets that can command the hardware to automatically resolve inter-packet dependencies (barrier AND & barrier OR packet), kernel dispatch packets and agent dispatch packets.

In addition to user mode queues and AQL, the HSA runtime exposes various virtual address ranges that can be accessed by one or more of the system's graphics devices, and possibly the host. The exposed virtual address ranges either support a fine grained or a coarse grained access. Updates to memory in a fine grained region are immediately visible to all devices that can access it, but only one device can have access to a coarse grained allocation at a time. Ownership of a coarse grained region can be changed using the HSA runtime memory APIs, but this transfer of ownership must be explicitly done by the host application.

Programmers should consult the HSA Runtime Programmer's Reference Manual for a full description of the HSA Runtime APIs, AQL and the HSA memory policy.

Sample
******
The simplest way to check if the kernel, runtime and base development environment are installed correctly is to run a simple sample. A modified version of the vector_copy sample was taken from the HSA-Runtime-AMD repository and added to the ROCR repository to facilitate this. Build the sample and run it, using this series of commands:

cd ROCR-Runtime/sample && make && ./vector_copy

If the sample runs without generating errors, the installation is complete.

Known issues
**************
  *  The image extension is currently not supported for discrete GPUs. An image extension library is not provided in the binary    	package. The standard hsa_ext_image.h extension include file is provided for reference.
  *  Each HSA process creates and internal DMA queue, but there is a system-wide limit of four DMA queues. The fifth simultaneous    	  HSA process will fail hsa_init() with HSA_STATUS_ERROR_OUT_OF_RESOURCES. To run an unlimited number of simultaneous HSA 	   	processes, set the environment variable HSA_ENABLE_SDMA=0.

**Disclaimer**

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2016 Advanced Micro Devices, Inc. All rights reserved.
