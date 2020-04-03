
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


Overview
********

This repository contains the following libraries:

======= ============================================= =================
Name 	Comments 			               Dependencies
======= ============================================= =================
oclc*   `Open Compute library controls`_
ocml 	`Open Compute Math library`_                 	oclc*
ockl 	`Open Compute Kernel library`_                  oclc*
opencl 	OpenCL built-in library 			ocml,ockl,oclc*
hip     HIP built in library                            ocml,ockl,oclc*
hc 	Heterogeneous Compute built-in library 		ocml,ockl,oclc*
======= ============================================= =================

.. _Open Compute library controls: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/master/doc/OCML.md
.. _Open Compute Math Library: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/master/doc/OCML.md
.. _Open Compute Kernel library: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/master/doc/OCKL.md




Building
*********

The library sources should be compiled using a clang compiler built from sources in the amd-stg-open branch of AMD-modified llvm-project repository.
Use the following commands:

::

   git clone https://github.com/RadeonOpenCompute/llvm-project.git -b amd-stg-open llvm_amd
   cd llvm_amd
   mkdir -p build
   cd build
   cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
      ../llvm
   make
            

To build the library bitcodes, clone the amd_stg_open branch of this repository.
Run the following commands:

::
   git clone https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git -b amd-stg-open



and from its top level run the following commands:

::

   mkdir -p build
   cd build
   export LLVM_BUILD=... (path to LLVM build directory created above)
   CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD ..
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

To create packages for the libraray: make package



Using Bitcode Libraries
***************************
The ROCm language runtimes automatically add the required bitcode files during the LLVM linking stage invoked during the process of creating a code object. There are options to display the exact commands excecuted, but an approximation of the command the OpenCL runtime might use is as follows:

::

  $LLVM_BUILD/bin/clang -x cl -Xclang -finclude-default-header \
    -target amdgcn-amd-amdhsa -mcpu=gfx900 \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/opencl/opencl.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ocml/ocml.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ockl/ockl.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_correctly_rounded_sqrt_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_daz_opt_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_finite_only_off.amdgcn.bc \
       -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_unsafe_math_off.amdgcn.bc \
    -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_wavefrontsize64_off.amdgcn.bc \
    -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_isa_version_900.amdgcn.bc \
       test.cl -o test.so

Using from Cmake
*****************

The bitcode libraries are exported as CMake targets, organized in a CMake package. You can depend on this package using find_package(AMDDeviceLibs REQUIRED CONFIG) after ensuring the CMAKE_PREFIX_PATH includes either the build directory or install prefix of the bitcode libraries. The package defines a variable AMD_DEVICE_LIBS_TARGETS containing a list of the exported CMake targets.

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

Known issues
**************
 
  *  Each HSA process creates an internal DMA queue, but there is a system-wide limit of four DMA queues. When the limit is reached HSA processes will use internal kernels for copies.

**Disclaimer**

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2017 Advanced Micro Devices, Inc. All rights reserved.
