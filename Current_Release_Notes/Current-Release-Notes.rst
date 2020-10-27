.. image:: Currrent_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v3.9.0
================================
October, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v3.9.0 release.

`Download AMD ROCm Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


Support for Ubuntu 20.04.1
--------------------------

In this release, AMD ROCm extends support to Ubuntu 20.04.1, including
v5.4 and v5.6-oem.

Support for SLES 15 SP2
-----------------------

This release extends support to SLES 15 SP2.


List of Supported Operating Systems
-----------------------------------

The AMD ROCm platform is designed to support the following operating
systems:

The AMD ROCm platform is designed to support the following operating
systems:

-  Ubuntu 20.04.1 (5.4 and 5.6-oem) and 18.04.5 (Kernel 5.4)

-  CentOS 7.8 & RHEL 7.8 (Kernel 3.10.0-1127) (Using devtoolset-7
   runtime support)

-  CentOS 8.2 & RHEL 8.2 (Kernel 4.18.0 ) (devtoolset is not required)

-  SLES 15 SP1

-  SLES 15 SP2


Fresh Installation of AMD ROCm v3.8 Recommended
-----------------------------------------------

A fresh and clean installation of AMD ROCm v3.9 is recommended. An upgrade from previous releases to AMD ROCm v3.9 is not supported.

For more information, refer to the AMD ROCm Installation Guide at:

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

**Note**: AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.

**Note**: *render group* is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use *video group*.

-  For ROCm v3.5 and releases thereafter,the *clinfo* path is changed to
   - */opt/rocm/opencl/bin/clinfo*.

-  For ROCm v3.3 and older releases, the *clinfo* path remains unchanged
   - */opt/rocm/opencl/bin/x86_64/clinfo*.
   
ROCm MultiVersion Installation Update
---------------------------------------

With the AMD ROCm v3.9 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms are now deprecated for multi-version ROCm installs. For example, rocm-dkms3.7.0, rocm-dkms3.8.0.

-   Multi-version installation of ROCm should be performed by installing rocm-dev using each of the desired ROCm versions. For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.

-  Version files must be created for each multi-version rocm <= 3.9.0

   -  command: echo \| sudo tee /opt/rocm-/.info/version

   -  example: echo 3.9.0 \| sudo tee /opt/rocm-3.9.0/.info/version

-  The rock-dkms loadable kernel modules should be installed using a single rock-dkms package.

**NOTE**: The single version installation of the ROCm stack remains the same. The rocm-dkms package can be used for single version installs and is not deprecated at this time.



AMD ROCm Documentation Updates
-----------------------------------

AMD ROCm Installation Guide
================================

The AMD ROCm Installation Guide in this release includes:

-  Updated Supported Environments
-  Multi-version Installation Instructions

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

ROCm Compiler Documentation Updates
======================================

The ROCm Compiler documentation updates include,

-  OpenMP Extras v12.9-0
-  OpenMP-Extras Installation
-  OpenMP-Extras Source Build
-  AOMP-v11.9-0
-  AOMP Source Build

For more information, see

https://rocmdocs.amd.com/en/latest/Programming_Guides/openmp_support.html

For the updated ROCm SMI API Guide, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_Manual_v3.9.pdf


ROCm System Management Information
====================================

ROCM-SMI version: 1.4.1 \| Kernel version: 5.6.20

-  ROCm SMI and Command Line Interface
-  ROCm SMI APIs for Compute Unit Occupancy

   -  Usage
   -  Optional Arguments
   -  Display Options
   -  Topology
   -  Pages Information
   -  Hardware-related Information
   -  Software-related/controlled information
   -  Set Options
   -  Reset Options
   -  Auto-response Options
   -  Output Options

For more information, refer to

https://rocmdocs.amd.com/en/latest/ROCm_System_Managment/ROCm-System-Managment.html#rocm-command-line-interface

For ROCm SMI API Guide, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_Manual_v3.9.pdf


AMD ROCm - HIP Documentation Updates
=======================================

-  HIP Porting Guide â€“ CU_Pointer_Attribute_Memory_Type

For more information, refer to

https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html#hip-porting-guide



General AMD ROCm Documentation Links
------------------------------------

Access the following links for more information:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supped platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#build-amd-rocm

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#amd-rocm-version-history
   

What's New in This Release
-----------------------------

ROCm Compiler Enhancements
=============================

The ROCm compiler support in the llvm-amdgpu-12.0.dev-amd64.deb package is enhanced to include support for OpenMP. To utilize this support, the additional package openmp-extras_12.9-0_amd64.deb is required.

Note, by default, both packages are installed during the ROCm v3.9 installation. For information about ROCm installation, refer to the ROCm Installation Guide.

AMD ROCm supports the following compilers:

-  C++ compiler - Clang++
-  C compiler - Clang
-  Flang - FORTRAN compiler (FORTRAN 2003 standard)

**NOTE** : All of the above-mentioned compilers support:

-  OpenMP standard 4.5 and an evolving subset of the OpenMP 5.0 standard
-  OpenMP computational offloading to the AMD GPUs

For more information about AMD ROCm compilers, see the Compiler Documentation section at,

https://rocmdocs.amd.com/en/latest/index.html

Auxiliary Package Supporting OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The openmp-extras_12.9-0_amd64.deb auxiliary package supports OpenMP
within the ROCm compiler. It contains OpenMP specific header files,
which are installed in /opt/rocm/llvm/include as well as runtime
libraries, fortran runtime libraries, and device bitcode files in
/opt/rocm/llvm/lib. The auxiliary package also consists of examples in
the /opt/rocm/llvm/examples folder.

**NOTE**: The optional AOMP package resides in /opt/rocm//aomp/bin/clang
and the ROCm compiler, which supports OpenMP for AMDGPU, is located in
/opt/rocm/llvm/bin/clang.

AOMP Optional Package Deprecation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the AMD ROCm v3.9 release, the optional AOMP package provided support for OpenMP. While AOMP is available in this release, the optional package may be deprecated from ROCm in the future. It is recommended you transition to the ROCm compiler or AOMP standalone releases for OpenMP support.

Understanding ROCm Compiler OpenMP Support and AOMP OpenMP Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AOMP OpenMP support in ROCm v3.9 is based on the standalone AOMP v11.9-0, with LLVM v11 as the underlying system. However, the ROCm compiler's OpenMP support is based on LLVM v12 (upstream).

**NOTE**: Do not combine the object files from the two LLVM implementations. You must rebuild the application in its entirety using either the AOMP OpenMP or the ROCm OpenMP implementation.


Example - OpenMP Using the ROCm Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   $ cat helloworld.c
   #include <stdio.h>
   #include <omp.h>
    int main(void) {
     int isHost = 1; 
   #pragma omp target map(tofrom: isHost)
     {
       isHost = omp_is_initial_device();
       printf("Hello world. %d\n", 100);
       for (int i =0; i<5; i++) {
         printf("Hello world. iteration %d\n", i);
       }
     }
      printf("Target region executed on the %s\n", isHost ? "host" : "device");
     return isHost;
   }
   $ /opt/rocm/llvm/bin/clang  -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 helloworld.c -o helloworld
   $ export LIBOMPTARGET_KERNEL_TRACE=1
   $ ./helloworld
   DEVID: 0 SGN:1 ConstWGSize:256  args: 1 teamsXthrds:(   1X 256) reqd:(   1X   0) n:__omp_offloading_34_af0aaa_main_l7
   Hello world. 100
   Hello world. iteration 0
   Hello world. iteration 1
   Hello world. iteration 2
   Hello world. iteration 3
   Hello world. iteration 4
   Target region executed on the device

For more examples, see */opt/rocm/llvm/examples*.

.. _rocm-system-management-information-1:



Fixed Defects
=============

The following defects are fixed in this release:

-  GPU Kernel C++ Names Not Demangled
-  MIGraphX Fails for fp16 Datatype
-  Issue with Peer-to-Peer Transfers
-  *"rocprof"* option *“parallel-kernels" Not Supported in this Release


Known Issues
-------------------

Undefined Reference Issue in Statically Linked Libraries
===============================================================

Libraries and applications statically linked using flags *-rtlib=compiler-rt*, such as rocBLAS, have an implicit dependency on
gcc_s not captured in their CMAKE configuration.

Client applications may require linking with an additional library *-lgcc_s* to resolve the undefined reference to symbol *"_Unwind_ResumeGCC_3.0"*.


MIGraphX Pooling Operation Fails for Some Models
========================================================

MIGraphX does not work for some models with pooling operations and the following error appears:

*˜test_gpu_ops_test FAILED"*

This issue is currently under investigation and there is no known workaround currently.


MIVisionX Installation Error on CentOS/RHEL8.2 and SLES 15
=============================================================

Installing ROCm on MIVisionX results in the following error on CentOS/RHEL8.2 and SLES 15:

*"Problem: nothing provides opencv needed"*

As a workaround, install opencv before installing MIVisionX.


Deploying ROCm
-------------------

AMD hosts both Debian and RPM repositories for the ROCm v3.7.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html



DISCLAIMER 
----------------
The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information.  Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.  No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document.  Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. S
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

