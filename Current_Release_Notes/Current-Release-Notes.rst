.. image:: Currrent_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v3.8.0
================================
September, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v3.8.0 release.

`Download AMD ROCm v3.7.0 Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


Support for Vega 7nm Workstation
--------------------------------

This release extends support to the Vega 7nm Workstation (Vega20 GL-XE) version.

List of Supported Operating Systems
-----------------------------------

The AMD ROCm platform is designed to support the following operating
systems:

-  Ubuntu 20.04 (5.4 and 5.6-oem) and 18.04.5 (Kernel 5.4)
-  CentOS 7.8 & RHEL 7.8 (Kernel 3.10.0-1127) (Using devtoolset-7
   runtime support)
-  CentOS 8.2 & RHEL 8.2 (Kernel 4.18.0 ) (devtoolset is not required)
-  SLES 15 SP1

Fresh Installation of AMD ROCm v3.8 Recommended
-----------------------------------------------

A fresh and clean installation of AMD ROCm v3.8 is recommended. An upgrade from previous releases to AMD ROCm v3.8 is not supported.

For more information, refer to the AMD ROCm Installation Guide at:

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

**Note**: AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a
fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.

**Note**: *render group* is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use *video group*.

-  For ROCm v3.5 and releases thereafter,the *clinfo* path is changed to
   - */opt/rocm/opencl/bin/clinfo*.

-  For ROCm v3.3 and older releases, the *clinfo* path remains unchanged
   - */opt/rocm/opencl/bin/x86_64/clinfo*.

AMD ROCm Documentation Updates
==============================

AMD ROCm Installation Guide
---------------------------

The AMD ROCm Installation Guide in this release includes:

-  Updated Supported Environments
-  HIP Installation Instructions
-  Tensorflow ROCm Port: Basic Installations on RHEL v8.2

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

AMD ROCm - HIP Documentation Updates
------------------------------------

-  HIP Repository Information

For more information, see
https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html#hip-repository-information

ROCm Data Center Tool User Guide
--------------------------------

-  Error-Correction Codes Field and Output Documentation
-  Installation and Build instructions for SLES 15 Service Pack 1

For more information, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf

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
==========================

Hipfort-Interface for GPU Kernel Libraries
------------------------------------------

Hipfort is an interface library for accessing GPU Kernels. It provides support to the AMD ROCm architecture from within the Fortran programming
language. Currently, the gfortran and HIP-Clang compilers support hipfort. Note, the gfortran compiler belongs to the GNU Compiler
Collection (GCC). While hipfc wrapper calls hipcc for the non-fortran kernel source, gfortran is used for FORTRAN applications that call GPU
kernels.

The hipfort interface library is meant for Fortran developers with a focus on gfortran users.

For information on HIPFort installation and examples, see

https://github.com/ROCmSoftwarePlatform/hipfort


Error Correcting Code Fields in ROCm Data Center Tool
-----------------------------------------------------

The ROCm Data Center (RDC) tool is enhanced to provide counters to track correctable and uncorrectable errors. While a single bit per word error
can be corrected, double bit per word errors cannot be corrected.

The RDC tool now helps monitor and protect undetected memory data corruption. If the system is using ECC- enabled memory, the ROCm Data
Center tool can report the error counters to monitor the status of the memory.

.. image:: /Current_Release_Notes/forweb.PNG
    :align: center

For more information, refer to the ROCm Data Center User Guide at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf


Static Linking Libraries
------------------------

The underlying libraries of AMD ROCm are dynamic and are called shared objects (.so) in Linux. The AMD ROCm v3.8 release includes the
capability to build static ROCm libraries and link to the applications statically. CMake target files enable linking an application statically
to ROCm libraries and each component exports the required dependencies for linking. The static libraries are called Archives (.a) in Linux.

This release also comprises of the requisite changes required for all the components to work in a static environment. The components have been
successfully tested for basic functionalities like *rocminfo /rocm_bandwidth_test* and archives.

In the AMD ROCm v3.8 release, the following libraries support static linking:

.. image:: /Current_Release_Notes/staticlinkinglib.PNG
    :align: center


Fixed Defects
=============

The following defects are fixed in this release:

-  GPU Kernel C++ Names Not Demangled
-  MIGraphX Fails for fp16 Datatype
-  Issue with Peer-to-Peer Transfers
-  *"rocprof"* option *“parallel-kernels" Not Supported in this Release


Known Issues
============

ROCm Data Center Installation Issue on CentOS/RHEL 7.8/8.2 and SLES
-------------------------------------------------------------------

Installing ROCm Data Center on CentOS/RHEL v7.8/v8.2 and SLES may fail with an error.

This issue is under investigation and there is no known workaround currently.


Undefined Reference Issue in Statically Linked Libraries
--------------------------------------------------------

Libraries and applications statically linked using flags *-rtlib=compiler-rt*, such as rocBLAS, have an implicit dependency on
gcc_s not captured in their CMAKE configuration.

Client applications may require linking with an additional library *-lgcc_s* to resolve the undefined reference to symbol *"_Unwind_ResumeGCC_3.0"*.

MIGraphX Pooling Operation Fails for Some Models
------------------------------------------------

MIGraphX does not work for some models with pooling operations and the following error appears:

*˜test_gpu_ops_test FAILED"*

This issue is currently under investigation and there is no known workaround currently.

MIVisionX Installation Error on CentOS/RHEL8.2 and SLES 15
----------------------------------------------------------

Installing ROCm on MIVisionX results in the following error on CentOS/RHEL8.2 and SLES 15:

*"Problem: nothing provides opencv needed"*

As a workaround, install opencv before installing MIVisionX.


Deploying ROCm
==============

AMD hosts both Debian and RPM repositories for the ROCm v3.7.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html




DISCLAIMER 
===========
The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information.  Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.  No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document.  Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. S
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

