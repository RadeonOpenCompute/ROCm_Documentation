.. image:: /Current_Release_Notes/amdblack.jpg
|



=====================================
AMD ROCm™ Release Notes v4.2
=====================================
May, 2021

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v4.2.0 release.

`Download AMD ROCm Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


List of Supported Operating Systems
-----------------------------------

The AMD ROCm platform is designed to support the following operating systems:

-  Ubuntu 20.04.2 HWE (5.4 and 5.6-oem) and 18.04.5 (Kernel 5.4)

-  CentOS 7.9 (3.10.0-1127) & RHEL 7.9 (3.10.0-1160.6.1.el7) (Using devtoolset-7 runtime support)
   
-  CentOS 8.3 (4.18.0-193.el8)and RHEL 8.3 (4.18.0-193.1.1.el8) (devtoolset is not required)

-  SLES 15 SP2




Fresh Installation of AMD ROCm v4.2 Recommended
-----------------------------------------------

Complete uninstallation of previous ROCm versions is required before installing a new version of ROCm. An upgrade from previous releases to
AMD ROCm v4.2 is not supported. 

For more information, refer to the AMD ROCm Installation Guide at:

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

.. note::

   AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.
   
.. note::

   *render group* is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use *video group*.
   

-  For ROCm v3.5 and releases thereafter,the *clinfo* path is changed to
   - */opt/rocm/opencl/bin/clinfo*.

-  For ROCm v3.3 and older releases, the *clinfo* path remains unchanged
   - */opt/rocm/opencl/bin/x86_64/clinfo*.
   
   
.. note::

After an operating system upgrade, AMD ROCm may upgrade automatically and result in an error. This is because AMD ROCm does not support upgrades currently. You must uninstall and reinstall AMD ROCm after an operating system upgrade.

   
ROCm Multi Version Installation Update
---------------------------------------

With the AMD ROCm v4.2 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms are now deprecated for multi-version ROCm installs. For example, rocm-dkms3.8.0, rocm-dkms3.9.0.

-   Multi-version installation of ROCm should be performed by installing rocm-dev using each of the desired ROCm versions. For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.

-  The rock-dkms loadable kernel modules should be installed using a single rock-dkms package.

- ROCm v3.9 and above will not set any *ldconfig* entries for ROCm libraries for multi-version installation.  Users must set *LD_LIBRARY_PATH* to load the ROCm library version of choice.

.. note::

   The single version installation of the ROCm stack remains the same. The rocm-dkms package can be used for single version installs and is not deprecated at this time.



AMD ROCm Documentation Updates
-----------------------------------

ROCm Installation Guide
===========================

The AMD ROCm Installation Guide in this release includes:

-  Updated Supported Environments

-  Installation Instructions

-  HIP Installation Instructions


https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


HIP Documentation Updates
===========================

-  HIP Programming Guide v4.2

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD%20HIP%20Programming%20Guide_v4.2.pdf


-  HIP API Guide v4.2

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_API_Guide_4.2.pdf

-  HIP-Supported CUDA API Reference Guide v4.2

https://github.com/RadeonOpenCompute/ROCm/blob/master/HIP_Supported_CUDA_API_Reference_Guide_v4.2.pdf

-  HIP FAQ

   For more information, refer to

https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq


ROCm Data Center User and API Guide
====================================

-  ROCm Data Center Tool User Guide

   -  Reliability, Accessibility, and Serviceability (RAS) Plugin Integration

For more information, refer to the ROCm Data Center User Guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide_v4.2.pdf

-  ROCm Data Center Tool API Guide

For more information, refer to the ROCm Data Center API Guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_Data_Center_Tool_API_Guide_v4.2.pdf

   
ROCm SMI API Documentation Updates
===================================
   
-  ROCm SMI API Guide

For more information, refer to the ROCm SMI API Guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_Manual_4.2.pdf

   

ROC Debugger User and API Guide 
===================================

-  ROC Debugger User Guide
   https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_Debugger_User_Guide_v4.2.pdf

-  Debugger API Guide
   https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_Debugger_API_Guide_v4.2.pdf


General AMD ROCm Documentation Links
------------------------------------

Access the following links for more information:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supported platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Current_Release_Notes/ROCm-Version-History.html
   
   


==============================================
What's New in This Release and Other Features
==============================================

HIP Enhancements 
-------------------

HIP Target Platform Macro
===========================

The platform macros are updated to target either the AMD or NVIDIA platform in HIP projects. They now include corresponding headers and
libraries for compilation/linking.

-  **HIP_PLATFORM_AMD** is defined if the HIP platform targets AMD. Note, **HIP_PLATFORM_HCC** was used previously if the HIP platform      targeted AMD. This is now deprecated.

-  **HIP_PLATFORM_NVIDIA** is defined if the HIP platform targets NVIDIA. Note, \_HIP_PLATFORM_NVCC_\_ was used previously if the HIP platform targeted NVIDIA. This is now deprecated.

For example,

::

            #if (defined(__HIP_PLATFORM_AMD__)) && !(defined(__HIP_PLATFORM_NVIDIA__))

            #include <hip/amd_detail/hip_complex.h>

            #elif !(defined(__HIP_PLATFORM_AMD__)) && (defined(__HIP_PLATFORM_NVIDIA__))

            #include <hip/nvidia_detail/hip_complex.h>

::

Updated HIP 'Include' Directories
==================================

In the ROCm v4.2 release, HIP *include* header directories for platforms are updated as follows:

-  *amd_detail/* - includes source header details for the 'amd' platform implementation. In previous releases, the 'hcc_detail' directory was
   defined, and it it is now deprecated.

-  *nvidia_detail/* - includes source header details for the 'nvidia' platform implementation. In previous releases, the 'nvcc_detail'
   directory was defined, and it is now deprecated.


HIP Stream Memory Operations
=============================

The ROCm v4.2 extends support to Stream Memory Operations to enable direct synchronization between Network Nodes and GPU. The following new
APIs are added:

-  hipStreamWaitValue32
-  hipStreamWaitValue64
-  hipStreamWriteValue32
-  hipStreamWriteValue64

For more details, see the HIP API guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_API_Guide_4.2.pdf


HIP Events in Kernel Dispatch
===============================

HIP events in kernel dispatch using *hipExtLaunchKernelGGL/hipExtLaunchKernel* and passed in the API are not 
explicitly recorded and should only be used to get elapsed time for that specific launch.

Events used across multiple dispatches, for example, start and stop events from different *hipExtLaunchKernelGGL/hipExtLaunchKernel* calls,
are treated as invalid unrecorded events. In such scenarios, HIP will display the error *hipErrorInvalidHandle* from *hipEventElapsedTime*.

For more details, refer to the HIP API Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_API_Guide_4.2.pdf


Changed Environment Variables for HIP
======================================

In the ROCm v3.5 release, the Heterogeneous Compute Compiler (HCC) compiler was deprecated, and the HIP-Clang compiler was introduced for
compiling Heterogeneous-Compute Interface for Portability (HIP) programs. In addition, the HIP runtime API was implemented on top of
Radeon Open Compute Common Language Runtime (ROCclr). ROCclr is an abstraction layer that provides the ability to interact with different
runtime backends such as ROCr.

While the HIP_PLATFORM=hcc environment variable was functional in subsequent releases, in the ROCm v4.1 release, the following environment
variables were changed:

-  *HIP_PLATFORM=hcc to HIP_PLATFORM=amd*

-  *HIP_PLATFORM=nvcc to HIP_PLATFORM=nvidia*

Therefore, any applications continuing to use the *HIP_PLATFORM=hcc* variable will fail. You must update the environment variables to reflect
the changes as mentioned above.

       

ROCm Data Center Tool
---------------------

RAS Integration
================

The ROCm Data Center (RDC) Tool is enhanced with the Reliability, Accessibility, and Serviceability (RAS) plugin.

For more information about RAS integration and installation, refer to the ROCm Data Center Tool User guide at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide_v4.2.pdf



ROCm Math and Communication Libraries
-------------------------------------

rocBLAS
=========

Enhancements and fixes:

-  Added option to install script to build only rocBLAS clients with a
   pre-built rocBLAS library

-  Supported gemm ext for unpacked int8 input layout on gfx908 GPUs

   -  Added new flags rocblas_gemm_flags::rocblas_gemm_flags_pack_int8x4 to specify if using the packed layout

      -  Set the rocblas_gemm_flags_pack_int8x4 when using packed int8x;, this should be always set on GPUs before gfx908

      -  For gfx908 GPUs, unpacked int8 is supported. Setting of this flag is no longer required

      -  Notice the default flags 0 uses unpacked int8 and changes the behaviour of int8 gemm from ROCm 4.1.0

-  Added a query function rocblas_query_int8_layout_flag to get the preferable layout of int8 for gemm by device

For more information, refer to

https://rocblas.readthedocs.io/en/master/

rocRAND
=========

-  Performance fixes

For more information, refer to

https://rocrand.readthedocs.io/en/latest/


rocSOLVER
===========

Support for:

-  Multi-level logging functionality

-  Implementation of the Thin-SVD algorithm

-  Reductions of generalized symmetric- and hermitian-definite
   eigenproblems:

   -  SYGS2, SYGST (with batched and strided_batched versions)
   -  HEGS2, HEGST (with batched and strided_batched versions)

-  Symmetric and hermitian matrix eigensolvers:

   -  SYEV (with batched and strided_batched versions)
   -  HEEV (with batched and strided_batched versions)

-  Generalized symmetric- and hermitian-definite eigensolvers:

   -  SYGV (with batched and strided_batched versions)
   -  HEGV (with batched and strided_batched versions)

For more information, refer to

https://rocsolver.readthedocs.io/en/latest/


rocSPARSE
===========

Enhancements:

-  SpMM (CSR, COO)
-  Code coverage analysis

For more information, refer to

https://rocsparse.readthedocs.io/en/latest/usermanual.html#rocsparse-gebsrmv


hipSPARSE
==========

Enhancements:

-  Generic API support, including SpMM (CSR, COO)
-  csru2csr, csr2csru

For more information, refer to

https://rocsparse.readthedocs.io/en/latest/usermanual.html#types


Fixed Defects
---------------

Performance Impact for LDS-BOUND Kernels
==========================================

The following issue is fixed in the ROCm v4.2 release.

The compiler in ROCm v4.1 generates LDS load and stores instructions that incorrectly assume equal performance between aligned and misaligned
accesses. While this does not impact code correctness, it may result in sub-optimal performance.



Known Issues
--------------

The following are the known issues in this release.

Upgrade to AMD ROCm v4.2 Not Supported
==========================================

An upgrade from previous releases to AMD ROCm v4.2 is not supported. Complete uninstallation of previous ROCm versions is required before
installing a new version of ROCm.


Modulefile Fails to Install Automatically in ROCm Multi-Version Environment
============================================================================

The ROCm v4.2 release includes a preliminary implementation of environment modules to enable switching between multi versions of ROCm
installation. The modulefile in */opt/rocm-4.2/lib/rocmmod* fails to install automatically in the ROCm multi-version environment.

This is a known limitation for environment modules in ROCm, and the issue is under investigation at this time.

**Workaround**

Ensure you install the modulefile in */opt/rocm-4.2/lib/rocmmod* manually in a multi-version installation environment.

For general information about modules, see

http://modules.sourceforge.net/


Issue with Input/Output Types for Scan Algorithms in rocThrust
=================================================================

As rocThrust is updated to match CUDA Thrust 1.10, the different input/output types for scan algorithms in rocThrust/CUDA Thrust are no
longer officially supported. In this situation, the current C++ standard does not specify the intermediate accumulator type leading to
potentially incorrect results and ill-defined behavior.

As a workaround, users can:

-  Use the same types for input and output

Or

-  For exclusive_scan, explicitly specify an *InitialValueType* in the last argument

Or

-  For inclusive_scan, which does not have an initial value argument, use a transform_iterator to explicitly cast the input iterators to
   match the output's value_type
   
   
Precision Issue in AMD RADEON™ PRO VII and AMD RADEON™ VII
==============================================================

In AMD RADEON™ Pro VII AND AMD RADEON™ VII, a precision issue can occur when using the Tensorflow XLA path.

This issue is currently under investigation.


Deprecations
---------------

This section describes deprecations and removals in AMD ROCm.

Compiler Generated Code Object Version 2 Deprecation
======================================================

Compiler-generated code object version 2 is no longer supported and has been completely removed. Support for loading code object version 2 is
also deprecated with no announced removal release.

========================================
Driver Compability Issue in ROCm v4.1
========================================

In certain scenarios, the ROCm v4.1 or higher run-time and userspace environment are not compatible with ROCm v4.0 and older driver implementations for 7nm-based (Vega 20) hardware (MI50 and MI60). 

To mitigate issues, the ROCm v4.1 or newer userspace prevents running older drivers for these GPUs.

Users are notified in the following scenarios:

* Bare Metal 
* Containers
 
Bare Metal
------------

In the bare-metal environment, the following error message displays in the console: 

*“HSA Error: Incompatible kernel and userspace, Vega 20 disabled. Upgrade amdgpu.”*

To test the compatibility, run the ROCm v4.1 version of rocminfo using the following instruction: 

*/opt/rocm-4.1.0/bin/rocminfo 2>&1 | less*

Containers
------------

A container (built with error detection for this issue) using a ROCm v4.1/higher or newer run-time is initiated to execute on an older kernel. The container fails to start and the following warning appears:

*Error: Incompatible ROCm environment. The Docker container requires the latest kernel driver to operate correctly. Upgrade the ROCm kernel to v4.1 or newer, or use a container tagged for v4.0.1 or older.*

To inspect the version of the installed kernel driver,  run either: 

* dpkg --status rock-dkms [Debian-based]

or

* rpm -ql rock-dkms [RHEL, SUSE, and others]

To install or update the driver, follow the installation instructions at:

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html



===============
Deploying ROCm
===============

AMD hosts both Debian and RPM repositories for the ROCm v4.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


DISCLAIMER 
============
The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

* AMD®, the AMD Arrow logo, AMD Instinct™, Radeon™, ROCm® and combinations thereof are trademarks of Advanced Micro Devices, Inc. 

* Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.

* PCIe® is a registered trademark of PCI-SIG Corporation. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

