.. image:: Currrent_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v4.0
================================
December, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v4.0.0 release.

`Download AMD ROCm Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


List of Supported Operating Systems
-----------------------------------

The AMD ROCm platform is designed to support the following operating systems:

* Ubuntu 20.04.1 (5.4 and 5.6-oem) and 18.04.5 (Kernel 5.4)	

* CentOS 7.8 (3.10.0-1127) & RHEL 7.9 (3.10.0-1160.6.1.el7) (Using devtoolset-7 runtime support)

* CentOS 8.2 (4.18.0-193.el8) and RHEL 8.3 (4.18.0-240.1.1.el8) (devtoolset is not required)

* SLES 15 SP2



Fresh Installation of AMD ROCm v4.0 Recommended
-----------------------------------------------

A fresh and clean installation of AMD ROCm v4.0 is recommended. An upgrade from previous releases to AMD ROCm v4.0 is not supported.

For more information, refer to the AMD ROCm Installation Guide at:

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

**Note**: AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.

**Note**: *render group* is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use *video group*.

-  For ROCm v3.5 and releases thereafter,the *clinfo* path is changed to
   - */opt/rocm/opencl/bin/clinfo*.

-  For ROCm v3.3 and older releases, the *clinfo* path remains unchanged
   - */opt/rocm/opencl/bin/x86_64/clinfo*.
   
 **Note**: After an operating system upgrade, AMD ROCm may upgrade automatically and result in an error. This is because AMD ROCm does not support upgrades currently. You must uninstall and reinstall AMD ROCm after an operating system upgrade.

   
ROCm Multi Version Installation Update
---------------------------------------

With the AMD ROCm v4.0 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms are now deprecated for multi-version ROCm installs. For example, rocm-dkms3.8.0, rocm-dkms3.9.0.

-   Multi-version installation of ROCm should be performed by installing rocm-dev using each of the desired ROCm versions. For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.

-  Version files must be created for each multi-version rocm <= 4.0.0

   -  command: echo \| sudo tee /opt/rocm-/.info/version

   -  example: echo 4.0.0 \| sudo tee /opt/rocm-4.0.0/.info/version

-  The rock-dkms loadable kernel modules should be installed using a single rock-dkms package.

- ROCm v3.9 and above will not set any *ldconfig* entries for ROCm libraries for multi-version installation.  Users must set *LD_LIBRARY_PATH* to load the ROCm library version of choice.

**NOTE**: The single version installation of the ROCm stack remains the same. The rocm-dkms package can be used for single version installs and is not deprecated at this time.



AMD ROCm Documentation Updates
-----------------------------------

ROCm Installation Guide
===========================

The AMD ROCm Installation Guide in this release includes:

-  Updated Supported Environments

-  Installation Instructions

-  HIP Installation Instructions

- AMD ROCm and Mesa Multimedia Installation 

- Using CMake with AMD ROCm 


https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


ROCm SMI API Documentation Updates
===================================

-  System DMA (SDMA) Utilization API

-  ROCm-SMI Command Line Interface

-  Enhanced ROCm SMI Library for Events

For more information about ROCm SMI APIs, refer to the ROCm SMI API Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide_v3.10.pdf



HIP Documentation Updates
===========================
* HIP Programming Guide v4.0 

Add link

* HIP API Guide v4.0

Add link

* HIP FAQ 

For more information, see

https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq


General AMD ROCm Documentation Links
------------------------------------

Access the following links for more information:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supped platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Current_Release_Notes/ROCm-Version-History.html
   
   


===============================
INTRODUCING AMD INSTINCT™ MI100
===============================

The AMD Instinct™ MI100 accelerator is the world’s fastest HPC GPU, and a culmination of the AMD CDNA architecture, with all-new Matrix Core Technology, and AMD ROCm™ open ecosystem to deliver new levels of performance, portability, and productivity. AMD CDNA is an all-new GPU architecture from AMD to drive accelerated computing into the era of exascale computing. The new architecture augments scalar and vector processing with new Matrix Core Engines and adds Infinity Fabric™ technology to scale up to larger systems. The open ROCm ecosystem puts customers in control and is a robust, mature platform that is easy to develop for and capable of running the most critical applications. The overall result is that the MI100 is the first GPU to break the 10TFLOP/s FP64 barrier designed as the steppingstone to the next generation of Exascale systems that will deliver pioneering discoveries in machine learning and scientific computing.


Key Features of AMD Instinct™ MI100 
------------------------------------

Important features of the AMD Instinct™ MI100 accelerator include:

* Extended matrix core engine with Matrix Fused Multiply-Add (MFMA) for mixed-precision arithmetic and operates on KxN matrices (FP32, FP16, BF16, Int8) 

* Added native support for the bfloat16 data type

* 3 Infinity fabric connections per GPU enable a fully connected group of 4 GPUs in a ‘hive’ 


.. image:: /Current_Release_Notes/images/keyfeatures.PNG
   :align: center


Matrix Core Engines and GFX908 Considerations
----------------------------------------------

The AMD CDNA architecture builds on GCN’s foundation of scalars and vectors and adds matrices while simultaneously adding support for new numerical formats for machine learning and preserving backward compatibility for any software written for the GCN architecture. These Matrix Core Engines add a new family of wavefront-level instructions, the Matrix Fused MultiplyAdd or MFMA. The MFMA family performs mixed-precision arithmetic and operates on KxN matrices using four different types of input data: 8-bit integers (INT8), 16-bit half-precision FP (FP16), 16-bit brain FP (bf16), and 32-bit single-precision (FP32). All MFMA instructions produce either a 32-bit integer (INT32) or FP32 output, which reduces the likelihood of overflowing during the final accumulation stages of matrix multiplication.

On nodes with gfx908, MFMA instructions are available to substantially speed up matrix operations. This hardware feature is used only in matrix multiplications functions in rocBLAS and supports only three base types f16_r, bf16_r, and f32_r. 

* For half precision (f16_r and bf16_r) GEMM, use the function rocblas_gemm_ex, and set the compute_type parameter to f32_r.

* For single precision (f32_r) GEMM, use the function rocblas_sgemm.

* For single precision complex (f32_c) GEMM, use the function rocblas_cgemm.



References
------------

* For more information about bfloat16, see 

https://rocblas.readthedocs.io/en/master/usermanual.html

* For more details about AMD Instinct™ MI100 accelerator key features, see 

https://www.amd.com/system/files/documents/instinct-mi100-brochure.pdf

* For more information about the AMD Instinct MI100 accelerator, refer to the following sources:

 - AMD CDNA whitepaper at https://www.amd.com/system/files/documents/amd-cdna-whitepaper.pdf
 
 - MI100 datasheet at https://www.amd.com/system/files/documents/instinct-mi100-brochure.pdf

* AMD Instinct MI100/CDNA1 Shader Instruction Set Architecture (Dec. 2020) – This document describes the current environment, organization, and program state of AMD CDNA “Instinct MI100” devices. It details the instruction set and the microcode formats native to this family of processors that are accessible to programmers and compilers.

https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf



What's New in This Release
-----------------------------

RAS ENHANCEMENTS
~~~~~~~~~~~~~~~~~
RAS (Reliability, Availability, and Accessibility) features provide help with data center GPU management. It is a method provided to users to track and manage data points via options implemented in the ROCm-SMI Command Line Interface (CLI) tool. 

For more information about rocm-smi, see 

https://github.com/RadeonOpenCompute/ROC-smi 

The command options are wrappers of the system calls into the device driver interface as described here:

https://dri.freedesktop.org/docs/drm/gpu/amdgpu.html#amdgpu-ras-support



USING CMAKE WITH AMD ROCM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most components in AMD ROCm support CMake 3.5 or higher out-of-the-box and do not require any special Find modules. A Find module is often used downstream to find the files by guessing locations of files with platform-specific hints. Typically, the Find module is required when the upstream is not built with CMake or the package configuration files are not available.

AMD ROCm provides the respective config-file packages, and this enables find_package to be used directly. AMD ROCm does not require any Find module as the config-file packages are shipped with the upstream projects.

For more information, see 

UPDATE LINK



ROCm-SMI Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SDMA usage per-process is available using the following command,

::

   $ rocm-smi “showpids
   
   
::   


Enhanced ROCm SMI Library for Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm-SMI library clients can now register to receive the following
events:

-  GPU PRE RESET: This reset event is sent to the client just before a
   GPU is going to be RESET.

-  GPU POST RESET: This reset event is sent to the client after a
   successful GPU RESET.

-  GPU THERMAL THROTTLE: This Thermal throttling event is sent if GPU
   clocks are throttled
   
   
ROCm SMI Command Line Interface Hardware Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature provides a matrix representation of the GPUs present in a
system by providing information of the manner in which the nodes are
connected. This is represented in terms of weights, hops, and link types
between two given GPUs. It also provides the numa node and the CPU
affinity associated with every GPU.

.. image:: /Current_Release_Notes/images/CLI1.PNG
   :align: center

 

.. image:: /Current_Release_Notes/images/CLI2.PNG
   :align: center

 
 For more information about ROCm SMI API libraries, refer to the ROCm SMI API Guide at
 
 
 https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide_v3.10.pdf
 

ROCm Math and Communication Libraries
-------------------------------------

New rocSOLVER APIs
~~~~~~~~~~~~~~~~~~

The following new rocSOLVER APIs are added in this release:

.. image:: /Current_Release_Notes/images/rocsolverAPI.PNG
   :align: center

  

For more information, refer to

https://rocsolver.readthedocs.io/en/latest/userguide_api.html


RCCL Alltoallv Support in PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm v3.10 release includes a new API for ROCm Communication
Collectives Library (RCCL). This API sends data from all to all ranks
and each rank provides arrays of input/output data counts and offsets.

For details about the functions and parameters, see

https://rccl.readthedocs.io/en/master/allapi.html


ROCm AOMP ENHANCEMENTS
----------------------

AOMP Release 11.11-0
~~~~~~~~~~~~~~~~~~~~

The source code base for this release is the upstream LLVM 11 monorepo
release/11.x sources with the hash value

*176249bd6732a8044d457092ed932768724a6f06*

This release includes fixes to the internal Clang math headers:

-  This set of changes applies to clang internal headers to support
   OpenMP C, C++, and FORTRAN and for HIP C. This establishes
   consistency between NVPTX and AMDGCN offloading and between OpenMP,
   HIP, and CUDA. OpenMP uses function variants and header overlays to
   define device versions of functions. This causes clang LLVM IR
   codegen to mangled names of variants in both the definition and
   callsites of functions defined in the internal clang headers. These
   changes apply to headers found in the installation subdirectory
   lib/clang/11.0.0/include.

-  These changes temporarily eliminate the use of the libm bitcode
   libraries for C and C++. Although math functions are now defined with
   internal clang headers, a bitcode library of the C functions defined
   in the headers is still built for FORTRAN toolchain linking because
   FORTRAN cannot use c math headers. This bitcode library is installed
   in lib/libdevice/libm-.bc. The source build of this bitcode library
   is done with the aomp-extras repository and the component built
   script build_extras.sh. In the future, we will introduce across the
   board changes to eliminate massive header files for math libraries
   and replace them with linking to bitcode libraries.

-  Added support for -gpubnames in Flang Driver

-  Added an example category for Kokkos. The Kokkos example makefile
   detects if Kokkos is installed and, if not, it builds Kokkos from the
   Web. Refer to the script kokkos_build.sh in the bin directory on how
   to build Kokkos. Kokkos now builds cleanly with the OpenMP backend
   for simple test cases.

-  Fixed hostrpc cmake race condition in the build of openmp

-  Add a fatal error if missing -Xopenmp-target or -march options when
   -fopenmp-targets is specified. However, we do forgive this
   requirement for offloading to host when there is only a single target
   and that target is the host.

-  Fix a bug in InstructionSimplify pass where a comparison of two
   constants of different sizes found in the optimization pass. This
   fixes issue #182 which was causing kokkos build failure.

-  Fix openmp error message output for no_rocm_device_lib, was
   asserting.

-  Changed linkage on constant per-kernel symbols from external to
   weaklinkageonly to prevent duplicate symbols when building kokkos.



Fixed Defects
=============

The following defects are fixed in this release:

-  HIPfort failed to be installed

-  rocm-smi does not work as-is in 3.9, instead prints a reference to
   documentation

-  *showtopo*, weight and hop count shows wrong data

-  Unable to install RDC on CentOS/RHEL 7.8/8.2 & SLES

-  Unable to install mivisionx with error *Problem: nothing provides
   opencv needed*



Known Issues
--------------

Upgrade to AMD ROCm v3.10 Not Supported
========================================

An upgrade from previous releases to AMD ROCm v3.10 is not supported. A
fresh and clean installation of AMD ROCm v3.10 is recommended.


Deprecations
-------------------

This section describes deprecations and removals in AMD ROCm.

**WARNING: COMPILER-GENERATED CODE OBJECT VERSION 2 DEPRECATION**

Compiler-generated code object version 2 is no longer supported and will be removed shortly. AMD ROCm users must plan for the code object version 2 deprecation immediately. 

Support for loading code object version 2 is also being deprecated with no announced removal release.


Deploying ROCm
-------------------

AMD hosts both Debian and RPM repositories for the ROCm v3.10.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


DISCLAIMER 
----------------
The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

