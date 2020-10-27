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

The openmp-extras_12.9-0_amd64.deb auxiliary package supports OpenMP within the ROCm compiler. It contains OpenMP specific header files,
which are installed in /opt/rocm/llvm/include as well as runtime libraries, fortran runtime libraries, and device bitcode files in
/opt/rocm/llvm/lib. The auxiliary package also consists of examples in the /opt/rocm/llvm/examples folder.

**NOTE**: The optional AOMP package resides in /opt/rocm//aomp/bin/clang and the ROCm compiler, which supports OpenMP for AMDGPU, is located in
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



ROCm SYSTEM MANAGEMENT INFORMATION
----------------------------------

The AMD ROCm v3.9 release consists of the following ROCm System Management Information (SMI) enhancements:

-  Shows the hardware topology

-  The ROCm-SMI showpids option shows per-process Compute Unit (CU) Occupancy, VRAM usage, and SDMA usage

-  Support for GPU Reset Event and Thermal Throttling Event in ROCm-SMI Library

ROCm-SMI Hardware Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm-SMI Command Line Interface (CLI) is enhanced to include new options to denote GPU inter-connect topology in the system along with
the relative distance between each other and the closest NUMA (CPU) node for each GPU.

.. image:: https://github.com/RadeonOpenCompute/ROCm_Documentation/blob/doc_updates/Current_Release_Notes/images/ROCMCLI1.PNG
    :align: center

  

Compute Unit Occupancy
~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm stack now supports a user process in querying Compute Unit (CU) occupancy at a particular moment. This service can be accessed to
determine if a process P is using sufficient compute units.

A periodic collection is used to build the profile of a compute unit occupancy for a workload.

.. image:: /Current_Release_Notes/images/ROCMCLI2.PNG 
   :align: center

ROCm supports this capability only on GFX9 devices. Users can access the functionality in two ways:

-  indirectly from the SMI library

-  directly via Sysfs

**NOTE**: On systems that have both GFX9 and non-GFX9 devices, users should interpret the compute unit (CU) occupancy value carefully as the
service does not support non-GFX9 devices.

Accessing Compute Unit Occupancy Indirectly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm System Management Interface (SMI) library provides a convenient interface to determine the CU occupancy for a process. To get the CU
occupancy of a process reported in percentage terms, invoke the SMI interface using rsmi_compute_process_info_by_pid_get(). The value is
reported through the member field cu_occupancy of struct rsmi_process_info_t.

::

   /**
      * @brief Encodes information about a process
      * @cu_occupancy Compute Unit usage in percent
      */
     typedef struct {
         - - -,
         uint32_t cu_occupancy;
     } rsmi_process_info_t;

     /**
      * API to get information about a process
     rsmi_status_t
         rsmi_compute_process_info_by_pid_get(uint32_t pid,
             rsmi_process_info_t *proc);

Accessing Compute Unit Occupancy Directly Using SYSFS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Information provided by SMI library is built from sysfs. For every valid device, ROCm stack surfaces a file by the name cu_occupancy in Sysfs.
Users can read this file to determine how that device is being used by a particular workload. The general structure of the file path is
/proc//stats\_/cu_occupancy

::

   /**
      * CU occupancy files for processes P1 and P2 on two devices with 
      * ids: 1008 and 112326
      */
     /sys/devices/virtual/kfd/kfd/proc/<Pid_1>/stats_1008/cu_occupancy
     /sys/devices/virtual/kfd/kfd/proc/<Pid_1>/stats_2326/cu_occupancy
     /sys/devices/virtual/kfd/kfd/proc/<Pid_2>/stats_1008/cu_occupancy
     /sys/devices/virtual/kfd/kfd/proc/<Pid_2>/stats_2326/cu_occupancy
     
   // To get CU occupancy for a process P<i>
     for each valid-device from device-list {
       path-1 = Build path for cu_occupancy file;
       path-2 = Build path for file Gpu-Properties;
       cu_in_use += Open and Read the file path-1;
       cu_total_cnt += Open and Read the file path-2;
     }
     cu_percent = ((cu_in_use * 100) / cu_total_cnt);
     

GPU Reset Event and Thermal Throttling Event
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm-SMI library clients can now register for the following events:

.. image:: /Current_Release_Notes/images/ROCMCLI3.PNG 
   :align: center




ROCm Math and Communication Libraries
-------------------------------------

"rocfft_execution_info_set_stream" API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rocFFT is a software library for computing Fast Fourier Transforms (FFT). It is part of AMDâ€™s software ecosystem based on ROCm. In addition
to AMD GPU devices, the library can be compiled with the CUDA compiler using HIP tools for running on Nvidia GPU devices.

The ˜rocfft_execution_info_set_stream" API is a function to specify optional and additional information to control execution. This API
specifies the compute stream, which must be invoked before the call to rocfft_execute. Compute stream is the underlying device queue/stream
where the library computations are inserted.

PREREQUISITES
^^^^^^^^^^^^^

Using the compute stream API makes the following assumptions:

-  This stream already exists in the program and assigns work to the stream

-  The stream must be of type hipStream_t. Note, it is an error to pass the address of a hipStream_t object

PARAMETERS
^^^^^^^^^^

Input

-  info execution info handle
-  stream underlying compute stream

Improved GEMM Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, rocblas_gemm_ext2() supports matrix multiplication D <= alpha \* A \* B + beta \* C, where the A, B, C, and D matrices are
single-precision float, column-major, and non-transposed, except that the row stride of C may equal 0. This means the first row of C is
broadcast M times in C:

.. image:: /Current_Release_Notes/images/GEMM.PNG
   :align: center

If an optimized kernel solution for a particular problem is not available, a slow fallback algorithm is used, and the first time a
fallback algorithm is used, the following message is printed to standard error:

*Warning: Using slow on-host algorithm, because it is not implemented in Tensile yet.*

**NOTE**: ROCBLAS_LAYER controls the logging of the calls. It is recommended to use logging with the rocblas_gemm_ext2() feature, to
identify the precise parameters which are passed to it.

-  Setting the ROCBLAS_LAYER environment variable to 2 will print the problem parameters as they are being executed.

-  Setting the ROCBLAS_LAYER environment variable to 4 will collect all of the sizes, and print them out at the end of program execution.

For more logging information, refer to

https://rocblas.readthedocs.io/en/latest/logging.html.

New Matrix Pruning Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this release, the following new Matrix Pruning functions are introduced.

.. image:: /Current_Release_Notes/images/matrix.PNG 
   :align: center

   
rocSOLVER General Matrix Singular Value Decomposition API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rocSOLVER General Matrix Singular Value Decomposition (GESVD) API is now available in the AMD ROCm v3.9 release.

GESVD computes the Singular Values and, optionally, the Singular Vectors of a general m-by-n matrix A (Singular Value Decomposition).

The SVD of matrix A is given by:

::

   A = U * S * V'

For more information, refer to

https://rocsolver.readthedocs.io/en/latest/userguide_api.html


ROCm AOMP ENHANCEMENTS
----------------------

AOMP v11.08-0
~~~~~~~~~~~~~

The source code base for this release is the upstream LLVM 11 monorepo release/11.x sources as of August 18, 2020 with the hash value

*aabff0f7d564b22600b33731e0d78d2e70d060b4*

The amd-llvm-project branch used to build this release is amd-stg-openmp. In addition to complete source tarball, the artifacts of this release includes the file llvm-project.patch. This file shows the delta from the llvm-project upstream release/11.x which is currently at 32715 lines in 240 files. These changes include support for flang driver, OMPD support and the hsa libomptarget plugin. Our goal is to reduce this with continued upstreaming activity.

These are the major changes for this release of AOMP:

-  Switch to the LLVM 11.x stable code base.

-  OMPD updates for flang.

-  To support debugging OpenMP, selected OpenMP runtime sources are included in lib-debug/src/openmp. The ROCgdb debugger will find these
   automatically.

-  Threadsafe hsa plugin for libomptarget.

-  Updates to support device libraries.

-  Openmpi configure issue with real16 resolved.

-  DeviceRTL memory use is now independent of number of openmp binaries.

-  Startup latency on first kernel launch reduced by order of magnitude.


AOMP v11.07-1
~~~~~~~~~~~~~

The source code base for this release is the upstream LLVM 11 monorepo development sources as July 10, 2020 with hash valued 979c5023d3f0656cf51bd645936f52acd62b0333 The amd-llvm-project branch used to build this release is amd-stg-openmp. In addition to complete source tarball, the artifacts of this release includes the file
llvm-project.patch. This file shows the delta from the llvm-project upstream trunk which is currently at 34121 lines in 277 files. Our goal is to reduce this with continued upstreaming activity.

-  Inclusion of OMPD support which is not yet upstream

-  Build of ROCgdb

-  Host runtime optimisation. GPU image information is now mostly read on the host instead of from the GPU.

-  Fixed the source build scripts so that building from the source tarball does not fail because of missing test directories. This fixes issue #116.



Fixed Defects
=============

The following defects are fixed in this release:

-  Random Soft Hang Observed When Running ResNet-Based Models

-  (AOMP) "Undefined Hidden Symbol" Linker Error Causes Compilation Failure in HIP

-  MIGraphx -> test_gpu_ops_test FAILED

-  Unable to install RDC on CentOS/RHEL 7.8/8.2 & SLES


Known Issues
-------------------

The following are the known issues in this release.

(AOMP) HIP EXAMPLE DEVICE_LIB FAILS TO COMPILE
----------------------------------------------

The HIP example device_lib fails to compile and displays the following error:

*lld: error: undefined hidden symbol: inc_arrayval*

The recommended workaround is to use */opt/rocm/hip/bin/hipcc to compile HIP applications*.


HIPFORT INSTALLATION FAILURE
----------------------------

Hipfort fails to install during the ROCm installation.

As a workaround, you may force install hipfort using the following instructions:

Ubuntu
~~~~~~

::

   sudo apt-get -o Dpkg::Options::="--force-overwrite" install hipfort

SLES
~~~~

Zypper gives you an option to continue with the overwrite during the installation.

CentOS
~~~~~~

Download hipfort to a temporary location and force install with rpm:

::

   yum install --downloadonly --downloaddir=/tmp/hipfort hipfort
   rpm -i --replacefiles hipfort<package-version>


MEMORY FAULT ACCESS ERROR DURING MEMORY TEST OF ROCM VALIDATION SUITE 
-----------------------------------------------------------------------

When the ROCm Validation Suite (RVS) is installed using the prebuilt Debian/rpm package and run for the first time, the memory module displays the following error message, 

*“Memory access fault by GPU node-<x> (Agent handle: 0xa55170) on address 0x7fc268c00000. Reason: Page not present or supervisor privilege.
Aborted (core dumped).”*

As a workaround, run the test again. Subsequent runs appear to fix the error.

**NOTE**: The error may appear after a system reboot. Run the test again to fix the issue.  

Note, reinstallation of ROCm Validation Suite is not required. 



Deprecations
-------------------

This section describes deprecations and removals in AMD ROCm.

**WARNING: COMPILER-GENERATED CODE OBJECT VERSION 2 DEPRECATION**

Compiler-generated code object version 2 is no longer supported and will be removed shortly. AMD ROCm users must plan for the code object version 2 deprecation immediately. 

Support for loading code object version 2 is also being deprecated with no announced removal release.


Deploying ROCm
-------------------

AMD hosts both Debian and RPM repositories for the ROCm v3.9.x packages.

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

