.. image:: Currrent_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v4.1
================================
March, 2021

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v4.0.0 release.

`Download AMD ROCm Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


List of Supported Operating Systems
-----------------------------------

The AMD ROCm platform is designed to support the following operating systems:

- Ubuntu 20.04.1 (5.4 and 5.6-oem) and 18.04.5 (Kernel 5.4)
-  CentOS 7.9 (3.10.0-1127) & RHEL 7.9 (3.10.0-1160.6.1.el7) (Using
   devtoolset-7 runtime support)
-  CentOS 8.3 (4.18.0-193.el8) and RHEL 8.3 (4.18.0-193.1.1.el8)
   (devtoolset is not required)
-  SLES 15 SP2



Fresh Installation of AMD ROCm v4.1 Recommended
-----------------------------------------------

A complete uninstallation of previous ROCm versions is required before installing a new version of ROCm. An upgrade from previous releases to
AMD ROCm v4.1 is not supported. 

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

With the AMD ROCm v4.1 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms are now deprecated for multi-version ROCm installs. For example, rocm-dkms3.8.0, rocm-dkms3.9.0.

-   Multi-version installation of ROCm should be performed by installing rocm-dev using each of the desired ROCm versions. For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.

-  Version files must be created for each multi-version rocm <= 4.1.0

   -  command: echo \| sudo tee /opt/rocm-/.info/version

   -  example: echo 4.1.0 \| sudo tee /opt/rocm-4.1.0/.info/version

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

-  HIP Programming Guide v4.1

   https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_Programming_Guide_v4.1.pdf

-  HIP API Guide v4.1

   https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_API_Guide_v4.1.pdf

-  HIP-Supported CUDA API Reference Guide v4.1

   https://github.com/RadeonOpenCompute/ROCm/blob/master/HIP_Supported_CUDA_API_Reference_Guide_v4.1.pdf

-  HIP FAQ

   For more information, refer to

   https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq


ROCm Data Center User and API Guide
====================================

-  ROCm Data Center Tool User Guide

   -  Grafana Plugin Integration

   For more information, refer to the ROCm Data Center User Guide at,

   https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide_v4.1.pdf

-  ROCm Data Center Tool API Guide

   For more information, refer to the ROCm Data Center API Guide at,

   https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_Data_Center_Tool_API_Manual_4.1.pdf
   
   
ROCm SMI API Documentation Updates
===================================
   
-  ROCm SMI API Guide

   For more information, refer to the ROCm SMI API Guide at,

   https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_GUIDE_v4.1.pdf



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

TARGETID FOR MULTIPLE CONFIGURATIONS
--------------------------------------

The new TargetID functionality allows compilations to specify various configurations of the supported hardware.

Previously, ROCm supported only a single configuration per target.

With the TargetID enhancement, ROCm supports configurations for Linux, PAL and associated configurations such as XNACK. This feature addresses
configurations for the same target in different modes and allows applications to build executables that specify the supported
configurations, including the option to be agnostic for the desired setting.


New Code Object Format Version for TargetID
============================================

-  A new clang option *-mcode-object-version* can be used to request the legacy code object version 3 or code object version 2.    For more information, refer to

   https://llvm.org/docs/AMDGPUUsage.html#elf-code-object

-  A new clang *offload-arch=* option is introduced to specify the offload target architecture(s) for the HIP language.

-  The clang's *offload-arch=* and *-mcpu* options accept a new Target ID syntax. This allows both the processor and target      feature settings to be specified. 
   
   For more details, refer to

   https://llvm.org/docs/AMDGPUUsage.html#amdgpu-target-id

   -  If a target feature is not specified, it defaults to a new concept of "any". The compiler, then, produces code, which executes on a target configured for           either value of the setting impacting the overall performance. It is recommended to explicitly specify the setting for more efficient performance.

   -  In particular, the setting for XNACK now defaults to produce less performant code than previous ROCm releases.

   -  The legacy clang *-mxnack*, *-mno-xnack*, *-msram-ecc*, and *-mno-sram-ecc* options are deprecated. They are still supported, however, they will be removed in       a future release.

   -  The new Target ID syntax renames the SRAM ECC feature from *sram-ecc* to *sramecc*.

-  The clang offload bundler uses the new offload hipv4 for HIP code object version 4. For more information, see
   https://clang.llvm.org/docs/ClangOffloadBundler.html

-  ROCm v4.1 corrects code object loading to enforce target feature settings of the code object to match the setting of the agent. It
   also corrects the recording of target feature settings in the code object. As a consequence, the legacy code objects may no longer load
   due to mismatches.

-  gfx802, gfx803, and gfx805 do not support the XNACK target feature in the ROCm v4.1 release.


USING CMAKE WITH AMD ROCM
===========================

Most components in AMD ROCm support CMake 3.5 or higher out-of-the-box and do not require any special Find modules. A Find module is often used downstream to find the files by guessing locations of files with platform-specific hints. Typically, the Find module is required when the upstream is not built with CMake or the package configuration files are not available.

AMD ROCm provides the respective config-file packages, and this enables find_package to be used directly. AMD ROCm does not require any Find module as the config-file packages are shipped with the upstream projects.

For more information, see 

https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html


AMD ROCM AND MESA MULTIMEDIA 
===============================

AMD ROCm extends support to Mesa Multimedia. Mesa is an open-source software implementation of OpenGL, Vulkan, and other graphics API specifications. Mesa translates these specifications to vendor-specific graphics hardware drivers.

For detailed installation instructions, refer to

https://rocmdocs.amd.com/en/latest/Installation_Guide/Mesa-Multimedia-Installation.html


ROCM – SYSTEM MANAGEMENT INTERFACE
====================================

The following enhancements are made to ROCm System Management Interface (SMI).


Support for Printing PCle Information on AMD Instinct™100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD ROCm extends support for printing PCle information on AMD Instinct MI100. 

To check the pp_dpm_pcie file, use "rocm-smi --showclocks".

 */opt/rocm-4.0.0-6132/bin/rocm_smi.py  --showclocks*

.. image:: /Current_Release_Notes/images/SMI.PNG
   :align: center
   

New API for xGMI 
~~~~~~~~~~~~~~~~~~

Rocm_smi_lib now provides an API that exposes xGMI (inter-chip Global Memory Interconnect) throughput from one node to another. Refer to the rocm_smi_lib API documentation for more details. 

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide_v4.0.pdf


AMD GPU Debugger Enhancements
=================================

In this release, AMD GPU Debugger has the following enhancements:

* ROCm v4.0 ROCgdb is based on gdb 10.1

* Extended support for AMD Instinct™ MI100 


Known Issues
--------------

Upgrade to AMD ROCm v4.0 Not Supported
========================================

An upgrade from previous releases to AMD ROCm v4.0 is not supported. A fresh and clean installation of AMD ROCm v4.0 is recommended.


Deprecations
--------------

This section describes deprecations and removals in AMD ROCm.

COMPILER-GENERATED CODE OBJECT VERSION 2
=========================================

*WARNING: COMPILER-GENERATED CODE OBJECT VERSION 2 DEPRECATION*

Compiler-generated code object version 2 is no longer supported and will be removed shortly. AMD ROCm users must plan for the code object version 2 deprecation immediately. 

Support for loading code object version 2 is also being deprecated with no announced removal release.


ROCr RUNTIME DEPRECATIONS
============================

The following ROCr Runtime enumerations, functions, and structs are deprecated in the AMD ROCm v4.0 release.

Deprecated ROCr Runtime Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* hsa_isa_get_info

* hsa_isa_compatible

* hsa_executable_create

* hsa_executable_get_symbol

* hsa_executable_iterate_symbols

* hsa_code_object_serialize

* hsa_code_object_deserialize

* hsa_code_object_destroy

* hsa_code_object_get_info

* hsa_executable_load_code_object

* hsa_code_object_get_symbol

* hsa_code_object_get_symbol_from_name

* hsa_code_symbol_get_info

* hsa_code_object_iterate_symbols


Deprecated ROCr Runtime Enumerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* HSA_ISA_INFO_CALL_CONVENTION_COUNT

* HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE

* HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT

* HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH

* HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME

* HSA_EXECUTABLE_SYMBOL_INFO_AGENT

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST

* HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION

* HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION

   - hsa_code_object_type_t
 
   - hsa_code_object_info_t
 
   - hsa_code_symbol_info_t
   

Deprecated ROCr Runtime Structs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* hsa_code_object_t

* hsa_callback_data_t

* hsa_code_symbol_


AOMP DEPRECATION
====================

As of AMD ROCm v4.0, AOMP (aomp-amdgpu) is deprecated. OpenMP support has moved to the openmp-extras auxiliary package, which leverages the ROCm compiler on LLVM 12.

For more information, refer to 

https://rocmdocs.amd.com/en/latest/Programming_Guides/openmp_support.html


Deploying ROCm
-------------------

AMD hosts both Debian and RPM repositories for the ROCm v4.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


DISCLAIMER 
----------------
The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

* AMD®, the AMD Arrow logo, AMD Instinct™, Radeon™, ROCm® and combinations thereof are trademarks of Advanced Micro Devices, Inc. 

* Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.

* PCIe® is a registered trademark of PCI-SIG Corporation. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

