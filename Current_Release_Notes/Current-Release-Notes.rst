.. image:: /Current_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v3.6.0
================================
July, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v3.6.0 release.

-  ` Download AMD ROCm v3.6 Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


-  `Supported Operating Systems and Documentation
   Updates <#Supported-Operating-Systems-and-Documentation-Updates>`__

   -  `Supported Operating Systems <#Supported-Operating-Systems>`__
   -  `Documentation Updates <#Documentation-Updates>`__
   -  `AMD ROCm General Documentation Links <#AMD-ROCm-General-Documentation-Links>`__

-  `What's New in This Release <#Whats-New-in-This-Release>`__

   -  `Upgrading to This Release <#Upgrading-to-This-Release>`__
   -  `AMD ROCm Collective Groups<#AMD-ROCm-Collective-Groups>`__
   -  `AMD ROCm Data Center Tool <#AMD-ROCm-Data-Center-Tool>`__
   -  `AMD ROCm System Management Interface <#AMD-ROCm-System-Management-Interface>`__
   -  `AMD ROCm GNU Debugger <#AMD-ROCm-GNU-Debugger>`__
   -  `AMD ROCm Debugger API Library <#AMD-ROCm-Debugger-API-Library>`_
   -  `ROCm Communications Collective Library<#ROCm-Communications-Collective-Library>`__
   -  `AMD MIVisionX <#AMD-MIVisionX>`__
   
-  `Known Issues <#Known-Issues>`__

-  `Deploying ROCm <#Deploying-ROCm>`__

-  `Hardware and Software Support <#Hardware-and-Software-Support>`__

-  `Machine Learning and High Performance Computing Software Stack for
   AMD
   GPU <#Machine-Learning-and-High-Performance-Computing-Software-Stack-for-AMD-GPU>`__

   -  `ROCm Binary Package Structure <#ROCm-Binary-Package-Structure>`__
   -  `ROCm Platform Packages <#ROCm-Platform-Packages>`__

Supported Operating Systems and Documentation Updates
=====================================================

Supported Operating Systems
---------------------------

Support for RHEL v8.2
~~~~~~~~~~~~~~~~~~~~~

In this release, AMD ROCm extends support to RHEL v8.2.

Support for CentoS v7.8
~~~~~~~~~~~~~~~~~~~~~~~

In this release, AMD ROCm extends support to CentOS v7.8.

Support for CentOS v8.1
~~~~~~~~~~~~~~~~~~~~~~~

In this release, AMD ROCm extends support to CentOS v8.1.

List of Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm platform is designed to support the following operating
systems:

-  Ubuntu 16.04.6 (Kernel 4.15) and 18.04.4 (Kernel 5.3)
-  CentOS 7.7 (Kernel 3.10-1062) and RHEL 7.8(Kernel 3.10.0-1127)(Using
   devtoolset-7 runtime support)
-  SLES 15 SP1
-  CentOS and RHEL 8.1(Kernel 4.18.0-147)

Documentation Updates
---------------------

AMD ROCm Data Center Tool Guides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  AMD ROCm Data Center Toolâ„¢ User Guide
-  AMD ROCm Data Center Tool API Guide

HIP-Clang Compiler
~~~~~~~~~~~~~~~~~~

-  `HIP Installation
   Instructions <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`__

AMD ROCDebugger (ROCgdb)
~~~~~~~~~~~~~~~~~~~~~~~~

-  `ROCgdb User
   Guide <https://github.com/RadeonOpenCompute/ROCm/blob/master/gdb.pdf>`__
-  `ROCgdbapi
   Guide <https://github.com/RadeonOpenCompute/ROCm/blob/master/amd-dbgapi.pdf>`__

AMD ROCm Systems Management Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `System Management Interface Event API
   Guide <https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_Manual.pdf>`__

AMD ROCm Glossary of Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Updated Glossary of Terms and
   Definitions <https://rocmdocs.amd.com/en/latest/ROCm_Glossary/ROCm-Glossary.html>`__


AMD ROCm General Documentation Links
------------------------------------

Access the following links for more information on:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supported platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#machine-learning-and-high-performance-computing-software-stack-for-amd-gpu-v3-3-0

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#amd-rocm-version-history


What's New in This Release
==========================


Upgrading to This Release
-------------------------

A fresh and clean installation of AMD ROCm v3.6 is recommended. An upgrade from previous releases to AMD ROCm v3.6 is not supported.

For more information, refer to the `Installation
Guide <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`__


AMD ROCm Collective Groups
--------------------------

AMD ROCm introduces the Collective Groups feature for defining and synchronizing groups of threads and sharing data to perform efficient
collective computations. The sharing of data varies from algorithm to algorithm, so the thread synchronization must be flexible to ensure
modularity.

The Cooperative Groups feature in AMD ROCm adds the following two important mechanisms:

-  Kernel-wide barriers to synchronize between the workgroups in a
   kernel running on a single GPU.
-  System-wide barriers to synchronize between multiple kernels running
   on multiple GPUs.

AMD ROCm Data Center Tool
-------------------------

The ROCm Data Center Tool simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter
environments. The important features of this tool are:

* GPU telemetry
* GPU statistics for jobs â€¢ Integration with third-party tools â€¢ Open
  source

The Radeon Data Center Tool can be used in the standalone mode if all components are installed. The same set of features is also available in
a library format that can be used by existing management tools.

.. figure:: RDCComponentsGit.png
   :alt: ScreenShot

Refer to the Radeon Data Center Tool User Guide for more details on the different modes of operation.

**NOTE**: The Radeon Data Center User Guide is intended to provide an overview of ROCm Data Center Tool features and how system administrators
and Data Center (or HPC) users can administer and configure AMD GPUs. The guide also provides an overview of its components and open source
developer handbook. For more information, refer the Radeon Data Center User Guide at

*Add doc link*

AMD ROCm Data Center Tool API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RDC Tool API supports the following components:

-  Discovery, Grouping, fieldgrouping, dmon, Job Statistics

-  The rdcd daemon runs as the gPRC server. You can call RDC API to
   handle the client requests.

-  The rdci command-line tool can run in standalone mode and embedded
   mode. In the standalone mode, rdci connects to daemon via the RDC
   API. In the embedded mode, the rdci link to the RDC library directly
   and no daemon is required.

-  Third-party tools such as collectd integration

For more information, refer the Radeon Data Center API Guide at

*Add doc link*

AMD ROCm System Management Interface
------------------------------------

Hardware Topology
~~~~~~~~~~~~~~~~~

The Hardware Topology feature is enhanced to include functions to the ROCm-SMI library to query the hardware topology for a system. The new
functions enable you to retrieve the following details about the connection types, weights, and distances between GPUs for applications:

-  NUMA CPU node number for a device.
-  Weight for a connection between 2 GPUs.
-  Hops and connection types between 2 GPUs.

**NOTE**: Applications can also query the numa node number for a GPU.

**Parameters**

-  in - dv_ind - a device index

-  in, out - numa_node - A pointer to an uint32_t to which the numa node
   number should be written.

**RETURNS and VALUES**

-  RSMI_STATUS_SUCCESS - The call is successful

-  RSMI_STATUS_INVALID_ARGS - The provided arguments are not valid

For more details, refer the ROCm SMI API Guide at

*Add doc link to API Guide*

Get Process ID API
~~~~~~~~~~~~~~~~~~

The existing get process information API is now enhanced to include information about the VRAM usage.

A new VRAM usage file is created for each GPU as follows:

*/sys/class/kfd/kfd/proc//vram\_*

For example, for a system with multiple GPUs, you can have a VRAM usage file for each GPU as:

*vram\_, vram\_, vram\_*

Note, the VRAM usage file stores the VRAM memory currently in use (in bytes) by the process with PID on the GPU having GPUID .

For more information about the original and the enhanced APIs, refer the AMD ROCm SMI API Guide at

*Add Doc Link*

AMD ROCm GNU Debugger
----------------------

AMD ROCm v3.6.0 ROC Debugger (ROCgdb) is a multi-architecture debugger.that has a full standard x86_64 and HIP source language standard gdb support for amdgcn.
The following enhancements are available in the AMD ROCm v3.6 release.

Fixed AMD GPU Thread List
~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD GPU thread list is correctly refreshed after Ctrl-C or a host
breakpoint. This ensures the AMD GPU threads are displayed correctly,
and the all stop mode will stop all AMD GPU threads.

Support for Function Call Debug Information for Call Back Traces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler and debugger now support function call debug information
for call back traces. This allows stepping over, into, and out of
functions to work correctly.

Support for Address Watch
~~~~~~~~~~~~~~~~~~~~~~~~~

Support is now extended to the Address Watch feature.

Enhanced AMD GPU Virtual Registers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD GPU virtual registers are available for flat_scratch and xnack_mask.

Libraries Enhancement
~~~~~~~~~~~~~~~~~~~~~

Loaded AMD GPU shared libraries are displayed using file URI syntax.

The AMD ROCm Debugger User Guide is available as a PDF at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/gdb.pdf

For more information about GNU Debugger (GDB), refer to the GNU Debugger
(GDB) web site at: http://www.gnu.org/software/gdb

AMD ROCm Debugger API Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm Debugger API Library (ROCdbgapi) implements an AMD GPU debugger application programming interface (API) that provides the
support necessary for a client of the library to control the execution and inspect the state of AMD GPU devices.

The following AMD GPU architectures are supported: 

* Vega 10 

* Vega 7nm

The AMD ROCm Debugger API Library is installed by the rocm-dbgapi ackage. The rocm-gdb package is part of the rocm-dev meta-package,
which is in the rocm-dkms package. The AMD ROCm Debugger API Specification is available as a PDF at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/amd-dbgapi.pdf


ROCm Communications Collective Library
---------------------------------------

rocBLAS and hipBLAS Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following rocBLAS and hipBLAS enhancements are made in the AMD ROCm v3.6 release:

rocBLAS
^^^^^^^

-  L1 dot function optimized to utilize shuffle instructions
   (improvements on bf16, f16, f32 data types)

-  L1 dot function added x dot x optimized kernel

-  Standardization of L1 rocblas-bench to use device pointer mode to
   focus on GPU memory bandwidth

-  Adjustments for hipcc (hip-clang) compiler as standard build compiler
   and Centos8 support

-  Added Fortran support for all rocBLAS functions

hipBLAS
^^^^^^^

-  Fortran support for BLAS 1, BLAS 2, BLAS 3

-  hemm, hemm_batched, and hemm_strided_batched

-  symm, symm_batched, and symm_strided_batched

-  complex versions of geam, along with geam_batched and
   geam_strided_batched

-  gemm_batched_ex and gemm_strided_batched_ex

-  tbsv, tbsv_batched, and tbsv_strided_batched

AMD MIVisionX
-------------

AMD Radeon Augmentation Libraryâ„¢
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deep learning applications require loading and pre-processing of data efficiently to achieve high processing throughput. This requires
creating efficient processing pipelines fully utilizing the underlying hardware capabilities. Some examples are load and decode data, perform a
variety of augmentations, color-format conversions, and others. Deep learning frameworks require supporting multiple data formats and
augmentations to adapt to a variety of data-sets and models.

AMD Radeon Augmentation Library (RALI) is now designed to efficiently perform such processing pipelines from both images and video as well as
from a variety of storage formats. These pipelines are programmable by the user using both C++ and Python APIs. Some of the key features of
RALI are:

-  Process pipeline support for data_loading, meta-data loading,
   augmentations, and data-format conversions for training and inference

-  Process on CPU or Radeon GPU (with OpenCL or HIP backend)

-  Ease of integration with framework plugins in Python

-  Support a variety of augmentation operations through AMDâ€™s Radeon
   Performance Primitives (RPP).

-  Available in public and open-source platforms

For more information and installation instructions, see
https://github.com/rrawther/MIVisionX/tree/master/rali/docs

Known Issues
============

The following are the known issues in the v3.6.0 release.

Use of ROCgdb on Cooperative Queues Results in System Failure on Vega 10 and 7nm
-----------------------------------------------------------------------------------

In this release, using ROC Debugger (ROCgdb) on Cooperative queues can lead to a system failure on Vega 10 and 7nm. Cooperative queues are HSA queues created with the type HSA_QUEUE_TYPE_COOPERATIVE. The HIP runtime creates such queues when using: 

* Cooperative Groups features that launch a kernel to the device: 

	* hipLaunchCooperativeKernel()
      
 	* hipLaunchCooperativeKernelMultiDevice()
      
* Peer-to-peer transfers on systems without PCIe large BAR support

If a system crash occurs, examine the messages in ‘dmesg’ before rebooting the system. 

There is no known workaround at this time.
.

NaN Loss during ImageNet Training on Tensorflow
-----------------------------------------------

[Need content from Ryan/Subhani/Gowtham and workaround if any]

ROC Debugger Freezes with hipMemcpyWithStream
---------------------------------------------

[Need content from Tony and workaround if any]

Debug Agent Encounters an Error and Fails When Using Thunk API
--------------------------------------------------------------

[Need content from Qingchuan and workaround if any]

ROCgdb Fails to Recognize Code Objects Loaded by the Deprecated Runtime Loader API
----------------------------------------------------------------------------------

ROCgdb does not recognize code objects loaded using the deprecated ROCm runtime code object loader API. The deprecated loader API specifies the
code object using an argument of type hsa_code_object_t. The ROCgdb info sharedlibrary command does not list these code objects, thus, preventing
ROCgdb from displaying source information or setting breakpoints by source position in these code objects.

There is no workaround available at this time.

Calling thrust::sort() and thrust::sort_by_key() Not Supported from Device Code
-------------------------------------------------------------------------------

ROCm support for device malloc has been disabled. As a result, the rocThrust functionality which is dependent on device malloc does not
work. The use of the device malloc launched thrust::sort and thrust::sort_by_key is, therefore, not recommended.

**Note**: Host launched functionality is not impacted.

**Workaround**: A partial enablement of device malloc is possible by setting **HIP_ENABLE_DEVICE_MALLOC** to 1. Thrust::sort and
thrust::sort_by_key may work on certain input sizes.


Deploying ROCm
=================

AMD hosts both Debian and RPM repositories for the ROCm v3.5.x packages.

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

