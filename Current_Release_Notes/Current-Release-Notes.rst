.. image:: /Current_Release_Notes/amdblack.jpg

|

=====================================
AMD ROCm Release Notes v3.5.1
=====================================
June, 2020

AMD ROCm released a maintenance patch release v3.5.1. For more information, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_Release_Notes_v3.5.1.pdf

Known Issue with AMD ROCm v3.5.1 Installation
===============================================

**Multiple Release Installations of AMD ROCm on a Server Fail**

The Side-by-Side install feature does not behave as expected, and multiple release installations of AMD ROCm on a server fail. Attempts to install AMD ROCm v3.5.0 and AMD ROCm v3.5.1 side-by-side can result in breaking the existing ROCm v3.5.0 installation.

This known issue is currently under investigation.




AMD ROCm™ Release Notes v3.5.0
================================
July, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v3.6.0 release.

`Download AMD ROCm v3.6.0 Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__



-  `Supported Operating Systems and Documentation
   Updates <#Supported-Operating-Systems-and-Documentation-Updates>`__

   -  `Supported Operating Systems <#Supported-Operating-Systems>`__
   -  `Documentation Updates <#Documentation-Updates>`__
   -  `AMD ROCm General Documentation Links <#AMD-ROCm-General-Documentation-Links>`__
   
   

-  `What's New in This Release <#Whats-New-in-This-Release>`__

   -  `Upgrading to This Release <#Upgrading-to-This-Release>`__
   -  `AMD ROCm Cooperative Groups <#AMD-ROCm-Cooperative-Groups>`__
   -  `AMD ROCm Data Center Tool <#AMD-ROCm-Data-Center-Tool>`__
   -  `AMD ROCm System Management Interface <#AMD-ROCm-System-Management-Interface>`__
   -  `AMD ROCm GNU Debugger <#AMD-ROCm-GNU-Debugger>`__
   -  `AMD ROCm Debugger API Library <#AMD-ROCm-Debugger-API-Library>`_
   -  `ROCm Communications Collective Library <#ROCm-Communications-Collective-Library>`__
   -  `AMD MIVisionX <#AMD-MIVisionX>`__
   
-  `Known Issues <#Known-Issues>`__

-  `Deploying ROCm <#Deploying-ROCm>`__


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

-  `AMD ROCm Data Center Tool User Guide <https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_Data_Center_Tool_User_Guide.pdf>`_
-  `AMD ROCm Data Center Tool API Guide - Alpha Release <https://github.com/Rmalavally/ROCm/blob/master/RDC_API_Guide_Alpha.pdf>`_ 

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
   Guide <https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide.pdf>`__

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



AMD ROCm Cooperative Groups
----------------------------

AMD ROCm 3.6 introduces the Cooperative Groups feature into HIP. Cooperative Groups introduces additional mechanisms for synchronizing groups of threads within and between GPUs, which can be used to increase the performance and ease programmability for a number of different programming paradigms.

The Cooperative Groups feature in AMD ROCm adds the following two important mechanisms into the HIP programming environment:

* Inter-thread-block barriers to synchronize between the thread blocks in a kernel running on a single GPU.

* System-wide barriers to synchronize between multiple kernels running on multiple GPUs.

   
You may use the Cooperative Groups feature in HIP applications by:

- Including the header file "hip/hip_cooperative_groups.h"

- Querying the target GPU(s) to ensure that they support the Cooperative Groups functionality

- Writing a GPU kernel that uses the new features of the cooperative_groups name-space

- Launching the GPU kernel using the new Cooperative Groups host-side APIs

Limitations

- This feature is currently only supported on AMD "gfx9" devices.

- This features does not currently support the classes: cooperative_groups::thread_group, cooperative_groups::thread_block, or cooperative_groups::coalesced_group, or cooperative_groups::thread_block_tile<>.



AMD ROCm Data Center Tool
-------------------------

The ROCm Data Center Tool simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter
environments. The important features of this tool are:

* GPU telemetry
* GPU statistics for jobs â€¢ Integration with third-party tools â€¢ Open
  source

The ROCm Data Center Tool can be used in the standalone mode if all components are installed. The same set of features is also available in
a library format that can be used by existing management tools.

.. figure:: RDCComponentsrevised.png
   :alt: ScreenShot

Refer to the ROCm Data Center Tool User Guide for more details on the different modes of operation.

**NOTE**: The ROCm Data Center User Guide is intended to provide an overview of ROCm Data Center Tool features and how system administrators
and Data Center (or HPC) users can administer and configure AMD GPUs. The guide also provides an overview of its components and open source
developer handbook. For more information, refer the ROCm Data Center User Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_Data_Center_Tool_User_Guide.pdf

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

For more information, refer the ROCm Data Center Tool API Guide (Alpha Release) at

https://github.com/RadeonOpenCompute/ROCm/blob/master/RDC_API_Guide_Alpha.pdf



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

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide.pdf

Get Process ID API
~~~~~~~~~~~~~~~~~~

The existing get process information API is now enhanced to include information about the VRAM usage.

A new VRAM usage file is created for each GPU as follows:

*/sys/class/kfd/kfd/proc//vram\_*

For example, for a system with multiple GPUs, you can have a VRAM usage file for each GPU as:

*vram\_, vram\_, vram\_*

Note, the VRAM usage file stores the VRAM memory currently in use (in bytes) by the process with PID on the GPU having GPUID .

For more information about the original and the enhanced APIs, refer the AMD ROCm SMI API Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/ROCm_SMI_API_Guide.pdf



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

AMD Radeon Augmentation Library
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

Hipify-Clang Installation Fails on CentOS/RHEL
------------------------------------------------

Installation of Hipify-Clang fails on CentOS/RHEL. Note, this issue is not observed on Ubuntu and SLES. 

**Workaround**: You can download and install the hipify-clang package manually using “—force”.

::

$ sudo rpm -ivh --force hipify-clang-11.0.0.x86_64.rpm




Use of ROCgdb on Cooperative Queues Results in System Failure on Vega 10 and 7nm
-----------------------------------------------------------------------------------

In this release, using ROC Debugger (ROCgdb) on Cooperative queues can lead to a system failure on Vega 10 and 7nm. Cooperative queues are HSA queues created with the type HSA_QUEUE_TYPE_COOPERATIVE. The HIP runtime creates such queues when using: 

* Cooperative Groups features that launch a kernel to the device: 

 - hipLaunchCooperativeKernel()
      
 - hipLaunchCooperativeKernelMultiDevice()
      
* Peer-to-peer transfers on systems without PCIe large BAR support

If a system crash occurs, examine the messages in ‘dmesg’ before rebooting the system. 

There is no known workaround at this time.



NaN Loss during ImageNet Training on Tensorflow
-----------------------------------------------

A Not a Number (NaN) loss error is observed while running Imagenet training on Tensorflow.

This issue is under investigation and there is no known workaround at this time.



Debug Agent Encounters an Error and Fails When Using Thunk API
--------------------------------------------------------------

The Debug Agent encounters the following error and, as a result, fails when using the Thunk API.

*“Error: Debug Agent: Cannot get queue info from KMT”

This known issue is under investigation and the Debug Agent functionality is unavailable in the AMD ROCm v3.6 release.



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



Memory Fault Error for fp16 operations in MIGraphX 
-----------------------------------------------------

fp16 operations in MIGraphX result in a memory fault error in ROCm v3.6.

There is no known workaround for the issue at this time.



Random Hang while Running Tensorflow Benchmarks
-----------------------------------------------------

A software hang has been observed with Tensorflow benchmarks at random times. The hang occurs more frequently on systems with multiple GPUs than a single GPU. 

You can kill the process or use  CNTRL + C to recover the system. 

There is no known workaround at this time.




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






Features and enhancements introduced in previous versions of ROCm can be found in :ref:`Current-Release-Notes`.


AMD ROCm Version History
=========================

This file contains archived version history information for the ROCm project.

New features and enhancements in ROCm v3.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release Notes: https://github.com/RadeonOpenCompute/ROCm/tree/roc-3.5.0

**rocProf Command Line Tool Python Requirement**
SQLite3 is a required Python module for the rocprof command-line tool. You can install the SQLite3 Python module using the pip utility and set env var ROCP_PYTHON_VERSION to the Python version, which includes the SQLite3 module.

**Heterogeneous-Compute Interface for Portability**
In this release, the Heterogeneous Compute Compiler (HCC) compiler is deprecated and the HIP-Clang compiler is introduced for compiling Heterogeneous-Compute Interface for Portability (HIP) programs.

**Radeon Open Compute Common Language Runtime**
In this release, the HIP runtime API is implemented on top of Radeon Open Compute Common Language Runtime (ROCclr). ROCclr is an abstraction layer that provides the ability to interact with different runtime backends such as ROCr.

**OpenCL Runtime**
The following OpenCL runtime changes are made in this release:

-AMD ROCm OpenCL Runtime extends support to OpenCL2.2
-The developer branch is changed from master to master-next

**AMD ROCm GNU Debugger (ROCgdb)**
The AMD ROCm Debugger (ROCgdb) is the AMD ROCm source-level debugger for Linux based on the GNU Debugger (GDB). It enables heterogeneous debugging on the AMD ROCm platform of an x86-based host architecture along with AMD GPU architectures and supported by the AMD Debugger API Library (ROCdbgapi).

**AMD ROCm Debugger API Library**
The AMD ROCm Debugger API Library (ROCdbgapi) implements an AMD GPU debugger application programming interface (API) that provides the support necessary for a client of the library to control the execution and inspect the state of AMD GPU devices.

**rocProfiler Dispatch Callbacks Start Stop API**
In this release, a new rocprofiler start/stop API is added to enable/disable GPU kernel HSA dispatch callbacks. The callback can be registered with the 'rocprofiler_set_hsa_callbacks' API. The API helps you eliminate some profiling performance impact by invoking the profiler only for kernel dispatches of interest. This optimization will result in significant performance gains.

**ROCm Communications Collective Library**
The ROCm Communications Collective Library (RCCL) consists of the following enhancements:

-Re-enable target 0x803
-Build time improvements for the HIP-Clang compiler

**NVIDIA Communications Collective Library Version Compatibility**
AMD RCCL is now compatible with NVIDIA Communications Collective Library (NCCL) v2.6.4 and provides the following features:

Network interface improvements with API v3
Network topology detection
Improved CPU type detection
Infiniband adaptive routing support

**MIOpen Optional Kernel Package Installation**
MIOpen provides an optional pre-compiled kernel package to reduce startup latency.

**New SMI Event Interface and Library**
An SMI event interface is added to the kernel and ROCm SMI lib for system administrators to get notified when specific events occur. On the kernel side, AMDKFD_IOC_SMI_EVENTS input/output control is enhanced to allow notifications propagation to user mode through the event channel.

**API for CPU Affinity**
A new API is introduced for aiding applications to select the appropriate memory node for a given accelerator(GPU).

**Radeon Performance Primitives Library**
The new Radeon Performance Primitives (RPP) library is a comprehensive high-performance computer vision library for AMD (CPU and GPU) with the HIP and OpenCL backend. The target operating system is Linux.


New features and enhancements in ROCm v3.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release Notes: https://github.com/RadeonOpenCompute/ROCm/tree/roc-3.3.0

**Multi-Version Installation**
Users can install and access multiple versions of the ROCm toolkit simultaneously. Previously, users could install only a single version of the ROCm toolkit.

**GPU Process Information**
A new functionality to display process information for GPUs is available in this release. For example, you can view the process details to determine if the GPU(s) must be reset.

**Support for 3D Pooling Layers**
AMD ROCm is enhanced to include support for 3D pooling layers. The implementation of 3D pooling layers now allows users to run 3D convolutional networks, such as ResNext3D, on AMD Radeon Instinct GPUs.

**ONNX Enhancements**
Open Neural Network eXchange (ONNX) is a widely-used neural net exchange format. The AMD model compiler & optimizer support the pre-trained models in ONNX, NNEF, & Caffe formats. Currently, ONNX versions 1.3 and below are supported.


New features and enhancements in ROCm v3.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This release was not productized.


New features and enhancements in ROCm v3.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'Release Notes: https://github.com/RadeonOpenCompute/ROCm/tree/roc-3.1.0

**Change in ROCm Installation Directory Structure**

A fresh installation of the ROCm toolkit installs the packages in the /opt/rocm-<version> folder. 
Previously, ROCm toolkit packages were installed in the /opt/rocm folder.

**Reliability, Accessibility, and Serviceability Support for Vega 7nm**

The Reliability, Accessibility, and Serviceability (RAS) support for Vega7nm is now available. 

**SLURM Support for AMD GPU**

SLURM (Simple Linux Utility for Resource Management) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. 


New features and enhancements in ROCm v3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release Notes: https://github.com/RadeonOpenCompute/ROCm/tree/roc-3.0.0

* Support for CentOS RHEL v7.7
* Support is extended for CentOS/RHEL v7.7 in the ROCm v3.0 release. For more information about the CentOS/RHEL v7.7 release, see:

* CentOS/RHEL

* Initial distribution of AOMP 0.7-5 in ROCm v3.0
The code base for this release of AOMP is the Clang/LLVM 9.0 sources as of October 8th, 2019. The LLVM-project branch used to build this release is AOMP-191008. It is now locked. With this release, an artifact tarball of the entire source tree is created. This tree includes a Makefile in the root directory used to build AOMP from the release tarball. You can use Spack to build AOMP from this source tarball or build manually without Spack.

* Fast Fourier Transform Updates
The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform. Fast Fourier transforms are used in signal processing, image processing, and many other areas. The following real FFT performance change is made in the ROCm v3.0 release:

* Implement efficient real/complex 2D transforms for even lengths.

Other improvements:

• More 2D test coverage sizes.

• Fix buffer allocation error for large 1D transforms.

• C++ compatibility improvements.

MemCopy Enhancement for rocProf
In the v3.0 release, the rocProf tool is enhanced with an additional capability to dump asynchronous GPU memcopy information into a .csv file. You can use the '-hsa-trace' option to create the results_mcopy.csv file. Future enhancements will include column labels.

New features and enhancements in ROCm v2.10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rocBLAS Support for Complex GEMM

The rocBLAS library is a gpu-accelerated implementation of the standard Basic Linear Algebra Subroutines (BLAS). rocBLAS is designed to enable you to develop algorithms, including high performance computing, image analysis, and machine learning.

In the AMD ROCm release v2.10, support is extended to the General Matrix Multiply (GEMM) routine for multiple small matrices processed simultaneously for rocBLAS in AMD Radeon Instinct MI50. Both single and double precision, CGEMM and ZGEMM, are now supported in rocBLAS.

Support for SLES 15 SP1

In the AMD ROCm v2.10 release, support is added for SUSE Linux® Enterprise Server (SLES) 15 SP1. SLES is a modular operating system for both multimodal and traditional IT.

Code Marker Support for rocProfiler and rocTracer Libraries

Code markers provide the external correlation ID for the calling thread. This function indicates that the calling thread is entering and leaving an external API region.

New features and enhancements in ROCm 2.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Initial release for Radeon Augmentation Library(RALI)

The AMD Radeon Augmentation Library (RALI) is designed to efficiently decode and process images from a variety of storage formats and modify them through a processing graph programmable by the user. RALI currently provides C API.

Quantization in MIGraphX v0.4

MIGraphX 0.4 introduces support for fp16 and int8 quantization. For additional details, as well as other new MIGraphX features, see MIGraphX documentation.

rocSparse csrgemm

csrgemm enables the user to perform matrix-matrix multiplication with two sparse matrices in CSR format.

Singularity Support

ROCm 2.9 adds support for Singularity container version 2.5.2.

Initial release of rocTX

ROCm 2.9 introduces rocTX, which provides a C API for code markup for performance profiling. This initial release of rocTX supports annotation of code ranges and ASCII markers. 

* Added support for Ubuntu 18.04.3
* Ubuntu 18.04.3 is now supported in ROCm 2.9.

New features and enhancements in ROCm 2.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for NCCL2.4.8 API

Implements ncclCommAbort() and ncclCommGetAsyncError() to match the NCCL 2.4.x API

New features and enhancements in ROCm 2.7.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This release is a hotfix for ROCm release 2.7.

Issues fixed in ROCm 2.7.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A defect in upgrades from older ROCm releases has been fixed.
* rocprofiler --hiptrace and --hsatrace fails to load roctracer library
* In ROCm 2.7.2, rocprofiler --hiptrace and --hsatrace fails to load roctracer library defect has been fixed.
* To generate traces, please provide directory path also using the parameter: -d <$directoryPath> for example:

/opt/rocm/bin/rocprof  --hsa-trace -d $PWD/traces /opt/rocm/hip/samples/0_Intro/bit_extract/bit_extract
All traces and results will be saved under $PWD/traces path

Upgrading from ROCm 2.7 to 2.7.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To upgrade, please remove 2.7 completely as specified for ubuntu or for centos/rhel, and install 2.7.2 as per instructions install instructions

Other notes
To use rocprofiler features, the following steps need to be completed before using rocprofiler:

Step-1: Install roctracer
Ubuntu 16.04 or Ubuntu 18.04:
sudo apt install roctracer-dev
CentOS/RHEL 7.6:
sudo yum install roctracer-dev

Step-2: Add /opt/rocm/roctracer/lib to LD_LIBRARY_PATH
New features and enhancements in ROCm 2.7
[rocFFT] Real FFT Functional
Improved real/complex 1D even-length transforms of unit stride. Performance improvements of up to 4.5x are observed. Large problem sizes should see approximately 2x.

rocRand Enhancements and Optimizations

Added support for new datatypes: uchar, ushort, half.

Improved performance on "Vega 7nm" chips, such as on the Radeon Instinct MI50

mtgp32 uniform double performance changes due generation algorithm standardization. Better quality random numbers now generated with 30% decrease in performance

Up to 5% performance improvements for other algorithms

RAS

Added support for RAS on Radeon Instinct MI50, including:

* Memory error detection
* Memory error detection counter
* ROCm-SMI enhancements
* Added ROCm-SMI CLI and LIB support for FW version, compute running processes, utilization rates, utilization counter, link error counter, and unique ID.

New features and enhancements in ROCm 2.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCmInfo enhancements

ROCmInfo was extended to do the following: For ROCr API call errors including initialization determine if the error could be explained by:

ROCk (driver) is not loaded / available
User does not have membership in appropriate group - "video"
If not above print the error string that is mapped to the returned error code
If no error string is available, print the error code in hex
Thrust - Functional Support on Vega20

ROCm2.6 contains the first official release of rocThrust and hipCUB. rocThrust is a port of thrust, a parallel algorithm library. hipCUB is a port of CUB, a reusable software component library. Thrust/CUB has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: rocThrust and hipCUB library replaces https://github.com/ROCmSoftwarePlatform/thrust (hip-thrust), i.e. hip-thrust has been separated into two libraries, rocThrust and hipCUB. Existing hip-thrust users are encouraged to port their code to rocThrust and/or hipCUB. Hip-thrust will be removed from official distribution later this year.

MIGraphX v0.3

MIGraphX optimizer adds support to read models frozen from Tensorflow framework. Further details and an example usage at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.3

MIOpen 2.0

This release contains several new features including an immediate mode for selecting convolutions, bfloat16 support, new layers, modes, and algorithms.

MIOpenDriver, a tool for benchmarking and developing kernels is now shipped with MIOpen. BFloat16 now supported in HIP requires an updated rocBLAS as a GEMM backend.

Immediate mode API now provides the ability to quickly obtain a convolution kernel.

MIOpen now contains HIP source kernels and implements the ImplicitGEMM kernels. This is a new feature and is currently disabled by default. Use the environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation this feature. ImplicitGEMM requires an up to date HIP version of at least 1.5.9211.

A new "loss" catagory of layers has been added, of which, CTC loss is the first. See the API reference for more details. 2.0 is the last release of active support for gfx803 architectures. In future releases, MIOpen will not actively debug and develop new features specifically for gfx803.

System Find-Db in memory cache is disabled by default. Please see build instructions to enable this feature. Additional documentation can be found here: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/

Bloat16 software support in rocBLAS/Tensile

Added mixed precision bfloat16/IEEE f32 to gemm_ex. The input and output matrices are bfloat16. All arithmetic is in IEEE f32.

AMD Infinity Fabric™ Link enablement

The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in two hives or two Radeon Instinct MI60 or Radeon Instinct MI50 boards in four hives via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

ROCm-smi features and bug fixes

mGPU & Vendor check

Fix clock printout if DPM is disabled

Fix finding marketing info on CentOS

Clarify some error messages

ROCm-smi-lib enhancements

Documentation updates

Improvements to *name_get functions

RCCL2 Enablement

RCCL2 supports collectives intranode communication using PCIe, Infinity Fabric™, and pinned host memory, as well as internode communication using Ethernet (TCP/IP sockets) and Infiniband/RoCE (Infiniband Verbs). Note: For Infiniband/RoCE, RDMA is not currently supported.

rocFFT enhancements

Added: Debian package with FFT test, benchmark, and sample programs
Improved: hipFFT interfaces
Improved: rocFFT CPU reference code, plan generation code and logging code

New features and enhancements in ROCm 2.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UCX 1.6 support

Support for UCX version 1.6 has been added.

BFloat16 GEMM in rocBLAS/Tensile

Software support for BFloat16 on Radeon Instinct MI50, MI60 has been added. This includes:

Mixed precision GEMM with BFloat16 input and output matrices, and all arithmetic in IEEE32 bit

Input matrix values are converted from BFloat16 to IEEE32 bit, all arithmetic and accumulation is IEEE32 bit. Output values are rounded from IEEE32 bit to BFloat16

Accuracy should be correct to 0.5 ULP

ROCm-SMI enhancements

CLI support for querying the memory size, driver version, and firmware version has been added to ROCm-smi.

[PyTorch] multi-GPU functional support (CPU aggregation/Data Parallel)

Multi-GPU support is enabled in PyTorch using Dataparallel path for versions of PyTorch built using the 06c8aa7a3bbd91cda2fd6255ec82aad21fa1c0d5 commit or later.

rocSparse optimization on Radeon Instinct MI50 and MI60

This release includes performance optimizations for csrsv routines in the rocSparse library.

[Thrust] Preview

Preview release for early adopters. rocThrust is a port of thrust, a parallel algorithm library. Thrust has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: This library will replace https://github.com/ROCmSoftwarePlatform/thrust in a future release. The package for rocThrust (this library) currently conflicts with version 2.5 package of thrust. They should not be installed together.

Support overlapping kernel execution in same HIP stream

HIP API has been enhanced to allow independent kernels to run in parallel on the same stream.

AMD Infinity Fabric™ Link enablement

The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in one hive via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

New features and enhancements in ROCm 2.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow 2.0 support

ROCm 2.4 includes the enhanced compilation toolchain and a set of bug fixes to support TensorFlow 2.0 features natively

AMD Infinity Fabric™ Link enablement

ROCm 2.4 adds support to connect two Radeon Instinct MI60 or Radeon Instinct MI50 boards via AMD Infinity Fabric™ Link GPU interconnect technology.

New features and enhancements in ROCm 2.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mem usage per GPU

Per GPU memory usage is added to rocm-smi. Display information regarding used/total bytes for VRAM, visible VRAM and GTT, via the --showmeminfo flag

MIVisionX, v1.1 - ONNX

ONNX parser changes to adjust to new file formats

MIGraphX, v0.2

MIGraphX 0.2 supports the following new features:

New Python API

* Support for additional ONNX operators and fixes that now enable a large set of Imagenet models
* Support for RNN Operators
* Support for multi-stream Execution
* [Experimental] Support for Tensorflow frozen protobuf files

See: Getting-started:-using-the-new-features-of-MIGraphX-0.2 for more details

MIOpen, v1.8 - 3d convolutions and int8

This release contains full 3-D convolution support and int8 support for inference.
Additionally, there are major updates in the performance database for major models including those found in Torchvision.
See: MIOpen releases

Caffe2 - mGPU support

Multi-gpu support is enabled for Caffe2.

rocTracer library, ROCm tracing API for collecting runtimes API and asynchronous GPU activity traces
HIP/HCC domains support is introduced in rocTracer library.

BLAS - Int8 GEMM performance, Int8 functional and performance
Introduces support and performance optimizations for Int8 GEMM, implements TRSV support, and includes improvements and optimizations with Tensile.

Prioritized L1/L2/L3 BLAS (functional)
Functional implementation of BLAS L1/L2/L3 functions

BLAS - tensile optimization
Improvements and optimizations with tensile

MIOpen Int8 support
Support for int8

New features and enhancements in ROCm 2.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rocSparse Optimization on Vega20
Cache usage optimizations for csrsv (sparse triangular solve), coomv (SpMV in COO format) and ellmv (SpMV in ELL format) are available.

DGEMM and DTRSM Optimization
Improved DGEMM performance for reduced matrix sizes (k=384, k=256)

Caffe2
Added support for multi-GPU training

New features and enhancements in ROCm 2.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RocTracer v1.0 preview release – 'rocprof' HSA runtime tracing and statistics support -
Supports HSA API tracing and HSA asynchronous GPU activity including kernels execution and memory copy

Improvements to ROCM-SMI tool -
Added support to show real-time PCIe bandwidth usage via the -b/--showbw flag

DGEMM Optimizations -
Improved DGEMM performance for large square and reduced matrix sizes (k=384, k=256)

New features and enhancements in ROCm 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds support for RHEL 7.6 / CentOS 7.6 and Ubuntu 18.04.1

Adds support for Vega 7nm, Polaris 12 GPUs

Introduces MIVisionX
A comprehensive computer vision and machine intelligence libraries, utilities and applications bundled into a single toolkit.
Improvements to ROCm Libraries
rocSPARSE & hipSPARSE
rocBLAS with improved DGEMM efficiency on Vega 7nm

MIOpen
This release contains general bug fixes and an updated performance database
Group convolutions backwards weights performance has been improved

RNNs now support fp16
Tensorflow multi-gpu and Tensorflow FP16 support for Vega 7nm
TensorFlow v1.12 is enabled with fp16 support
PyTorch/Caffe2 with Vega 7nm Support

fp16 support is enabled

Several bug fixes and performance enhancements

Known Issue: breaking changes are introduced in ROCm 2.0 which are not addressed upstream yet. Meanwhile, please continue to use ROCm fork at https://github.com/ROCmSoftwarePlatform/pytorch

Improvements to ROCProfiler tool

Support for Vega 7nm

Support for hipStreamCreateWithPriority

Creates a stream with the specified priority. It creates a stream on which enqueued kernels have a different priority for execution compared to kernels enqueued on normal priority streams. The priority could be higher or lower than normal priority streams.

OpenCL 2.0 support

ROCm 2.0 introduces full support for kernels written in the OpenCL 2.0 C language on certain devices and systems.  Applications can detect this support by calling the “clGetDeviceInfo” query function with “parame_name” argument set to “CL_DEVICE_OPENCL_C_VERSION”.  

In order to make use of OpenCL 2.0 C language features, the application must include the option “-cl-std=CL2.0” in options passed to the runtime API calls responsible for compiling or building device programs.  The complete specification for the OpenCL 2.0 C language can be obtained using the following link: https://www.khronos.org/registry/OpenCL/specs/opencl-2.0-openclc.pdf

Improved Virtual Addressing (48 bit VA) management for Vega 10 and later GPUs

Fixes Clang AddressSanitizer and potentially other 3rd-party memory debugging tools with ROCm

Small performance improvement on workloads that do a lot of memory management

Removes virtual address space limitations on systems with more VRAM than system memory
Kubernetes support

New features and enhancements in ROCm 1.9.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RDMA(MPI) support on Vega 7nm

Support ROCnRDMA based on Mellanox InfiniBand

Improvements to HCC

Improved link time optimization

Improvements to ROCProfiler tool

General bug fixes and implemented versioning APIs

New features and enhancements in ROCm 1.9.2

RDMA(MPI) support on Vega 7nm

Support ROCnRDMA based on Mellanox InfiniBand

Improvements to HCC

Improved link time optimization

Improvements to ROCProfiler tool

General bug fixes and implemented versioning APIs

Critical bug fixes

New features and enhancements in ROCm 1.9.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added DPM support to Vega 7nm

Dynamic Power Management feature is enabled on Vega 7nm.

Fix for 'ROCm profiling' that used to fail with a “Version mismatch between HSA runtime and libhsa-runtime-tools64.so.1” error

New features and enhancements in ROCm 1.9.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preview for Vega 7nm
Enables developer preview support for Vega 7nm

System Management Interface
Adds support for the ROCm SMI (System Management Interface) library, which provides monitoring and management capabilities for AMD GPUs.

Improvements to HIP/HCC
Support for gfx906

Added deprecation warning for C++AMP. This will be the last version of HCC supporting C++AMP.

Improved optimization for global address space pointers passing into a GPU kernel

Fixed several race conditions in the HCC runtime

Performance tuning to the unpinned copy engine

Several codegen enhancement fixes in the compiler backend

Preview for rocprof Profiling Tool

Developer preview (alpha) of profiling tool rocProfiler. It includes a command-line front-end, rpl_run.sh, which enables:

Cmd-line tool for dumping public per kernel perf-counters/metrics and kernel timestamps

Input file with counters list and kernels selecting parameters

Multiple counters groups and app runs supported

Output results in CSV format

The tool can be installed from the rocprofiler-dev package. It will be installed into: /opt/rocm/bin/rpl_run.sh

Preview for rocr Debug Agent rocr_debug_agent

The ROCr Debug Agent is a library that can be loaded by ROCm Platform Runtime to provide the following functionality:

Print the state for wavefronts that report memory violation or upon executing a "s_trap 2" instruction.
Allows SIGINT (ctrl c) or SIGTERM (kill -15) to print wavefront state of aborted GPU dispatches.
It is enabled on Vega10 GPUs on ROCm1.9.
The ROCm1.9 release will install the ROCr Debug Agent library at /opt/rocm/lib/librocr_debug_agent64.so

New distribution support
Binary package support for Ubuntu 18.04
ROCm 1.9 is ABI compatible with KFD in upstream Linux kernels.
Upstream Linux kernels support the following GPUs in these releases: 4.17: Fiji, Polaris 10, Polaris 11 4.18: Fiji, Polaris 10, Polaris 11, Vega10

Some ROCm features are not available in the upstream KFD:

More system memory available to ROCm applications
Interoperability between graphics and compute
RDMA
IPC
To try ROCm with an upstream kernel, install ROCm as normal, but do not install the rock-dkms package. Also add a udev rule to control /dev/kfd permissions:

    echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
    
New features as of ROCm 1.8.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm 1.8.3 is a minor update meant to fix compatibility issues on Ubuntu releases running kernel 4.15.0-33

New features as of ROCm 1.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DKMS driver installation

Debian packages are provided for DKMS on Ubuntu

RPM packages are provided for CentOS/RHEL 7.4 and 7.5

See the ROCT-Thunk-Interface and ROCK-Kernel-Driver for additional documentation on driver setup

New distribution support

Binary package support for Ubuntu 16.04 and 18.04

Binary package support for CentOS 7.4 and 7.5

Binary package support for RHEL 7.4 and 7.5

Improved OpenMPI via UCX support

UCX support for OpenMPI

ROCm RDMA

New Features as of ROCm 1.7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DKMS driver installation

New driver installation uses Dynamic Kernel Module Support (DKMS)

Only amdkfd and amdgpu kernel modules are installed to support AMD hardware

Currently only Debian packages are provided for DKMS (no Fedora suport available)

See the ROCT-Thunk-Interface and ROCK-Kernel-Driver for additional documentation on driver setup

New Features as of ROCm 1.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developer preview of the new OpenCL 1.2 compatible language runtime and compiler

OpenCL 2.0 compatible kernel language support with OpenCL 1.2 compatible runtime

Supports offline ahead of time compilation today; during the Beta phase we will add in-process/in-memory compilation.

Binary Package support for Ubuntu 16.04

Binary Package support for Fedora 24 is not currently available

Dropping binary package support for Ubuntu 14.04, Fedora 23

IPC support
                 



DISCLAIMER 
===========
The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information.  Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.  No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document.  Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. S
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

