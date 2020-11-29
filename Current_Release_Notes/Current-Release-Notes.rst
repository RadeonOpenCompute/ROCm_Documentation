.. image:: Currrent_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v3.10
================================
November, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v3.10.0 release.

`Download AMD ROCm Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__


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

-  SLES 15 SP2


Fresh Installation of AMD ROCm v3.10 Recommended
-----------------------------------------------

A fresh and clean installation of AMD ROCm v3.10 is recommended. An upgrade from previous releases to AMD ROCm v3.10 is not supported.

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

With the AMD ROCm v3.10 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms are now deprecated for multi-version ROCm installs. For example, rocm-dkms3.8.0, rocm-dkms3.9.0.

-   Multi-version installation of ROCm should be performed by installing rocm-dev using each of the desired ROCm versions. For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.

-  Version files must be created for each multi-version rocm <= 3.10.0

   -  command: echo \| sudo tee /opt/rocm-/.info/version

   -  example: echo 3.10.0 \| sudo tee /opt/rocm-3.10.0/.info/version

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

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


ROCm SMI API Documentation Updates
===================================

-  System DMA (SDMA) Utilization API

-  ROCm-SMI Command Line Interface

-  Enhanced ROCm SMI Library for Events


ROCm Data Center Tool User Guide
==================================

The ROCm Data Center Tool User Guide includes the following
enhancements:

-  ROCm Data Center Tool Python Binding

-  Prometheus plugin integration

For more information, refer to the ROCm Data Center Tool User Guide at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf



HIP Documentation Updates
===========================

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

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#software-stack-for-amd-gpu

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Current_Release_Notes/Current-Release-Notes.html#amd-rocm-version-history
   
   

What's New in This Release
-----------------------------

ROCm DATA CENTER TOOL
========================

The following enhancements are made to the ROCm Data Center Tool.

Prometheus Plugin for ROCm Data Center Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm Data Center (RDC) Tool now provides the Prometheus plugin, a
Python client to collect the telemetry data of the GPU. The RDC uses
Python binding for Prometheus and the collected plugin. The Python
binding maps the RDC C APIs to Python using ctypes. The functions
supported by C APIs can also be used in the Python binding.

For more information, refer to

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf


Python Binding
~~~~~~~~~~~~~~

The ROCm Data Center (RDC) Tool now uses PyThon Binding for Prometheus
and collectd plugins. PyThon binding maps the RDC C APIs to PyThon using
ctypes. All the functions supported by C APIs can also be used in PyThon
binding. A generic PyThon class RdcReader is created to simplify the
usage of the RDC:

-  Users can only specify the fields they want to monitor. RdcReader
   creates groups and fieldgroups, watches the fields, and fetches the
   fields.

-  The RdcReader can support both the Embedded and Standalone mode.
   Standalone mode can be used with and without authentication.

-  In the Standalone mode, the RdcReader can automatically reconnect to
   rdcd when connection is lost.When rdcd is restarted, the previously
   created group and fieldgroup may lose. The RdcReader can re-create
   them and watch the fields after a reconnect.

-  If the client is restarted, RdcReader can detect the groups and
   fieldgroups created previously, and, therefore, can avoid recreating
   them.

-  Users can pass the unit converter if they do not want to use the RDC
   default unit.

See the following sample program to monitor the power and GPU
utilization using the RdcReader:

::


   from RdcReader import RdcReader
   from RdcUtil import RdcUtil
   from rdc_bootstrap import *
    
   default_field_ids = [
           rdc_field_t.RDC_FI_POWER_USAGE,
           rdc_field_t.RDC_FI_GPU_UTIL
   ]
    
   class SimpleRdcReader(RdcReader):
       def __init__(self):
           RdcReader.__init__(self,ip_port=None, field_ids = default_field_ids, update_freq=1000000)
       def handle_field(self, gpu_index, value):
           field_name = self.rdc_util.field_id_string(value.field_id).lower()
           print("%d %d:%s %d" % (value.ts, gpu_index, field_name, value.value.l_int))
    
   if __name__ == '__main__':
       reader = SimpleRdcReader()
       while True:
           time.sleep(1)
           reader.process()
::


For more information about RDC Python binding and the Prometheus plugin
integration, refer to the ROCm Data Center Tool User Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide.pdf



ROCm SYSTEM MANAGEMENT INFORMATION
----------------------------------

System DMA (SDMA) Utilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-process, the SDMA usage is exposed via the ROCm SMI library. The
structure rsmi_process_info_t is extended to include sdma_usage.
sdma_usage is a 64-bit value that counts the duration (in microseconds)
for which the SDMA engine was active during that processâ€™s lifetime.

For example, see the rsmi_compute_process_info_by_pid_get() API below.

::


   /**
   * @brief This structure contains information specific to a process.
   */
     typedef struct {
         - - -,
         uint64_t sdma_usage; // SDMA usage in microseconds
     } rsmi_process_info_t;
     rsmi_status_t
         rsmi_compute_process_info_by_pid_get(uint32_t pid,
             rsmi_process_info_t *proc);
             
::


ROCm-SMI Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SDMA usage per-process is available using the following command,

::

   $ rocm-smi â€“showpids
   
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
   clocks are throttled.
   
ROCm SMI Command Line Interface Hardware Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This feature provides a matrix representation of the GPUs present in a
system by providing information of the manner in which the nodes are
connected. This is represented in terms of weights, hops, and link types
between two given GPUs. It also provides the numa node and the CPU
affinity associated with every GPU.

.. image:: https://github.com/Rmalavally/ROCm/blob/master/images/CLI1.PNG
   :align: center

 

.. image:: https://github.com/Rmalavally/ROCm/blob/master/images/CLI2.PNG
   :align: center

 



ROCm Math and Communication Libraries
-------------------------------------

New rocSOLVER APIs
~~~~~~~~~~~~~~~~~~

The following new rocSOLVER APIs are added in this release:

.. image:: https://github.com/Rmalavally/ROCm/blob/master/images/rocsolverAPI.PNG
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

