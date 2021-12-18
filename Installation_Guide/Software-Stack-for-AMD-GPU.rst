.. image:: amdblack.jpg


.. _Machine Learning and High Performance Computing Software Stack for AMD GPU:
|

============================
Software Stack for AMD GPU
============================

Machine Learning and High Performance Computing Software Stack for AMD GPU 
----------------------------------------------------------------------------


.. _ROCm Binary Package Structure:

ROCm Binary Package Structure
#################################

ROCm is a collection of software ranging from drivers and runtimes to libraries and developer tools. In AMD's package distributions, these software projects are provided as a separate packages. This allows users to install only the packages they need, if they do not wish to install all of ROCm. These packages will install most of the ROCm software into ``/opt/rocm/`` by default.

The packages for each of the major ROCm components are:

ROCm Core Components
=====================

-  ROCk Kernel Driver: ``rock-dkms rock-dkms-firmware``
-  ROCr Runtime: ``hsa-rocr-dev``
-  ROCt Thunk Interface: ``hsakmt-roct``, ``hsakmt-roct-dev``


ROCm Support Software
======================

-  ROCm SMI: ``rocm-smi``
-  ROCm cmake: ``rocm-cmake``
-  rocminfo: ``rocminfo``
-  ROCm Bandwidth Test: ``rocm_bandwidth_test``
     
    
ROCm Compilers
================

-  Clang compiler: ``llvm-amdgpu``
-  HIP: ``hip_base``, ``hip_doc``, ``hip_rocclr``, ``hip_samples``     
-  ROCM Clang-OCL Kernel Compiler: ``rocm-clang-ocl``
     

ROCm Device Libraries
===========================
     
-  ROCm Device Libraries: ``rocm-device-libs``     
-  ROCm OpenCL: ``rocm-opencl``, ``rocm-opencl-devel`` (on RHEL/CentOS), ``rocm-opencl-dev`` (on Ubuntu)
     
     
ROCm Development ToolChain
===========================
     
-  Asynchronous Task and Memory Interface (ATMI): ``atmi``     
-  ROCm Debug Agent: ``rocm_debug_agent``     
-  ROCm Code Object Manager: ``comgr``     
-  ROC Profiler: ``rocprofiler-dev``     
-  ROC Tracer: ``roctracer-dev``     
      

ROCm Libraries
==============
 
-  rocALUTION: ``rocalution``
-  rocBLAS: ``rocblas``
-  hipBLAS: ``hipblas``
-  hipCUB: ``hipCUB``
-  rocFFT: ``rocfft``
-  rocRAND: ``rocrand``
-  rocSPARSE: ``rocsparse``
-  hipSPARSE: ``hipsparse``
-  ROCm SMI Lib: ``rocm-smi-lib64``
-  rocThrust: ``rocThrust``
-  MIOpen: ``MIOpen-HIP`` (for the HIP version), ``MIOpen-OpenCL`` (for the OpenCL version)
-  MIOpenGEMM: ``miopengemm``
-  MIVisionX: ``mivisionx``
-  RCCL: ``rccl``


To make it easier to install ROCm, the AMD binary repositories provide a number of meta-packages that will automatically install multiple other packages. For example, ``rocm-dkms`` is the primary meta-package that is used to install most of the base technology needed for ROCm to operate. It will install the ``rock-dkms`` kernel driver, and another meta-package (``rocm-dev``) which installs most of the user-land ROCm core components, support software, and development tools.
 

The *rocm-utils* meta-package will install useful utilities that, while not required for ROCm to operate, may still be beneficial to have. Finally, the *rocm-libs* meta-package will install some (but not all) of the libraries that are part of ROCm.

The chain of software installed by these meta-packages is illustrated below:


::

   └── rocm-dkms
    ├── rock-dkms
    └── rocm-dev
        ├── comgr
        ├── hip-base
        ├── hip-doc
        ├── hip-rocclr
        ├── hip-samples
        ├── hsa-amd-aqlprofile
        ├── hsakmt-roct
        ├── hsakmt-roct-dev
        ├── hsa-rocr-dev
        ├── llvm-amdgpu
        ├── rocm-cmake
        ├── rocm-dbgapi
        ├── rocm-debug-agent
        ├── rocm-device-libs
        ├── rocm-gdb
        ├── rocm-smi
        ├── rocm-smi-lib64
        ├── rocprofiler-dev
        └── roctracer-dev
	├── rocm-utils
            │   ├── rocm-clang-ocl
            │   └── rocminfo

  rocm-libs
    |--miopen
    |--hipblas
    |--hipcub
    |--hipsparse
    |--rocalution
    |--rocblas
    |--rocfft
    |--rocprim
    |--rocrand
    |--rocsolver
    |--rocsparse
    \--rocthrust




These meta-packages are not required but may be useful to make it easier to install ROCm on most systems.

Note: Some users may want to skip certain packages. For instance, a user that wants to use the upstream kernel drivers (rather than those supplied by AMD) may want to skip the rocm-dkms and rock-dkms packages. Instead, they could directly install rocm-dev.

Similarly, a user that only wants to install OpenCL support instead of HCC and HIP may want to skip the rocm-dkms and rocm-dev packages. Instead, they could directly install rock-dkms, rocm-opencl, and rocm-opencl-dev and their dependencies.

.. _ROCm Platform Packages:


ROCm Platform Packages
^^^^^^^^^^^^^^^^^^^^^^^

The following platform packages are for ROCm v4.1.0:

Drivers, ToolChains, Libraries, and Source Code
==================================================

The latest supported version of the drivers, tools, libraries and source code for the ROCm platform have been released and are available from the following GitHub repositories:

**ROCm Core Components**

-  `ROCk Kernel Driver`_
-  `ROCr Runtime`_
-  `ROCt Thunk Interface`_

**ROCm Support Software**

-  `ROCm SMI`_
-  `ROCm cmake`_
-  `rocminfo`_
-  `ROCm Bandwidth Test`_

**ROCm Compilers**

-  `HIP`_
-  `ROCM Clang-OCL Kernel Compiler`_
  
Example Applications:

-  `HIP Examples`_
  
**ROCm Device Libraries and Tools**
  
-  `ROCm Device Libraries`_
-  `ROCm OpenCL Runtime`_
-  `ROCm LLVM OCL`_
-  `ROCm Device Libraries OCL`_
-  `Asynchronous Task and Memory Interface`_
-  `ROCr Debug Agent`_
-  `ROCm Code Object Manager`_
-  `ROC Profiler`_
-  `ROC Tracer`_
-  `AOMP`_
-  `Radeon Compute Profiler`_
-  `ROCm Validation Suite`_



**ROCm Libraries**

-  `rocBLAS`_
-  `hipBLAS`_
-  `rocFFT`_
-  `rocRAND`_
-  `rocSPARSE`_
-  `hipSPARSE`_
-  `rocALUTION`_
-  `MIOpenGEMM`_
-  `mi open`_
-  `rocThrust`_
-  `ROCm SMI Lib`_
-  `RCCL`_
-  `MIVisionX`_
-  `hipCUB`_
-  `AMDMIGraphX`_


..  ROCm Core Components

.. _ROCk Kernel Driver: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/rocm-4.1.0

.. _ROCr Runtime: https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/rocm-4.1.0

.. _ROCt Thunk Interface: https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/rocm-4.1.0


.. ROCm Support Software

.. _ROCm SMI: https://github.com/RadeonOpenCompute/ROC-smi/tree/rocm-4.1.0

.. _ROCm cmake: https://github.com/RadeonOpenCompute/rocm-cmake/tree/rocm-4.1.0

.. _rocminfo: https://github.com/RadeonOpenCompute/rocminfo/tree/rocm-4.1.0

.. _ROCm Bandwidth Test: https://github.com/RadeonOpenCompute/rocm_bandwidth_test/tree/rocm-4.1.0


.. ROCm Compilers

.. _HIP: https://github.com/ROCm-Developer-Tools/HIP/tree/rocm-4.1.0

.. _HIP Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/rocm-4.1.0



.. ROCm Device Libraries and Tools

.. _ROCm Device Libraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-4.1.0

.. _ROCm OpenCL Runtime: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/tree/rocm-4.1.0

.. _ROCm LLVM OCL: https://github.com/RadeonOpenCompute/llvm-project/tree/rocm-ocl-4.1.0

.. _ROCm Device Libraries OCL: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-4.1.0

.. _ROCM Clang-OCL Kernel Compiler: https://github.com/RadeonOpenCompute/clang-ocl/tree/rocm-4.1.0

.. _Asynchronous Task and Memory Interface: https://github.com/RadeonOpenCompute/atmi/tree/rocm-4.1.0

.. _ROCr Debug Agent: https://github.com/ROCm-Developer-Tools/rocr_debug_agent/tree/rocm-4.1.0

.. _ROCm Code Object Manager: https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/tree/rocm-4.1.0

.. _ROC Profiler: https://github.com/ROCm-Developer-Tools/rocprofiler/tree/rocm-4.1.0

.. _ROC Tracer: https://github.com/ROCm-Developer-Tools/roctracer/tree/rocm-4.1.0

.. _AOMP: https://github.com/ROCm-Developer-Tools/aomp/tree/rocm-4.1.0

.. _Radeon Compute Profiler: https://github.com/GPUOpen-Tools/RCP/tree/3a49405

.. _ROCm Validation Suite: https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/tree/rocm-4.1.0


.. ROCm Libraries

.. _rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/rocm-4.1.0

.. _hipBLAS: https://github.com/ROCmSoftwarePlatform/hipBLAS/tree/rocm-4.1.0

.. _rocFFT: https://github.com/ROCmSoftwarePlatform/rocFFT/tree/rocm-4.1.0

.. _rocRAND: https://github.com/ROCmSoftwarePlatform/rocRAND/tree/rocm-4.1.0

.. _rocSPARSE: https://github.com/ROCmSoftwarePlatform/rocSPARSE/tree/rocm-4.1.0

.. _hipSPARSE: https://github.com/ROCmSoftwarePlatform/hipSPARSE/tree/rocm-4.1.0

.. _rocALUTION: https://github.com/ROCmSoftwarePlatform/rocALUTION/tree/rocm-4.1.0

.. _MIOpenGEMM: https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/tree/rocm-4.1.0

.. _mi open: https://github.com/ROCmSoftwarePlatform/MIOpen/tree/rocm-4.1.0

.. _rocThrust: https://github.com/ROCmSoftwarePlatform/rocThrust/tree/rocm-4.1.0

.. _ROCm SMI Lib: https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/rocm-4.1.0

.. _RCCL: https://github.com/ROCmSoftwarePlatform/rccl/tree/rocm-4.1.0

.. _hipCUB: https://github.com/ROCmSoftwarePlatform/hipCUB/tree/rocm-4.1.0

.. _MIVisionX: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/rocm-4.1.0

.. _AMDMIGraphX: https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/rocm-4.1.0




List of ROCm Packages for Supported Operating Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm-Library Meta Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------+---------------------------------------------------------+
|Package                            |  Debian 	            |   RPM						      |	
+===================================+=======================+=========================================================+
| rocFFT	                    |   Yes	            |  Yes				                      |	 
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocRAND	                    |   Yes	            |  Yes 			                              | 	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocBLAS 	                    |   Yes 	            |  Yes            		                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocSPARSE    	                    |   Yes	            |  Yes			                              | 
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocALUTION  		            |   Yes	            |  Yes  			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocPRIM			    |   Yes 	            |  Yes			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocTHRUST	                    |   Yes	            |  Yes			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocSOLVER	                    |   Yes                 |  Yes			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| hipBLAS	                    |   Yes 	            |  Yes				                      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| hipSPARSE 			    |   Yes	            |  Yes 				                      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
| hipcub			    |   Yes 	            |  Yes				                      |
+-----------------------------------+-----------------------+---------------------------------------------------------+


Meta Packages
~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------+---------------------------------------------------------+
|Package                            |  Debian 	            |   RPM						      |	
+===================================+=======================+=========================================================+
|ROCm Master Package 	            |   rocm 	            |  rocm-1.6.77-Linux.rpm				      |	 
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm Developer Master Package 	    |   rocm-dev 	    |  rocm-dev-1.6.77-Linux.rpm  			      | 	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm Libraries Master Package 	    |   rocm-libs 	    |  rocm-libs-1.6.77-Linux.rpm            		      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ATMI       	                    |   atmi     	    |  atmi-0.3.7-45-gde867f2-Linux.rpm			      | 
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Core 	                    |   hip_base 	    |  hip_base-1.2.17263.rpm				      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Documents 			    |   hip_doc 	    |  hip_doc-1.2.17263.rpm				      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Compiler 			    |   hip_hcc 	    |  hip_hcc-1.2.17263.rpm				      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Samples 			    |   hip_samples 	    |  hip_samples-1.2.17263.rpm.			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIPBLAS 			    |   hipblas 	    |  hipblas-0.4.0.3-Linux.rpm			      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|MIOpen OpenCL Lib 		    |   miopen-opencl. 	    |  MIOpen-OpenCL-1.0.0-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|rocBLAS 	                    |   rocblas 	    |  rocblas-0.4.2.3-Linux.rpm      		              |		 
+-----------------------------------+-----------------------+---------------------------------------------------------+ 
|rocFFT 	                    |   rocfft 	            |  rocm-device-libs-0.0.1-Linux.rpm			      |
+-----------------------------------+-----------------------+---------------------------------------------------------+        
|ROCm Device Libs 		    |   rocm-device-libs    |  rocm-device-libs-0.0.1-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm OpenCL for Dev with CL headers|    rocm-opencl-dev    |  rocm-opencl-devel-1.2.0-1424893.x86_64.rpm	      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm GDB 	                    |   rocm-gdb 	    |  rocm-gdb-1.5.265-gc4fb045.x86_64.rpm     	      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|RCP profiler 	                    |   rocm-profiler 	    | rocm-profiler-5.1.6386-gbaddcc9.x86_64.rpm	      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm SMI Tool 	                    |   rocm-smi 	    |  rocm-smi-1.0.0_24_g68893bc-1.x86_64.rpm  	      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm Utilities 	            |   rocm-utils 	    |  rocm-utils-1.0.0-Linux.rpm			      |
+-----------------------------------+-----------------------+---------------------------------------------------------+




