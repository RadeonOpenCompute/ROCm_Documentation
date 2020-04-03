.. _rocm-debug:

ROCm-GPUDebugSDK
================

The ROCm-GPUDebugSDK repository provides the components required to build a GPU kernel debugger for Radeon Open Compute platforms (ROCm). The ROCm GPU Debug SDK components are used by ROCm-GDB and CodeXL debugger to support debugging GPU kernels on ROCm.

Package Contents
****************
The ROCm GPU Debug SDK includes the source code and libraries briefly listed below

* Source code
      *  HSA Debug Agent: The HSA Debug Agent is a library injected into an HSA application by the ROCR-Runtime. The source code for 		 the Agent is provided in src/HSADebugAgent.
      *  Debug Facilities: The Debug Facilities is a utility library to perform symbol processing for ROCm code object. The header 	    file FacilitiesInterface.h is in the include folder while the source code is provided in src/HwDbgFacilities.
      *  Matrix multiplication example: A sample HSA application that runs a matrix multiplication kernel.
* Header files and libraries
      * libAMDGPUDebugHSA-x64: This library provides the low level hardware control required to enable debugging a kernel executing 	    on ROCm. The functionality of this library is exposed by the header file AMDGPUDebug.h in include/. The HSA Debug Agent 	library uses this interface
      * libelf: A libelf library compatible with the ROCm and its corresponding header files. The HSA Debug Agent library uses this 		libelf.

Build Steps
************

1.Install ROCm using the instruction `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-ubuntu>`_
    
2.Clone the Debug SDK repository

:: 
    git clone https://github.com/RadeonOpenCompute/ROCm-GPUDebugSDK.git

3. Build the AMD HSA Debug Agent Library and the Matrix multiplication examples by calling make in the src/HSADebugAgent and the samples/MatrixMultiplication directories respectively

::
    cd src/HSADebugAgent
    make
  
* Note that matrixMul_kernel.hsail is included for reference only. This sample will load the pre-built hsa binary (matrixMul_kernel.brig) to run the kernel.
   
   
::
  
   cd samples/MatrixMultiplication
  
::
 
    make

4. Build the Debug Facilities library by calling make in the src/HwDbgFacilities directory

::

    cd src/HwDbgFacilities
    make

5. Build the ROCm-GDB debugger as shown in the GDB repository.

