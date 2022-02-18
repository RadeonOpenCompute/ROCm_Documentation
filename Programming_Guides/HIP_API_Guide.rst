==========================
HIP API Documentation v4.5
==========================


You can access the latest Doxygen-generated HIP API Guide at the following location:

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD-HIP-API-4.5.pdf


============================================
HIP-Supported CUDA API Reference Guide v4.5
============================================

You can access the latest Reference guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_HIP_Supported_CUDA_API_Reference_Guide.pdf


========================================
AMD ROCm Compiler Reference Guide v4.5
========================================

You can access and download the AMD ROCm Compiler Reference Guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_Compiler_Reference_Guide_v4.5.pdf


Supported CUDA APIs 
---------------------

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_HIP_Supported_CUDA_API_Reference_Guide.pdf

To access the following supported CUDA APIs, see

https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html#hip-faq-porting-guide-and-programming-guide

* Runtime API

* Driver API

* cuComplex API

* cuBLAS

* cuRAND

* cuDNN

* cuFFT

* cuSPARSE


Deprecated HIP APIs
-------------------

HIP Context Management APIs
=============================

CUDA supports cuCtx API, the Driver API that defines "Context" and "Devices" as separate entities. Contexts contain a single device, and a device can theoretically have multiple contexts. HIP initially added limited support for APIs to facilitate easy porting from existing driver codes. The APIs are marked as deprecated now as there is a better alternate interface (such as hipSetDevice or the stream API) to achieve the required functions.

* hipCtxPopCurrent

* hipCtxPushCurrent

* hipCtxSetCurrent

* hipCtxGetCurrent

* hipCtxGetDevice

* hipCtxGetApiVersion

* hipCtxGetCacheConfig

* hipCtxSetCacheConfig

* hipCtxSetSharedMemConfig

* hipCtxGetSharedMemConfig

* hipCtxSynchronize

* hipCtxGetFlags

* hipCtxEnablePeerAccess

* hipCtxDisablePeerAccess






   
 
==========================
OpenCL Programming Guide
==========================

* :ref:`Opencl-Programming-Guide`

OpenCL Best Practices
######################

* :ref:`Optimization-Opencl`


   

   

