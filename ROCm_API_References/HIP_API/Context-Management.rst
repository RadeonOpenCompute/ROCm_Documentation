.. _Context-Management:

====================
Context Management
====================

*  `Current HIP APIs`_

* `Deprecated HIP APIs`_


Current HIP APIs
==================

hipDevicePrimaryCtxGetState 
-----------------------------
.. doxygenfunction:: hipDevicePrimaryCtxGetState 

hipDevicePrimaryCtxRelease
----------------------------
.. doxygenfunction:: hipDevicePrimaryCtxRelease

hipDevicePrimaryCtxRetain
--------------------------
.. doxygenfunction:: hipDevicePrimaryCtxRetain

hipDevicePrimaryCtxReset
---------------------------
.. doxygenfunction:: hipDevicePrimaryCtxReset 

hipDevicePrimaryCtxSetFlags 
----------------------------
.. doxygenfunction:: hipDevicePrimaryCtxSetFlags 



Deprecated HIP APIs
========================

HIP Management APIs
####################

**hipMallocHost**

Use "hipHostMalloc" 


**hipMemAllocHost** 

Use "hipHostMalloc" 


**hipHostAlloc**

Use "hipHostMalloc" 


For more information on 'hipHostMalloc', see 

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html?highlight=hipHostMalloc#hiphostmalloc


**hipFreeHost**

Use "hipHostFree" 

**Note**: "hipHostFree" has the same input as the deprecated "hipFreeHost" API.

For more information, see

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html?highlight=hipFreeHost#hipfreehost


hipCtxCreate
----------------
.. doxygenfunction::  hipCtxCreate

hipCtxDestroy
----------------
.. doxygenfunction:: hipCtxDestroy

hipCtxPopCurrent
----------------
.. doxygenfunction:: hipCtxPopCurrent

hipCtxPushCurrent 
------------------
.. doxygenfunction:: hipCtxPushCurrent  

hipCtxSetCurrent 
----------------
.. doxygenfunction:: hipCtxSetCurrent 

hipCtxGetCurrent 
----------------
.. doxygenfunction:: hipCtxGetCurrent 

hipCtxGetDevice 
----------------
.. doxygenfunction:: hipCtxGetDevice 

hipCtxGetApiVersion 
--------------------
.. doxygenfunction:: hipCtxGetApiVersion  

hipCtxGetCacheConfig 
----------------------
.. doxygenfunction:: hipCtxGetCacheConfig 

hipCtxSetSharedMemConfig
--------------------------
.. doxygenfunction:: hipCtxSetSharedMemConfig

hipCtxGetSharedMemConfig
--------------------------
.. doxygenfunction:: hipCtxGetSharedMemConfig

hipCtxSynchronize 
------------------
.. doxygenfunction:: hipCtxSynchronize 

hipCtxGetFlags 
----------------
.. doxygenfunction:: hipCtxGetFlags 

hipCtxEnablePeerAccess 
------------------------
.. doxygenfunction:: hipCtxEnablePeerAccess 

hipCtxDisablePeerAccess  
------------------------
.. doxygenfunction:: hipCtxDisablePeerAccess 
























