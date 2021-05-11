==========================
HIP API Documentation 4.2
==========================

HIP API Guide 
---------------

You can access the latest Doxygen-generated HIP API Guide at the following location:

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_API_Guide_4.2.pdf


Supported CUDA APIs
--------------------

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

HIP Memory Management APIs
===========================

hipMallocHost
**************
Use "hipHostMalloc"

hipMemAllocHost
*****************
Use "hipHostMalloc"

hipHostAlloc
**************
Use "hipHostMalloc"

hipFreeHost
************
Use "hipHostFree"

Supported HIP Math APIs
------------------------

You can access the supported HIP Math APIs at:

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP-MATH.html#hip-math


Related Topics
---------------

HIP Programming Guide 
======================

For the latest HIP Programming Guide, see

https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html


============================================
HIP-Supported CUDA API Reference Guide v4.1
============================================

You can access the latest HIP-Supported CUDA API Reference Guide at

https://github.com/RadeonOpenCompute/ROCm/blob/master/HIP_Supported_CUDA_API_Reference_Guide_v4.1.pdf



HIP API Documentation v3.x 
=============================


HIP Language Runtime API
##############################

* :ref:`HIP-API`


HIP Math API
#############

* :ref:`HIP-MATH`


Supported CUDA APIs
#######################

 * :ref:`CUDAAPIHIP`
 * :ref:`CUDAAPIHIPTEXTURE`
 * `cuComplex API <https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/cuComplex_API_supported_by_HIP.md>`_
 * `cuBLAS <https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/CUBLAS_API_supported_by_HIP.md>`_
 * `cuRAND <https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/CURAND_API_supported_by_HIP.md>`_
 * `cuDNN <https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/CUDNN_API_supported_by_HIP.md>`_
 * `cuFFT <https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/CUFFT_API_supported_by_HIP.md>`_
 * `cuSPARSE <https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/CUSPARSE_API_supported_by_HIP.md>`_
 * `Developer/CONTRIBUTING Info <https://github.com/ROCm-Developer-Tools/HIP/blob/master/CONTRIBUTING.md>`_
 * `Release Notes <https://github.com/ROCm-Developer-Tools/HIP/blob/master/RELEASE.md>`_

    
    
    
Deprecated HIP APIs
##############################

HIP Memory Management API 
*************************


*hipMallocHost*

.. doxygenfunction:: hipMallocHost

**Recommendation**: Use "hipHostMalloc" 

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html?highlight=hipHostMalloc#hiphostmalloc


*hipHostAlloc* 

.. doxygenfunction:: hipHostAlloc 

Recommendation: Use "hipHostMalloc" 

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html?highlight=hipHostMalloc#hiphostmalloc


*hipFreeHost*

.. doxygenfunction:: hipFreeHost


**Recommendation**: Use "hipHostFree" 

**Note**: "hipHostFree" has the same input as the deprecated "hipFreeHost" API.

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html?highlight=hipFreeHost#hipfreehost


*hipMemAllocHost* 

Recommendation: Use "hipHostMalloc" 

https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Memory-Management.html?highlight=hipHostMalloc#hiphostmalloc



.. _Context-Management:

HIP Context Management APIs 
***************************


*hipCtxPopCurrent*

.. doxygenfunction:: hipCtxPopCurrent

*hipCtxPushCurrent* 

.. doxygenfunction:: hipCtxPushCurrent  

*hipCtxSetCurrent* 

.. doxygenfunction:: hipCtxSetCurrent 

*hipCtxGetCurrent* 

.. doxygenfunction:: hipCtxGetCurrent 

*hipCtxGetDevice* 

.. doxygenfunction:: hipCtxGetDevice 

*hipCtxGetApiVersion* 

.. doxygenfunction:: hipCtxGetApiVersion  

*hipCtxGetCacheConfig* 

.. doxygenfunction:: hipCtxGetCacheConfig 

*hipCtxSetSharedMemConfig*

.. doxygenfunction:: hipCtxSetSharedMemConfig

*hipCtxGetSharedMemConfig*

.. doxygenfunction:: hipCtxGetSharedMemConfig

*hipCtxSynchronize* 

.. doxygenfunction:: hipCtxSynchronize 

*hipCtxGetFlags* 

.. doxygenfunction:: hipCtxGetFlags 

*hipCtxEnablePeerAccess* 

.. doxygenfunction:: hipCtxEnablePeerAccess 

*hipCtxDisablePeerAccess*  

.. doxygenfunction:: hipCtxDisablePeerAccess 



==========================
OpenCL Programming Guide
==========================

* :ref:`Opencl-Programming-Guide`

OpenCL Best Practices
######################

* :ref:`Optimization-Opencl`


   

   

====================================
HCC Programming Guide (Deprecated)
=====================================

**What is the Heterogeneous Compute (HC) API ?**

It’s a C++ dialect with extensions to launch kernels and manage accelerator memory. It closely tracks the evolution of C++ and will incorporate parallelism and concurrency features as the C++ standard does. For example, HC includes early support for the C++17 Parallel STL. At the recent ISO C++ meetings in Kona and Jacksonville, the committee was excited about enabling the language to express all forms of parallelism, including multicore CPU, SIMD and GPU. We’ll be following these developments closely, and you’ll see HC move quickly to include standard C++ capabilities.

The Heterogeneous Compute Compiler (HCC) provides two important benefits:

Ease of development


   * A full C++ API for managing devices, queues and events
   * C++ data containers that provide type safety, multidimensional-array indexing and automatic data management
   * C++ kernel-launch syntax using parallel_for_each plus C++11 lambda functions
   * A single-source C++ programming environment---the host and source code can be in the same source file and use the same C++     	 language; templates and classes work naturally across the host/device boundary
   * HCC generates both host and device code from the same compiler, so it benefits from a consistent view of the source code using   	   the same Clang-based language parser

Full control over the machine


    * Access AMD scratchpad memories (“LDS”)
    * Fully control data movement, prefetch and discard
    * Fully control asynchronous kernel launch and completion
    * Get device-side dependency resolution for kernel and data commands (without host involvement)
    * Obtain HSA agents, queues and signals for low-level control of the architecture using the HSA Runtime API
    * Use `direct-to-ISA <https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA>`_ compilation

**When to Use HC**
 Use HC when you're targeting the AMD ROCm platform: it delivers a single-source, easy-to-program C++ environment without compromising performance or control of the machine.

HC Best Practices
##################

HC comes with two header files as of now:

    * `hc.hpp <http://scchan.github.io/hcc/hc_8hpp.html>`_ : Main header file for HC
    * `hc_math.hpp <http://scchan.github.io/hcc/hc__math_8hpp_source.html>`_ : Math functions for HC

Most HC APIs are stored under "hc" namespace, and the class name is the same as their counterpart in C++AMP "Concurrency" namespace. Users of C++AMP should find it easy to switch from C++AMP to HC.

================================== =====================
C++AMP 		       			             HC
================================== =====================
``Concurrency::accelerator``	     ``hc::accelerator``
``Concurrency::accelerator_view``  ``hc::accelerator_view``
``Concurrency::extent``		         ``hc::extent``
``Concurrency::index``		         ``hc::index``
``Concurrency::completion_future`` ``hc::completion_future``
``Concurrency::array``		         ``hc::array``
``Concurrency::array_view``     	 ``hc::array_view``
================================== =====================

HC-specific features
#####################

   * relaxed rules in operations allowed in kernels
   * new syntax of tiled_extent and tiled_index
   * dynamic group segment memory allocation
   * true asynchronous kernel launching behavior
   * additional HSA-specific APIs

Differences between HC API and C++ AMP
######################################
Despite HC and C++ AMP sharing many similar program constructs (e.g. parallel_for_each, array, array_view, etc.), there are several significant differences between the two APIs.

**Support for explicit asynchronous parallel_for_each**
In C++ AMP, the parallel_for_each appears as a synchronous function call in a program (i.e. the host waits for the kernel to complete); howevever, the compiler may optimize it to execute the kernel asynchronously and the host would synchronize with the device on the first access of the data modified by the kernel. For example, if a parallel_for_each writes the an array_view, then the first access to this array_view on the host after the parallel_for_each would block until the parallel_for_each completes.

HC supports the automatic synchronization behavior as in C++ AMP. In addition, HC's parallel_for_each supports explicit asynchronous execution. It returns a completion_future (similar to C++ std::future) object that other asynchronous operations could synchronize with, which provides better flexibility on task graph construction and enables more precise control on optimization.


**Annotation of device functions**

C++ AMP uses the restrict(amp) keyword to annotate functions that runs on the device.

.. code:: cpp

 void foo() restrict(amp) { .. } ... parallel_for_each(...,[=] () restrict(amp) { foo(); });

HC uses a function attribute ([[hc]] or __attribute__((hc)) ) to annotate a device function.

.. code:: cpp

  void foo() [[hc]] { .. } ... parallel_for_each(...,[=] () [[hc]] { foo(); });

The [[hc]] annotation for the kernel function called by parallel_for_each is optional as it is automatically annotated as a device function by the hcc compiler. The compiler also supports partial automatic [[hc]] annotation for functions that are called by other device functions within the same source file:

Since bar is called by foo, which is a device function, the hcc compiler will automatically annotate bar as a device function ``void bar() { ... } void foo() [[hc]] { bar(); }``


**Dynamic tile size**

C++ AMP doesn't support dynamic tile size. The size of each tile dimensions has to be a compile-time constant specified as template arguments to the tile_extent object:

 `extent<2> <http://scchan.github.io/hcc/classConcurrency_1_1extent.html>`_  ex(x, y)

 To create a tile extent of 8x8 from the extent object,note that the tile dimensions have to be constant values:

   tiled_extent<8,8> t_ex(ex)

parallel_for_each(t_ex, [=](tiled_index<8,8> t_id) restrict(amp) { ... });

    HC supports both static and dynamic tile size:

   `extent<2> <http://scchan.github.io/hcc/classConcurrency_1_1extent.html>`_ ex(x,y)

To create a tile extent from dynamically calculated values,note that the the tiled_extent template takes the rank instead of dimensions

         tx = test_x ? tx_a : tx_b;

         ty = test_y ? ty_a : ty_b;

         tiled_extent<2> t_ex(ex, tx, ty);

         parallel_for_each(t_ex, [=](tiled_index<2> t_id) [[hc]] { ... });

**Support for memory pointer**

C++ AMP doesn't support lambda capture of memory pointer into a GPU kernel.

HC supports capturing memory pointer by a GPU kernel.

allocate GPU memory through the HSA API
.. code:: sh

 int* gpu_pointer; hsa_memory_allocate(..., &gpu_pointer); ... parallel_for_each(ext, [=](index i) [[hc]] { gpu_pointer[i[0]]++; }

For HSA APUs that supports system wide shared virtual memory, a GPU kernel can directly access system memory allocated by the host:
.. code:: sh

 int* cpu_memory = (int*) malloc(...); ... parallel_for_each(ext, [=](index i) [[hc]] { cpu_memory[i[0]]++; });


