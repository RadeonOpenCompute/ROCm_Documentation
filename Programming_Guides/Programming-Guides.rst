.. _Programming-Guides:

=================
Programming Guide
=================

ROCm Languages
================


ROCm, Lingua Franca,  C++, OpenCL and Python
#############################################
The open-source ROCm stack offers multiple programming-language choices. The goal is to give you a range of tools to help solve the
problem at hand. Here, we describe some of the options and how to choose among them.


HCC: Heterogeneous Compute Compiler
####################################

**HCC : An open source C++ compiler for heterogeneous devices**

This repository hosts the HCC compiler implementation project. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as HC, C++ 17 ParallelSTL and transforms it into the AMD GCN ISA.

Deprecation Notice
*******************

AMD is deprecating HCC to put more focus on HIP development and on other languages supporting heterogeneous compute. We will no longer develop any new feature in HCC and we will stop maintaining HCC after its final release, which is planned for June 2019. If your application was developed with the hc C++ API, we would encourage you to transition it to other languages supported by AMD, such as HIP or OpenCL. HIP and hc language share the same compiler technology, so many hc kernel language features (including inline assembly) are also available through the HIP compilation path.

The project is based on LLVM+CLANG. For more information, please visit :ref:`HCCguide`

HIP: Heterogeneous-Computing Interface for Portability
#########################################################
What is Heterogeneous-Computing Interface for Portability (HIP)? It’s a C++ dialect designed to ease conversion of Cuda applications to portable C++ code. It provides a C-style API and a C++ kernel language. The C++ interface can use templates and classes across the
host/kernel boundary.

The Hipify tool automates much of the conversion work by performing a source-to-source transformation from Cuda to HIP. HIP code can run on AMD hardware (through the HCC compiler) or Nvidia hardware (through the NVCC compiler) with no performance loss compared with the original Cuda code.

Programmers familiar with other GPGPU languages will find HIP very easy to learn and use. AMD platforms implement this language using the HC dialect described above, providing similar low-level control over the machine.

When to Use HIP
****************
Use HIP when converting Cuda applications to portable C++ and for new projects that require portability between AMD and Nvidia. HIP provides a C++ development language and access to the best development tools on both platforms.

OpenCL™: Open Compute Language
################################
What is OpenCL ?  It’s a framework for developing programs that can execute across a wide variety of heterogeneous platforms. AMD, Intel
and Nvidia GPUs support version 1.2 of the specification, as do x86 CPUs and other devices (including FPGAs and DSPs). OpenCL provides a C run-time API and C99-based kernel language.

When to Use OpenCL
*******************
Use OpenCL when you have existing code in that language and when you need portability to multiple platforms and devices. It runs on
Windows, Linux and Mac OS, as well as a wide variety of hardware platforms (described above).

Anaconda Python With Numba
###########################
What is Anaconda ?  It’s a modern open-source analytics platform powered by Python. Continuum Analytics, a ROCm platform partner,  is the driving force behind it. Anaconda delivers high-performance capabilities including acceleration of HSA APUs, as well as
ROCm-enabled discrete GPUs via Numba. It gives superpowers to the people who are changing the world.

Numba
******
Numba gives you the power to speed up your applications with high-performance functions written directly in Python. Through a few
annotations, you can just-in-time compile array-oriented and math-heavy Python code to native machine instructions---offering
performance similar to that of C, C++ and Fortran---without having to switch languages or Python interpreters.

Numba works by generating optimized machine code using the LLVM compiler infrastructure at import time, run time or statically
(through the included Pycc tool). It supports Python compilation to run on either CPU or GPU hardware and is designed to integrate with Python scientific software stacks, such as NumPy.

  * `Anaconda® with Numba acceleration <http://numba.pydata.org/numba-doc/latest/index.html>`_

When to Use Anaconda
*********************
Use Anaconda when you’re handling large-scale data-analytics,
scientific and engineering problems that require you to manipulate
large data arrays.

Wrap-Up
#######
From a high-level perspective, ROCm delivers a rich set of tools that
allow you to choose the best language for your application.

 * HCC (Heterogeneous Compute Compiler) supports HC dialects
 * HIP is a run-time library that layers on top of HCC (for AMD ROCm platforms; for Nvidia, it uses the NVCC compiler)
 * The following will soon offer native compiler support for the GCN ISA:
    * OpenCL 1.2+
    * Anaconda (Python) with Numba

All are open-source projects, so you can employ a fully open stack from the language down to the metal. AMD is committed to providing an open ecosystem that gives developers the ability to choose; we are excited about innovating quickly using open source and about
interacting closely with our developer community. More to come soon!

Table Comparing Syntax for Different Compute APIs
**************************************************



+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Term 	          	|  CUDA 	|       HIP       |       HC 	        |      C++AMP 	         |  OpenCL                   |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Device 	        | int deviceId  | int deviceId 	  | hc::accelerator     |  concurrency::	 |  cl_device                |
|			|		|		  |		        |  accelerator 	         |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Queue 	                | cudaStream_t  |   hipStream_t   | hc:: 	        | concurrency::          | cl_command_queue          |
|			|		|	     	  | accelerator_view    | accelerator_view       |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Event 	                | cudaEvent_t 	|   hipEvent_t 	  | hc::                | concurrency::          |                           |
|                       |               |                 | completion_future   | completion_future      |   cl_event                |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Memory                 |   void * 	|    void * 	  |void *; hc::array;   | concurrency::array;    |   cl_mem                  |
|			|		|                 |hc::array_view       |concurrency::array_view |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |     grid      |    grid         |	extent	        |	      extent	 |	   NDRange	     |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
|                       |    block      |    block	  |      tile           |	       tile 	 |	  work-group	     |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
|                       |    thread     |    thread       |      thread         |	      thread 	 |	work-item            |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
|                       |     warp      |    warp         |    wavefront        |	       N/A	 |  sub-group                |
|                       |---------------|-----------------|---------------------|------------------------|---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Thread index           |threadIdx.x    | hipThreadIdx_x  |  t_idx.local[0]     |    t_idx.local[0]      |  get_local_id(0)          |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Block index            | blockIdx.x    | hipBlockIdx_x   |  t_idx.tile[0]      |    t_idx.tile[0]       | get_group_id(0)           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Block  dim             | blockDim.x    | hipBlockDim_x   | t_ext.tile_dim[0]   |  t_idx.tile_dim0       |get_local_size(0)          |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Grid-dim               | gridDim.x     | hipGridDim_x    |   	t_ext[0]        |      t_ext[0]          |get_global_size(0)         |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Device Function        | __device__    | __device__      |[[hc]] (detected     |                        |Implied in device          |
|                       |               |                 |automatically in     |    restrict(amp)       |Compilation                |
|                       |               |                 |many case)           |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Host Function          | __host_       |__host_ (default)|[[cpu]] (default)    |                        |Implied in host            |
|                       |  (default)    |                 |                     |  strict(cpu) (default) |Compilation                |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
| Host +                |  __host__     |  __host_        |[[hc]] [[cpu]]       |                        |No equivalent              |
| Device                |  __device__   | __device__      |                     |  restrict(amp,cpu)     |                           |
| Function              |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|Kernel Launch          |               |                 |                     | concurrency::          |                           |
|                       |   <<< >>>     | hipLaunchKernel |hc::                 | parallel_for_each      |clEnqueueND-               |
|                       |               |      GGL        |parallel_for_each    |                        |RangeKernel                |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |                        |                           |
|Global Memory          |  __global__   |   __global__    |Unnecessary/         |  Unnecessary/Implied   |  __global                 |
|                       |               |                 |Implied              |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |                        |                           |
|Group Memory           | __shared__    | __shared__      | tile_static         |   tile_static          |   __local                 |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |Unnecessary/         |                        |                           |
|Constant               | __constant__  |   __constant__  |Implied              |Unnecessary / Implied   |   __constant              |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |__syncthreads  |__syncthreads    |tile_static.barrier()| 	t_idx.barrier()  |barrier(CLK_LOCAL_MEMFENCE)|
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |   concurrency::        |                           |
|Atomic Builtins        | atomicAdd     |  atomicAdd      |hc::atomic_fetch_add |   atomic_fetch_add     |      atomic_add           |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |                     |                        |                           |
|Precise Math           |  cos(f)       |   cos(f)        | hc::                |   concurrency::        |      	cos(f)       |
|                       |               |                 | precise_math::cos(f)|   precise_math::cos(f) |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |hc::fast_math::cos(f)|   concurrency::        |                           |
|Fast Math              | __cos(f)      |  __cos(f)       |                     |   fast_math::cos(f)    |    native_cos(f)          |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |hc::                 |concurrency::           |                           |
|Vector                 |   float4      |   	float4    |short_vector::float4 |graphics::float_4       |         float4            |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+



Notes
******

1. For HC and C++AMP, assume a captured _tiled_ext_ named "t_ext" and captured _extent_ named "ext".  These languages use captured variables to pass information to the kernel rather than using special built-in functions so the exact variable name may vary.
2. The indexing functions (starting with `thread-index`) show the terminology for a 1D grid.  Some APIs use reverse order of xyz / 012 indexing for 3D grids.
3. HC allows tile dimensions to be specified at runtime while C++AMP requires that tile dimensions be specified at compile-time.  Thus hc syntax for tile dims is ``t_ext.tile_dim[0]``  while C++AMP is ``t_ext.tile_dim0``.
4. **From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**


HC Programming Guide
====================

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


HIP Programing Guide
====================

**What is this repository for?**

HIP allows developers to convert CUDA code to portable C++. The same source code can be compiled to run on NVIDIA or AMD GPUs. Key features include:

 * HIP is very thin and has little or no performance impact over coding directly in CUDA or hcc "HC" mode.
 *   HIP allows coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, and more.
 *  HIP allows developers to use the "best" development environment and tools on each target platform.
 *   The "hipify" tool automatically converts source from CUDA to HIP.
 *   Developers can specialize for the platform (CUDA or hcc) to tune for performance or handle tricky cases

New projects can be developed directly in the portable HIP C++ language and can run on either NVIDIA or AMD platforms. Additionally, HIP provides porting tools which make it easy to port existing CUDA codes to the HIP layer, with no loss of performance as compared to the original CUDA application. HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to complete the port.

**Repository branches:**

The HIP repository maintains several branches. The branches that are of importance are:

    master branch: This is the stable branch. All stable releases are based on this branch.
    developer-preview branch: This is the branch were the new features still under development are visible. While this maybe of interest to many, it should be noted that this branch and the features under development might not be stable.

**Release tagging:**

HIP releases are typically of two types. The tag naming convention is different for both types of releases to help differentiate them.

    release_x.yy.zzzz: These are the stable releases based on the master branch. This type of release is typically made once a month.
    preview_x.yy.zzzz: These denote pre-release code and are based on the developer-preview branch. This type of release is typically made once a week

**More Info**

HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in compute kernels, including classes, namespaces, operator overloading, templates and more. Additionally, it defines other language features designed specifically to target accelerators, such as the following:

   * A kernel-launch syntax that uses standard C++, resembles a function call and is portable to all HIP targets
   * Short-vector headers that can serve on a host or a device
   * Math functions resembling those in the "math.h" header included with standard C++ compilers
   * Built-in functions for accessing specific GPU hardware capabilities

This section describes the built-in variables and functions accessible from the HIP kernel. It’s intended for readers who are familiar with Cuda kernel syntax and want to understand how HIP is different.

  * :ref:`HIP-GUIDE`


HIP Best Practices
###################

 * :ref:`HIP-IN`
 * :ref:`HIP-FAQ`
 * :ref:`Kernel_language`
 * `HIP Runtime API (Doxygen) <https://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/HIP-API.html#hip-api>`_
 * :ref:`HIP-porting-guide`
 * :ref:`hip-p`
 * :ref:`hip-pro`
 * :ref:`hip_profiling`
 * :ref:`HIP_Debugging`
 * :ref:`HIP-terminology`
 * :ref:`HIP-Term2`
 * `hipify-clang <https://github.com/ROCm-Developer-Tools/HIP/blob/master/hipify-clang/README.md>`_
Supported CUDA APIs:
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

**Simple Example**

The HIP API includes functions such as hipMalloc, hipMemcpy, and hipFree.
Programmers familiar with CUDA will also be able to quickly learn and start coding with the HIP API. Compute kernels are launched with the "hipLaunchKernelGGL" macro call. Here is simple example showing a snippet of HIP API code:

hipMalloc(&A_d, Nbytes));
hipMalloc(&C_d, Nbytes));

hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice);

const unsigned blocks = 512;
const unsigned threadsPerBlock = 256;
hipLaunchKernelGGL(vector_square,   /* compute kernel*/
                dim3(blocks), dim3(threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                C_d, A_d, N);  /* arguments to the compute kernel */

hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost); 

The HIP kernel language defines builtins for determining grid and block coordinates, math functions, short vectors, atomics, and timer functions. It also specifies additional defines and keywords for function types, address spaces, and optimization controls. (See the  :ref:`Kernel_language` for a full description). Here's an example of defining a simple 'vector_square' kernel.

template <typename T>
__global__ void
vector_square(T *C_d, const T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

The HIP Runtime API code and compute kernel definition can exist in the same source file - HIP takes care of generating host and device code appropriately.

**HIP Portability and Compiler Technology**

HIP C++ code can be compiled with either :

    * On the NVIDIA CUDA platform, HIP provides header file which translate from the HIP runtime APIs to CUDA runtime APIs. The header file contains mostly inlined functions and thus has very low overhead - developers coding in HIP should expect the same performance as coding in native CUDA. The code is then compiled with nvcc, the standard C++ compiler provided with the CUDA SDK. Developers can use any tools supported by the CUDA SDK including the CUDA profiler and debugger.
    * On the AMD ROCm platform, HIP provides a header and runtime library built on top of hcc compiler. The HIP runtime implements HIP streams, events, and memory APIs, and is a object library that is linked with the application. The source code for all headers and the library implementation is available on GitHub.
    HIP developers on ROCm can use AMD's CodeXL for debugging and profiling.

Thus HIP source code can be compiled to run on either platform. Platform-specific features can be isolated to a specific platform using conditional compilation. Thus HIP provides source portability to either platform. HIP provides the hipcc compiler driver which will call the appropriate toolchain depending on the desired platform.

**Examples and Getting Started:**

    * A sample and blog that uses hipify to convert a simple app from CUDA to HIP:
 ::

 cd samples/01_Intro/square
 # follow README / blog steps to hipify the application.


    * A sample and blog demonstrating platform specialization:
 ::

 cd samples/01_Intro/bit_extract
 make


    * Guide to Porting a New Cuda Project

**More Examples**

The GitHub repository `HIP-Examples <https://github.com/ROCm-Developer-Tools/HIP-Examples.git>`_ contains a hipified version of the popular Rodinia benchmark suite. The README with the procedures and tips the team used during this porting effort is here: `Rodinia Porting Guide <https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/rodinia_3.0/hip/README.hip_porting>`_

**Tour of the HIP Directories**

    **include:**
        * **hip_runtime_api.h** : Defines HIP runtime APIs and can be compiled with many standard Linux compilers (hcc, GCC, ICC, CLANG, etc), in either C or C++ mode.
        * **hip_runtime.h** : Includes everything in hip_runtime_api.h PLUS hipLaunchKernelGGL and syntax for writing device kernels and device functions. hip_runtime.h can only be compiled with hcc.
        * **hcc_detail/** , nvcc_detail/** : Implementation details for specific platforms. HIP applications should not include these files directly.
        * **hcc.h** : Includes interop APIs for HIP and HCC

   * **bin:** Tools and scripts to help with hip porting
        * **hipify :** Tool to convert CUDA code to portable CPP. Converts CUDA APIs and kernel builtins.
        * **hipcc :** Compiler driver that can be used to replace nvcc in existing CUDA code. hipcc will call nvcc or hcc depending on platform, and include appropriate platform-specific headers and libraries.
        * **hipconfig :** Print HIP configuration (HIP_PATH, HIP_PLATFORM, CXX config flags, etc)
        * **hipexamine.sh** : Script to scan directory, find all code, and report statistics on how much can be ported with HIP (and identify likely features not yet supported)

    * **doc:** Documentation - markdown and doxygen info

OpenCL Programing Guide
========================

* :ref:`Opencl-Programming-Guide`

OpenCL Best Practices
######################

* :ref:`Optimization-Opencl`
