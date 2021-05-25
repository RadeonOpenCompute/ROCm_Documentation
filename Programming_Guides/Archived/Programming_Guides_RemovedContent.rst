The following content is removed from the main Programming_Guides.rst file.



HIP Best Practices
=====================

 * :ref:`HIP-IN`
 * :ref:`Kernel_language`
 * `HIP Runtime API (Doxygen) <https://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/HIP-API.html#hip-api>`_
 * :ref:`hip-p`
 * :ref:`hip_profiling`
 * :ref:`HIP_Debugging`
 * :ref:`HIP-terminology`
 * :ref:`HIP-Term2`
 * `hipify-clang <https://github.com/ROCm-Developer-Tools/HIP/blob/master/hipify-clang/README.md>`_





HIP Documentation v3.x
=========================

ROCm Supported Languages
############################

ROCm, Lingua Franca,  C++, OpenCL and Python

The open-source ROCm stack offers multiple programming-language choices. The goal is to give you a range of tools to help solve the
problem at hand. Here, we describe some of the options and how to choose among them.

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
|Grid-dim               | gridDim.x     | hipGridDim_x    |   	t_ext[0]        |      t_ext[0]          |get_global_size(0)          |
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
|Precise Math           |  cos(f)       |   cos(f)        | hc::                |   concurrency::        |      	cos(f)              |
|                       |               |                 | precise_math::cos(f)|   precise_math::cos(f) |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |hc::fast_math::cos(f)|   concurrency::        |                           |
|Fast Math              | __cos(f)      |  __cos(f)       |                     |   fast_math::cos(f)    |    native_cos(f)          |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+
|                       |               |                 |hc::                 |concurrency::           |                           |
|Vector                 |   float4      |   	float4    |short_vector::float4 |graphics::float_4          |         float4            |
|                       |               |                 |                     |                        |                           |
+-----------------------+---------------+-----------------+---------------------+------------------------+---------------------------+



Notes
******

1. For HC and C++AMP, assume a captured _tiled_ext_ named "t_ext" and captured _extent_ named "ext".  These languages use captured variables to pass information to the kernel rather than using special built-in functions so the exact variable name may vary.
2. The indexing functions (starting with `thread-index`) show the terminology for a 1D grid.  Some APIs use reverse order of xyz / 012 indexing for 3D grids.
3. HC allows tile dimensions to be specified at runtime while C++AMP requires that tile dimensions be specified at compile-time.  Thus hc syntax for tile dims is ``t_ext.tile_dim[0]``  while C++AMP is ``t_ext.tile_dim0``.
4. **From ROCm version 2.0 onwards C++AMP is no longer available in HCC.**

HIP Repository Information
###########################


**HIP is a C++ Runtime API and Kernel Language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source
code.**

Key features include:

-  HIP is very thin and has little or no performance impact over coding
   directly in CUDA mode.
-  HIP allows coding in a single-source C++ programming language
   including features such as templates, C++11 lambdas, classes,
   namespaces, and more.
-  HIP allows developers to use the best development environment and
   tools on each target platform.
-  The
   `HIPIFY <https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/README.md>`__
   tools automatically convert source from CUDA to HIP.
-  Developers can specialize for the platform (CUDA or AMD) to tune for
   performance or handle tricky cases.

New projects can be developed directly in the portable HIP C++ language
and can run on either NVIDIA or AMD platforms. Additionally, HIP
provides porting tools which make it easy to port existing CUDA codes to
the HIP layer, with no loss of performance as compared to the original
CUDA application. HIP is not intended to be a drop-in replacement for
CUDA, and developers should expect to do some manual coding and
performance tuning work to complete the port.

Repository Branches
#######################

The HIP repository maintains several branches. The branches that are of
importance are:

-  master branch: This is the stable branch. All stable releases are based on this branch.

-  developer-preview branch: This is the branch were the new features still under development are visible. While this maybe of interest to
   many, it should be noted that this branch and the features under development might not be stable.


Release Tagging
#######################

HIP releases are typically of two types. The tag naming convention is different for both types of releases to help differentiate them.

-  release_x.yy.zzzz: These are the stable releases based on the master
   branch. This type of release is typically made once a month.
   
-  preview_x.yy.zzzz: These denote pre-release code and are based on the
   developer-preview branch. This type of release is typically made once
   a week.

  * :ref:`HIP-GUIDE`

HIP FAQ and HIP Porting Guide
################################

 * :ref:`HIP-FAQ`
 * :ref:`HIP-porting-guide`
 * :ref:`hip-pro`
 
 
How to Install
###############

Refer to the Installation Guide at https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#hip-installation-instructions

HIP API Code - Example
#######################

The HIP API includes functions such as hipMalloc, hipMemcpy, and hipFree. Programmers familiar with CUDA will also be able to quickly
learn and start coding with the HIP API. Compute kernels are launched with the hipLaunchKernel's macro call. Here is an example showing a
snippet of HIP API code:

.. code:: cpp

   hipMalloc(&A_d, Nbytes));
   hipMalloc(&C_d, Nbytes));

   hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice);

   const unsigned blocks = 512;
   const unsigned threadsPerBlock = 256;
   hipLaunchKernel(vector_square,   /* compute kernel*/
                   dim3(blocks), dim3(threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                   C_d, A_d, N);  /* arguments to the compute kernel */

   hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost);

The HIP kernel language defines builtins for determining grid and block coordinates, math functions, short vectors, atomics, and timer
functions. It also specifies additional defines and keywords for function types, address spaces, and optimization controls (See the `HIP
Kernel Language <docs/markdown/hip_kernel_language.md>`__ for a full description). Here's an example of defining a simple 'vector_square'
kernel.

.. code:: cpp

   template <typename T>
   __global__ void
   vector_square(T *C_d, const T *A_d, size_t N)
   {
       size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
       size_t stride = hipBlockDim_x * hipGridDim_x;

       for (size_t i=offset; i<N; i+=stride) {
           C_d[i] = A_d[i] * A_d[i];
       }
   }

The HIP Runtime API code and compute kernel definition can exist in the
same source file - HIP takes care of generating host and device code
appropriately.

HIP Portability and Compiler Technology
=======================================

HIP C++ code can be compiled with either, - On the NVIDIA CUDA platform,
HIP provides header file which translate from the HIP runtime APIs to
CUDA runtime APIs. The header file contains mostly inlined functions and
thus has very low overhead - developers coding in HIP should expect the
same performance as coding in native CUDA. The code is then compiled
with nvcc, the standard C++ compiler provided with the CUDA SDK.
Developers can use any tools supported by the CUDA SDK including the
CUDA profiler and debugger. - On the AMD ROCm platform, HIP provides a
header and runtime library built on top of HIP-Clang compiler. The HIP
runtime implements HIP streams, events, and memory APIs, and is a object
library that is linked with the application. The source code for all
headers and the library implementation is available on GitHub. HIP
developers on ROCm can use AMDâ€™s ROCgdb
(https://github.com/ROCm-Developer-Tools/ROCgdb) for debugging and
profiling.

Thus HIP source code can be compiled to run on either platform.
Platform-specific features can be isolated to a specific platform using
conditional compilation. Thus HIP provides source portability to either
platform. HIP provides the *hipcc* compiler driver which will call the
appropriate toolchain depending on the desired platform.

Examples and Getting Started
###############################

-  A sample and
   `blog <http://gpuopen.com/hip-to-be-squared-an-introductory-hip-tutorial>`__
   that uses any of
   `HIPIFY <https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/README.md>`__
   tools to convert a simple app from CUDA to HIP:

.. code:: shell

   cd samples/01_Intro/square
   # follow README / blog steps to hipify the application.

-  A sample and
   `blog <http://gpuopen.com/platform-aware-coding-inside-hip/>`__
   demonstrating platform specialization:

.. code:: shell

   cd samples/01_Intro/bit_extract
   make

-  Guide to `Porting a New Cuda
   Project <docs/markdown/hip_porting_guide.md#porting-a-new-cuda-project%22>`__

More Examples
##############

The GitHub repository
`HIP-Examples <https://github.com/ROCm-Developer-Tools/HIP-Examples.git>`__
contains a hipified version of the popular Rodinia benchmark suite. The README with the procedures and tips the team used during this porting
effort is here: `Porting Guide <https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/rodinia_3.0/hip/README.hip_porting>`__

Tour of the HIP Directories
###############################

-  **include**:

   -  **hip_runtime_api.h** : Defines HIP runtime APIs and can be
      compiled with many standard Linux compilers (hcc, GCC, ICC, CLANG,
      etc), in either C or C++ mode.
      
   -  **hip_runtime.h** : Includes everything in hip_runtime_api.h PLUS
      hipLaunchKernel and syntax for writing device kernels and device
      functions. hip_runtime.h can only be compiled with hcc.
      
   -  **hcc_detail/**\ \*\* , **nvcc_detail/**\ \*\* : Implementation
      details for specific platforms. HIP applications should not
      include these files directly.
      
   -  **hcc.h** : Includes interop APIs for HIP and HCC

-  **bin**: Tools and scripts to help with hip porting

   -  **hipify-perl** : Script based tool to convert CUDA code to
      portable CPP. Converts CUDA APIs and kernel builtins.
   -  **hipcc** : Compiler driver that can be used to replace nvcc in
      existing CUDA code. hipcc will call nvcc or HIP-Clang depending on
      platform and include appropriate platform-specific headers and
      libraries.
   -  **hipconfig** : Print HIP configuration (HIP_PATH, HIP_PLATFORM,
      HIP_COMPILER, HIP_RUNTIME, CXX config flags, etc.)
   -  **hipexamine-perl.sh** : Script to scan the directory, find all
      code, and report statistics on how much can be ported with HIP
      (and identify likely features not yet supported).
   -  **hipconvertinplace-perl.sh** : Script to scan the directory, find
      all code, and convert the found CUDA code to HIP reporting all
      unconverted things.

-  **doc**: Documentation - markdown and doxygen information.

Reporting an Issue
######################

Use the `GitHub issue tracker <https://github.com/ROCm-Developer-Tools/HIP/issues>`__. 

If reporting a bug, include the output of 'hipconfig' 'full' and samples/1_hipInfo/hipInfo (if possible).

    
    
