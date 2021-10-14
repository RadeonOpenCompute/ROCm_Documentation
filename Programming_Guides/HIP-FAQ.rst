.. _HIP-FAQ:



==========
HIP-FAQ
==========

.. contents::



What APIs and features does HIP support?
----------------------------------------

HIP provides the following: 

- Devices (hipSetDevice(), hipGetDeviceProperties())

- Memory management (hipMalloc(), hipMemcpy(), hipFree())

- Streams (hipStreamCreate(),hipStreamSynchronize(), hipStreamWaitEvent())

- Events (hipEventRecord(), hipEventElapsedTime())

- Kernel launching (hipLaunchKernel is a standard C/C++ function that replaces <<< >>>) 

- HIP Module API to control when adn how code is loaded. 

- CUDA-style kernel coordinate functions (threadIdx, blockIdx, blockDim, gridDim) 

- Cross-lane instructions including shfl, ballot, any, all - Most device-side math built-ins. 

- Error reporting (hipGetLastError(), hipGetErrorString())

The HIP API documentation describes each API and its limitations, if any, compared with the equivalent CUDA API.

What is not supported?
----------------------

Runtime/Driver API features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

At a high-level, the following features are not supported: 

- Textures (partial support available) 

- Dynamic parallelism (CUDA 5.0) 

- Managed memory (CUDA 6.5) 

- Graphics interoperability with OpenGL or Direct3D 

CUDA IPC Functions (Under Development) 

- CUDA array, mipmappedArray and pitched memory 

- Queue priority controls

See the `API Support Table <CUDA_Runtime_API_functions_supported_by_HIP.md>`__ for more detailed information.

Kernel language features
~~~~~~~~~~~~~~~~~~~~~~~~

-  C++-style device-side dynamic memory allocations (free, new, delete)
   (CUDA 4.0)
-  Virtual functions, indirect functions and try/catch (CUDA 4.0)
-  ``__prof_trigger``
- PTX assembly (CUDA 4.0).  HIP-Clang supports inline GCN assembly..

Several kernel features are under development. See the `HIP Kernel Language <hip_kernel_language.md>`__ for more information. This includes:

   -  printf

Is HIP a drop-in replacement for CUDA?
--------------------------------------

No. HIP provides porting tools which do most of the work to convert CUDA code into portable C++ code that uses the HIP APIs. Most developers will port their code from CUDA to HIP and then maintain the HIP version. HIP code provides the same performance as native CUDA code, plus the benefits of running on AMD platforms.

What specific version of CUDA does HIP support?
-----------------------------------------------

HIP APIs and features do not map to a specific CUDA version. HIP provides a strong subset of the functionality provided in CUDA, and the
hipify tools can scan code to identify any unsupported CUDA functions. This is useful for identifying the specific features required by a given application.

However, we can provide a rough summary of the features included in each CUDA SDK and the support level in HIP. Each bullet below lists the major new language features in each CUDA release and then indicate which are supported/not supported in HIP:

-  CUDA 4.0 and earlier :

   -  HIP supports CUDA 4.0 except for the limitations described above.

-  CUDA 5.0 :

   -  Dynamic Parallelism (not supported)
   -  cuIpc functions (under development).

-  CUDA 5.5 :

   -  CUPTI (not directly supported, `AMD
      GPUPerfAPI <http://developer.amd.com/tools-and-sdks/graphics-development/gpuperfapi/>`__
      can be used as an alternative in some cases)

-  CUDA 6.0 :

   -  Managed memory (under development)

-  CUDA 6.5 :

   -  \__shfl intriniscs (supported)

-  CUDA 7.0 :

   -  Per-thread-streams (under development)
   - C++11 (Hip-Clang supports all of C++11, all of C++14 and some C++17 features)
  
-  CUDA 7.5 :

   -  float16 (supported)

-  CUDA 8.0 :

   -  Page Migration including cudaMemAdvise, cudaMemPrefetch, other cudaMem\* APIs(not supported)
   
- CUDA 9.0 :

 - Cooperative Launch, Surface Object Management, Version Management

What libraries does HIP support?
--------------------------------

HIP includes growing support for the four key math libraries using hcBlas, hcFft, hcrng and hcsparse, as well as MIOpen for machine
intelligence applications. These offer pointer-based memory interfaces (as opposed to opaque buffers) and can be easily interfaced with other HIP applications. The hip interfaces support both ROCm and CUDA paths, with familiar library interfaces.

-  `hipBlas <https://github.com/ROCmSoftwarePlatform/hipBLAS>`__, which
   utilizes
   `rocBlas <https://github.com/ROCmSoftwarePlatform/rocBLAS>`__.
-  `hipfft <https://github.com/ROCmSoftwarePlatform/hcFFT>`__
-  `hipsparse <https://github.com/ROCmSoftwarePlatform/hcSPARSE>`__
-  `hiprng <https://github.com/ROCmSoftwarePlatform/hcrng>`__

Additionally, some of the cublas routines are automatically converted to hipblas equivalents by the HIPIFY tools. These APIs use cublas or hcblas depending on the platform and replace the need to use conditional compilation.

How does HIP compare with OpenCL?
---------------------------------

Both AMD and Nvidia support OpenCL 1.2 on their devices so that developers can write portable code. HIP offers several benefits over OpenCL: 

- Developers can code in C++ as well as mix host and device C++ code in their source files. HIP C++ code can use templates, lambdas, classes and so on. 

- The HIP API is less verbose than OpenCL and is familiar to CUDA developers. 

- Because both CUDA and HIP are C++ languages, porting from CUDA to HIP is significantly easier than porting from CUDA to OpenCL. 

- HIP uses the best available development tools on each platform: on Nvidia GPUs, HIP code compiles using NVCC and can
employ the nSight profiler and debugger (unlike OpenCL on Nvidia GPUs).

- HIP provides pointers and host-side pointer arithmetic.

- HIP provides device-level control over memory allocation and placement. 

- HIP offers an offline compilation model.

How does porting CUDA to HIP compare to porting CUDA to OpenCL?
---------------------------------------------------------------

Both HIP and CUDA are dialects of C++, and thus porting between them is relatively straightforward. Both dialects support templates, classes, lambdas, and other C++ constructs. As one example, the hipify-perl tool was originally a Perl script that used simple text conversions from CUDA to HIP. HIP and CUDA provide similar math library calls as well. In summary, the HIP philosophy was to make the HIP language close enough to CUDA that the porting effort is relatively simple. This reduces the potential for error, and also makes it easy to automate the translation. HIP's goal is to quickly get the ported program running on both platforms with little manual intervention, so that the programmer can focus on performance optimizations.

There have been several tools that have attempted to convert CUDA into OpenCL, such as CU2CL. OpenCL is a C99-based kernel language (rather than C++) and also does not support single-source compilation. As a result, the OpenCL syntax is different from CUDA, and the porting tools have to perform some heroic transformations to bridge this gap. The tools also struggle with more complex CUDA applications, in particular, those that use templates, classes, or other C++ features inside the kernel.

What hardware does HIP support?
-------------------------------

- For a list of AMD-supported platforms, refer to the HIP Programming Guide.  
-  For Nvidia platforms, HIP requires Unified Memory and should run on any device supporting CUDA SDK 6.0 or newer. We have tested the NVIDIA Titan and Tesla K40.

Do HIPIFY tools automatically convert all source code?
------------------------------------------------------

Typically, HIPIFY tools can automatically convert almost all run-time code, and the coordinate indexing device code ( threadIdx.x -> hipThreadIdx_x ). Most device code needs no additional conversion since HIP and CUDA have similar names for math and built-in functions. The hipify-clang tool will automatically modify the kernel signature as needed (automating a step that used to be done manually). Additional porting may be required to deal with architecture feature queries or with CUDA capabilities that HIP doesn't support. In general, developers should always expect to perform some platform-specific tuning and optimization.

What is NVCC?
-------------

NVCC is Nvidia's compiler driver for compiling CUDA C++ code into PTX or device code for Nvidia GPUs. It's a closed-source binary compiler that is provided by the CUDA SDK.


What is HIP-Clang?
------------------
HIP-Clang is a Clang/LLVM based compiler to compile HIP programs, which can run on the AMD platform.


Why use HIP rather than supporting CUDA directly?
-------------------------------------------------

While HIP is a strong subset of the CUDA, it is a subset. The HIP layer allows that subset to be clearly defined and documented. Developers who code to the HIP API can be assured their code will remain portable across Nvidia and AMD platforms. In addition, HIP defines portable mechanisms to query architectural features and supports a larger 64-bit wavesize which expands the return type for cross-lane functions like ballot and shuffle from 32-bit ints to 64-bit ints.

Can I develop HIP code on NVIDIA CUDA platform?
--------------------------------------------------

Yes. HIP's CUDA path only exposes the APIs and functionality that work on both NVCC and AMDGPU back-ends. APIs, parameters, and features which exist in CUDA but not in HIP-Clang will typically result in compile-time or run-time errors. Developers need to use the HIP API for most accelerator code and bracket any CUDA-specific code with preprocessor conditionals. Developers concerned about portability should, of course, run on both platforms, and should expect to tune for performance. In some cases, CUDA has a richer set of modes for some APIs, and some C++ capabilities such as virtual functions - see the HIP @API documentation for more details.

Can I develop HIP code on an AMD HIP-Clang platform?
----------------------------------------------------

Yes. HIP-Clang path only exposes the APIs and functions that work on AMD runtime back ends. APIs, parameters, and features that appear in HIP-Clang but not CUDA will typically cause compile or run-time errors. Developers must use the HIP API for most accelerator code and bracket any HIP-Clang specific code with preprocessor conditionals. Those concerned about portability should, of course, test their code on both platforms and should tune it for performance.

Typically, HIP-Clang supports a more modern set of C++11/C++14/C++17 features, so HIP developers who want portability should be careful when using advanced C++ features on the HIP-Clang path. 

How to use HIP-Clang to build HIP programs?
-------------------------------------------

The environment variable can be used to set compiler path:  

- HIP_CLANG_PATH: path to hip-clang. When set, this variable let hipcc to use hip-clang for compilation/linking. 

There is an alternative environment variable to set compiler path: 

- HIP_ROCCLR_HOME: path to root directory of the HIP-ROCclr runtime. When set, this variable let hipcc use hip-clang from the ROCclr distribution.

NOTE: If HIP_ROCCLR_HOME is set, there is no need to set HIP_CLANG_PATH since hipcc will deduce them from HIP_ROCCLR_HOME.

What is ROCclr?
---------------

ROCclr (Radeon Open Compute Common Language Runtime) is a virtual device interface that compute runtimes interact with backends such as ROCr on Linux, as well as PAL on Windows.

Can a HIP binary run on both AMD and NVIDIA platforms?
------------------------------------------------------

HIP is a source-portable language that can be compiled to run on either AMD or NVIDIA platform. HIP tools don't create a fat binary that can run on either platform.



Linking HIP code with host code compiled with another compiler on HIP Clang
----------------------------------------------------------------------------------

Yes. HIP generates the object code which conforms to the GCC ABI, and also links with libstdc++. This means you can compile host code with the compiler of your choice and link the generated object code with GPU code compiled with HIP. Larger projects often contain a mixture of accelerator code (initially written in CUDA with nvcc) and host code (compiled with gcc, icc, or clang). These projects can convert the accelerator code to HIP, compile that code with hipcc, and link with object code from their preferred compiler.

Installing CUDA SDK and HIP-Clang on the same machine
---------------------------------------------------------

Yes. You can use HIP_PLATFORM to choose which path hipcc targets. This configuration can be useful when using HIP to develop an application which is portable to both AMD and NVIDIA.

HIP detects my platform incorrectly
--------------------------------------
HIP sets the platform to AMD and use HIP-Clang as the compiler if the AMD graphics driver is installed and has detected an AMD GPU.

If this is not what you want, you can force HIP to recognize the platform by setting the following,

::

         export HIP_PLATFORM=amd


HIP then sets and uses the correct AMD compiler and runtime:

*HIP_COMPILER=clang* 

*HIP_RUNTIME=rocclr*

To choose the NVIDIA platform, you can set,

::

      export HIP_PLATFORM=nvidia


In this case, HIP will set and use the following,

*HIP_COMPILER=nvcc*

*HIP_RUNTIME=cuda*


A symptom of this problem is the error message: 

*'an unknown error(11) at square.hipref.cpp:56'*

This error can occur if you have a CUDA installation on an AMD platform, and HIP incorrectly detects the platform as nvcc.
HIP may be able to compile the application using the nvcc tool-chain, however, it will generate this error at runtime as the platform does not have a CUDA device. 

On CUDA, can I mix CUDA code with HIP code?
-------------------------------------------

Yes. Most HIP data structures (hipStream_t, hipEvent_t) are typedefs to CUDA equivalents and can be intermixed. Both CUDA and HIP use integer device ids. One notable exception is that hipError_t is a new type, and cannot be used where a cudaError_t is expected. In these cases, refactor the code to remove the expectation. Alternatively, hip_runtime_api.h defines functions which convert between the error code spaces:

hipErrorToCudaError hipCUDAErrorTohipError hipCUResultTohipError

If platform portability is important, use #ifdef **HIP_PLATFORM_NVCC** to guard the CUDA-specific code.



How do I trace HIP application flow?
------------------------------------

See the `HIP Profiling Guide <hip_porting_guide.md>`__ for more information.


Maximum limit of generic kernel launching parameter
----------------------------------------------------

Product of block.x, block.y, and block.z should be less than 1024.

Shuffle functions supported on HIP platform
-------------------------------------------------------

__shfl_*_sync is not supported on HIP but for NVCC path CUDA 9.0. Above all, shuffle calls get redirected to its sync version.

How to create a guard for code that is specific to the host or the GPU?
--------------------------------------------------------------------------
The compiler defines the `__HIP_DEVICE_COMPILE__` macro only when compiling the code for the GPU.  It could be used to guard code that is specific to the host or the GPU.

OpenMP is undefined when compiling with fopenmp
-------------------------------------------------------

When compiling an OpenMP source file with `hipcc -fopenmp`, the compiler may generate an error if there is a reference to the `_OPENMP` macro.  This is due to a limitation in hipcc that treats any source file type (e.g., `.cpp`) as HIP translation unit leading to some conflicts with the OpenMP language switch.  If the OpenMP source file doesn't contain any HIP language construct, you could work around this issue by adding the `-x c++` switch to force the compiler to treat the file as regular C++.  

Another approach would be to guard the OpenMP code with `#ifdef _OPENMP` so that the code block is disabled when compiling for the GPU.  The `__HIP_DEVICE_COMPILE__` macro defined by the HIP compiler when compiling the GPU code could also be used for guarding code paths specific to the host or the GPU.

Does the HIP-Clang compiler support extern shared declarations?
-----------------------------------------------------------------

Previously, it was required to declare dynamic shared memory using the HIP_DYNAMIC_SHARED macro for accuracy, as using static shared memory in the same kernel could result in overlapping memory ranges and data-races.

Now, the HIP-Clang compiler provides support for extern shared declarations, and the HIP_DYNAMIC_SHARED option is no longer required. You may use the standard extern definition:

*extern __shared__ type var[];*

How is the HIP version defined?
---------------------------------

The HIP version definition is updated since the ROCm v4.2 release as follows:

::


      HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)


The HIP version can be queried from the following HIP API call, 

::

      hipRuntimeGetVersion(&runtimeVersion);


The version returned will always be greater than the versions in previous ROCm releases.

**NOTE**: The version definition of HIP runtime is different from CUDA. On the AMD platform, the function returns HIP runtime version, while on the NVIDIA platform, it returns CUDA runtime version. There is no mapping or a correlation between HIP version and CUDA version.





