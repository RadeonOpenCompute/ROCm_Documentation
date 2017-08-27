.. _HIP-FAQ:

*********
HIP Programing FAQ
*********


What APIs and features does HIP support?
###################

HIP provides the following:

* Devices (hipSetDevice(), hipGetDeviceProperties(), etc.)
* Memory management (hipMalloc(), hipMemcpy(), hipFree(), etc.)
* Streams (hipStreamCreate(),hipStreamSynchronize(), hipStreamWaitEvent(),  etc.)
* Events (hipEventRecord(), hipEventElapsedTime(), etc.)
* Kernel launching (hipLaunchKernel is a standard C/C++ function that replaces <<< >>>)
* HIP Module API to control when adn how code is loaded.
* CUDA*style kernel coordinate functions (threadIdx, blockIdx, blockDim, gridDim)
* Cross*lane instructions including shfl, ballot, any, all
* Most device*side math built*ins
* Error reporting (hipGetLastError(), hipGetErrorString())

The HIP API documentation describes each API and its limitations, if any, compared with the equivalent CUDA API.

What is not supported?
################

Runtime/Driver API features
*******************

At a high*level, the following features are not supported:

* Textures 
* Dynamic parallelism (CUDA 5.0)
* Managed memory (CUDA 6.5)
* Graphics interoperability with OpenGL or Direct3D
* CUDA Driver API
* CUDA IPC Functions (Under Development)
* CUDA array, mipmappedArray and pitched memory
* MemcpyToSymbol functions
* Queue priority controls

See the [API Support Table](CUDA_Runtime_API_functions_supported_by_HIP.md) for more detailed information.

Kernel language features
**************************

* Device*side dynamic memory allocations (malloc, free, new, delete) (CUDA 4.0)
* Virtual functions, indirect functions and try/catch (CUDA 4.0)
* `__prof_trigger` 
* PTX assembly (CUDA 4.0).  HCC supports inline GCN assembly.
* Several kernel features are under development.  See the `HIP Kernel Language <hip_kernel_language.md>`_ for more information.  

These include

  * printf
  * assert
  * `__restrict__`
  * `__threadfence*_`, `__syncthreads*`
  * Unbounded loop unroll



Is HIP a drop*in replacement for CUDA?
******************************

No. HIP provides porting tools which do most of the work to convert CUDA code into portable C++ code that uses the HIP APIs.
Most developers will port their code from CUDA to HIP and then maintain the HIP version. 
HIP code provides the same performance as native CUDA code, plus the benefits of running on AMD platforms.

What specific version of CUDA does HIP support?
*************************************

HIP APIs and features do not map to a specific CUDA version. HIP provides a strong subset of functionality provided in CUDA, and the hipify tools can 
scan code to identify any unsupported CUDA functions * this is useful for identifying the specific features required by a given application.

However, we can provide a rough summary of the features included in each CUDA SDK and the support level in HIP:

* CUDA 4.0 and earlier :  
    * HIP supports CUDA 4.0 except for the limitations described above.
* CUDA 5.0 : 
    * Dynamic Parallelism (not supported) 
    * cuIpc functions (under development).
* CUDA 5.5 : 
    * CUPTI (not directly supported), `AMD GPUPerfAPI <http://developer.amd.com/tools*and*sdks/graphics*development/gpuperfapi/>`_ can be used as an alternative in some cases)
* CUDA 6.0
    * Managed memory (under development)
* CUDA 6.5
    * __shfl instriniscs (supported)
* CUDA 7.0
    * Per*thread*streams (under development)
    * C++11 (HCC supports all of C++11, all of C++14 and some C++17 features)
* CUDA 7.5
    * float16
* CUDA 8.0
    * TBD.

What libraries does HIP support?
*****************************

HIP includes growing support for the 4 key math libraries using hcBlas, hcFft, hcrng and hcsparse.
These offer pointer*based memory interfaces (as opposed to opaque buffers) and can be easily interfaced with other HCC applications.  Developers should use conditional compilation if portability to nvcc systems is desired * using calls to cu* routines on one path and hc* routines on the other.  

* `rocblas <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_
* `rocfft <https://github.com/ROCmSoftwarePlatform/rocFFT>`_
* `MIOpen <https://github.com/ROCmSoftwarePlatform/MIOpen>`_
* hipRAND Under Development 
   
Additionally, some of the cublas routines are automatically converted to hipblas equivalents by the hipify*clang tool.  These APIs use cublas or hcblas depending on the platform, and replace the need
to use conditional compilation. 

How does HIP compare with OpenCL?
*****************************

Both AMD and Nvidia support OpenCL 1.2 on their devices, so developers can write portable code.
HIP offers several benefits over OpenCL:

* Developers can code in C++ as well as mix host and device C++ code in their source files. HIP C++ code can use templates, lambdas, classes and so on.
* The HIP API is less verbose than OpenCL and is familiar to CUDA developers.
* Because both CUDA and HIP are C++ languages, porting from CUDA to HIP is significantly easier than porting from CUDA to OpenCL.
* HIP uses the best available development tools on each platform: on Nvidia GPUs, HIP code compiles using NVCC and can employ the nSight profiler and debugger (unlike OpenCL on Nvidia GPUs).
* HIP provides pointers and host*side pointer arithmetic.
* HIP provides device*level control over memory allocation and placement.
* HIP offers an offline compilation model.

How does porting CUDA to HIP compare to porting CUDA to OpenCL?
*****************************

Both HIP and CUDA are dialects of C++, and thus porting between them is relatively straightforward.

Both dialects support templates, classes, lambdas, and other C++ constructs.

As one example, the hipify tool was originally a Perl script that used simple text conversions from CUDA to HIP.
HIP and CUDA provide similar math library calls as well.  In summary, the HIP philosophy was to make the HIP language close enough to CUDA that the porting effort is relatively simple.

This reduces the potential for error, and also makes it easy to automate the translation.  HIP's goal is to quickly get the ported program running on both platforms with little manual intervention,
so that the programmer can focus on performance optimizations.

There have been several tools that have attempted to convert CUDA into OpenCL, such as CU2CL.  OpenCL is a C99*based kernel language (rather than C++) and also does not support single*source compilation.  
As a result, the OpenCL syntax is different from CUDA, and the porting tools have to perform some heroic transformations to bridge this gap.

The tools also struggle with more complex CUDA applications, in particular those that use templates, classes, or other C++ features inside the kernel.  


What hardware does HIP support?
*****************************

* For AMD platforms, HIP runs on the same hardware that the HCC "hc" mode supports.  See the ROCm documentation for the list of supported platforms.
* For Nvidia platforms, HIP requires Unified Memory and should run on any device supporting CUDA SDK 6.0 or newer. We have tested the Nvidia Titan and Tesla K40.

Does Hipify automatically convert all source code?
*****************************

Typically, hipify can automatically convert almost all run*time code, and the coordinate indexing device code ( threadIdx.x *> hipThreadIdx_x ).  

Most device code needs no additional conversion, since HIP and CUDA have similar names for math and built*in functions. 
The hipify*clang tool will automatically modify the kernel signature as needed (automating a step that used to be done manually)

Additional porting may be required to deal with architecture feature queries or with CUDA capabilities that HIP doesn't support. 

In general, developers should always expect to perform some platform*specific tuning and optimization.

What is NVCC?
*****************************

NVCC is Nvidia's compiler driver for compiling "CUDA C++" code into PTX or device code for Nvidia GPUs. It's a closed*source binary compiler that is provided by the CUDA SDK.

What is HCC?
*****************************

HCC is AMD's compiler driver which compiles "heterogeneous C++" code into HSAIL or GCN device code for AMD GPUs.  It's an open*source compiler based on recent versions of CLANG/LLVM.

Why use HIP rather than supporting CUDA directly?
*****************************
While HIP is a strong subset of the CUDA, it is a subset.  The HIP layer allows that subset to be clearly defined and documented.

Developers who code to the HIP API can be assured their code will remain portable across Nvidia and AMD platforms.  
In addition, HIP defines portable mechanisms to query architectural features, and supports a larger 64*bit wavesize which expands the return type for cross*lane functions like ballot and shuffle from 32*bit ints to 64*bit ints.  

Can I develop HIP code on an Nvidia CUDA platform?
*****************************

Yes.  HIP's CUDA path only exposes the APIs and functionality that work on both NVCC and HCC back*ends.
"Extra" APIs, parameters, and features which exist in CUDA but not in HCC will typically result in compile* or run*time errors.

Developers need to use the HIP API for most accelerator code, and bracket any CUDA*specific code with preprocessor conditionals.

Developers concerned about portability should of course run on both platforms, and should expect to tune for performance.
In some cases CUDA has a richer set of modes for some APIs, and some C++ capabilities such as virtual functions * see the HIP @API documentation for more details.

Can I develop HIP code on an AMD HCC platform?
*****************************

Yes. HIP's HCC path only exposes the APIs and functions that work on both NVCC and HCC back ends. "Extra" APIs, parameters and features that appear in HCC but not CUDA will typically cause compile* or run*time errors. Developers must use the HIP API for most accelerator code and bracket any HCC*specific code with preprocessor conditionals. 

Those concerned about portability should, of course, test their code on both platforms and should tune it for performance. Typically, HCC supports a more modern set of C++11/C++14/C++17 features, so HIP developers who want portability should be careful when using advanced C++ features on the hc path.


