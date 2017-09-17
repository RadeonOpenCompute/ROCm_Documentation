---
layout: default
title: ROCm Languages
---

## ROCm, Lingua Franca,  C++, OpenCL and Python

The open-source ROCm stack offers multiple programming-language
choices. The goal is to give you a range of tools to help solve the
problem at hand. Here, we describe some of the options and how to
choose among them.



### HCC: Heterogeneous Compute Compiler

What is the Heterogeneous Compute (HC) API? It’s a C++ dialect with
extensions to launch kernels and manage accelerator memory. It closely
tracks the evolution of C++ and will incorporate parallelism and
concurrency features as the C++ standard does. For example, HC
includes early support for the C++17 Parallel STL. At the recent ISO
C++ meetings in Kona and Jacksonville, the committee was excited about
enabling the language to express all forms of parallelism, including
multicore CPU, SIMD and GPU. We’ll be following these developments
closely, and you’ll see HC move quickly to include standard C++
capabilities.

The Heterogeneous Compute Compiler (HCC) provides two important
benefits:

*Ease of development*

 * A full C++ API for managing devices, queues and events
 * C++ data containers that provide type safety,
   multidimensional-array indexing and automatic data management
 * C++ kernel-launch syntax using parallel_for_each plus C++11 lambda
   functions
 * A single-source C++ programming environment---the host and source
   code can be in the same source file and use the same C++ language;
   templates and classes work naturally across the host/device
   boundary
 * HCC generates both host and device code from the same compiler, so
   it benefits from a consistent view of the source code using the
   same Clang-based language parser

*Full control over the machine*

 * Access AMD scratchpad memories (“LDS”)
 * Fully control data movement, prefetch and discard
 * Fully control asynchronous kernel launch and completion
 * Get device-side dependency resolution for kernel and data commands (without host involvement)
 * Obtain HSA agents, queues and signals for low-level control of the architecture using the HSA Runtime API
 * Use [direct-to-ISA](https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA) compilation

#### When to Use HC

Use HC when you're targeting the AMD ROCm platform: it delivers a single-source, easy-to-program C++ environment without compromising
performance or control of the machine.

### HIP: Heterogeneous-Computing Interface for Portability

What is Heterogeneous-Computing Interface for Portability (HIP)? It’s
a C++ dialect designed to ease conversion of Cuda applications to
portable C++ code. It provides a C-style API and a C++ kernel
language. The C++ interface can use templates and classes across the
host/kernel boundary.

The Hipify tool automates much of the conversion work by performing a
source-to-source transformation from Cuda to HIP. HIP code can run on
AMD hardware (through the HCC compiler) or Nvidia hardware (through
the NVCC compiler) with no performance loss compared with the original
Cuda code.

Programmers familiar with other GPGPU languages will find HIP very
easy to learn and use. AMD platforms implement this language using the
HC dialect described above, providing similar low-level control over
the machine.

#### When to Use HIP

Use HIP when converting Cuda applications to portable C++ and for new
projects that require portability between AMD and Nvidia. HIP provides
a C++ development language and access to the best development tools on
both platforms.

### OpenCL™: Open Compute Language

What is OpenCL? It’s a framework for developing programs that can
execute across a wide variety of heterogeneous platforms. AMD, Intel
and Nvidia GPUs support version 1.2 of the specification, as do x86
CPUs and other devices (including FPGAs and DSPs). OpenCL provides a C
run-time API and C99-based kernel language.

#### When to Use OpenCL

Use OpenCL when you have existing code in that language and when you
need portability to multiple platforms and devices. It runs on
Windows, Linux and Mac OS, as well as a wide variety of hardware
platforms (described above).

### Anaconda Python With Numba

What is Anaconda? It’s a modern open-source analytics platform powered
by Python. Continuum Analytics, a ROCm platform partner,  is the
driving force behind it. Anaconda delivers high-performance
capabilities including acceleration of HSA APUs, as well as
ROCm-enabled discrete GPUs via Numba. It gives superpowers to the
people who are changing the world.

### Numba

Numba gives you the power to speed up your applications with
high-performance functions written directly in Python. Through a few
annotations, you can just-in-time compile array-oriented and
math-heavy Python code to native machine instructions---offering
performance similar to that of C, C++ and Fortran---without having to
switch languages or Python interpreters.

Numba works by generating optimized machine code using the LLVM
compiler infrastructure at import time, run time or statically
(through the included Pycc tool). It supports Python compilation to
run on either CPU or GPU hardware and is designed to integrate with
Python scientific software stacks, such as NumPy.

#### When to Use Anaconda

Use Anaconda when you’re handling large-scale data-analytics,
scientific and engineering problems that require you to manipulate
large data arrays.

### Wrap-Up

From a high-level perspective, ROCm delivers a rich set of tools that
allow you to choose the best language for your application.

 * HCC (Heterogeneous Compute Compiler) supports HC dialects
 * HIP is a run-time library that layers on top of HCC (for AMD ROCm platforms; for Nvidia, it uses the NVCC compiler)
 * The following will soon offer native compiler support for the GCN ISA:
    * OpenCL 1.2+
    * Anaconda (Python) with Numba

All are open-source projects, so you can employ a fully open stack
from the language down to the metal. AMD is committed to providing an
open ecosystem that gives developers the ability to choose; we are
excited about innovating quickly using open source and about
interacting closely with our developer community. More to come soon!

# Table Comparing Syntax for Different Compute APIs

|Term|CUDA|HIP|HC|C++AMP|OpenCL|
|---|---|---|---|---|---|
|Device|`int deviceId`|`int deviceId`|`hc::accelerator`|`concurrency::`<br>`accelerator`|`cl_device`
|Queue|`cudaStream_t`|`hipStream_t`|`hc::`<br>`accelerator_view`|`concurrency::`<br>`accelerator_view`|`cl_command_queue`
|Event|`cudaEvent_t`|`hipEvent_t`|`hc::`<br>`completion_future`|`concurrency::`<br>`completion_future`|`cl_event`
|Memory|`void *`|`void *`|`void *`; `hc::array`; `hc::array_view`|`concurrency::array`;<br>`concurrency::array_view`|`cl_mem`
|||||
| |grid|grid|extent|extent|NDRange
| |block|block|tile|tile|work-group
| |thread|thread|thread|thread|work-item
| |warp|warp|wavefront|N/A|sub-group
|||||
|Thread-<br>index | threadIdx.x | hipThreadIdx_x | t_idx.local[0] | t_idx.local[0] | get_local_id(0) |
|Block-<br>index  | blockIdx.x  | hipBlockIdx_x  | t_idx.tile[0]  | t_idx.tile[0]  | get_group_id(0) |
|Block-<br>dim    | blockDim.x  | hipBlockDim_x  | t_ext.tile_dim[0]| t_idx.tile_dim0 | get_local_size(0) |
|Grid-dim     | gridDim.x   | hipGridDim_x   | t_ext[0]| t_ext[0] | get_global_size(0) |
|||||
|Device Kernel|`__global__`|`__global__`|lambda inside `hc::`<br>`parallel_for_each` or [[hc]]|`restrict(amp)`|`__kernel`
|Device Function|`__device__`|`__device__`|`[[hc]]` (detected automatically in many case)|`restrict(amp)`|Implied in device compilation
|Host Function|`__host_` (default)|`__host_` (default)|`[[cpu]]`  (default)|`restrict(cpu)` (default)|Implied in host compilation.
|Host + Device Function|`__host__` `__device__`|`__host__` `__device__`|  `[[hc]]` `[[cpu]]`|`restrict(amp,cpu)`|No equivalent
|Kernel Launch|`<<< >>>`|`hipLaunchKernel`|`hc::`<br>`parallel_for_each`|`concurrency::`<br>`parallel_for_each`|`clEnqueueNDRangeKernel`
||||||
|Global Memory|`__global__`|`__global__`|Unnecessary / Implied|Unnecessary / Implied|`__global`
|Group Memory|`__shared__`|`__shared__`|`tile_static`|`tile_static`|`__local`
|Constant|`__constant__`|`__constant__`|Unnecessary / Implied|Unnecessary / Implied|`__constant`
||||||
||`__syncthreads`|`__syncthreads`|`tile_static.barrier()`|`t_idx.barrier()`|`barrier(CLK_LOCAL_MEMFENCE)`
|Atomic Builtins|`atomicAdd`|`atomicAdd`|`hc::atomic_fetch_add`|`concurrency::`<br>`atomic_fetch_add`|`atomic_add`
|Precise Math|`cos(f)`|`cos(f)`|`hc::`<br>`precise_math::cos(f)`|`concurrency::`<br>`precise_math::cos(f)`|`cos(f)`
|Fast Math|`__cos(f)`|`__cos(f)`|`hc::`<br>`fast_math::cos(f)`|`concurrency::`<br>`fast_math::cos(f)`|`native_cos(f)`
|Vector|`float4`|`float4`|`hc::`<br>`short_vector::float4`|`concurrency::`<br>`graphics::float_4`|`float4`

###Notes
1. For HC and C++AMP, assume a captured _tiled_ext_ named "t_ext" and captured _extent_ named "ext".  These languages use captured variables to pass information to the kernel rather than using special built-in functions so the exact variable name may vary.
2. The indexing functions (starting with `thread-index`) show the terminology for a 1D grid.  Some APIs use reverse order of xyz / 012 indexing for 3D grids.
3. HC allows tile dimensions to be specified at runtime while C++AMP requires that tile dimensions be specified at compile-time.  Thus hc syntax for tile dims is `t_ext.tile_dim[0]` while C++AMP is t_ext.tile_dim0.


