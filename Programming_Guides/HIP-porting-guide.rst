.. _HIP-porting-guide:



HIP Porting Guide
=================

In addition to providing a portable C++ programming environment for
GPUs, HIP is designed to ease the porting of existing CUDA code into the
HIP environment. This section describes the available tools and provides
practical suggestions on how to port CUDA code and work through common
issues.


-  `Porting a New CUDA Project <#porting-a-new-cuda-project>`__

   -  `General Tips <#general-tips>`__
   -  `Scanning existing CUDA code to scope the porting
      effort <#scanning-existing-cuda-code-to-scope-the-porting-effort>`__
   -  `Converting a project
      in-place <#converting-a-project-in-place>`__
   -  `CUDA to HIP Math Library Equivalents <#library-equivalents>`__

-  `Distinguishing Compiler Modes <#distinguishing-compiler-modes>`__

   -  `Identifying HIP Target
      Platform <#identifying-hip-target-platform>`__
   -  `Identifying the Compiler: hip-clang, or
      nvcc <#identifying-the-compiler-hip-clang-or-nvcc>`__
   -  `Identifying Current Compilation Pass: Host or
      Device <#identifying-current-compilation-pass-host-or-device>`__
   -  `Compiler Defines: Summary <#compiler-defines-summary>`__

-  `Identifying Architecture
   Features <#identifying-architecture-features>`__

   -  `HIP_ARCH Defines <#hip_arch-defines>`__
   -  `Device-Architecture
      Properties <#device-architecture-properties>`__
   -  `Table of Architecture
      Properties <#table-of-architecture-properties>`__

-  `Finding HIP <#finding-hip>`__
-  `Identifying HIP Runtime <#identifying-hip-runtime>`__
-  `hipLaunchKernel <#hiplaunchkernel>`__
-  `Compiler Options <#compiler-options>`__
-  `Linking Issues <#linking-issues>`__

   -  `Linking With hipcc <#linking-with-hipcc>`__
   -  `-lm Option <#-lm-option>`__

-  `Linking Code With Other
   Compilers <#linking-code-with-other-compilers>`__

   -  `libc++ and libstdc++ <#libc-and-libstdc>`__
   -  `HIP Headers (hip_runtime.h,
      hip_runtime_api.h) <#hip-headers-hip_runtimeh-hip_runtime_apih>`__
   -  `Using a Standard C++ Compiler <#using-a-standard-c-compiler>`__

      -  `cuda.h <#cudah>`__

   -  `Choosing HIP File Extensions <#choosing-hip-file-extensions>`__

-  `Workarounds <#workarounds>`__

   -  `warpSize <#warpsize>`__
   -  `Kernel launch with group size >
      256 <#kernel-launch-with-group-size--256>`__

-  `memcpyToSymbol <#memcpytosymbol>`__
-  `CU Pointer Attribute Memory Type  <#CU-Pointer-Attribute-Memory-Type>`__
-  `threadfence_system <#threadfence_system>`__

   -  `Textures and Cache Control <#textures-and-cache-control>`__

-  `More Tips <#more-tips>`__

   -  `HIP Logging <#hip-logging>`__
   -  `Debugging hipcc <#debugging-hipcc>`__
   -  `What Does This Error Mean? <#what-does-this-error-mean>`__

      -  `/usr/include/c++/v1/memory:5172:15: error: call to implicitly
         deleted default constructor of std::__1::bad_weak_ptr throw
         bad_weak_ptr(); <#usrincludecv1memory517215-error-call-to-implicitly-deleted-default-constructor-of-std__1bad_weak_ptr-throw-bad_weak_ptr>`__

   -  `Editor Highlighting <#editor-highlighting>`__

.. raw:: html

   <!-- tocstop -->

Porting a New CUDA Project
--------------------------

General Tips
~~~~~~~~~~~~

-  Starting the port on a CUDA machine is often the easiest approach,
   since you can incrementally port pieces of the code to HIP while
   leaving the rest in CUDA. (Recall that on CUDA machines HIP is just a
   thin layer over CUDA, so the two code types can interoperate on nvcc
   platforms.) Also, the HIP port can be compared with the original CUDA
   code for function and performance.
-  Once the CUDA code is ported to HIP and is running on the CUDA
   machine, compile the HIP code using the HIP compiler on an AMD
   machine.
-  HIP ports can replace CUDA versions: HIP can deliver the same
   performance as a native CUDA implementation, with the benefit of
   portability to both Nvidia and AMD architectures as well as a path to
   future C++ standard support. You can handle platform-specific
   features through conditional compilation or by adding them to the
   open-source HIP infrastructure.
-  Use
   `bin/hipconvertinplace-perl.sh <https://github.com/ROCm-Developer-Tools/HIP/blob/master/bin/hipconvertinplace-perl.sh>`__
   to hipify all code files in the CUDA source directory.

Scanning existing CUDA code to scope the porting effort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hipexamine-perl.sh tool will scan a source directory to determine
which files contain CUDA code and how much of that code can be
automatically hipified.

::

   > cd examples/rodinia_3.0/cuda/kmeans
   > $HIP_DIR/bin/hipexamine-perl.sh.
   info: hipify ./kmeans.h =====>
   info: hipify ./unistd.h =====>
   info: hipify ./kmeans.c =====>
   info: hipify ./kmeans_cuda_kernel.cu =====>
     info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:3 other:0 ) warn:0 LOC:185
   info: hipify ./getopt.h =====>
   info: hipify ./kmeans_cuda.cu =====>
     info: converted 49 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:0 math:0 stream:0 event:0 err:0 def:0 tex:12 other:0 ) warn:0 LOC:311
   info: hipify ./rmse.c =====>
   info: hipify ./cluster.c =====>
   info: hipify ./getopt.c =====>
   info: hipify ./kmeans_clustering.c =====>
   info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
     kernels (1 total) :   kmeansPoint(1)

hipexamine-perl scans each code file (cpp, c, h, hpp, etc.) found in the
specified directory:

-  Files with no CUDA code (ie kmeans.h) print one line summary just
   listing the source file name.
-  Files with CUDA code print a summary of what was found - for example
   the kmeans_cuda_kernel.cu file:

::

   info: hipify ./kmeans_cuda_kernel.cu =====>
     info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 

-  Interesting information in kmeans_cuda_kernel.cu :

   -  How many CUDA calls were converted to HIP (40)
   -  Breakdown of the CUDA functionality used (dev:0 mem:0 etc). This
      file uses many CUDA builtins (37) and texture functions (3).
   -  Warning for code that looks like CUDA API but was not converted (0
      in this file).
   -  Count Lines-of-Code (LOC) - 185 for this file.

-  hipexamine-perl also presents a summary at the end of the process for
   the statistics collected across all files. This has similar format to
   the per-file reporting, and also includes a list of all kernels which
   have been called. An example from above:

.. code:: shell

   info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
     kernels (1 total) :   kmeansPoint(1)

Converting a project in-place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   > hipify-perl --inplace

For each input file FILE, this script will: - If FILE.prehip file does
not exist, copy the original code to a new file with extension.prehip.
Then hipify the code file. - IfFILE.prehip" file exists, hipify
FILE.prehip and save to FILE.

This is useful for testing improvements to the hipify toolset.

The
`hipconvertinplace-perl.sh <https://github.com/ROCm-Developer-Tools/HIP/blob/master/bin/hipconvertinplace-perl.sh>`__
script will perform inplace conversion for all code files in the
specified directory. This can be quite handy when dealing with an
existing CUDA code base since the script preserves the existing
directory structure and filenames - and includes work. After converting
in-place, you can review the code to add additional parameters to
directory names.

.. code:: shell

   > hipconvertinplace-perl.sh MY_SRC_DIR

Library Equivalents
~~~~~~~~~~~~~~~~~~~

+-----------------------+-----------------------------+----------------+
| CUDA Library          | ROCm Library                | Comment        |
+=======================+=============================+================+
| cuBLAS                | rocBLAS                     | Basic Linear   |
|                       |                             | Algebra        |
|                       |                             | Subroutines    |
+-----------------------+-----------------------------+----------------+
| cuFFT                 | rocFFT                      | Fast Fourier   |
|                       |                             | Transfer       |
|                       |                             | Library        |
+-----------------------+-----------------------------+----------------+
| cuSPARSE              | rocSPARSE                   | Sparse BLAS +  |
|                       |                             | SPMV           |
+-----------------------+-----------------------------+----------------+
| cuSolver              | rocSOLVER                   | Lapack library |
+-----------------------+-----------------------------+----------------+
| AMG-X                 | rocALUTION                  | Sparse         |
|                       |                             | iterative      |
|                       |                             | solvers and    |
|                       |                             | p              |
|                       |                             | reconditioners |
|                       |                             | with Geometric |
|                       |                             | and Algebraic  |
|                       |                             | MultiGrid      |
+-----------------------+-----------------------------+----------------+
| Thrust                | rocThrust                   | C++ parallel   |
|                       |                             | algorithms     |
|                       |                             | library        |
+-----------------------+-----------------------------+----------------+
| CUB                   | rocPRIM                     | Low Level      |
|                       |                             | Optimized      |
|                       |                             | Parallel       |
|                       |                             | Primitives     |
+-----------------------+-----------------------------+----------------+
| cuDNN                 | MIOpen                      | Deep learning  |
|                       |                             | Solver Library |
+-----------------------+-----------------------------+----------------+
| cuRAND                | rocRAND                     | Random Number  |
|                       |                             | Generator      |
|                       |                             | Library        |
+-----------------------+-----------------------------+----------------+
| EIGEN                 | EIGEN â€“ HIP port            | C++ template   |
|                       |                             | library for    |
|                       |                             | linear         |
|                       |                             | algebra:       |
|                       |                             | matrices,      |
|                       |                             | vectors,       |
|                       |                             | numerical      |
|                       |                             | solvers,       |
+-----------------------+-----------------------------+----------------+
| NCCL                  | RCCL                        | Communications |
|                       |                             | Primitives     |
|                       |                             | Library based  |
|                       |                             | on the MPI     |
|                       |                             | equivalents    |
+-----------------------+-----------------------------+----------------+

Distinguishing Compiler Modes
-----------------------------

Identifying HIP Target Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All HIP projects target either AMD or NVIDIA platform. The platform
affects which headers are included and which libraries are used for
linking.

-  ``HIP_PLATFORM_HCC`` is defined if the HIP platform targets AMD

-  ``HIP_PLATFORM_NVCC`` is defined if the HIP platform targets NVIDIA

On AMD platform, the compiler was hcc, but is deprecated in ROCM v3.5
release, and HIP-Clang compiler is introduced for compiling HIP
programs.

For most HIP applications, the transition from hcc to HIP-Clang is
transparent. HIPCC and HIP cmake files automatically choose compilation
options for HIP-Clang and hide the difference between the hcc and
hip-clang code. However, minor changes may be required as HIP-Clang has
stricter syntax and semantic checks compared to hcc.

Many projects use a mixture of an accelerator compiler (AMD or NVIDIA)
and a standard compiler (e.g.Â g++). These defines are set for both
accelerator and standard compilers and thus are often the best option
when writing code that uses conditional compilation.

Identifying the Compiler: hip-clang or nvcc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, it is useful to know whether the underlying compiler is HIP-Clang
or nvcc. This knowledge can guard platform-specific code or aid in
platform-specific performance tuning.

::

   #ifdef __HIP_PLATFORM_HCC__
   // Compiled with HIP-Clang

::

   #if defined(__HCC__) || (defined(__clang__) && defined(__HIP__))
   #define __HIP_PLATFORM_HCC__
   #endif
   // Compiled with HIP-Clang

::

   #ifdef __NVCC__
   // Compiled with nvcc
   //  Could be compiling with CUDA language extensions enabled (for example, a ".cu file)
   //  Could be in pass-through mode to an underlying host compile OR (for example, a .cpp file)

::

   #ifdef __CUDACC__
   // Compiled with nvcc (CUDA language extensions enabled)

Compiler directly generates the host code (using the Clang x86 target)
and passes the code to another host compiler. Thus, they have no
equivalent of the \__CUDA_ACC define.

Identifying Current Compilation Pass: Host or Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nvcc makes two passes over the code: one for host code and one for
device code. HIP-Clang will have multiple passes over the code: one for
the host code, and one for each architecture on the device code.
``__HIP_DEVICE_COMPILE__`` is set to a nonzero value when the compiler
(HIP-Clang or nvcc) is compiling code for a device inside a
``__global__`` kernel or for a device function.
``__HIP_DEVICE_COMPILE__`` can replace #ifdef checks on the
``__CUDA_ARCH__`` define.

::

   // #ifdef __CUDA_ARCH__
   #if __HIP_DEVICE_COMPILE__

Unlike ``__CUDA_ARCH__``, the ``__HIP_DEVICE_COMPILE__`` value is 1 or
undefined, and it does not represent the feature capability of the target
device.

Compiler Defines: Summary
~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Define          | HIP-Clang       | nvcc            | Other (GCC,     |
|                 |                 |                 | ICC, Clang,     |
|                 |                 |                 | etc.)           |
+=================+=================+=================+=================+
| HIP-related     |                 |                 |                 |
| defines:        |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| ``__HIP_        |                 |                 |                 |
|PLATFORM_HCC__`` | Defined         | Undefined       | Defined if      |
|                 |                 | targeting AMD   |platform;        |
|                 |                 |                 | undefined       |
|                 |                 |                 | otherwise       |
+-----------------+-----------------+-----------------+-----------------+
| ``__HIP_        |                 |                 |                 |
|PLATFORM_NVCC__``| Undefined       | Defined         | Defined if      |
|                 |                 |                 | targeting nvcc  |
|                 |                 |                 | platform;       |
|                 |                 |                 | undefined       |
|                 |                 |                 | otherwise       |
+-----------------+-----------------+-----------------+-----------------+
| ``              |                 |                 |                 |
|__HIP_DEVICE     |                 |                 |                 |
|_COMPILE__``     | 1 if compiling  | 1 if compiling  | Undefined       |
|                 | for device;     | for device;     |                 |
|                 | undefined if    | undefined if    |                 |
|                 | compiling for   | compiling for   |                 |
|                 | host            | host            |                 |
+-----------------+-----------------+-----------------+-----------------+
| ``__HIPCC__``   | Defined         | Defined         | Undefined       |
+-----------------+-----------------+-----------------+-----------------+
| `               | 0 or 1          | 0 or 1          | 0               |
| `__HIP_ARCH_*`` | depending on    | depending on    |                 |
|                 | feature support | feature support |                 |
|                 | (see below)     | (see below)     |                 |
+-----------------+-----------------+-----------------+-----------------+
| nvcc-related    |                 |                 |                 |
| defines:        |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| ``__CUDACC__``  | Defined if      | Undefined       |                 |
|                 | source code is  |                 |                 |
|                 | compiled by     |                 |                 |
|                 | nvcc; undefined |                 |                 |
|                 | otherwise       |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| ``__NVCC__``    | Undefined       | Defined         | Undefined       |
+-----------------+-----------------+-----------------+-----------------+
| ``              | Undefined       | Unsigned        | Undefined       |
| __CUDA_ARCH__`` |                 | representing    |                 |
|                 |                 | compute         |                 |
|                 |                 | capability      |                 |
|                 |                 | (e.g.,130)      |                 |
|                 |                 | if in device    |                 |
|                 |                 | code; 0 if in   |                 |
|                 |                 | host code       |                 |
+-----------------+-----------------+-----------------+-----------------+
| hip-clang       |                 |                 |                 |
|related defines: |                 |                 |                 |
|                 |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| ``__HIP__``     | Defined         | Undefined       | Undefined       |
+-----------------+-----------------+-----------------+-----------------+
| HIP-Clang       |                 |                 |                 |
| common defines: |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| ``__clang__``   | Defined         | Defined         | Undefined       |
+-----------------+-----------------+-----------------+-----------------+

Identifying Architecture Features
---------------------------------

HIP_ARCH Defines
~~~~~~~~~~~~~~~~

Some CUDA code tests ``__CUDA_ARCH__`` for a specific value to determine
whether the machine supports a certain architectural feature. For
instance,

::

   #if (__CUDA_ARCH__ >= 130)
   // doubles are supported

This type of code requires special attention, since hcc/AMD and
nvcc/CUDA devices have different architectural capabilities. Moreover,
you cannnot determine the presence of a feature using a simple comparison
against an architecture's version number. HIP provides a set of defines
and device properties to query whether a specific architectural feature
is supported.

The ``__HIP_ARCH_*`` defines can replace comparisons of
``__CUDA_ARCH__`` values:

::

   //#if (__CUDA_ARCH__ >= 130)   // non-portable
   if __HIP_ARCH_HAS_DOUBLES__ {  // portable HIP feature query
      // doubles are supported
   }

For host code, the ``__HIP_ARCH__*`` defines are set to 0. You should
only use the **HIP_ARCH** fields in device code.

Device-Architecture Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host code should query the architecture feature flags in the device
properties that hipGetDeviceProperties returns, rather than testing the
'major'and 'minor'fields directly:

::

   hipGetDeviceProperties(&deviceProp, device);
   //if ((deviceProp.major == 1 && deviceProp.minor < 2))  // non-portable
   if (deviceProp.arch.hasSharedInt32Atomics) {            // portable HIP feature query
       // has shared int32 atomic operations ...
   }

Table of Architecture Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below shows the full set of architectural properties that HIP
supports.

+-----------------------+-----------------------------+----------------+
| Define (use only in   | Device Property (run-time   | Comment        |
| device code)          | query)                      |                |
+=======================+=============================+================+
| 32-bit atomics:       |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_GLO  | hasGlobalInt32Atomics       | 32-bit integer |
| BAL_INT32_ATOMICS__`` |                             | atomics for    |
|                       |                             | global memory  |
+-----------------------+-----------------------------+----------------+
| ``_                   | hasGlobalFloatAtomicExch    | 32-bit float   |
| _HIP_ARCH_HAS_GLOBAL_ |                             | atomic         |
| FLOAT_ATOMIC_EXCH__`` |                             | exchange for   |
|                       |                             | global memory  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_SHA  | hasSharedInt32Atomics       | 32-bit integer |
| RED_INT32_ATOMICS__`` |                             | atomics for    |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| ``_                   | hasSharedFloatAtomicExch    | 32-bit float   |
| _HIP_ARCH_HAS_SHARED_ |                             | atomic         |
| FLOAT_ATOMIC_EXCH__`` |                             | exchange for   |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS      | hasFloatAtomicAdd           | 32-bit float   |
| _FLOAT_ATOMIC_ADD__`` |                             | atomic add in  |
|                       |                             | global and     |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| 64-bit atomics:       |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_GLO  | hasGlobalInt64Atomics       | 64-bit integer |
| BAL_INT64_ATOMICS__`` |                             | atomics for    |
|                       |                             | global memory  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_SHA  | hasSharedInt64Atomics       | 64-bit integer |
| RED_INT64_ATOMICS__`` |                             | atomics for    |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| Doubles:              |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP               | hasDoubles                  | Do             |
| _ARCH_HAS_DOUBLES__`` |                             | uble-precision |
|                       |                             | floating point |
+-----------------------+-----------------------------+----------------+
| Warp cross-lane       |                             |                |
| operations:           |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_A             | hasWarpVote                 | Warp vote      |
| RCH_HAS_WARP_VOTE__`` |                             | instructions   |
|                       |                             | (any, all)     |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARC           | hasWarpBallot               | Warp ballot    |
| H_HAS_WARP_BALLOT__`` |                             | instructions   |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH          | hasWarpShuffle              | Warp shuffle   |
| _HAS_WARP_SHUFFLE__`` |                             | operations     |
|                       |                             | (shfl_*)       |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_     | hasFunnelShift              | Funnel shift   |
| WARP_FUNNEL_SHIFT__`` |                             | two input      |
|                       |                             | words into one |
+-----------------------+-----------------------------+----------------+
| Sync:                 |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_TH   | hasThreadFenceSystem        | thre           |
| READ_FENCE_SYSTEM__`` |                             | adfence_system |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HA       | hasSyncThreadsExt           | sync           |
| S_SYNC_THREAD_EXT__`` |                             | threads_count, |
|                       |                             | sy             |
|                       |                             | ncthreads_and, |
|                       |                             | syncthreads_or |
+-----------------------+-----------------------------+----------------+
| Miscellaneous:        |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_         | hasSurfaceFuncs             |                |
| HAS_SURFACE_FUNCS__`` |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HI                | has3dGrid                   | Grids and      |
| P_ARCH_HAS_3DGRID__`` |                             | groups are 3D  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS      | hasDynamicParallelism       |                |
| _DYNAMIC_PARALLEL__`` |                             |                |
+-----------------------+-----------------------------+----------------+

Finding HIP
-----------

Makefiles can use the following syntax to conditionally provide a
default HIP_PATH if one does not exist:

::

   HIP_PATH ?= $(shell hipconfig --path)

Identifying HIP Runtime
-----------------------

HIP can depend on ROCclr, or NVCC as runtime

-  AMD platform ``HIP_ROCclr`` is defined on AMD platform that HIP use
   Radeon Open Compute Common Language Runtime, called ROCclr.

ROCclr is a virtual device interface that HIP runtimes interact with
different backends which allows runtimes to work on Linux , as well as
Windows without much efforts.

-  NVIDIA platform On Nvidia platform, HIP is just a thin layer on top
   of CUDA. On non-AMD platform, HIP runtime determines if nvcc is
   available and can be used. If available, HIP_PLATFORM is set to nvcc
   and underneath CUDA path is used.

hipLaunchKernel
---------------

hipLaunchKernel is a variadic macro which accepts as parameters the
launch configurations (grid dims, group dims, dynamic shared size, stream)
followed by a variable number of kernel arguments. This sequence
is then expanded into the appropriate kernel launch syntax depending on
the platform. While this can be a convenient single-line kernel launch
syntax, the macro implementation can cause issues when nested inside
other macros. For example, consider the following:

::

   // Will cause compile error:
   #define MY_LAUNCH(command, doTrace) \
   {\
       if (doTrace) printf ("TRACE: %s\n", #command); \
       (command);   /* The nested ( ) will cause compile error */\
   }

   MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");

Avoid nesting macro parameters inside parenthesis - here is an
alternative that will work:

::

   #define MY_LAUNCH(command, doTrace) \
   {\
       if (doTrace) printf ("TRACE: %s\n", #command); \
       command;\ 
   }

   MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");

Compiler Options
----------------

hipcc is a portable compiler driver that will call nvcc or HIP-Clang
(depending on the target system) and attach all required include and
library options. It passes options through to the target compiler. Tools
that call hipcc must ensure the compiler options are appropriate for the
target compiler. The ``hipconfig`` script may helpful in identifying the
target platform, compiler and runtime. It can also help set options
appropriately.

Linking Issues
--------------

Linking With hipcc
~~~~~~~~~~~~~~~~~~

hipcc adds the necessary libraries for HIP as well as for the
accelerator compiler (nvcc or AMD compiler). We recommend linking with
hipcc since it automatically links the binary to the necessary HIP
runtime libraries. It also has knowledge on how to link and to manage
the GPU objects.

-lm Option
~~~~~~~~~~

hipcc adds -lm by default to the link command.

Linking Code With Other Compilers
---------------------------------

CUDA code often uses nvcc for accelerator code (defining and launching
kernels, typically defined in .cu or .cuh files). It also uses a
standard compiler (g++) for the rest of the application. nvcc is a
preprocessor that employs a standard host compiler (gcc) to generate the
host code. Code compiled using this tool can employ only the
intersection of language features supported by both nvcc and the host
compiler. In some cases, you must take care to ensure the data types and
alignment of the host compiler are identical to those of the device
compiler. Only some host compilers are supported. For example, recent
nvcc versions lack Clang host-compiler capability.

hcc generates both device and host code using the same Clang-based
compiler. The code uses the same API as gcc, which allows code generated
by different gcc-compatible compilers to be linked together. For
example, code compiled using hcc can link with code compiled using
standard compilers (such as gcc, ICC and Clang). Take care to ensure
all compilers use the same standard C++ header and library formats.

libc++ and libstdc++
~~~~~~~~~~~~~~~~~~~~

hipcc links to libstdc++ by default. This provides better compatibility
between g++ and HIP.

If you pass stdlib=libc++ to hipcc, hipcc will use the libc++
library. Generally, libc++ provides a broader set of C++ features while
libstdc++ is the standard for more compilers (notably including g++).

When cross-linking C++ code, any C++ functions that use types from the
C++ standard library (including std::string, std::vector and other
containers) must use the same standard-library implementation. They
include the following:

-  Functions or kernels defined in hcc that are called from a standard
   compiler
-  Functions defined in a standard compiler that are called from hcc.

Applications with these interfaces should use the default libstdc++
linking.

Applications which are compiled entirely with hipcc, and which benefit
from advanced C++ features not supported in libstdc++, and which do not
require portability to nvcc, may choose to use libc++.

HIP Headers (hip_runtime.h, hip_runtime_api.h)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hip_runtime.h and hip_runtime_api.h files define the types,
functions and enumerations needed to compile a HIP program:

-  hip_runtime_api.h: defines all the HIP runtime APIs (e.g., hipMalloc)
   and the types required to call them. A source file that is only
   calling HIP APIs but neither defines nor launches any kernels can
   include hip_runtime_api.h. hip_runtime_api.h uses no custom hc
   language features and can be compiled using a standard C++ compiler.
-  hip_runtime.h: included in hip_runtime_api.h. It additionally
   provides the types and defines required to create and launch kernels.
   hip_runtime.h does use custom hc language features, but they are
   guarded by ifdef checks. It can be compiled using a standard C++
   compiler but will expose a subset of the available functions.

CUDA has slightly different contents for these two files. In some cases
you may need to convert hipified code to include the richer
hip_runtime.h instead of hip_runtime_api.h.

Using a Standard C++ Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can compile hip_runtime_api.h using a standard C or C++ compiler
(e.g., gcc or ICC). The HIP include paths and defines
(``__HIP_PLATFORM_HCC__`` or ``__HIP_PLATFORM_NVCC__``) must pass to the
standard compiler; hipconfig then returns the necessary options:

::

   > hipconfig --cxx_config
    -D__HIP_PLATFORM_HCC__ -I/home/user1/hip/include

You can capture the hipconfig output and passed it to the standard
compiler; below is a sample makefile syntax:

::

   CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)

nvcc includes some headers by default. However, HIP does not include
default headers, and instead all required files must be explicitly
included. Specifically, files that call HIP run-time APIs or define HIP
kernels must explicitly include the appropriate HIP headers. If the
compilation process reports that it cannot find necessary APIs (for
example, error: identifier 'hipSetDevice' is undefined, ensure that
the file includes hip_runtime.h (or hip_runtime_api.h, if appropriate).
The hipify-perl script automatically converts 'cuda_runtime.h' to
'hip_runtime.h', and it converts 'cuda_runtime_api.h' to
'hip_runtime_api.h', but it may miss nested headers or macros.

cuda.h
^^^^^^

The hcc path provides an empty cuda.h file. Some existing CUDA programs
include this file but does not require any of the functions.

Choosing HIP File Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many existing CUDA projects use the '.cu' and '.cuh' file extensions to
indicate code that should be run through the nvcc compiler. For quick
HIP ports, leaving these file extensions unchanged is often easier, as
it minimizes the work required to change file names in the directory and
#include statements in the files.

For new projects or ports which can be re-factored, we recommend the use
of the extension '.hip.cpp' for source files, and '.hip.h' or '.hip.hpp'
for header files. This indicates that the code is standard C++ code, but
also provides a unique indication for make tools to run hipcc when
appropriate.

Workarounds
-----------

warpSize
~~~~~~~~

Code should not assume a warp size of 32 or 64. See `Warp Cross-Lane
Functions <hip_kernel_language.md#warp-cross-lane-functions>`__ for
information on how to write portable wave-aware code.

Kernel launch with group size > 256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel code should use
``__attribute__((amdgpu_flat_work_group_size(<min>,<max>)))``.

For example:

::

   __global__ void dot(double *a,double *b,const int n) __attribute__((amdgpu_flat_work_group_size(1, 512)))

memcpyToSymbol
--------------

HIP support for hipMemcpyToSymbol is complete. This feature allows a
kernel to define a device-side data symbol which can be accessed on the
host side. The symbol can be in \__constant or device space.

Note that the symbol name needs to be encased in the HIP_SYMBOL macro,
as shown in the code example below. This also applies to
hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.

For example:

Device Code:

::

   #include<hip/hip_runtime.h>
   #include<hip/hip_runtime_api.h>
   #include<iostream>

   #define HIP_ASSERT(status) \
       assert(status == hipSuccess)

   #define LEN 512
   #define SIZE 2048

   __constant__ int Value[LEN];

   __global__ void Get(hipLaunchParm lp, int *Ad)
   {
       int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
       Ad[tid] = Value[tid];
   }

   int main()
   {
       int *A, *B, *Ad;
       A = new int[LEN];
       B = new int[LEN];
       for(unsigned i=0;i<LEN;i++)
       {
           A[i] = -1*i;
           B[i] = 0;
       }

       HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));

       HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
       hipLaunchKernel(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);
       HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

       for(unsigned i=0;i<LEN;i++)
       {
           assert(A[i] == B[i]);
       }
       std::cout<<"Passed"<<std::endl;
   }
   

CU POINTER ATTRIBUTE MEMORY TYPE
------------------------------------

To get pointer's memory type in HIP/HIP-Clang one should use hipPointerGetAttributes API. First parameter of the API is hipPointerAttribute_t which has 'memoryType' as member variable. 'memoryType' indicates input pointer is allocated on device or host.

For example:

::

   double * ptr;
   hipMalloc(reinterpret_cast<void**>(&ptr), sizeof(double));
   hipPointerAttribute_t attr;
   hipPointerGetAttributes(&attr, ptr); /*attr.memoryType will have value as hipMemoryTypeDevice*/

   double* ptrHost;
   hipHostMalloc(&ptrHost, sizeof(double));
   hipPointerAttribute_t attr;
   hipPointerGetAttributes(&attr, ptrHost); /*attr.memoryType will have value as hipMemoryTypeHost*/



threadfence_system
------------------

Threadfence_system makes all device memory writes, all writes to mapped
host memory, and all writes to peer memory visible to CPU and other GPU
devices. Some implementations can provide this behavior by flushing the
GPU L2 cache. HIP/HIP-Clang does not provide this functionality. As a
workaround, users can set the environment variable
``HSA_DISABLE_CACHE=1`` to disable the GPU L2 cache. This will affect
all accesses and for all kernels and so may have a performance impact.

Textures and Cache Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute programs sometimes use textures either to access dedicated
texture caches or to use the texture-sampling hardware for interpolation
and clamping. The former approach uses simple point samplers with linear
interpolation, essentially only reading a single point. The latter
approach uses the sampler hardware to interpolate and combine multiple
samples. AMD hardware, as well as recent competing hardware, has a
unified texture/L1 cache, so it no longer has a dedicated texture cache.
But the nvcc path often caches global loads in the L2 cache, and some
programs may benefit from explicit control of the L1 cache contents. We
recommend the \__ldg instruction for this purpose.

AMD compilers currently load all data into both the L1 and L2 caches, so
\__ldg is treated as a no-op.

We recommend the following for functional portability:

-  For programs that use textures only to benefit from improved caching,
   use the \__ldg instruction
-  Programs that use texture object and reference APIs, work well on HIP

More Tips
---------

HIP Logging
~~~~~~~~~~~

On an AMD platform, set the AMD_LOG_LEVEL environment variable to log
HIP application execution information.

The value of the setting controls different logging level,

::

   enum LogLevel {
   LOG_NONE = 0,
   LOG_ERROR = 1,
   LOG_WARNING = 2,
   LOG_INFO = 3,
   LOG_DEBUG = 4
   };

Logging mask is used to print types of functionalities during the
execution of HIP application. It can be set as one of the following
values,

::

   enum LogMask {
     LOG_API       = 0x00000001, //!< API call
     LOG_CMD       = 0x00000002, //!< Kernel and Copy Commands and Barriers
     LOG_WAIT      = 0x00000004, //!< Synchronization and waiting for commands to finish
     LOG_AQL       = 0x00000008, //!< Decode and display AQL packets
     LOG_QUEUE     = 0x00000010, //!< Queue commands and queue contents
     LOG_SIG       = 0x00000020, //!< Signal creation, allocation, pool
     LOG_LOCK      = 0x00000040, //!< Locks and thread-safety code.
     LOG_KERN      = 0x00000080, //!< kernel creations and arguments, etc.
     LOG_COPY      = 0x00000100, //!< Copy debug
     LOG_COPY2     = 0x00000200, //!< Detailed copy debug
     LOG_RESOURCE  = 0x00000400, //!< Resource allocation, performance-impacting events.
     LOG_INIT      = 0x00000800, //!< Initialization and shutdown
     LOG_MISC      = 0x00001000, //!< misc debug, not yet classified
     LOG_AQL2      = 0x00002000, //!< Show raw bytes of AQL packet
     LOG_CODE      = 0x00004000, //!< Show code creation debug
     LOG_CMD2      = 0x00008000, //!< More detailed command info, including barrier commands
     LOG_LOCATION  = 0x00010000, //!< Log message location
     LOG_ALWAYS    = 0xFFFFFFFF, //!< Log always even mask flag is zero
   };

Debugging hipcc
~~~~~~~~~~~~~~~

To see the detailed commands that hipcc issues, set the environment
variable HIPCC_VERBOSE to 1. Doing so will print to stderr the HIP-clang
(or nvcc) commands that hipcc generates.

::

   export HIPCC_VERBOSE=1
   make
   ...
   hipcc-cmd: /opt/hcc/bin/hcc  -hc -I/opt/hcc/include -stdlib=libc++ -I../../../../hc/include -I../../../../include/hcc_detail/cuda -I../../../../include -x c++ -I../../common -O3 -c backprop_cuda.cu

What Does This Error Mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~

/usr/include/c++/v1/memory:5172:15: error: call to implicitly deleted default constructor of 'std::__1::bad_weak_ptr' throw bad_weak_ptr();
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you pass a '.cu' file, hcc will attempt to compile it as a CUDA
language file. You must tell hcc that it is, infact, a C++ file: use the
-x c++ option.

Editor Highlighting
~~~~~~~~~~~~~~~~~~~

See the utils/vim or utils/gedit directories to add handy highlighting
to hip files.



