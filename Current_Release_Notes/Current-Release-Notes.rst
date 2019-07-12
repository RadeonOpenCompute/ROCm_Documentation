
.. _Current-Release-Notes:

=====================
Current Release Notes
=====================

New features and enhancements in ROCm 2.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCmInfo enhancements
^^^^^^^^^^^^^^^^^^^^^^^

ROCmInfo was extended to do the following: For ROCr API call errors including initialization determine if the error could be explained by:

   * ROCk (driver) is not loaded / available
   * User does not have membership in appropriate group - "video"
   * If not above print the error string that is mapped to the returned error code
   * If no error string is available, print the error code in hex

[Thrust] Functional Support on Vega20
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm2.6 contains the first official release of rocThrust and hipCUB. rocThrust is a port of thrust, a parallel algorithm library. hipCUB is a port of CUB, a reusable software component library. Thrust/CUB has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

::

Note: rocThrust and hipCUB library replaces `hip-thrust <https://github.com/ROCmSoftwarePlatform/thrust>`_ , i.e. hip-thrust has been separated into two libraries, rocThrust and hipCUB. Existing hip-thrust users are encouraged to port their code to rocThrust and/or hipCUB. Hip-thrust will be removed from official distribution later this year.

MIGraphX v0.3
^^^^^^^^^^^^^^^

MIGraphX optimizer adds support to read models frozen from Tensorflow framework. Further details and an example usage at `<https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.3>`_

MIOpen 2.0
^^^^^^^^^^^^

    * This release contains several new features including an immediate mode for selecting convolutions, bfloat16 support, new layers,  
      modes, and algorithms.     
    * MIOpenDriver, a tool for benchmarking and developing kernels is now shipped with MIOpen. BFloat16 now supported in HIP requires an     
      updated rocBLAS as a GEMM backend.
    * Immediate mode API now provides the ability to quickly obtain a convolution kernel.
    * MIOpen now contains HIP source kernels and implements the ImplicitGEMM kernels. This is a new feature and is currently disabled by   
      default. Use the environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation this feature. ImplicitGEMM requires an  
      up to  date HIP version of at least 1.5.9211.     
    * A new "loss" catagory of layers has been added, of which, CTC loss is the first. See the API reference for more details. 2.0 is the   
      last release of active support for gfx803 architectures. In future releases, MIOpen will not actively debug and develop new features   
      specifically for gfx803.
    * System Find-Db in memory cache is disabled by default. Please see build instructions to enable this feature. Additional documentation  
      can be found `here <https://rocmsoftwareplatform.github.io/MIOpen/doc/html/>`_

Bloat16 software support in rocBLAS/Tensile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added mixed precision bfloat16/IEEE f32 to gemm_ex. The input and output matrices are bfloat16. All arithmetic is in IEEE f32.

AMD Infinity Fabric™ Link enablement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in two hives or two Radeon Instinct MI60 or Radeon Instinct MI50 boards in four hives via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

ROCm-smi features and bug fixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * mGPU & Vendor check
    * Fix clock printout if DPM is disabled
    * Fix finding marketing info on CentOS
    * Clarify some error messages

ROCm-smi-lib enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^

    * Documentation updates
    * Improvements to *name_get functions

RCCL2 Enablement
^^^^^^^^^^^^^^^^^^

RCCL2 supports collectives intranode communication using PCIe, Infinity Fabric™, and pinned host memory, as well as internode communication using Ethernet (TCP/IP sockets) and Infiniband/RoCE (Infiniband Verbs). Note: For Infiniband/RoCE, RDMA is not currently supported.

rocFFT enhancements
^^^^^^^^^^^^^^^^^^^^

   * Added: Debian package with FFT test, benchmark, and sample programs
   * Improved: hipFFT interfaces
   * Improved: rocFFT CPU reference code, plan generation code and logging code

Features and enhancements introduced in previous versions of ROCm can be found in `version_history.md <https://github.com/RadeonOpenCompute/ROCm/blob/master/version_history.md>`_

New features and enhancements in ROCm 2.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UCX 1.6 support
^^^^^^^^^^^^^^^

Support for UCX version 1.6 has been added.

BFloat16 GEMM in rocBLAS/Tensile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Software support for BFloat16 on Radeon Instinct MI50, MI60 has been added. This includes:

   * Mixed precision GEMM with BFloat16 input and output matrices, and all arithmetic in IEEE32 bit
   * Input matrix values are converted from BFloat16 to IEEE32 bit, all arithmetic and accumulation is IEEE32 bit.Output values are rounded    from IEEE32 bit to BFloat16
   * Accuracy should be correct to 0.5 ULP

ROCm-SMI enhancements
^^^^^^^^^^^^^^^^^^^^^

CLI support for querying the memory size, driver version, and firmware version has been added to ROCm-smi.

[PyTorch] multi-GPU functional support (CPU aggregation/Data Parallel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-GPU support is enabled in PyTorch using Dataparallel path for versions of PyTorch built using the 06c8aa7a3bbd91cda2fd6255ec82aad21fa1c0d5 commit or later.

rocSparse optimization on Radeon Instinct MI50 and MI60
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This release includes performance optimizations for csrsv routines in the rocSparse library.

[Thrust] Preview
^^^^^^^^^^^^^^^^^

Preview release for early adopters. rocThrust is a port of thrust, a parallel algorithm library. Thrust has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: This library will replace `thrust`_ in a future release. The package for rocThrust (this library) currently conflicts with version 2.5 package of thrust. They should not be installed together.

.. _thrust: https://github.com/ROCmSoftwarePlatform/thrust

Support overlapping kernel execution in same HIP stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HIP API has been enhanced to allow independent kernels to run in parallel on the same stream.

AMD Infinity Fabric™ Link enablement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in one hive via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

Features and enhancements introduced in previous versions of ROCm can be found in `version_history.md`_

.. _version_history.md: https://github.com/RadeonOpenCompute/ROCm/blob/master/version_history.md


New features and enhancements in ROCm 2.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow 2.0 support
^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm 2.4 includes the enhanced compilation toolchain and a set of bug fixes to support TensorFlow 2.0 features natively

AMD Infinity Fabric™ Link enablement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm 2.4 adds support to connect two Radeon Instinct MI60 or Radeon Instinct MI50 boards via AMD Infinity Fabric™ Link GPU interconnect technology.


New features and enhancements in ROCm 2.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mem usage per GPU
^^^^^^^^^^^^^^^^^^^^^

Per GPU memory usage is added to rocm-smi. Display information regarding used/total bytes for VRAM, visible VRAM and GTT, via the --showmeminfo flag

MIVisionX, v1.1 - ONNX
^^^^^^^^^^^^^^^^^^^^^^^^

ONNX parser changes to adjust to new file formats

MIGraphX, v0.2
^^^^^^^^^^^^^^^^^

MIGraphX 0.2 supports the following new features:

   * New Python API
   * Support for additional ONNX operators and fixes that now enable a large set of Imagenet models
   * Support for RNN Operators
   * Support for multi-stream Execution
   * [Experimental] Support for Tensorflow frozen protobuf files

See: `Getting-started:-using-the-new-features-of-MIGraphX-0.2`_ for more details

.. _Getting-started:-using-the-new-features-of-MIGraphX-0.2: https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.2

MIOpen, v1.8 - 3d convolutions and int8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   * This release contains full 3-D convolution support and int8 support for inference.
   * Additionally, there are major updates in the performance database for major models including those found in Torchvision.

See: `MIOpen releases`_

.. _MIOpen releases: https://github.com/ROCmSoftwarePlatform/MIOpen/releases

Caffe2 - mGPU support
^^^^^^^^^^^^^^^^^^^^^^^

Multi-gpu support is enabled for Caffe2.

rocTracer library, ROCm tracing API for collecting runtimes API and asynchronous GPU activity traces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HIP/HCC domains support is introduced in rocTracer library.

BLAS - Int8 GEMM performance, Int8 functional and performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduces support and performance optimizations for Int8 GEMM, implements TRSV support, and includes improvements and optimizations with Tensile.

Prioritized L1/L2/L3 BLAS (functional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functional implementation of BLAS L1/L2/L3 functions

BLAS - tensile optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Improvements and optimizations with tensile

MIOpen Int8 support
^^^^^^^^^^^^^^^^^^^^^
Support for int8

New features and enhancements in ROCm 2.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rocSparse Optimization on Vega20
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cache usage optimizations for csrsv (sparse triangular solve), coomv (SpMV in COO format) and ellmv (SpMV in ELL format) are available.

DGEMM and DTRSM Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Improved DGEMM performance for reduced matrix sizes (k=384, k=256)

Caffe2
^^^^^^^^^^
Added support for multi-GPU training


New features and enhancements in ROCm 2.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RocTracer v1.0 preview release – 'rocprof' HSA runtime tracing and statistics support - 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
Supports HSA API tracing and HSA asynchronous GPU activity including kernels execution and memory copy

Improvements to ROCM-SMI tool -
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added support to show real-time PCIe bandwidth usage via the -b/--showbw flag

DGEMM Optimizations -
^^^^^^^^^^^^^^^^^^^^^^

Improved DGEMM performance for large square and reduced matrix sizes (k=384, k=256)


New features and enhancements in ROCm 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Features and enhancements introduced in previous versions of ROCm can be found in version_history.md

Adds support for RHEL 7.6 / CentOS 7.6 and Ubuntu 18.04.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds support for Vega 7nm, Polaris 12 GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduces MIVisionX
^^^^^^^^^^^^^^^^^^^^^
A comprehensive computer vision and machine intelligence libraries, utilities and applications bundled into a single toolkit.

Improvements to ROCm Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * rocSPARSE & hipSPARSE
   * rocBLAS with improved DGEMM efficiency on Vega 7nm

MIOpen
^^^^^^^^
    * This release contains general bug fixes and an updated performance database
    * Group convolutions backwards weights performance has been improved
    * RNNs now support fp16

Tensorflow multi-gpu and Tensorflow FP16 support for Vega 7nm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * TensorFlow v1.12 is enabled with fp16 support

PyTorch/Caffe2 with Vega 7nm Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * fp16 support is enabled
    * Several bug fixes and performance enhancements
    * Known Issue: breaking changes are introduced in ROCm 2.0 which are not addressed upstream yet. Meanwhile, please continue to use ROCm fork at https://github.com/ROCmSoftwarePlatform/pytorch

Improvements to ROCProfiler tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * Support for Vega 7nm

Support for hipStreamCreateWithPriority
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * Creates a stream with the specified priority. It creates a stream on which enqueued kernels have a different priority for execution compared to kernels enqueued on normal priority streams. The priority could be higher or lower than normal priority streams.

OpenCL 2.0 support
^^^^^^^^^^^^^^^^^^
    * ROCm 2.0 introduces full support for kernels written in the OpenCL 2.0 C language on certain devices and systems.  Applications can detect this support by calling the “clGetDeviceInfo” query function with “parame_name” argument set to “CL_DEVICE_OPENCL_C_VERSION”.  In order to make use of OpenCL 2.0 C language features, the application must include the option “-cl-std=CL2.0” in options passed to the runtime API calls responsible for compiling or building device programs.  The complete specification for the OpenCL 2.0 C language can be obtained using the following link: https://www.khronos.org/registry/OpenCL/specs/opencl-2.0-openclc.pdf

Improved Virtual Addressing (48 bit VA) management for Vega 10 and later GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * Fixes Clang AddressSanitizer and potentially other 3rd-party memory debugging tools with ROCm
    * Small performance improvement on workloads that do a lot of memory management
    * Removes virtual address space limitations on systems with more VRAM than system memory

Kubernetes support
^^^^^^^^^^^^^^^^^^^

Removed features
^^^^^^^^^^^^^^^^

- HCC: removed support for C++AMP

New features and enhancements in ROCm 1.9.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RDMA(MPI) support on Vega 7nm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Support ROCnRDMA based on Mellanox InfiniBand.

Improvements to HCC
^^^^^^^^^^^^^^^^^^^

-  Improved link time optimization.

Improvements to ROCProfiler tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  General bug fixes and implemented versioning APIs.

Critical bug fixes
^^^^^^^^^^^^^^^^^^

New features and enhancements in ROCm 1.9.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added DPM support to Vega 7nm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dynamic Power Management feature is enabled on Vega 7nm.

Fix for 'ROCm profiling' "Version mismatch between HSA runtime and libhsa-runtime-tools64.so.1" error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New features and enhancements in ROCm 1.9.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preview for Vega 7nm
^^^^^^^^^^^^^^^^^^^^

-  Enables developer preview support for Vega 7nm

System Management Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Adds support for the ROCm SMI (System Management Interface) library,
   which provides monitoring and management capabilities for AMD GPUs.

Improvements to HIP/HCC
^^^^^^^^^^^^^^^^^^^^^^^

-  Support for gfx906
-  Added deprecation warning for C++AMP. This will be the last version
   of HCC supporting C++AMP.
-  Improved optimization for global address space pointers passing into
   a GPU kernel
-  Fixed several race conditions in the HCC runtime
-  Performance tuning to the unpinned copy engine
-  Several codegen enhancement fixes in the compiler backend

Preview for rocprof Profiling Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Developer preview (alpha) of profiling tool 'rpl\_run.sh', cmd-line
front-end for rocProfiler, enables: \* Cmd-line tool for dumping public
per kernel perf-counters/metrics and kernel timestamps \* Input file
with counters list and kernels selecting parameters \* Multiple counters
groups and app runs supported \* Output results in CSV format The tool
location is: /opt/rocm/bin/rpl\_run.sh

Preview for rocr Debug Agent rocr\_debug\_agent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCr Debug Agent is a library that can be loaded by ROCm Platform
Runtime to provide the following functionality: \* Print the state for
wavefronts that report memory violation or upon executing a "s\_trap 2"
instruction. \* Allows SIGINT (``ctrl c``) or SIGTERM (``kill -15``) to
print wavefront state of aborted GPU dispatches. \* It is enabled on
Vega10 GPUs on ROCm1.9. The ROCm1.9 release will install the ROCr Debug
Agent library at /opt/rocm/lib/librocr\_debug\_agent64.so

New distribution support
^^^^^^^^^^^^^^^^^^^^^^^^

-  Binary package support for Ubuntu 18.04

ROCm 1.9 is ABI compatible with KFD in upstream Linux kernels.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Upstream Linux kernels support the following GPUs in these releases:
4.17: Fiji, Polaris 10, Polaris 11 4.18: Fiji, Polaris 10, Polaris 11,
Vega10

Some ROCm features are not available in the upstream KFD: \* More system
memory available to ROCm applications \* Interoperability between
graphics and compute \* RDMA \* IPC

To try ROCm with an upstream kernel, install ROCm as normal, but do not
install the rock-dkms package. Also add a udev rule to control /dev/kfd
permissions:

.. code:: sh

    echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
