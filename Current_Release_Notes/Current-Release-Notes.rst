
.. _Current-Release-Notes:

=====================
Current Release Notes
=====================

New features and enhancements in ROCm 2.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow 2.0 support
^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm 2.4 includes the enhanced compilation toolchain and a set of bug fixes to support TensorFlow 2.0 features natively

AMD Infinity Fabric™ Link enablement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm 2.4 adds support to connect two Radeon Instinct MI60 or Radeon Instinct MI50 boards via AMD Infinity Fabric™ Link GPU interconnect technology.

Features and enhancements introduced in previous versions of ROCm can be found in `version_history.md`_

.. _version_history.md: https://github.com/RadeonOpenCompute/ROCm/blob/master/version_history.md

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
