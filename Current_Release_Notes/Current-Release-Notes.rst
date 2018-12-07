
.. _Current-Release-Notes:

=====================
Current Release Notes
=====================

New features and enhancements in ROCm 2.0 (*Work In Progress*)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
