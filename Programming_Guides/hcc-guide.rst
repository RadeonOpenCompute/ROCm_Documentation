.. _HCCguide:

HCC: Heterogeneous Compute Compiler
####################################

**HCC is an Open Source, Optimizing C++ Compiler for Heterogeneous Devices**

This repository hosts the HCC compiler implementation project. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP, and transforms it into the AMD GCN ISA.

The project is based on LLVM+CLANG. For more information, please visit the :ref:`HCCwiki`

The Heterogeneous Compute Compiler (HCC) provides two important benefits:

**Ease of development**

 * A full C++ API for managing devices, queues and events
 * C++ data containers that provide type safety, multidimensional-array indexing and automatic data management
 * C++ kernel-launch syntax using parallel_for_each plus C++11 lambda functions
 * A single-source C++ programming environment---the host and device code can be in the same source file and use the same C++        	language;templates and classes work naturally across the host/device boundary
 * HCC generates both host and device code from the same compiler, so it benefits from a consistent view of the source code using the
   same Clang-based language parser

**Full control over the machine**

 * Access AMD scratchpad memories (“LDS”)
 * Fully control data movement, prefetch and discard
 * Fully control asynchronous kernel launch and completion
 * Get device-side dependency resolution for kernel and data commands (without host involvement)
 * Obtain HSA agents, queues and signals for low-level control of the architecture using the HSA Runtime API
 * Use [direct-to-ISA](https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA) compilation

Download HCC
###############

The project now employs git submodules to manage external components it depends upon. It it advised to add --recursive when you clone the project so all submodules are fetched automatically.

For example:

.. code:: sh

  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

For more information about git submodules, please refer to `git documentation <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_.


Build HCC from source
#######################

To configure and build HCC from source, use the following steps:

.. code:: sh

  mkdir -p build; cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make

To install it, use the following steps:

.. code:: sh

  sudo make install

Use HCC
########

For HC source codes:

.. code:: sh

  hcc -hc foo.cpp -o foo


Multiple ISA
###############
HCC now supports having multiple GCN ISAs in one executable file. You can do it in different ways:

**use ``--amdgpu-target=`` command line option**

It's possible to specify multiple `` --amdgpu-target= `` option.

Example:

.. code:: sh

 # ISA for Fiji(gfx803) and Vega10(gfx900) would
 # be produced
  hcc -hc \
    --amdgpu-target=gfx803 \
    --amdgpu-target=gfx900 \
    foo.cpp

**configure HCC use CMake ``HSA_AMDGPU_GPU_TARGET`` variable**

If you build HCC from source, it's possible to configure it to automatically produce multiple ISAs via `HSA_AMDGPU_GPU_TARGET` CMake variable.

Use ``;`` to delimit each AMDGPU target.
Example:

.. code:: sh

 # ISA for Fiji(gfx803) and Vega10(gfx900) would
 # be produced by default
 cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
    ../hcc

CodeXL Activity Logger
#########################

To enable the `CodeXL Activity Logger <https://github.com/RadeonOpenCompute/ROCm-Profiler/tree/master/CXLActivityLogger>`_, use the  ``USE_CODEXL_ACTIVITY_LOGGER`` environment variable.

Configure the build in the following way:

.. code:: sh

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CODEXL_ACTIVITY_LOGGER=1 \
    <ToT HCC checkout directory>

In your application compiled using hcc, include the CodeXL Activiy Logger header:

.. code:: sh

  #include <CXLActivityLogger.h>

For information about the usage of the Activity Logger for profiling, please refer to its `documentation <https://github.com/RadeonOpenCompute/ROCm-Profiler/blob/master/CXLActivityLogger/doc/AMDTActivityLogger.pdf>`_.

HCC with ThinLTO Linking
#########################

To enable the ThinLTO link time, use the ``KMTHINLTO`` environment variable.

Set up your environment in the following way:

.. code:: sh

  export KMTHINLTO=1

**ThinLTO Phase 1 - Implemented**

For applications compiled using hcc, ThinLTO could significantly improve link-time performance. This implementation will maintain kernels in their .bc file format, create module-summaries for each, perform llvm-lto's cross-module function importing and then perform clamp-device (which uses opt and llc tools) on each of the kernel files. These files are linked with lld into one .hsaco per target specified.

**ThinLTO Phase 2 - Under development**
This ThinLTO implementation which will use llvm-lto LLVM tool to replace clamp-device bash script. It adds an optllc option into ThinLTOGenerator, which will perform in-program opt and codegen in parallel.

To use HCC Printf Functions
#############################

Set up environmental variable:

.. code:: sh

export HCC_ENABLE_PRINTF=1

Then compile the printf kernel with ``HCC_ENABLE_ACCELERATOR_PRINTF`` macro defined.

.. code:: sh

~/build/bin/hcc -hc -DHCC_ENABLE_ACCELERATOR_PRINTF -lhc_am -o printf.out ~/hcc/tests/Unit/HSA/printf.cpp


HCC built-in macros
#######################
Built-in macros:

====================== ===============================================================================
Macro                  Meaning
====================== ===============================================================================
``__HCC__``		         always be 1
``__hcc_major__``	     major version number of HCC
``__hcc_minor__``	     minor version number of HCC
``__hcc_patchlevel__`` patchlevel of HCC
``__hcc_version__``	   combined string of ``__hcc_major__``, ``__hcc_minor__``, ``__hcc_patchlevel__``
====================== ===============================================================================

The rule for ``__hcc_patchlevel__`` is: yyWW-(HCC driver git commit #)-(HCC clang git commit #)

   * yy stands for the last 2 digits of the year
   * WW stands for the week number of the year

Macros for language modes in use:

================== ==========================================================================
 Macro             Meaning
================== ==========================================================================
``__KALMAR_AMP__`` 1 in case in C++ AMP mode (-std=c++amp; **Removed from ROCm 2.0 onwards**)
``__KALMAR_HC__``  1 in case in HC mode (-hc)
================== ==========================================================================

Compilation mode: HCC is a single-source compiler where kernel codes and host codes can reside in the same file. Internally HCC would trigger 2 compilation iterations, and the following macros can be used by user programs to determine which mode the compiler is in.

========================== ===============================================================
Macro           		       Meaning
========================== ===============================================================
``__KALMAR_ACCELERATOR__`` not 0 in case the compiler runs in kernel code compilation mode
``__KALMAR_CPU__``         not 0 in case the compiler runs in host code compilation mode
========================== ===============================================================




For more examples on how to use printf, see tests in tests/Unit/HSA/printf*.cpp.
