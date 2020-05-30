.. image:: /Current_Release_Notes/amdblack.jpg

|

=============================================================
AMD Radeon Open Compute platforM (ROCm) Release Notes v3.5
=============================================================
June 03, 2020

AMD ROCm Release Notes v3.5.0
=============================

This page describes the features, fixed issues, and information about
downloading and installing the ROCm software. It also covers known
issues and deprecated features in the ROCm v3.5.0 release.

-  `Supported Operating Systems and Documentation
   Updates <#Supported-Operating-Systems-and-Documentation-Updates>`__

   -  `Supported Operating Systems <#Supported-Operating-Systems>`__
   -  `Documentation Updates <#Documentation-Updates>`__

-  `What's New in This Release <#Whats-New-in-This-Release>`__

   -  `Upgrading to This Release <#Upgrading-to-This-Release>`__
   -  `AMD ROCm Tools <#AMD-ROCm-Tools>`__
   -  `AMD ROCm Math and Communications
      Libraries <#AMD-ROCm-Math-and-Communications-Libraries>`__
   -  `AMD ROCm Deep Learning <#AMD-ROCm-Deep-Learning>`__
   -  `AMD ROCm System Management
      Interface <#AMD-ROCm-System-Management-Interface>`__
   -  `AMD ROCm MIVision <#AMD-ROCm-MIVision>`__

-  `Fixed Issues <#Fixed-Issues>`__

-  `Known Issues <#Known-Issues>`__

-  `Deprecations <#Deprecations>`__

   -  `Heterogeneous Compute
      Compiler <#Heterogeneous-Compute-Compiler>`__

-  `Deploying ROCm <#Deploying-ROCm>`__

-  `Hardware and Software Support <#Hardware-and-Software-Support>`__

-  `Machine Learning and High Performance Computing Software Stack for
   AMD
   GPU <#Machine-Learning-and-High-Performance-Computing-Software-Stack-for-AMD-GPU>`__

   -  `ROCm Binary Package Structure <#ROCm-Binary-Package-Structure>`__
   -  `ROCm Platform Packages <#ROCm-Platform-Packages>`__

Supported Operating Systems and Documentation Updates
=====================================================

Supported Operating Systems
---------------------------

The AMD ROCm v3.5.x platform is designed to support the following
operating systems:

-  Ubuntu 16.04.6(Kernel 4.15) and 18.04.4(Kernel 5.3)
-  CentOS 7.7 (Kernel 3.10-1062) and RHEL 7.8(Kernel 3.10.0-1127)(Using
   devtoolset-7 runtime support)
-  SLES 15 SP1
-  CentOS and RHEL 8.1(Kernel 4.18.0-147)

**NOTE**: Framework support (TensorFlow, pyTorch & Caffe2) for v8.1 is
not available.

Documentation Updates
---------------------

HIP-Clang Compile
~~~~~~~~~~~~~~~~~

-  `HIP FAQ - Transition from HCC to
   HIP-Clang <https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq>`__
-  `HIP-Clang Porting
   Guide <https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html#hip-porting-guide>`__
-  `HIP - Glossary of
   Terms <https://rocmdocs.amd.com/en/latest/ROCm_Glossary/ROCm-Glossary.html>`__

AMD ROCDebugger (ROCbdg)
~~~~~~~~~~~~~~~~~~~~~~~~

-  ROCgdb User Guide
-  ROCgdbapi Library

AMD ROCm Systems Management Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  System Management Interface Event Library

AMD ROCm Deep Learning
~~~~~~~~~~~~~~~~~~~~~~

-  `MIOpen API <https://github.com/ROCmSoftwarePlatform/MIOpen>`__

AMD ROCm Glossary of Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Updated Glossary of Terms and
   Definitions <https://rocmdocs.amd.com/en/latest/ROCm_Glossary/ROCm-Glossary.html>`__

General AMD ROCm Documentatin Links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access the following links for more information on:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supported platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#machine-learning-and-high-performance-computing-software-stack-for-amd-gpu-v3-3-0

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#amd-rocm-version-history

What's New in This Release
==========================

Upgrading to This Release
-------------------------

You must perform a fresh and a clean AMD ROCm install to successfully
upgrade from v3.3 to v3.5. The following changes apply in this release:

-  HCC is deprecated and replaced with the HIP-Clang compiler
-  HIP-HCC runtime is changed to Radeon Open Compute Common Language
   Runtime (HIP-ROCClr)
-  In the v3.5 release, the firmware is separated from the kernel
   package. The difference is as follows:

   -  v3.5 release has two separate rock-dkms and rock-dkms-firmware
      packages
   -  v3.3 release had the firmware as part of the rock-dkms package

AMD ROCm Compilers
------------------

Heterogeneous-Compute Interface for Portability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this release, the Heterogeneous Compute Compiler (HCC) compiler is
deprecated and the HIP-Clang compiler is introduced for compiling
Heterogeneous-Compute Interface for Portability (HIP) programs.

NOTE: The HCC environment variables will be gradually deprecated in
subsequent releases.

The majority of the codebase for the HIP-Clang compiler has been
upstreamed to the Clang trunk. The HIP-Clang implementation has
undergone a strict code review by the LLVM/Clang community and
comprehensive tests consisting of LLVM/Clang build bots. These reviews
and tests resulted in higher productivity, code quality, and lower cost
of maintenance.

.. figure:: HIPClang2.png
   :alt: ScreenShot

   ScreenShot

For most HIP applications, the transition from HCC to HIP-Clang is
transparent and efficient as the HIPCC and HIP cmake files automatically
choose compilation options for HIP-Clang and hide the difference between
the HCC and HIP-Clang code. However, minor changes may be required as
HIP-Clang has a stricter syntax and semantic checks compared to HCC.

NOTE: Native HCC language features are no longer supported.

Radeon Open Compute Common Language Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Radeon Open Compute Common Language Runtime (ROCclr) is a virtual device
interface that computes runtime interaction with backends such as ROCr
or PAL.

In this release, HIP is implemented on top of ROCclr, which is a layer
abstracting ROCm and PAL (Platform Abstraction Library) APIs. This
abstraction allows runtimes to work easily on Linux and Windows
machines.

The following image summarizes the HIP stack for HIP-Clang.

.. figure:: HipClang2.1.png
   :alt: ScreenShot

   ScreenShot

OpenCL Runtime
~~~~~~~~~~~~~~

The following OpenCL runtime changes are made in this release:

-  AMD ROCm OpenCL Runtime extends support to OpenCL2.2
-  The developer branch is changed from master to master-next

AMD ROCm Tools
--------------

AMD ROCm GNU Debugger (ROCgdb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm Debugger (ROCgdb) is the AMD ROCm source-level debugger for
Linux based on the GNU Debugger (GDB). It enables heterogeneous
debugging on the AMD ROCm platform of an x86-based host architecture
along with AMD GPU architectures and supported by the AMD Debugger API
Library (ROCdbgapi).

The AMD ROCm Debugger is installed by the rocm-gdb package. The rocm-gdb
package is part of the rocm-dev meta-package, which is in the rocm-dkms
package.

The current AMD ROCm Debugger (ROCgdb) is an initial prototype that
focuses on source line debugging. Note, symbolic variable debugging
capabilities are not currently supported.

You can use the standard GDB commands for both CPU and GPU code
debugging. For more information about ROCgdb, refer to the ROCgdb User
Guide, which is installed at:

-  /opt/rocm/share/info/gdb.info as a texinfo file
-  /opt/rocm/share/doc/gdb/gdb.pdf as a PDF file

The AMD ROCm Debugger User Guide is available as a PDF at:

-  <<>>

For more information about GNU Debugger (GDB), refer to the GNU Debugger
(GDB) web site at: http://www.gnu.org/software/gdb

AMD ROCm Debugger API Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMD ROCm Debugger API Library (ROCdbgapi) implements an AMD GPU
debugger application programming interface (API) that provides the
support necessary for a client of the library to control the execution
and inspect the state of AMD GPU devices.

The following AMD GPU architectures are supported: \* Vega 10 \* Vega
7nm

The AMD ROCm Debugger API Library is installed by the rocm-dbgapi
package. The rocm-gdb package is part of the rocm-dev meta-package,
which is in the rocm-dkms package. The AMD ROCm Debugger API
Specification is available as a PDF at:

(Enter Doc link)

rocProfiler Dispatch Callbacks Start/Stop API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this release, a new rocprofiler start/stop API is added to
enable/disable GPU kernel HSA dispatch callbacks. The callback can be
registered with the â€˜rocprofiler_set_hsa_callbacksâ€™ API. The API helps
you eliminate some profiling performance impact by invoking the profiler
only for kernel dispatches of interest. This optimization will result in
significant performance gains.

The API provides the following functions: \* *hsa_status_t
rocprofiler_start_queue_callbacks();* is used to start profiling \*
*hsa_status_t rocprofiler_stop_queue_callbacks();* is used to stop
profiling.

For more information on kernel dispatches, see the HSA Platform System
Architecture Specification guide at
http://www.hsafoundation.com/standards/.

AMD ROCm Math and Communications Libraries
------------------------------------------

ROCm Communications Collective Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm Communications Collective Library (RCCL) consists of the
following enhancements: \* Re-enable target 0x803 \* Build time
improvements for the HIP-Clang compiler

NVIDIA Communications Collective Library Version Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD RCCL is now compatible with NVIDIA Communications Collective Library
(NCCL) v2.6.4 and provides the following features: \* Network interface
improvements with API v3 \* Network topology detection \* Improved CPU
type detection \* Infiniband adaptive routing support

.. _amd-rocm-deep-learning-1:

AMD ROCm Deep Learning
----------------------

MIOpen - Optional Kernel Package Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIOpen provides an optional pre-compiled kernel package to reduce
startup latency.

NOTE: The installation of this package is optional. MIOpen will continue
to function as expected even if you choose to not install the
pre-compiled kernel package. This is because MIOpen compiles the kernels
on the target machine once the kernel is run. However, the compilation
step may significantly increase the startup time for different
operations.

To install the kernel package for your GPU architecture, use the
following command:

*apt-get install miopen-kernels--*

-   is the GPU architecture. Ror example, gfx900, gfx906
-   is the number of CUs available in the GPU. Ffor example, 56 or 64

AMD ROCm System Management Interface
------------------------------------

New SMI Event Interface and Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An SMI event interface is added to the kernel and ROCm SMI lib for
system administrators to get notified when specific events occur. On the
kernel side, AMDKFD_IOC_SMI_EVENTS input/output control is enhanced to
allow notifications propagation to user mode through the event channel.

On the ROCm SMI lib side, APIs are added to set an event mask and
receive event notifications with a timeout option. Further, ROCm SMI API
details can be found in the PDF generated by Doxygen from source or by
referring to the rocm_smi.h header file (see the
rsmi_event_notification_\* functions).

For the more details about ROCm SMI API, see

(enter doc link after updating the website)

API for CPU Affinity
~~~~~~~~~~~~~~~~~~~~

A new API is introduced for aiding applications to select the
appropriate memory node for a given accelerator(GPU).

The API for CPU affinity has the following signature:

*rsmi_status_t rsmi_topo_numa_affinity_get(uint32_t dv_ind,
uint32_t*\ numa_node);\*

This API takes as input, device index (dv_ind), and returns the NUMA
node (CPU affinity), stored at the location pointed by numa_node
pointer, associated with the device.

Non-Uniform Memory Access (NUMA) is a computer memory design used in
multiprocessing, where the memory access time depends on the memory
location relative to the processor.

AMD ROCm MIVision
-----------------

Radeon Performance Primitives Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new Radeon Performance Primitives (RPP) library is a comprehensive
high-performance computer vision library for AMD (CPU and GPU) with the
HIP and OpenCL backend. The target operating system is Linux.

.. figure:: RPP.png
   :alt: ScreenShot

   ScreenShot

For more information about prerequisites and library functions, see

https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docs

Fixed Issues
============

Device printf Support for HIP-Clang
-----------------------------------

HIP now supports the use of printf in the device code. The parameters
and return value for the device-side printf follow the POSIX.1 standard,
with the exception that the â€œ%nâ€ specifier is not supported. A call to
printf blocks the calling wavefront until the operation is completely
processed by the host.

No host-side runtime calls by the application are needed to cause the
output to appear. There is also no limit on the number of device-side
calls to printf or the amount of data that is printed.

For more details, refer the HIP Programming Guide at:
https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html#hip-guide

Assertions in HIP Device Code
-----------------------------

Previously, a failing assertion caused early termination of kernels and
the application to exit with a line number, file, and failing condition
printed to the screen. This issue is now fixed and the assert() and
abort() functions are implemented for HIP device code. NOTE: There may
be a performance impact in the use of device assertions in its current
form.

You may choose to disable the assertion in the production code. For
example, to disable an assertion of:

*assert(foo != 0);*

you may comment it out as:

*//assert(foo != 0);*

NOTE: Assertions are currently enabled by default.

Known Issues
============

The following are the known issues in the v3.5.x release.

Deprecations
============

Heterogeneous Compute Compiler
------------------------------

In this release, the Heterogeneous Compute Compiler (HCC) compiler is
deprecated and the HIP-Clang compiler is introduced for compiling
Heterogeneous-Compute Interface for Portability (HIP) programs.

For more information, see HIP documentation at:
https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html

Deploying ROCm
--------------

AMD hosts both Debian and RPM repositories for the ROCm v3.5.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

Hardware and Software Support
-----------------------------

ROCm is focused on using AMD GPUs to accelerate computational tasks such
as machine learning, engineering workloads, and scientific computing. In
order to focus our development efforts on these domains of interest,
ROCm supports a targeted set of hardware configurations which are
detailed further in this section.

Supported GPUs
~~~~~~~~~~~~~~

Because the ROCm Platform has a focus on particular computational
domains, we offer official support for a selection of AMD GPUs that are
designed to offer good performance and price in these domains.

ROCm officially supports AMD GPUs that use following chips:

-  GFX8 GPUs

   -  â€œFijiâ€ chips, such as on the AMD Radeon R9 Fury X and Radeon
      Instinct MI8
   -  â€œPolaris 10â€ chips, such as on the AMD Radeon RX 580 and Radeon
      Instinct MI6

-  GFX9 GPUs

   -  â€œVega 10â€ chips, such as on the AMD Radeon RX Vega 64 and Radeon
      Instinct MI25
   -  â€œVega 7nmâ€ chips, such as on the Radeon Instinct MI50, Radeon
      Instinct MI60 or AMD Radeon VII

ROCm is a collection of software ranging from drivers and runtimes to
libraries and developer tools. Some of this software may work with more
GPUs than the â€œofficially supportedâ€ list above, though AMD does not
make any official claims of support for these devices on the ROCm
software platform. The following list of GPUs are enabled in the ROCm
software, though full support is not guaranteed:

-  GFX8 GPUs

   -  â€œPolaris 11â€ chips, such as on the AMD Radeon RX 570 and Radeon
      Pro WX 4100
   -  â€œPolaris 12â€ chips, such as on the AMD Radeon RX 550 and Radeon RX
      540

-  GFX7 GPUs

   -  â€œHawaiiâ€ chips, such as the AMD Radeon R9 390X and FirePro W9100

As described in the next section, GFX8 GPUs require PCI Express 3.0
(PCIe 3.0) with support for PCIe atomics. This requires both CPU and
motherboard support. GFX9 GPUs require PCIe 3.0 with support for PCIe
atomics by default, but they can operate in most cases without this
capability.

The integrated GPUs in AMD APUs are not officially supported targets for
ROCm. As described `below <#limited-support>`__, â€œCarrizoâ€, â€œBristol
Ridgeâ€, and â€œRaven Ridgeâ€ APUs are enabled in our upstream drivers and
the ROCm OpenCL runtime. However, they are not enabled in our HCC or HIP
runtimes, and may not work due to motherboard or OEM hardware
limitations. As such, they are not yet officially supported targets for
ROCm.

For a more detailed list of hardware support, please see `the following
documentation <https://rocm.github.io/hardware.html>`__.

Supported CPUs
~~~~~~~~~~~~~~

As described above, GFX8 GPUs require PCIe 3.0 with PCIe atomics in
order to run ROCm. In particular, the CPU and every active PCIe point
between the CPU and GPU require support for PCIe 3.0 and PCIe atomics.
The CPU root must indicate PCIe AtomicOp Completion capabilities and any
intermediate switch must indicate PCIe AtomicOp Routing capabilities.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

-  AMD Ryzen CPUs
-  The CPUs in AMD Ryzen APUs
-  AMD Ryzen Threadripper CPUs
-  AMD EPYC CPUs
-  Intel Xeon E7 v3 or newer CPUs
-  Intel Xeon E5 v3 or newer CPUs
-  Intel Xeon E3 v3 or newer CPUs
-  Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e.Â Haswell
   family or newer)
-  Some Ivy Bridge-E systems

Beginning with ROCm 1.8, GFX9 GPUs (such as Vega 10) no longer require
PCIe atomics. We have similarly opened up more options for number of
PCIe lanes. GFX9 GPUs can now be run on CPUs without PCIe atomics and on
older PCIe generations, such as PCIe 2.0. This is not supported on GPUs
below GFX9, e.g.Â GFX8 cards in the Fiji and Polaris families.

If you are using any PCIe switches in your system, please note that PCIe
Atomics are only supported on some switches, such as Broadcom PLX. When
you install your GPUs, make sure you install them in a PCIe 3.1.0 x16,
x8, x4, or x1 slot attached either directly to the CPUâ€™s Root I/O
controller or via a PCIe switch directly attached to the CPUâ€™s Root I/O
controller.

In our experience, many issues stem from trying to use consumer
motherboards which provide physical x16 connectors that are electrically
connected as e.g.Â PCIe 2.0 x4, PCIe slots connected via the Southbridge
PCIe I/O controller, or PCIe slots connected through a PCIe switch that
does not support PCIe atomics.

If you attempt to run ROCm on a system without proper PCIe atomic
support, you may see an error in the kernel log (``dmesg``):

::

   kfd: skipped device 1002:7300, PCI rejects atomics

Experimental support for our Hawaii (GFX7) GPUs (Radeon R9 290, R9 390,
FirePro W9100, S9150, S9170) does not require or take advantage of PCIe
Atomics. However, we still recommend that you use a CPU from the list
provided above for compatibility purposes.

Not supported or limited support under ROCm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Limited support
^^^^^^^^^^^^^^^

-  ROCm 2.9.x should support PCIe 2.0 enabled CPUs such as the AMD
   Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and older
   Intel Xeon and Intel Core Architecture and Pentium CPUs. However, we
   have done very limited testing on these configurations, since our
   test farm has been catering to CPUs listed above. This is where we
   need community support. *If you find problems on such setups, please
   report these issues*.
-  Thunderbolt 1, 2, and 3 enabled breakout boxes should now be able to
   work with ROCm. Thunderbolt 1 and 2 are PCIe 2.0 based, and thus are
   only supported with GPUs that do not require PCIe 3.1.0 atomics
   (e.g.Â Vega 10). However, we have done no testing on this
   configuration and would need community support due to limited access
   to this type of equipment.
-  AMD â€œCarrizoâ€ and â€œBristol Ridgeâ€ APUs are enabled to run OpenCL, but
   do not yet support HCC, HIP, or our libraries built on top of these
   compilers and runtimes.

   -  As of ROCm 2.1, â€œCarrizoâ€ and â€œBristol Ridgeâ€ require the use of
      upstream kernel drivers.
   -  In addition, various â€œCarrizoâ€ and â€œBristol Ridgeâ€ platforms may
      not work due to OEM and ODM choices when it comes to key
      configurations parameters such as inclusion of the required CRAT
      tables and IOMMU configuration parameters in the system BIOS.
   -  Before purchasing such a system for ROCm, please verify that the
      BIOS provides an option for enabling IOMMUv2 and that the system
      BIOS properly exposes the correct CRAT table. Inquire with your
      vendor about the latter.

-  AMD â€œRaven Ridgeâ€ APUs are enabled to run OpenCL, but do not yet
   support HCC, HIP, or our libraries built on top of these compilers
   and runtimes.

   -  As of ROCm 2.1, â€œRaven Ridgeâ€ requires the use of upstream kernel
      drivers.
   -  In addition, various â€œRaven Ridgeâ€ platforms may not work due to
      OEM and ODM choices when it comes to key configurations parameters
      such as inclusion of the required CRAT tables and IOMMU
      configuration parameters in the system BIOS.
   -  Before purchasing such a system for ROCm, please verify that the
      BIOS provides an option for enabling IOMMUv2 and that the system
      BIOS properly exposes the correct CRAT table. Inquire with your
      vendor about the latter.

Not supported
^^^^^^^^^^^^^

-  â€œTongaâ€, â€œIcelandâ€, â€œVega Mâ€, and â€œVega 12â€ GPUs are not supported in
   ROCm 2.9.x
-  We do not support GFX8-class GPUs (Fiji, Polaris, etc.) on CPUs that
   do not have PCIe 3.0 with PCIe atomics.

   -  As such, we do not support AMD Carrizo and Kaveri APUs as hosts
      for such GPUs.
   -  Thunderbolt 1 and 2 enabled GPUs are not supported by GFX8 GPUs on
      ROCm. Thunderbolt 1 & 2 are based on PCIe 2.0.

ROCm support in upstream Linux kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of ROCm 1.9.0, the ROCm user-level software is compatible with the
AMD drivers in certain upstream Linux kernels. As such, users have the
option of either using the ROCK kernel driver that are part of AMDâ€™s
ROCm repositories or using the upstream driver and only installing ROCm
user-level utilities from AMDâ€™s ROCm repositories.

These releases of the upstream Linux kernel support the following GPUs
in ROCm: \* 4.17: Fiji, Polaris 10, Polaris 11 \* 4.18: Fiji, Polaris
10, Polaris 11, Vega10 \* 4.20: Fiji, Polaris 10, Polaris 11, Vega10,
Vega 7nm

The upstream driver may be useful for running ROCm software on systems
that are not compatible with the kernel driver available in AMDâ€™s
repositories. For users that have the option of using either AMDâ€™s or
the upstreamed driver, there are various tradeoffs to take into
consideration:

+---+-------------------------------------------------------------+----+
|   | Using AMDâ€™s ``rock-dkms`` package                           | U  |
|   |                                                             | si |
|   |                                                             | ng |
|   |                                                             | t  |
|   |                                                             | he |
|   |                                                             | up |
|   |                                                             | st |
|   |                                                             | re |
|   |                                                             | am |
|   |                                                             | ke |
|   |                                                             | rn |
|   |                                                             | el |
|   |                                                             | dr |
|   |                                                             | iv |
|   |                                                             | er |
+===+=============================================================+====+
| P | More GPU features, and they are enabled earlier             | In |
| r |                                                             | cl |
| o |                                                             | ud |
| s |                                                             | es |
|   |                                                             | t  |
|   |                                                             | he |
|   |                                                             | la |
|   |                                                             | te |
|   |                                                             | st |
|   |                                                             | L  |
|   |                                                             | in |
|   |                                                             | ux |
|   |                                                             | ke |
|   |                                                             | rn |
|   |                                                             | el |
|   |                                                             | fe |
|   |                                                             | at |
|   |                                                             | ur |
|   |                                                             | es |
+---+-------------------------------------------------------------+----+
|   | Tested by AMD on supported distributions                    | M  |
|   |                                                             | ay |
|   |                                                             | wo |
|   |                                                             | rk |
|   |                                                             | on |
|   |                                                             | o  |
|   |                                                             | th |
|   |                                                             | er |
|   |                                                             | d  |
|   |                                                             | is |
|   |                                                             | tr |
|   |                                                             | ib |
|   |                                                             | ut |
|   |                                                             | io |
|   |                                                             | ns |
|   |                                                             | a  |
|   |                                                             | nd |
|   |                                                             | wi |
|   |                                                             | th |
|   |                                                             | cu |
|   |                                                             | st |
|   |                                                             | om |
|   |                                                             | k  |
|   |                                                             | er |
|   |                                                             | ne |
|   |                                                             | ls |
+---+-------------------------------------------------------------+----+
|   | Supported GPUs enabled regardless of kernel version         |    |
+---+-------------------------------------------------------------+----+
|   | Includes the latest GPU firmware                            |    |
+---+-------------------------------------------------------------+----+
| C | May not work on all Linux distributions or versions         | Fe |
| o |                                                             | at |
| n |                                                             | ur |
| s |                                                             | es |
|   |                                                             | a  |
|   |                                                             | nd |
|   |                                                             | ha |
|   |                                                             | rd |
|   |                                                             | wa |
|   |                                                             | re |
|   |                                                             | s  |
|   |                                                             | up |
|   |                                                             | po |
|   |                                                             | rt |
|   |                                                             | va |
|   |                                                             | ri |
|   |                                                             | es |
|   |                                                             | d  |
|   |                                                             | ep |
|   |                                                             | en |
|   |                                                             | di |
|   |                                                             | ng |
|   |                                                             | on |
|   |                                                             | ke |
|   |                                                             | rn |
|   |                                                             | el |
|   |                                                             | v  |
|   |                                                             | er |
|   |                                                             | si |
|   |                                                             | on |
+---+-------------------------------------------------------------+----+
|   | Not currently supported on kernels newer than 5.4           | Li |
|   |                                                             | mi |
|   |                                                             | ts |
|   |                                                             | G  |
|   |                                                             | PU |
|   |                                                             | â€™s |
|   |                                                             | u  |
|   |                                                             | sa |
|   |                                                             | ge |
|   |                                                             | of |
|   |                                                             | sy |
|   |                                                             | st |
|   |                                                             | em |
|   |                                                             | me |
|   |                                                             | mo |
|   |                                                             | ry |
|   |                                                             | to |
|   |                                                             | 3  |
|   |                                                             | /8 |
|   |                                                             | of |
|   |                                                             | sy |
|   |                                                             | st |
|   |                                                             | em |
|   |                                                             | me |
|   |                                                             | mo |
|   |                                                             | ry |
|   |                                                             | (  |
|   |                                                             | be |
|   |                                                             | fo |
|   |                                                             | re |
|   |                                                             | 5  |
|   |                                                             | .6 |
|   |                                                             | ). |
|   |                                                             | F  |
|   |                                                             | or |
|   |                                                             | 5  |
|   |                                                             | .6 |
|   |                                                             | a  |
|   |                                                             | nd |
|   |                                                             | b  |
|   |                                                             | ey |
|   |                                                             | on |
|   |                                                             | d, |
|   |                                                             | bo |
|   |                                                             | th |
|   |                                                             | DK |
|   |                                                             | MS |
|   |                                                             | a  |
|   |                                                             | nd |
|   |                                                             | up |
|   |                                                             | st |
|   |                                                             | re |
|   |                                                             | am |
|   |                                                             | k  |
|   |                                                             | er |
|   |                                                             | ne |
|   |                                                             | ls |
|   |                                                             | a  |
|   |                                                             | ll |
|   |                                                             | ow |
|   |                                                             | u  |
|   |                                                             | se |
|   |                                                             | of |
|   |                                                             | 1  |
|   |                                                             | 5/ |
|   |                                                             | 16 |
|   |                                                             | of |
|   |                                                             | sy |
|   |                                                             | st |
|   |                                                             | em |
|   |                                                             | m  |
|   |                                                             | em |
|   |                                                             | or |
|   |                                                             | y. |
+---+-------------------------------------------------------------+----+
|   |                                                             | I  |
|   |                                                             | PC |
|   |                                                             | a  |
|   |                                                             | nd |
|   |                                                             | RD |
|   |                                                             | MA |
|   |                                                             | ca |
|   |                                                             | pa |
|   |                                                             | bi |
|   |                                                             | li |
|   |                                                             | ti |
|   |                                                             | es |
|   |                                                             | a  |
|   |                                                             | re |
|   |                                                             | n  |
|   |                                                             | ot |
|   |                                                             | y  |
|   |                                                             | et |
|   |                                                             | e  |
|   |                                                             | na |
|   |                                                             | bl |
|   |                                                             | ed |
+---+-------------------------------------------------------------+----+
|   |                                                             | N  |
|   |                                                             | ot |
|   |                                                             | te |
|   |                                                             | st |
|   |                                                             | ed |
|   |                                                             | by |
|   |                                                             | A  |
|   |                                                             | MD |
|   |                                                             | to |
|   |                                                             | t  |
|   |                                                             | he |
|   |                                                             | sa |
|   |                                                             | me |
|   |                                                             | l  |
|   |                                                             | ev |
|   |                                                             | el |
|   |                                                             | as |
|   |                                                             | `  |
|   |                                                             | `r |
|   |                                                             | oc |
|   |                                                             | k- |
|   |                                                             | dk |
|   |                                                             | ms |
|   |                                                             | `` |
|   |                                                             | p  |
|   |                                                             | ac |
|   |                                                             | ka |
|   |                                                             | ge |
+---+-------------------------------------------------------------+----+
|   |                                                             | Do |
|   |                                                             | es |
|   |                                                             | n  |
|   |                                                             | ot |
|   |                                                             | i  |
|   |                                                             | nc |
|   |                                                             | lu |
|   |                                                             | de |
|   |                                                             | mo |
|   |                                                             | st |
|   |                                                             | up |
|   |                                                             | -t |
|   |                                                             | o- |
|   |                                                             | da |
|   |                                                             | te |
|   |                                                             | fi |
|   |                                                             | rm |
|   |                                                             | wa |
|   |                                                             | re |
+---+-------------------------------------------------------------+----+

Software Support
----------------

As of AMD ROCm v1.9.0, the ROCm user-level software is compatible with
the AMD drivers in certain upstream Linux kernels. You have the
following options:

â€¢ Use the ROCk kernel driver that is a part of AMDâ€™s ROCm repositories
or â€¢ Use the upstream driver and only install ROCm user-level utilities
from AMDâ€™s ROCm repositories

The releases of the upstream Linux kernel support the following GPUs in
ROCm:

â€¢ Fiji, Polaris 10, Polaris 11 â€¢ Fiji, Polaris 10, Polaris 11, Vega10 â€¢
Fiji, Polaris 10, Polaris 11, Vega10, Vega 7nm


DISCLAIMER 
===========
The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information.  Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.  No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document.  Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. S
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

