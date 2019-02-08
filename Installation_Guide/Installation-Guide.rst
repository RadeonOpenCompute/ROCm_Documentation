=======================
ROCm Installation Guide
=======================
            
Current ROCm Version: 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~

Hardware Support
~~~~~~~~~~~~~~~~

ROCm is focused on using AMD GPUs to accelerate computational tasks such
as machine learning, engineering workloads, and scientific computing. In
order to focus our development efforts on these domains of interest,
ROCm supports a targeted set of hardware configurations which are
detailed further in this section.

Supported GPUs
^^^^^^^^^^^^^^

Because the ROCm Platform has a focus on particular computational
domains, we offer official support for a selection of AMD GPUs that are
designed to offer good performance and price in these domains.

ROCm officially supports AMD GPUs that use following chips:

-  GFX8 GPUs

   -  “Fiji” chips, such as on the AMD Radeon R9 Fury X and Radeon
      Instinct MI8
   -  “Polaris 10” chips, such as on the AMD Radeon RX 580 and Radeon
      Instinct MI6
   -  “Polaris 11” chips, such as on the AMD Radeon RX 570 and Radeon
      Pro WX 4100
   -  “Polaris 12” chips, such as on the AMD Radeon RX 550 and Radeon RX
      540

-  GFX9 GPUs

   -  “Vega 10” chips, such as on the AMD Radeon RX Vega 64 and Radeon
      Instinct MI25
   -  “Vega 7nm” chips

ROCm is a collection of software ranging from drivers and runtimes to
libraries and developer tools. Some of this software may work with more
GPUs than the “officially supported” list above, though AMD does not
make any official claims of support for these devices on the ROCm
software platform. The following list of GPUs are enabled in the ROCm
software, though full support is not guaranteed:

-  GFX7 GPUs

   -  “Hawaii” chips, such as the AMD Radeon R9 390X and FirePro W9100

As described in the next section, GFX8 GPUs require PCI Express 3.0
(PCIe 3.0) with support for PCIe atomics. This requires both CPU and
motherboard support. GFX9 GPUs, by default, also require PCIe 3.0 with
support for PCIe atomics, but they can operate in most cases without
this capability.

At this time, the integrated GPUs in AMD APUs are not officially
supported targets for ROCm. As descried below, “Carrizo”, “Bristol
Ridge”, and “Raven Ridge” APUs are enabled in our upstream drivers and
the ROCm OpenCL runtime. However, they are not enabled in our HCC or HIP
runtimes, and may not work due to motherboard or OEM hardware
limitations. As such, they are not yet officially supported targets for
ROCm.

For a more detailed list of hardware support, please see `the following
documentation`_.

.. _the following documentation: https://rocm.github.io/hardware.html

Supported CPUs
^^^^^^^^^^^^^^

As described above, GFX8 GPUs require PCIe 3.0 with PCIe atomics in
order to run ROCm. In particular, the CPU and every active PCIe point
between the CPU and GPU require support for PCIe 3.0 and PCIe atomics.
The CPU root must indicate PCIe AtomicOp Completion capabilities and any
intermediate switch must indicate PCIe AtomicOp Routing capabilities.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

-  AMD Ryzen CPUs;
-  The CPUs in AMD Ryzen APUs;
-  AMD Ryzen Threadripper CPUs
-  AMD EPYC CPUs;
-  Intel Xeon E7 v3 or newer CPUs;
-  Intel Xeon E5 v3 or newer CPUs;
-  Intel Xeon E3 v3 or newer CPUs;
-  Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e. Haswell
   family or newer).
-  Some Ivy Bridge-E systems

Beginning with ROCm 1.8, GFX9 GPUs (such as Vega 10) no longer require
PCIe atomics. We have similarly opened up more options for number of
PCIe lanes. GFX9 GPUs can now be run on CPUs without PCIe atomics and on
older PCIe generations, such as PCIe 2.0. This is not supported on GPUs
below GFX9, e.g. GFX8 cards in the Fiji and Polaris families.

If you are using any PCIe switches in your system, please note that PCIe
Atomics are only supported on some switches, such as Broadcom PLX. When
you install your GPUs, make sure you install them in a PCIe 3.0 x16, x8,
x4, or x1 slot attached either directly to the CPU’s Root I/O controller
or via a PCIe switch directly attached to the CPU’s Root I/O controller.

In our experience, many issues stem from trying to use consumer
motherboards which provide physical x16 connectors that are electrically
connected as e.g. PCIe 2.0 x4, PCIe slots connected via the Southbridge
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Limited support
'''''''''''''''

-  ROCm 2.0.x should support PCIe 2.0 enabled CPUs such as the AMD
   Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and older
   Intel Xeon and Intel Core Architecture and Pentium CPUs. However, we
   have done very limited testing on these configurations, since our
   test farm has been catering to CPUs listed above. This is where we
   need community support. *If you find problems on such setups, please
   report these issues*.
-  Thunderbolt 1, 2, and 3 enabled breakout boxes should now be able to
   work with ROCm. Thunderbolt 1 and 2 are PCIe 2.0 based, and thus are
   only supported with GPUs that do not require PCIe 3.0 atomics
   (e.g. Vega 10). However, we have done no testing on this
   configuration and would need community support due to limited access
   to this type of equipment.
-  AMD “Carrizo” and “Bristol Ridge” APUs are enabled to run OpenCL, but
   do not yet support HCC, HIP, or our libraries built on top of these
   compilers and runtimes.

   -  As of ROCm 2.0, “Carrizo” and “Bristol Ridge” require the use of
      upstream kernel drivers.
   -  In addition, various “Carrizo” and “Bristol Ridge” platforms may
      not work due to OEM and ODM choices when it comes to key
      configurations parameters such as inclusion of the required CRAT
      tables and IOMMU configuration parameters in the system BIOS.
   -  Before purchasing such a system for ROCm, please verify that the
      BIOS provides an option for enabling IOMMUv2 and that the system
      BIOS properly exposes the correct CRAT table. Inquire with your
      vendor about the latter.

-  AMD “Raven Ridge” APUs are enabled to run OpenCL, but do not yet
   support HCC, HIP, or our libraries built on top of these compilers
   and runtimes.

   -  As of ROCm 2.0, “Raven Ridge” requires the use of upstream kernel
      drivers.
   -  In addition, various “Raven Ridge” platforms may not work due to
      OEM and ODM choices when it comes to key configurations parameters
      such as inclusion of the required CRAT tables and IOMMU
      configuration parameters in the system BIOS.
   -  Before purchasing such a system for ROCm, please verify that the
      BIOS provides an option for enabling IOMMUv2 and that the system
      BIOS properly exposes the correct CRAT table. Inquire with your
      vendor about the latter.

Not supported
'''''''''''''''

-  “Tonga”, “Iceland”, “Vega M”, and “Vega 12” GPUs are not supported in
   ROCm 2.0.x
-  We do not support GFX8-class GPUs (Fiji, Polaris, etc.) on CPUs that
   do not have PCIe 3.0 with PCIe atomics.

   -  As such, we do not support AMD Carrizo and Kaveri APUs as hosts for
      such GPUs.
   -  Thunderbolt 1 and 2 enabled GPUs are not supported by GFX8 GPUs on
      ROCm. Thunderbolt 1 & 2 are based on PCIe 2.0.

The latest ROCm platform - ROCm 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latest supported version of the drivers, tools, libraries and source
code for the ROCm platform have been released and are available from the
following GitHub repositories:

-  ROCm Core Components

   -  `ROCk Kernel Driver`_
   -  `ROCr Runtime`_
   -  `ROCt Thunk Interface`_

-  ROCm Support Software

   -  `ROCm SMI`_
   -  `ROCm cmake`_
   -  `rocminfo`_
   -  `ROCm Bandwidth Test`_

-  ROCm Development Tools

   -  `HCC compiler`_
   -  `HIP`_
   -  `ROCm Device Libraries`_
   -  ROCm OpenCL, which is created from the following components:

      -  `ROCm OpenCL Runtime`_
      -  `ROCm OpenCL Driver`_
      -  The ROCm OpenCL compiler, which is created from the following
         components:

         -  `ROCm LLVM`_
         -  `ROCm Clang`_
         -  `ROCm lld`_
         -  `ROCm Device Libraries`_

   -  `ROCM Clang-OCL Kernel Compiler`_
   -  `Asynchronous Task and Memory Interface (ATMI)`_
   -  `ROCr Debug Agent`_
   -  `ROCm Code Object Manager`_
   - `ROC Profiler`_
   - `Radeon Compute Profiler`_

   -  Example Applications:

      -  `HCC Examples`_
      -  `HIP Examples`_

-  ROCm Libraries

   -  `rocBLAS`_
   -  `hipBLAS`_
   -  `rocFFT`_
   -  `rocRAND`_
   -  `rocSPARSE`_
   -  `hipSPARSE`_
   -  `rocALUTION`_
   -  `MIOpenGEMM`_
   -  `MIOpen`_
   -  `HIP Thrust`_
   -  `ROCm SMI Lib`_
   -  `RCCL`_
   -  `MIVisionX`_
   -  `CUB HIP`_

.. _ROCk Kernel Driver: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/roc-2.0.0
.. _ROCr Runtime: https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/roc-2.0.0
.. _ROCt Thunk Interface: https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/roc-2.0.0
.. _ROCm SMI: https://github.com/RadeonOpenCompute/ROC-smi/tree/roc-2.0.0
.. _ROCm cmake: https://github.com/RadeonOpenCompute/rocm-cmake/tree/ac45c6e6
.. _rocminfo: https://github.com/RadeonOpenCompute/rocminfo/tree/1bb0ccc7
.. _ROCm Bandwidth Test: https://github.com/RadeonOpenCompute/rocm_bandwidth_test/tree/roc-2.0.0
.. _HCC compiler: https://github.com/RadeonOpenCompute/hcc/tree/roc-2.0.0
.. _HIP: https://github.com/ROCm-Developer-Tools/HIP/tree/roc-2.0.0
.. _ROCm Device Libraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/roc-2.0.0
.. _ROCm OpenCL Runtime: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/tree/roc-2.0.0
.. _ROCm OpenCL Driver: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Driver/tree/roc-2.0.0
.. _ROCm LLVM: http://github.com/RadeonOpenCompute/llvm/tree/roc-2.0.0
.. _ROCm Clang: http://github.com/RadeonOpenCompute/clang/tree/roc-2.0.0
.. _ROCm lld: http://github.com/RadeonOpenCompute/lld/tree/roc-2.0.0
.. _ROCM Clang-OCL Kernel Compiler: https://github.com/RadeonOpenCompute/clang-ocl/tree/688fe5d9
.. _Asynchronous Task and Memory Interface (ATMI): https://github.com/RadeonOpenCompute/atmi/tree/4dd14ad8
.. _ROCr Debug Agent: https://github.com/ROCm-Developer-Tools/rocr_debug_agent/tree/roc-2.0.0
.. _ROCm Code Object Manager: https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/tree/roc-2.0.0
.. _ROC Profiler: https://github.com/ROCm-Developer-Tools/rocprofiler/tree/roc-2.0.0
.. _Radeon Compute Profiler: https://github.com/GPUOpen-Tools/RCP/tree/v5.6
.. _HCC Examples: https://github.com/ROCm-Developer-Tools/HCC-Example-Application/tree/ffd65333
.. _HIP Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/roc-2.0.x
.. _rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/v2.0.0
.. _hipBLAS: https://github.com/ROCmSoftwarePlatform/hipBLAS/tree/v0.12.1.0
.. _rocFFT: https://github.com/ROCmSoftwarePlatform/rocFFT/tree/v0.8.8
.. _rocRAND: https://github.com/ROCmSoftwarePlatform/rocRAND/tree/7278524e
.. _rocSPARSE: https://github.com/ROCmSoftwarePlatform/rocSPARSE/tree/v1.0.1
.. _hipSPARSE: https://github.com/ROCmSoftwarePlatform/hipSPARSE/tree/v1.0.2
.. _rocALUTION: https://github.com/ROCmSoftwarePlatform/rocALUTION/tree/v1.3.7
.. _MIOpenGEMM: https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/tree/9547fb9e
.. _MIOpen: https://github.com/ROCmSoftwarePlatform/MIOpen/tree/1.7.0
.. _HIP Thrust: https://github.com/ROCmSoftwarePlatform/Thrust/tree/e0b8fe2a
.. _ROCm SMI Lib: https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/roc-2.0.0
.. _RCCL: https://github.com/ROCmSoftwarePlatform/rccl/tree/0.7.1
.. _MIVisionX: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/1.0.0
.. _CUB HIP: https://github.com/ROCmSoftwarePlatform/cub-hip/tree/hip_port_1.7.4


Supported Operating Systems - New operating systems available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm 2.0.x platform supports the following operating systems:

-  Ubuntu 16.04.x and 18.04.x (Version 16.04.3 and newer or kernels 4.13
   and newer)
-  CentOS 7.4, 7.5, and 7.6 (Using devtoolset-7 runtime support)
-  RHEL 7.4, 7.5, and 7.6 (Using devtoolset-7 runtime support)

ROCm support in upstream Linux kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of ROCm 1.9.0, the ROCm user-level software is compatible with the
AMD drivers in certain upstream Linux kernels. As such, users have the
option of either using the ROCK kernel driver that are part of AMD’s
ROCm repositories or using the upstream driver and only installing ROCm
user-level utilities from AMD’s ROCm repositories.

These releases of the upstream Linux kernel support the following GPUs
in ROCm:

-  4.17: Fiji, Polaris 10, Polaris 11
-  4.18: Fiji, Polaris 10, Polaris 11, Vega10

The upstream driver may be useful for running ROCm software on systems
that are not compatible with the kernel driver available in AMD’s
repositories. For users that have the option of using either AMD’s or
the upstreamed driver, there are various tradeoffs to take into
consideration:

+------+-------------------------------------------------+-------------------------------------------+
|      | Using AMD's `rock-dkms` package                 | Using the upstream kernel driver          |
+======+=================================================+===========================================+
| Pros | More GPU features, and they are enabled earlier | Includes the latest Linux kernel features |
+------+-------------------------------------------------+-------------------------------------------+
|      | Tested by AMD on supported distributions        | May work on other distributions and with  |
|      |                                                 | custom kernels                            |
+------+-------------------------------------------------+-------------------------------------------+
|      | Supported GPUs enabled regardless of            +                                           |
|      | kernel version                                  |                                           |
+------+-------------------------------------------------+-------------------------------------------+
|      | Includes the latest GPU firmware                |                                           |
+------+-------------------------------------------------+-------------------------------------------+
| Cons | May not work on all Linx distributions or       | Features and hardware support varies      |
|      | versions                                        | depending on kernel version               |
+------+-------------------------------------------------+-------------------------------------------+
|      | Not currently supported on kernels newer        | Limits GPU's usage of system memory to    |
|      | than 4.18.                                      | 3/8 of system memory                      |
+------+-------------------------------------------------+-------------------------------------------+
|      |                                                 | IPC and RDMA capabilities not yet enabled |
+------+-------------------------------------------------+-------------------------------------------+
|      |                                                 | Not tested by AMD to the same level as    |
|      |                                                 | `rock-dkms` package                       |
+------+-------------------------------------------------+-------------------------------------------+
|      |                                                 | Does not include most up-to-date firmware |
+------+-------------------------------------------------+-------------------------------------------+

Installing from AMD ROCm repositories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD hosts both `Debian`_ and `RPM`_ repositories for the ROCm 2.0.x
packages at this time.

The packages in the Debian repository have been signed to ensure package
integrity.

.. _Debian: http://repo.radeon.com/rocm/apt/debian/
.. _RPM: http://repo.radeon.com/rocm/yum/rpm/

ROCm Binary Package Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm is a collection of software ranging from drivers and runtimes to
libraries and developer tools. In AMD’s package distributions, these
software projects are provided as a separate packages. This allows users
to install only the packages they need, if they do not wish to install
all of ROCm. These packages will install most of the ROCm software into
``/opt/rocm/`` by default.

The packages for each of the major ROCm components are:

-  ROCm Core Components

   -  ROCk Kernel Driver: ``rock-dkms``
   -  ROCr Runtime: ``hsa-rocr-dev``, ``hsa-ext-rocr-dev``
   -  ROCt Thunk Interface: ``hsakmt-roct``, ``hsakmt-roct-dev``

-  ROCm Support Software

   -  ROCm SMI: ``rocm-smi``
   -  ROCm cmake: ``rocm-cmake``
   -  rocminfo: ``rocminfo``
   -  ROCm Bandwidth Test: ``rocm_bandwidth_test``

-  ROCm Development Tools

   -  HCC compiler: ``hcc``
   -  HIP: ``hip_base``, ``hip_doc``, ``hip_hcc``, ``hip_samples``
   -  ROCm Device Libraries: ``rocm-device-libs``
   -  ROCm OpenCL: ``rocm-opencl``, ``rocm-opencl-devel`` (on
      RHEL/CentOS), ``rocm-opencl-dev`` (on Ubuntu)
   -  ROCM Clang-OCL Kernel Compiler: ``rocm-clang-ocl``
   -  Asynchronous Task and Memory Interface (ATMI): ``atmi``
   -  ROCr Debug Agent: ``rocr_debug_agent``
   -  ROCm Code Object Manager: ``comgr``
   -  ROC Profiler: ``rocprofiler-dev``
   -  Radeon Compute Profiler: ``rocm-profiler``

-  ROCm Libraries

   -  rocBLAS: ``rocblas``
   -  hipBLAS: ``hipblas``
   -  rocFFT: ``rocfft``
   -  rocRAND: ``rocrand``
   -  rocSPARSE: ``rocsparse``
   -  hipSPARSE: ``hipsparse``
   -  rocALUTION: ``rocalution:``
   -  MIOpenGEMM: ``miopengemm``
   -  MIOpen: ``MIOpen-HIP`` (for the HIP version), ``MIOpen-OpenCL``
      (for the OpenCL version)
   -  HIP Thrust: ``thrust`` (on RHEL/CentOS), ``hip-thrust`` (on
      Ubuntu)
   -  ROCm SMI Lib: ``rocm_smi_lib64``
   -  RCCL: ``rccl``
   -  MIVisionX: ``mivisionx``
   -  CUB HIP: `cub-hip`

To make it easier to install ROCm, the AMD binary repos provide a number
of meta-packages that will automatically install multiple other
packages. For example, ``rocm-dkms`` is the primary meta-package that is
used to install most of the base technology needed for ROCm to operate.
It will install the ``rock-dkms`` kernel driver, and another
meta-package (``rocm-dev``) which installs most of the user-land ROCm
core components, support software, and development tools.

The ``rocm-utils`` meta-package will install useful utilities that,
while not required for ROCm to operate, may still be beneficial to have.
Finally, the ``rocm-libs`` meta-package will install some (but not all)
of the libraries that are part of ROCm.

The chain of software installed by these meta-packages is illustrated
below

::

   rocm-dkms
    |-- rock-dkms
    \-- rocm-dev
         |--hsa-rocr-dev
         |--hsa-ext-rocr-dev
         |--rocm-device-libs
         |--rocm-utils
             |-- rocminfo
             |-- rocm-cmake
             \-- rocm-clang-ocl # This will cause OpenCL to be installed
         |--hcc
         |--hip_base
         |--hip_doc
         |--hip_hcc
         |--hip_samples
         |--rocm-smi
         |--hsakmt-roct
         |--hsakmt-roct-dev
         |--hsa-amd-aqlprofile
         |--comgr
         \--rocr_debug_agent

   rocm-libs
    |-- rocblas
    |-- rocfft
    |-- rocrand
    \-- hipblas

These meta-packages are not required but may be useful to make it easier
to install ROCm on most systems. Some users may want to skip certain
packages. For instance, a user that wants to use the upstream kernel
drivers (rather than those supplied by AMD) may want to skip the
``rocm-dkms`` and ``rock-dkms`` packages, and instead directly install
``rocm-dev``.

Similarly, a user that only wants to install OpenCL support instead of
HCC and HIP may want to skip the ``rocm-dkms`` and ``rocm-dev``
packages. Instead, they could directly install ``rock-dkms``,
``rocm-opencl``, and ``rocm-opencl-dev`` and their dependencies.

Ubuntu Support - installing from a Debian repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following directions show how to install ROCm on supported
Debian-based systems such as Ubuntu 18.04. These directions may not work
as written on unsupported Debian-based distributions. For example, newer
versions of Ubuntu may not be compatible with the ``rock-dkms`` kernel
driver. As such, users may want to skip the ``rocm-dkms`` and
``rock-dkms`` packages, as described above, and instead use the
upstream kernel driver.

First make sure your system is up to date
'''''''''''''''''''''''''''''''''''''''''

.. code:: shell

   sudo apt update
   sudo apt dist-upgrade
   sudo apt install libnuma-dev
   sudo reboot

Add the ROCm apt repository
'''''''''''''''''''''''''''

For Debian-based systems like Ubuntu, configure the Debian ROCm
repository as follows:

.. code:: shell

   wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

The gpg key might change, so it may need to be updated when installing a
new release. If the key signature verification is failed while update,
please re-add the key from ROCm apt repository. The current rocm.gpg.key
is not available in a standard key ring distribution, but has the
following sha1sum hash:

``f7f8147431c75e505c58a6f3a3548510869357a6  rocm.gpg.key``

Install
'''''''

Next, update the apt repository list and install the ``rocm-dkms``
meta-package:

.. code:: shell

   sudo apt update
   sudo apt install rocm-dkms

Next set your permissions
'''''''''''''''''''''''''

Users will need to be in the ``video`` group in order to have access to
the GPU. As such, you should ensure that your user account is a member
of the ``video`` group prior to using ROCm. You can find which groups
you are a member of with the following command:

.. code:: shell

   groups

To add yourself to the video group you will need the sudo password and
can use the following command:

.. code:: shell

   sudo usermod -a -G video $LOGNAME 

You may want to ensure that any future users you add to your system are
put into the “video” group by default. To do that, you can run the
following commands:

.. code:: shell

   echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
   echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

Once complete, reboot your system.

Test basic ROCm installation
''''''''''''''''''''''''''''

After rebooting the system run the following commands to verify that the
ROCm installation was successful. If you see your GPUs listed by both of
these commands, you should be ready to go!

.. code:: shell

   /opt/rocm/bin/rocminfo 
   /opt/rocm/opencl/bin/x86_64/clinfo 

Note that, to make running ROCm programs easier, you may wish to put the
ROCm binaries in your PATH.

.. code:: shell

   echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh

If you have an `install issue`_ please read this FAQ.

Performing an OpenCL-only Installation of ROCm
''''''''''''''''''''''''''''''''''''''''''''''

Some users may want to install a subset of the full ROCm installation.
In particular, if you are trying to install on a system with a limited
amount of storage space, or which will only run a small collection of
known applications, you may want to install only the packages that are
required to run OpenCL applications. To do that, you can run the
following installation command **instead** of the command to install
``rocm-dkms``.

.. code:: shell

   sudo apt-get install dkms rock-dkms rocm-opencl-dev

How to uninstall from Ubuntu 16.04 or Ubuntu 18.04
''''''''''''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages installed in the above directions, you
can execute;

.. code:: shell

   sudo apt autoremove rocm-dkms rocm-dev rocm-utils

Installing development packages for cross compilation
'''''''''''''''''''''''''''''''''''''''''''''''''''''

It is often useful to develop and test on different systems. For
example, some development or build systems may not have an AMD GPU
installed. In this scenario, you may prefer to avoid installing the ROCK
kernel driver to your development system.

In this case, install the development subset of packages:

.. code:: shell

   sudo apt update
   sudo apt install rocm-dev

..

   **Note:** To execute ROCm enabled apps you will require a system with
   the full ROCm driver stack installed

.. _install issue: https://rocm.github.io/install_issues.html

Using Debian-based ROCm with upstream kernel drivers
''''''''''''''''''''''''''''''''''''''''''''''''''''

As described in the above section about upstream Linux kernel
support, users may want to try installing ROCm user-level software
without installing AMD’s custom ROCK kernel driver. Users who do want to
use upstream kernels can run the following commands instead of
installing ``rocm-dkms``

.. code:: shell

   sudo apt update
   sudo apt install rocm-dev
   echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules


CentOS/RHEL 7 (7.4, 7.5, 7.6) Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following directions show how to install ROCm on supported RPM-based
systems such as CentOS 7.6. These directions may not work as written on
unsupported RPM-based distributions. For example, Fedora may work but
may not be compatible with the ``rock-dkms`` kernel driver. As such,
users may want to skip the ``rocm-dkms`` and ``rock-dkms`` packages, as
described above, and instead use the upstream kernel driver.

Support for CentOS/RHEL 7 was added in ROCm 1.8, but ROCm requires a
special runtime environment provided by the RHEL Software Collections
and additional dkms support packages to properly install and run.

Preparing RHEL 7 (7.4, 7.5, 7.6) for installation
'''''''''''''''''''''''''''''''''''''''''''''''''

RHEL is a subscription-based operating system, and you must enable
several external repositories to enable installation of the devtoolset-7
environment and the DKMS support files. These steps are not required for
CentOS.

First, the subscription for RHEL must be enabled and attached to a pool
id. Please see Obtaining an RHEL image and license page for instructions
on registering your system with the RHEL subscription server and
attaching to a pool id.

Second, enable the following repositories:

.. code:: shell

   sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
   sudo subscription-manager repos --enable rhel-7-server-optional-rpms
   sudo subscription-manager repos --enable rhel-7-server-extras-rpms

Third, enable additional repositories by downloading and installing the
epel-release-latest-7 repository RPM:

.. code:: shell

   sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

Install and setup Devtoolset-7
''''''''''''''''''''''''''''''

To setup the Devtoolset-7 environment, follow the instructions on this
page:

https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

Note that devtoolset-7 is a Software Collections package, and it is not
supported by AMD.

Prepare CentOS/RHEL (7.4, 7.5, 7.6) for DKMS Install
''''''''''''''''''''''''''''''''''''''''''''''''''''

Installing kernel drivers on CentOS/RHEL 7.4/7.5/7.6 requires dkms tool
being installed:

.. code:: shell

   sudo yum install -y epel-release
   sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

Installing ROCm on the system
'''''''''''''''''''''''''''''

It is recommended to remove previous ROCm installations before
installing the latest version to ensure a smooth installation.

At this point ROCm can be installed on the target system. Create a
/etc/yum.repos.d/rocm.repo file with the following contents:

.. code:: shell

   [ROCm]
   name=ROCm
   baseurl=http://repo.radeon.com/rocm/yum/rpm
   enabled=1
   gpgcheck=0

The repo’s URL should point to the location of the repositories repodata
database. Install ROCm components using these commands:

.. code:: shell

   sudo yum install rocm-dkms

The rock-dkms component should be installed and the ``/dev/kfd`` device
should be available on reboot.

Set up permissions
''''''''''''''''''

Ensure that your user account is a member of the “video” or “wheel”
group prior to using the ROCm driver. You can find which groups you are
a member of with the following command:

.. code:: shell

   groups

To add yourself to the video (or wheel) group you will need the sudo
password and can use the following command:

.. code:: shell

   sudo usermod -a -G video $LOGNAME 

You may want to ensure that any future users you add to your system are
put into the “video” group by default. To do that, you can run the
following commands:

.. code:: shell

   echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
   echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

Current release supports CentOS/RHEL 7.4, 7.5, 7.6. If users want to
update the OS version, they should completely remove ROCm packages
before updating to the latest version of the OS, to avoid DKMS related
issues.

Once complete, reboot your system.

Test basic ROCm installation
                            

After rebooting the system run the following commands to verify that the
ROCm installation was successful. If you see your GPUs listed by both of
these commands, you should be ready to go!

.. code:: shell

   /opt/rocm/bin/rocminfo
   /opt/rocm/opencl/bin/x86_64/clinfo

Note that, to make running ROCm programs easier, you may wish to put the
ROCm binaries in your PATH.

.. code:: shell

   echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh

If you have an `install issue`_ please read this FAQ.

.. _install issue: https://rocm.github.io/install_issues.html

Performing an OpenCL-only Installation of ROCm
                                              

Some users may want to install a subset of the full ROCm installation.
In particular, if you are trying to install on a system with a limited
amount of storage space, or which will only run a small collection of
known applications, you may want to install only the packages that are
required to run OpenCL applications. To do that, you can run the
following installation command **instead** of the command to install
``rocm-dkms``.

.. code:: shell

   sudo yum install rock-dkms rocm-opencl-devel

Compiling applications using HCC, HIP, and other ROCm software
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

To compile applications or samples, please use gcc-7.2 provided by the
devtoolset-7 environment. To do this, compile all applications after
running this command:

.. code:: shell

   scl enable devtoolset-7 bash

How to uninstall ROCm from CentOS/RHEL 7.4, 7.5 and 7.6
'''''''''''''''''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages installed by the above directions, you
can execute:

.. code:: shell

   sudo yum autoremove rocm-dkms rock-dkms

Installing development packages for cross compilation
'''''''''''''''''''''''''''''''''''''''''''''''''''''

It is often useful to develop and test on different systems. For
example, some development or build systems may not have an AMD GPU
installed. In this scenario, you may prefer to avoid installing the ROCK
kernel driver to your development system.

In this case, install the development subset of packages:

.. code:: shell

   sudo yum install rocm-dev

..

   **Note:** To execute ROCm enabled apps you will require a system with
   the full ROCm driver stack installed

Using ROCm with upstream kernel drivers
'''''''''''''''''''''''''''''''''''''''

As described in the above section about upstream Linux kernel
support, use rs may want to try installing ROCm user-level software
without installing AMD’s custom ROCK kernel driver. Users who do want to
use upstream kernels can run the following commands instead of
installing ``rocm-dkms``

.. code:: shell

   sudo yum install rocm-dev
   echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules

Known issues / workarounds
~~~~~~~~~~~~~~~~~~~~~~~~~~

HCC: removed support for C++AMP in ROCm 2.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HipCaffe is supported on single GPU configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm SMI library calls to rsmi_dev_power_cap_set() and rsmi_dev_power_profile_set() will not work for all but the first gpu in multi-gpu set ups.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Closed source components
~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm platform relies on a few closed source components to provide
functionality such as HSA image support. These components are only
available through the ROCm repositories, and they will either be
deprecated or become open source components in the future. These
components are made available in the following packages:

-  hsa-ext-rocr-dev
 
Getting ROCm source code
~~~~~~~~~~~~~~~~~~~~~~~~

ROCm is built from open source software. As such, it is possible to make
modifications to the various components of ROCm by downloading the
source code, making modifications to it, and rebuilding the components.
The source code for ROCm components can be cloned from each of the
GitHub repositories using git. In order to make it easier to download
the correct versions of each of these tools, this ROCm repository
contains a `repo`_ manifest file, `default.xml`_. Interested users can
thus use this manifest file to download the source code for all of the
ROCm software.

Installing repo
^^^^^^^^^^^^^^^

Google’s repo tool allows you to manage multiple git repositories
simultaneously. You can install it by executing the following example
commands:

.. code:: shell

   mkdir -p ~/bin/
   curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
   chmod a+x ~/bin/repo

Note that you can choose a different folder to install repo into if you
desire. ``~/bin/`` is simply used as an example.

Downloading the ROCm source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example shows how to use the ``repo`` binary downloaded
above to download all of the ROCm source code. If you chose a directory
other than ``~/bin/`` to install ``repo``, you should use that directory
below.

.. code:: shell

   mkdir -p ~/ROCm/
   cd ~/ROCm/
   ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-2.0.0
   repo sync

This will cause repo to download all of the open source code associated
with this ROCm release. You may want to ensure that you have ssh-keys
configured on your machine for your GitHub ID.

Building the ROCm source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ROCm component repository contains directions for building that
component. As such, you should go to the repository you are interested
in building to find how to build it.

That said, AMD also offers `a project`_ that demonstrates how to
download, build, package, and install ROCm software on various
distributions. The scripts here may be useful for anyone looking to
build ROCm components.

Final notes
~~~~~~~~~~~

-  OpenCL Runtime and Compiler will be submitted to the Khronos Group
   for conformance testing prior to its final release.

.. _repo: https://gerrit.googlesource.com/git-repo/
.. _default.xml: default.xml
.. _a project: https://github.com/RadeonOpenCompute/Experimental_ROC
