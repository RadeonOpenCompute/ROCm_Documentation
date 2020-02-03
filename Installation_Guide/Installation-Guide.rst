==============================
Installation Guide
==============================

AMD ROCm Release Notes v3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues and deprecated features in the ROCm v3.0 release.

-  `What is ROCm`_

   -  `ROCm Components`_
   -  `Supported Operating Systems`_
   -  `Important ROCm Links`_


-  `Whats New in This Release`_

   -  `Support for CentOS RHEL v7.7`_
   -  `Initial distribution of AOMP 0.7-5 in ROCm v3.0`_
   -  `Fast Fourier Transform Updates`_
   -  `MemCopy Enhancement for rocProf`_

-  `Fixed Issues`_

   -  `MIGraph v0.5 Graph Optimizer`_


-  `Known Issues in This Release`_

   -  `Installation Issue with Red Hat Enterprise Linux v7.7`_
   -  `Error While Running rocProfiler on SLES`_
   -  `gpuOwl Fails with Memory Access Fault Error`_


-  `Deprecated Features`_
 
-  `MIOpen`_

   -  `SCGEMM Convolution Algorithmn`_
   -  `Text-Based Performance Database`_



-  `Deploying ROCm`_

   -  `Ubuntu`_
   -  `Centos RHEL`_



-  `ROCM Installation`_



-  `Getting the ROCm Source Code`_



-  `Hardware and Software Support`_



-  `Machine Learning and High Performance Computing Software Stack for AMD GPU`_


   -  `ROCm Binary Package Structure`_
   -  `ROCm Platform Packages`_
 

.. _What is ROCm:

What Is ROCm?
~~~~~~~~~~~~~~~~~~

ROCm is designed to be a universal platform for gpu-accelerated computing. This modular design allows hardware vendors to build drivers that support the ROCm framework. ROCm is also designed to integrate multiple programming languages and makes it easy to add support for other languages.

Note: You can also clone the source code for individual ROCm components from the GitHub repositories.

.. _ROCm Components:

ROCm Components
^^^^^^^^^^^^^^^^

The following components for the ROCm platform are released and available for the v3.0 release:

• Drivers

• Tools

• Libraries

• Source Code

You can access the latest supported version of drivers, tools, libraries, and source code for the ROCm platform at the following location: https://github.com/RadeonOpenCompute/ROCm

.. _Supported Operating Systems:

Supported Operating Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm v3.0.x platform is designed to support the following operating systems:

• SLES 15 SP1

• Ubuntu 16.04.6(Kernel 4.15) and 18.04.3(Kernel 5.0)

• CentOS 7.7 (Using devtoolset-7 runtime support)

• RHEL 7.7 (Using devtoolset-7 runtime support)

For details about deploying the ROCm v3.0.x on these operating systems, see the Deploying ROCm section later in the document.

.. _Important ROCm Links:

Important ROCm Links
^^^^^^^^^^^^^^^^^^^^^

Access the following links for more information on: 


• ROCm binary structure, see https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#rocm-binary-package-structure

• Common ROCm installation issues, see https://rocm.github.io/install_issues.html

• Instructions to install PyTorch after ROCm is installed – https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#pytorch

Note: These instructions reference the rocm/pytorch:rocm3.0_ubuntu16.04_py2.7_pytorch image. However, you can substitute the Ubuntu 18.04 image listed at https://hub.docker.com/r/rocm/pytorch/tags

.. _Whats New in This Release:

Whats New in This Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _Support for CentOS RHEL v7.7:

Support for CentOS RHEL v7.7
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support is extended for CentOS/RHEL v7.7 in the ROCm v3.0 release. For more information about the CentOS/RHEL v7.7 release, see:
https://centos.org/forums/viewtopic.php?t=71657.

.. _Initial distribution of AOMP 0.7-5 in ROCm v3.0:

Initial distribution of AOMP 0.7-5 in ROCm v3.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code base for this release of AOMP is the Clang/LLVM 9.0 sources as of October 8th, 2019. The LLVM-project branch used to build this release is AOMP-191008. It is now locked. With this release, an artifact tarball of the entire source tree is created. This tree includes a Makefile in the root directory used to build AOMP from the release tarball. You can use Spack to build AOMP from this source tarball or build manually without Spack.

For more information about AOMP 0.7-5, see: `AOMP <https://github.com/ROCm-Developer-Tools/aomp/tree/roc-3.0.0>`_

.. _Fast Fourier Transform Updates:

Fast Fourier Transform Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform. Fast Fourier transforms are used in signal processing, image processing, and many other areas. The following real FFT performance change is made in the ROCm v3.0 release:

• Implement efficient real/complex 2D transforms for even lengths.

Other improvements:

• More 2D test coverage sizes.

• Fix buffer allocation error for large 1D transforms.

• C++ compatibility improvements.

.. _MemCopy Enhancement for rocProf:

MemCopy Enhancement for rocProf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the v3.0 release, the rocProf tool is enhanced with an additional capability to dump asynchronous GPU memcopy information into a .csv file. You can use the '-hsa-trace' option to create the results_mcopy.csv file. Future enhancements will include column labels.

.. _Fixed Issues:

Fixed Issues in This Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _MIGraph v0.5 Graph Optimizer:

MIGraph v05 Graph Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm v3.0 release consists of performance updates and minor bug fixes for the MIGraphX graph optimizer. For more information, see

https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.5


.. _Known Issues in This Release:

Known Issues in This Release 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _Installation Issue with Red Hat Enterprise Linux v7.7:
   
Installation Issue with Red Hat Enterprise Linux v7.7
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue:** ROCm installation fails on Red Hat Enterprise Linux (RHEL) v7.7.

**Resolution:** Ensure the following repo is installed and available prior to installing ROCm on RHEL v7.7:

**Note:**

For workstations, use

:: 

   rhel-7-workstation-optional-rpms


For servers, use

::

   rhel-7-server-optional-rpms


**To install**

::

   $sudo subscription-manager repos --enable=rhel-7-workstation-optional-rpms


You will see the following message:

Repository 'rhel-7-workstation-optional-rpms' is enabled for this system.

If the following error message appears,

Error: 'rhel-7-workstation-optional-rpms' does not match a valid repository ID. Use "subscription-manager repos --list" to see valid repositories.

Use

::
 
   $sudo subscription-manager repos --enable=rhel-7-server-optional-rpms


You will see the following message:

Repository 'rhel-7-server-optional-rpms' is enabled for this system.

.. _Error While Running rocProfiler on SLES:

Error While Running rocProfiler on SLES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue:** Running rocprofiler: hip/hsa trace results in the following error. Note, this issue is noticed only on SLES.

ImportError: No module named sqlite3

**Resolution:** The following workarounds are recommended:

**Workaround 1**

    1. Run the following command

::

   sudo vi /opt/rocm/bin/rocprof


    2. Change Python to Python3.6.

    3. Save and run the test again.

**Workaround 2:**

• Run the following command:

::

   alias python=python3.6


.. _gpuOwl Fails with Memory Access Fault Error:

gpuOwl Fails with Memory Access Fault Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue:** gpuOwL is an OpenCL-based program for testing Mersenne numbers for primality. Currently, running gpuOwl for higher probable prime (PRP) values results in a Memory Access Fault error.

Note, the issue is noticed only when using higher PRP values.

**Resolution:** As a workaround, you may use lower PRP values.


.. _Deprecated Features:

Deprecated Features
~~~~~~~~~~~~~~~~~~~~~

The following features are deprecated in the AMD ROCm v3.0 release.

.. _MIOpen:

MIOpen
~~~~~~~

.. _SCGEMM Convolution Algorithmn:

SCGEMM Convolution Algorithmn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SCGEMM convolution algorithm is now disabled by default. This algorithm is deprecated and will be removed in future releases.

.. _Text-Based Performance Database:

Text-Based Performance Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An SQLite database has been added to replace the text-based performance database. While the text file still exists, by default, SQLite is used over the text-based performance database. The text-based performance database support is deprecated and will be removed in a future release.

.. _Deploying ROCm:

Deploying ROCm
~~~~~~~~~~~~~~~~

AMD hosts both Debian and RPM repositories for the ROCm v3.0x packages.

The following directions show how to install ROCm on supported Debian-based systems such as Ubuntu 18.04.x

Note: These directions may not work as written on unsupported Debian-based distributions. For example, newer versions of Ubuntu may not be compatible with the rock-dkms kernel driver. In this case, you can exclude the rocm-dkms and rock-dkms packages.

For more information on the ROCm binary structure, see https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#rocm-binary-package-structure

For information about upstream kernel drivers, see the Using Debian-based ROCm with Upstream Kernel Drivers section.

.. _Ubuntu:

Ubuntu
^^^^^^^^

Installing a ROCm Package from a Debian Repository
'''''''''''''''''''''''''''''''''''''''''''''''''''''

To install from a Debian Repository:

1. Run the following code to ensure that your system is up to date:

::

    sudo apt update

    sudo apt dist-upgrade

    sudo apt install libnuma-dev

    sudo reboot 




2. Add the ROCm apt repository.

  For Debian-based systems like Ubuntu, configure the Debian ROCm repository as follows:

::

    wget -qO –http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | 

    sudo apt-key add -echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | 

    sudo tee /etc/apt/sources.list.d/rocm.list


The gpg key may change; ensure it is updated when installing a new release. If the key signature verification fails while updating, re-add the key from the ROCm apt repository.

The current rocm.gpg.key is not available in a standard key ring distribution, but has the following sha1sum hash:

::

  e85a40d1a43453fe37d63aa6899bc96e08f2817a rocm.gpg.key




3. Install the ROCm meta-package. Update the appropriate repository list and install the rocm-dkms meta-package:

::

     sudo apt update

     sudo apt install rocm-dkms




4. Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

     groups




5. To add your user to the video group, use the following command for the sudo password:

::

     sudo usermod -a -G video $LOGNAME



6. By default, add any future users to the video group. Run the following command to add users to the video group:

::

     echo 'ADD_EXTRA_GROUPS=1' 
     sudo tee -a /etc/adduser.conf

     echo 'EXTRA_GROUPS=video'
     sudo tee -a /etc/adduser.conf



7. Restart the system.

8. Test the basic ROCm installation.

9. After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed by both commands, the installation is considered successful.

::

     /opt/rocm/bin/rocminfo
     /opt/rocm/opencl/bin/x86_64/clinfo

Note: To run the ROCm programs more efficiently, add the ROCm binaries in your PATH.

::

	echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | 
	sudo tee -a /etc/profile.d/rocm.sh

If you have an installation issue, refer the FAQ at: https://rocm.github.io/install_issues.html

Uninstalling ROCm Packages from Ubuntu
''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages from Ubuntu 1v6.04 or Ubuntu v18.04.x, run the following command:

::

  sudo apt autoremove rocm-dkms rocm-dev rocm-utils


Installing Development Packages for Cross Compilation
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

It is recommended that you develop and test development packages on different systems. For example, some development or build systems may not have an AMD GPU installed. In this scenario, you must avoid installing the ROCk kernel driver on the development system.

Instead, install the following development subset of packages:

::

  sudo apt update
  sudo apt install rocm-dev


Note: To execute ROCm enabled applications, you must install the full ROCm driver stack on your system.

Using Debian-based ROCm with Upstream Kernel Drivers
''''''''''''''''''''''''''''''''''''''''''''''''''''''

You can install the ROCm user-level software without installing the AMD's custom ROCk kernel driver. To use the upstream kernels, run the following commands instead of installing rocm-dkms:

::

  sudo apt update	
  sudo apt install rocm-dev	
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' 
  sudo tee /etc/udev/rules.d/70-kfd.rules


.. _CentOS RHEL:

CentOS RHEL
^^^^^^^^^^^^^^

This section describes how to install ROCm on supported RPM-based systems such as CentOS v7.7.

For more details, refer: https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#rocm-binary-package-structure


Preparing RHEL v7 (7.7) for Installation
'''''''''''''''''''''''''''''''''''''''''''

RHEL is a subscription-based operating system. You must enable the external repositories to install on the devtoolset-7 environment and the dkms support files.

Note: The following steps do not apply to the CentOS installation.

1. The subscription for RHEL must be enabled and attached to a pool ID. See the Obtaining an RHEL image and license page for instructions on registering your system with the RHEL subscription server and attaching to a pool id.

2. Enable the following repositories:

::
   
    sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms 
    sudo subscription-manager repos --enable rhel-7-server-optional-rpms
    sudo subscription-manager repos --enable rhel-7-server-extras-rpms


3. Enable additional repositories by downloading and installing the epel-release-latest-7 repository RPM:

::

   sudo rpm -ivh


For more details, see https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

4. Install and set up Devtoolset-7.

To setup the Devtoolset-7 environment, follow the instructions on this page: https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

Note: devtoolset-7 is a software collections package and is not supported by AMD.

Installing CentOS/RHEL (v7.7) for DKMS
''''''''''''''''''''''''''''''''''''''''

Use the dkms tool to install the kernel drivers on CentOS/RHEL v7.7:

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

.. _ROCM install:

ROCm Installation
~~~~~~~~~~~~~~~~~~~

Installing ROCm
^^^^^^^^^^^^^^^^

To install ROCm on your system, follow the instructions below:

1. Delete the previous versions of ROCm before installing the latest version.

2. Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

    [ROCm] 
    name=ROCm
    baseurl=http://repo.radeon.com/rocm/yum/rpm 
    enabled=1
    gpgcheck=0

Note: The URL of the repository must point to the location of the repositories’ repodata database.

3. Install ROCm components using the following command:

::

    sudo yum install rocm-dkms


4. Restart the system. The rock-dkms component is installed and the /dev/kfd device is now available.

Setting Permissions
'''''''''''''''''''''

To configure permissions, following the instructions below:

1. Ensure that your user account is a member of the "video" or "wheel" group prior to using the ROCm driver. You can find which groups you are a member of with the following command:

::

    groups

2. Add your user to the video (or wheel) group you will need the sudo password and can use the following command:

::

    sudo usermod -a -G video $LOGNAME


Note: All future users must be added to the "video" group by default. To add the users to the group, run the following commands

::

  echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
  echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

Note: The current release supports CentOS/RHEL v7.6. Before updating to the latest version of the operating system, delete the ROCm packages to avoid DKMS-related issues.

3. Restart the system.


Testing the ROCm Installation
'''''''''''''''''''''''''''''''


After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed, you are good to go!

::

  /opt/rocm/bin/rocminfo
  /opt/rocm/opencl/bin/x86_64/clinfo


Note: Add the ROCm binaries in your PATH for easy implementation of the ROCm programs.

::

  echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' |
  sudo tee -a /etc/profile.d/rocm.sh

For more information about installation issues, see: https://rocm.github.io/install_issues.html


Performing an OpenCL-only Installation of ROCm
''''''''''''''''''''''''''''''''''''''''''''''''

Some users may want to install a subset of the full ROCm installation. If you are trying to install on a system with a limited amount of storage space, or which will only run a small collection of known applications, you may want to install only the packages that are required to run OpenCL applications. To do that, you can run the following installation command instead of the command to install rocm-dkms.

::

  sudo yum install rock-dkms rocm-opencl-devel


Compiling Applications Using HCC, HIP, and Other ROCm Software
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


To compile applications or samples, run the following command to use gcc-7.2 provided by the devtoolset-7 environment:

::

  scl enable devtoolset-7 bash


Uninstalling ROCm from CentOS/RHEL v7.7
''''''''''''''''''''''''''''''''''''''''



To uninstall the ROCm packages, run the following command:

::

  sudo yum autoremove rocm-dkms rock-dkms


Installing Development Packages for Cross Compilation
'''''''''''''''''''''''''''''''''''''''''''''''''''''''

You can develop and test ROCm packages on different systems. For example, some development or build systems may not have an AMD GPU installed. In this scenario, you can avoid installing the ROCm kernel driver on your development system. Instead, install the following development subset of packages:

::

  sudo yum install rocm-dev


Note: To execute ROCm-enabled applications, you will require a system installed with the full ROCm driver stack.

Using ROCm with Upstream Kernel Drivers
'''''''''''''''''''''''''''''''''''''''''


You can install ROCm user-level software without installing AMD's custom ROCk kernel driver. To use the upstream kernel drivers, run the following commands

::

  sudo yum install rocm-dev
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' 
  sudo tee /etc/udev/rules.d/70-kfd.rules

Note: You can use this command instead of installing rocm-dkms.


ROCm Installation - Known Issues and Workarounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Closed source components
''''''''''''''''''''''''''

The ROCm platform relies on some closed source components to provide functionalities like HSA image support. These components are only available through the ROCm repositories, and they may be deprecated or become open source components in the future. These components are made available in the following packages:

• hsa-ext-rocr-dev


Getting the ROCm Source Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD ROCm is built from open source software. It is, therefore, possible to modify the various components of ROCm by downloading the source code and rebuilding the components. The source code for ROCm components can be cloned from each of the GitHub repositories using git. For easy access to download the correct versions of each of these tools, the ROCm repository contains a repo manifest file called default.xml. You can use this manifest file to download the source code for ROCm software.

Installing the Repo
^^^^^^^^^^^^^^^^^^^^^


The repo tool from Google® allows you to manage multiple git repositories simultaneously. Run the following commands to install the repo:

::

  mkdir -p ~/bin/
  curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
  chmod a+x ~/bin/repo

Note: You can choose a different folder to install the repo into if you desire. ~/bin/ is used as an example.

Downloading the ROCm Source Code
''''''''''''''''''''''''''''''''''

The following example shows how to use the repo binary to download the ROCm source code. If you choose a directory other than ~/bin/ to install the repo, you must use that chosen directory in the code as shown below:

::

  mkdir -p ~/ROCm/
  cd ~/ROCm/
  ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.0.0
  repo sync


Note: Using this sample code will cause the repo to download the open source code associated with this ROCm release. Ensure that you have ssh-keys configured on your machine for your GitHub ID prior to the download.

Building the ROCm Source Code
'''''''''''''''''''''''''''''''

Each ROCm component repository contains directions for building that component. You can access the desired component for instructions to build the repository.

.. _Hardware and Software Support:

Hardware and Software Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm is focused on using AMD GPUs to accelerate computational tasks such as machine learning, engineering workloads, and scientific computing. In order to focus our development efforts on these domains of interest, ROCm supports a targeted set of hardware configurations which are detailed further in this section.

Supported GPUs
^^^^^^^^^^^^^^^^


Because the ROCm Platform has a focus on particular computational domains, we offer official support for a selection of AMD GPUs that are designed to offer good performance and price in these domains.

ROCm officially supports AMD GPUs that use following chips:

    * GFX8 GPUs
        * "Fiji" chips, such as on the AMD Radeon R9 Fury X and Radeon Instinct MI8
        * "Polaris 10" chips, such as on the AMD Radeon RX 580 and Radeon Instinct MI6
    * GFX9 GPUs
        * "Vega 10" chips, such as on the AMD Radeon RX Vega 64 and Radeon Instinct MI25
        * "Vega 7nm" chips, such as on the Radeon Instinct MI50, Radeon Instinct MI60 or AMD Radeon VII

ROCm is a collection of software ranging from drivers and runtimes to libraries and developer tools. Some of this software may work with more GPUs than the "officially supported" list above, though AMD does not make any official claims of support for these devices on the ROCm software platform. The following list of GPUs are enabled in the ROCm software, though full support is not guaranteed:

    * GFX8 GPUs
        * "Polaris 11" chips, such as on the AMD Radeon RX 570 and Radeon Pro WX 4100
        * "Polaris 12" chips, such as on the AMD Radeon RX 550 and Radeon RX 540
    * GFX7 GPUs
        * "Hawaii" chips, such as the AMD Radeon R9 390X and FirePro W9100

As described in the next section, GFX8 GPUs require PCI Express 3.0 (PCIe 3.0) with support for PCIe atomics. This requires both CPU and motherboard support. GFX9 GPUs require PCIe 3.0 with support for PCIe atomics by default, but they can operate in most cases without this capability.

The integrated GPUs in AMD APUs are not officially supported targets for ROCm. As described `below <https://github.com/RadeonOpenCompute/ROCm#limited-support>`_, "Carrizo", "Bristol Ridge", and "Raven Ridge" APUs are enabled in our upstream drivers and the ROCm OpenCL runtime. However, they are not enabled in our HCC or HIP runtimes, and may not work due to motherboard or OEM hardware limitations. As such, they are not yet officially supported targets for ROCm.

For a more detailed list of hardware support, please see the `following documentation <https://rocm.github.io/hardware.html>`_.

Supported CPUs
^^^^^^^^^^^^^^^^
As described above, GFX8 GPUs require PCIe 3.0 with PCIe atomics in order to run ROCm. In particular, the CPU and every active PCIe point between the CPU and GPU require support for PCIe 3.0 and PCIe atomics. The CPU root must indicate PCIe AtomicOp Completion capabilities and any intermediate switch must indicate PCIe AtomicOp Routing capabilities.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

    * AMD Ryzen CPUs
    * The CPUs in AMD Ryzen APUs
    * AMD Ryzen Threadripper CPUs
    * AMD EPYC CPUs
    * Intel Xeon E7 v3 or newer CPUs
    * Intel Xeon E5 v3 or newer CPUs
    * Intel Xeon E3 v3 or newer CPUs
    * Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e. Haswell family or newer)
    * Some Ivy Bridge-E systems

Beginning with ROCm 1.8, GFX9 GPUs (such as Vega 10) no longer require PCIe atomics. We have similarly opened up more options for number of PCIe lanes. GFX9 GPUs can now be run on CPUs without PCIe atomics and on older PCIe generations, such as PCIe 2.0. This is not supported on GPUs below GFX9, e.g. GFX8 cards in the Fiji and Polaris families.

If you are using any PCIe switches in your system, please note that PCIe Atomics are only supported on some switches, such as Broadcom PLX. When you install your GPUs, make sure you install them in a PCIe 3.0 x16, x8, x4, or x1 slot attached either directly to the CPU's Root I/O controller or via a PCIe switch directly attached to the CPU's Root I/O controller.

In our experience, many issues stem from trying to use consumer motherboards which provide physical x16 connectors that are electrically connected as e.g. PCIe 2.0 x4, PCIe slots connected via the Southbridge PCIe I/O controller, or PCIe slots connected through a PCIe switch that does not support PCIe atomics.

If you attempt to run ROCm on a system without proper PCIe atomic support, you may see an error in the kernel log (dmesg):

::

  kfd: skipped device 1002:7300, PCI rejects atomics


Experimental support for our Hawaii (GFX7) GPUs (Radeon R9 290, R9 390, FirePro W9100, S9150, S9170) does not require or take advantage of PCIe Atomics. However, we still recommend that you use a CPU from the list provided above for compatibility purposes.

Not supported or limited support under ROCm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Limited support
''''''''''''''''

    * ROCm 2.9.x should support PCIe 2.0 enabled CPUs such as the AMD Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and older Intel Xeon and Intel Core Architecture and Pentium CPUs. However, we have done very limited testing on these configurations, since our test farm has been catering to CPUs listed above. This is where we need community support. If you find problems on such setups, please report these issues.
    * Thunderbolt 1, 2, and 3 enabled breakout boxes should now be able to work with ROCm. Thunderbolt 1 and 2 are PCIe 2.0 based, and thus are only supported with GPUs that do not require PCIe 3.0 atomics (e.g. Vega 10). However, we have done no testing on this configuration and would need community support due to limited access to this type of equipment.
    * AMD "Carrizo" and "Bristol Ridge" APUs are enabled to run OpenCL, but do not yet support HCC, HIP, or our libraries built on top of these compilers and runtimes.
        * As of ROCm 2.1, "Carrizo" and "Bristol Ridge" require the use of upstream kernel drivers.
        * In addition, various "Carrizo" and "Bristol Ridge" platforms may not work due to OEM and ODM choices when it comes to key configurations parameters such as inclusion of the required CRAT tables and IOMMU configuration parameters in the system BIOS.
        * Before purchasing such a system for ROCm, please verify that the BIOS provides an option for enabling IOMMUv2 and that the system BIOS properly exposes the correct CRAT table. Inquire with your vendor about the latter.
    * AMD "Raven Ridge" APUs are enabled to run OpenCL, but do not yet support HCC, HIP, or our libraries built on top of these compilers and runtimes.
        * As of ROCm 2.1, "Raven Ridge" requires the use of upstream kernel drivers.
        * In addition, various "Raven Ridge" platforms may not work due to OEM and ODM choices when it comes to key configurations parameters such as inclusion of the required CRAT tables and IOMMU configuration parameters in the system BIOS.
        * Before purchasing such a system for ROCm, please verify that the BIOS provides an option for enabling IOMMUv2 and that the system BIOS properly exposes the correct CRAT table. Inquire with your vendor about the latter.

Not supported
'''''''''''''''

    * "Tonga", "Iceland", "Vega M", and "Vega 12" GPUs are not supported in ROCm 2.9.x
    * We do not support GFX8-class GPUs (Fiji, Polaris, etc.) on CPUs that do not have PCIe 3.0 with PCIe atomics.
        * As such, we do not support AMD Carrizo and Kaveri APUs as hosts for such GPUs.
        * Thunderbolt 1 and 2 enabled GPUs are not supported by GFX8 GPUs on ROCm. Thunderbolt 1 & 2 are based on PCIe 2.0.

Supported Operating Systems - New operating systems available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm 2.9.x platform supports the following operating systems:

    * Ubuntu 16.04.5(Kernel 4.15) and 18.04.3(Kernel 4.15 and Kernel 4.18)
    * CentOS 7.7 (Using devtoolset-7 runtime support)
    * RHEL 7.7 (Using devtoolset-7 runtime support)

ROCm support in upstream Linux kernels
''''''''''''''''''''''''''''''''''''''''

As of ROCm 1.9.0, the ROCm user-level software is compatible with the AMD drivers in certain upstream Linux kernels. As such, users have the option of either using the ROCK kernel driver that are part of AMD's ROCm repositories or using the upstream driver and only installing ROCm user-level utilities from AMD's ROCm repositories.

These releases of the upstream Linux kernel support the following GPUs in ROCm:

    * 4.17: Fiji, Polaris 10, Polaris 11
    * 4.18: Fiji, Polaris 10, Polaris 11, Vega10
    * 4.20: Fiji, Polaris 10, Polaris 11, Vega10, Vega 7nm

The upstream driver may be useful for running ROCm software on systems that are not compatible with the kernel driver available in AMD's repositories. For users that have the option of using either AMD's or the upstreamed driver, there are various tradeoffs to take into consideration:

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


Software Support
^^^^^^^^^^^^^^^^^^

As of AMD ROCm v1.9.0, the ROCm user-level software is compatible with the AMD drivers in certain upstream Linux kernels. You have the following options:

• Use the ROCk kernel driver that is a part of AMD’s ROCm repositories or • Use the upstream driver and only install ROCm user-level utilities from AMD’s ROCm repositories

The releases of the upstream Linux kernel support the following GPUs in ROCm:

• Fiji, Polaris 10, Polaris 11 • Fiji, Polaris 10, Polaris 11, Vega10 • Fiji, Polaris 10, Polaris 11, Vega10, Vega 7nm



.. _Machine Learning and High Performance Computing Software Stack for AMD GPU:

Machine Learning and High Performance Computing Software Stack for AMD GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm Version 3.0

.. _ROCm Binary Package Structure:

ROCm Binary Package Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm is a collection of software ranging from drivers and runtimes to libraries and developer tools. In AMD's package distributions, these software projects are provided as a separate packages. This allows users to install only the packages they need, if they do not wish to install all of ROCm. These packages will install most of the ROCm software into ``/opt/rocm/`` by default.

The packages for each of the major ROCm components are:

-    ROCm Core Components

     -   ROCk Kernel Driver: ``rock-dkms``
     -   ROCr Runtime: ``hsa-rocr-dev``, ``hsa-ext-rocr-dev``
     -   ROCt Thunk Interface: ``hsakmt-roct``, ``hsakmt-roct-dev``

-    ROCm Support Software

     -   ROCm SMI: ``rocm-smi``
     -   ROCm cmake: ``rocm-cmake``
     -   rocminfo: ``rocminfo``
     -   ROCm Bandwidth Test: ``rocm_bandwidth_test``
    
-    ROCm Development Tools
     -   HCC compiler: ``hcc``
     -   HIP: ``hip_base``, ``hip_doc``, ``hip_hcc``, ``hip_samples``
     -   ROCm Device Libraries: ``rocm-device-libs``
     -   ROCm OpenCL: ``rocm-opencl``, ``rocm-opencl-devel`` (on RHEL/CentOS), ``rocm-opencl-dev`` (on Ubuntu)
     -   ROCM Clang-OCL Kernel Compiler: ``rocm-clang-ocl``
     -   Asynchronous Task and Memory Interface (ATMI): ``atmi``
     -   ROCr Debug Agent: ``rocr_debug_agent``
     -   ROCm Code Object Manager: ``comgr``
     -   ROC Profiler: ``rocprofiler-dev``
     -   ROC Tracer: ``roctracer-dev``
     -   Radeon Compute Profiler: ``rocm-profiler``

-    ROCm Libraries
 
     -  rocALUTION: ``rocalution``
     -  rocBLAS: ``rocblas``
     -  hipBLAS: ``hipblas``
     -  hipCUB: ``hipCUB``
     -  rocFFT: ``rocfft``
     -  rocRAND: ``rocrand``
     -  rocSPARSE: ``rocsparse``
     -  hipSPARSE: ``hipsparse``
     -  ROCm SMI Lib: ``rocm_smi_lib64``
     -  rocThrust: ``rocThrust``
     -  MIOpen: ``MIOpen-HIP`` (for the HIP version), ``MIOpen-OpenCL`` (for the OpenCL version)
     -  MIOpenGEMM: ``miopengemm``
     -  MIVisionX: ``mivisionx``
     -  RCCL: ``rccl``

To make it easier to install ROCm, the AMD binary repositories provide a number of meta-packages that will automatically install multiple other packages. For example, ``rocm-dkms`` is the primary meta-package that is
used to install most of the base technology needed for ROCm to operate.
It will install the ``rock-dkms`` kernel driver, and another meta-package 
 (``rocm-dev``) which installs most of the user-land ROCm core components, support software, and development tools.

The ``rocm-utils``meta-package will install useful utilities that,
while not required for ROCm to operate, may still be beneficial to have.
Finally, the ``rocm-libs``meta-package will install some (but not all)
of the libraries that are part of ROCm.

The chain of software installed by these meta-packages is illustrated below

::

   rocm-dkms
    |--rock-dkms
    \--rocm-dev
       |--comgr
       |--hcc
       |--hip_base
       |--hip_doc
       |--hip_hcc
       |--hip_samples
       |--hsakmt-roct
       |--hsakmt-roct-dev
       |--hsa-amd-aqlprofile
       |--hsa-ext-rocr-dev
       |--hsa-rocr-dev
       |--rocm-cmake
       |--rocm-device-libs
       |--rocm-smi
       |--rocprofiler-dev
       |--rocr_debug_agent
       \--rocm-utils
          |--rocminfo
          \--rocm-clang-ocl # This will cause OpenCL to be installed

  rocm-libs
    |--hipblas
    |--hipcub
    |--hipsparse
    |--rocalution
    |--rocblas
    |--rocfft
    |--rocprim
    |--rocrand
    |--rocsparse
    \--rocthrust




These meta-packages are not required but may be useful to make it easier to install ROCm on most systems.

Note:Some users may want to skip certain packages. For instance, a user that wants to use the upstream kernel drivers (rather than those supplied by AMD) may want to skip the rocm-dkms and rock-dkms packages. Instead, they could directly install rocm-dev.

Similarly, a user that only wants to install OpenCL support instead of HCC and HIP may want to skip the rocm-dkms and rocm-dev packages. Instead, they could directly install rock-dkms, rocm-opencl, and rocm-opencl-dev and their dependencies.

.. _ROCm Platform Packages:

ROCm Platform Packages
^^^^^^^^^^^^^^^^^^^^^^^

Drivers, ToolChains, Libraries, and Source Code

The latest supported version of the drivers, tools, libraries and source code for the ROCm platform have been released and are available from the following GitHub repositories:

-  **ROCm Core Components**

   -  `ROCk Kernel Driver`_
   -  `ROCr Runtime`_
   -  `ROCt Thunk Interface`_

-  **ROCm Support Software**

   -  `ROCm SMI`_
   -  `ROCm cmake`_
   -  `rocminfo`_
   -  `ROCm Bandwidth Test`_

-  **ROCm Development ToolChains**

   -  `HCC compiler`_
   -  `HIP`_
   -  `ROCm Device Libraries`_
   -  ROCm OpenCL, which is created from the following components:

      -  `ROCm OpenCL Runtime`_
      -  The ROCm OpenCL compiler, which is created from the following
         components:
      -  `ROCm LLVM OCL`_
      -  `ROCm DeviceLibraries`_
         
   -  `ROCM Clang-OCL Kernel Compiler`_
   -  `Asynchronous Task and Memory Interface`_
   -  `ROCr Debug Agent`_
   -  `ROCm Code Object Manager`_
   -  `ROC Profiler`_
   -  `ROC Tracer`_
   -  `AOMP`_
   -  `Radeon Compute Profiler`_
   -  `ROCm Validation Suite`_

   -  Example Applications:

      -  `HCC Examples`_
      -  `HIP Examples`_

-  **ROCm Libraries**

   -  `rocBLAS`_
   -  `hipBLAS`_
   -  `rocFFT`_
   -  `rocRAND`_
   -  `rocSPARSE`_
   -  `hipSPARSE`_
   -  `rocALUTION`_
   -  `MIOpenGEMM`_
   -  `mi open`_
   -  `rocThrust`_
   -  `ROCm SMI Lib`_
   -  `RCCL`_
   -  `MIVisionX`_
   -  `hipCUB`_
   -  `AMDMIGraphX`_

.. _ROCk Kernel Driver: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/roc-3.0.0
.. _ROCr Runtime: https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/roc-3.0.0
.. _ROCt Thunk Interface: https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/roc-3.0.0

.. _ROCm SMI: https://github.com/RadeonOpenCompute/ROC-smi/tree/roc-3.0.0
.. _ROCm cmake: https://github.com/RadeonOpenCompute/rocm-cmake/tree/roc-3.0.0
.. _rocminfo: https://github.com/RadeonOpenCompute/rocminfo/tree/roc-3.0.0
.. _ROCm Bandwidth Test: https://github.com/RadeonOpenCompute/rocm_bandwidth_test/tree/roc-3.0.0

.. _HCC compiler: https://github.com/RadeonOpenCompute/hcc/tree/roc-hcc-3.0.0
.. _HIP: https://github.com/ROCm-Developer-Tools/HIP/tree/roc-3.0.0
.. _ROCm Device Libraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/roc-hcc-3.0.0

.. _ROCm OpenCL Runtime: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/tree/roc-3.0.0

.. _ROCm LLVM OCL: http://github.com/RadeonOpenCompute/llvm/tree/roc-ocl-3.0.0
.. _ROCm DeviceLibraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/roc-ocl-3.0.0

.. _ROCM Clang-OCL Kernel Compiler: https://github.com/RadeonOpenCompute/clang-ocl/tree/3.0.0
.. _Asynchronous Task and Memory Interface: https://github.com/RadeonOpenCompute/atmi/tree/rocm_3.0.0

.. _ROCr Debug Agent: https://github.com/ROCm-Developer-Tools/rocr_debug_agent/tree/roc-3.0.0
.. _ROCm Code Object Manager: https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/tree/roc-3.0.0
.. _ROC Profiler: https://github.com/ROCm-Developer-Tools/rocprofiler/tree/roc-3.0.0
.. _ROC Tracer: https://github.com/ROCm-Developer-Tools/roctracer/tree/roc-3.0.x
.. _AOMP: https://github.com/ROCm-Developer-Tools/aomp/tree/roc-3.0.0
.. _Radeon Compute Profiler: https://github.com/GPUOpen-Tools/RCP/tree/3a49405
.. _ROCm Validation Suite: https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/tree/roc-3.0.0
.. _HCC Examples: https://github.com/ROCm-Developer-Tools/HCC-Example-Application/tree/ffd65333
.. _HIP Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/roc-3.0.0

.. _rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/rocm-3.0
.. _hipBLAS: https://github.com/ROCmSoftwarePlatform/hipBLAS/tree/rocm-3.0
.. _rocFFT: https://github.com/ROCmSoftwarePlatform/rocFFT/tree/rocm-3.0
.. _rocRAND: https://github.com/ROCmSoftwarePlatform/rocRAND/tree/3.0
.. _rocSPARSE: https://github.com/ROCmSoftwarePlatform/rocSPARSE/tree/rocm-3.0
.. _hipSPARSE: https://github.com/ROCmSoftwarePlatform/hipSPARSE/tree/rocm-3.0
.. _rocALUTION: https://github.com/ROCmSoftwarePlatform/rocALUTION/tree/rocm-3.0
.. _MIOpenGEMM: https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/tree/6275a879
.. _mi open: https://github.com/ROCmSoftwarePlatform/MIOpen/tree/roc-3.0.0
.. _rocThrust: https://github.com/ROCmSoftwarePlatform/rocThrust/tree/3.0.0
.. _ROCm SMI Lib: https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/roc.3.0.0
.. _RCCL: https://github.com/ROCmSoftwarePlatform/rccl/tree/3.0.0
.. _MIVisionX: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/1.5
.. _hipCUB: https://github.com/ROCmSoftwarePlatform/hipCUB/tree/3.0.0
.. _AMDMIGraphX: https://github.com/ROCmSoftwarePlatform/AMDMIGraphx/tree/0.5-hip-hcc




Features and enhancements introduced in previous versions of ROCm can be found in :ref:`Current-Release-Notes`.
                 
