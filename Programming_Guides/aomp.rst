

================
OpenMP Support
================

Overview
**********

AOMP is a scripted build of LLVM and supporting software. It has support for OpenMP target offload on AMD GPUs. Since AOMP is a clang/llvm compiler, it also supports GPU offloading with HIP, CUDA, and OpenCL.

Some sources to support OpenMP target offload on AMD GPUs have not yet been merged into the upstream LLVM trunk. However all sources used by AOMP are available in `AOMP repositories <https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md#repositories>`_. One of those repositories is a `mirror of the LLVM monorepo llvm-project <https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/README.md#repositories>`_ with a set of commits applied to a stable LLVM release branch.

The bin directory of this repository contains a README.md and build scripts needed to download, build, and install AOMP from source. In addition to the mirrored `LLVM project repository <https://github.com/ROCm-Developer-Tools/llvm-project>`_, AOMP uses a number of open-source ROCm components. The build scripts will download, build, and install all components needed for AOMP. However, we recommend that you install the latest release of the `debian or rpm package <https://github.com/ROCm-Developer-Tools/aomp/releases>`_ for AOMP described in the install section.

AOMP Install
**************

Platform Install Options:

    * Ubuntu or Debian
    * SUSE SLES-15-SP1
    * RHEL 7
    * Install Without Root
    * Build and Install from release source tarball
    * Development Source Build and Install


AOMP Debian/Ubuntu Install
----------------------------

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.

On Ubuntu 18.04 LTS (bionic beaver), run these commands:

::
  
  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1804_0.7-5_amd64.deb
  sudo dpkg -i aomp_Ubuntu1804_0.7-5_amd64.deb


On Ubuntu 16.04, run these commands:

::

  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1604_0.7-5_amd64.deb
  sudo dpkg -i aomp_Ubuntu1604_0.7-5_amd64.deb

The AOMP bin directory (which includes the standard clang and llvm binaries) is not intended to be in your PATH for typical operation.

Prerequisites
----------------

**AMD KFD Driver**

These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found `HERE <https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository>`_.

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  sudo apt install rock-dkms

  sudo reboot
  sudo usermod -a -G video $USER



**NVIDIA CUDA Driver**

If you build AOMP with support for nvptx GPUs, you must first install CUDA 10. Note these instructions reference the install for Ubuntu 16.04.

**Download Instructions for CUDA (Ubuntu 16.04)**

    Go to https://developer.nvidia.com/cuda-10.0-download-archive
    For Ubuntu 16.04, select Linux®, x86_64, Ubuntu, 16.04, deb(local) and then click Download. Note you can change these options for your specific distribution type.
    Navigate to the debian in your Linux® directory and run the following commands:

::

   sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
   sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
   sudo apt-get update
   sudo apt-get install cuda


Depending on your system the CUDA install could take a very long time.

AOMP SUSE SLES-15-SP1 Install
-------------------------------

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.

::

  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_SLES15_SP1-0.7-5.x86_64.rpm
  sudo rpm -i aomp_SLES15_SP1-0.7-5.x86_64.rpm


Confirm AOMP environment variable is set:

::

  echo $AOMP


**Prerequisites**

The ROCm kernel driver is required for AMD GPU support. Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

**AMD KFD DRIVER**

**Important Note:** There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:

::

  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu


**Set Group Access**

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

**NVIDIA CUDA Driver**

If you build AOMP with support for nvptx GPUs, you must first install CUDA 10.

Download Instructions for CUDA (SLES15)

    Go to https://developer.nvidia.com/cuda-10.0-download-archive
    For SLES-15, select Linux®, x86_64, SLES, 15.0, rpm(local) and then click Download.
    Navigate to the rpm in your Linux® directory and run the following commands:

::

  sudo rpm -i cuda-repo-sles15-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo zypper refresh
  sudo zypper install cuda


If prompted, select the 'always trust key' option. Depending on your system the CUDA install could take a very long time.

**Important Note:** If using a GUI on SLES-15-SP1, such as gnome, the installation of CUDA may cause the GUI to fail to load. This seems to be caused by a symbolic link pointing to nvidia-libglx.so instead of xorg-libglx.so. This can be fixed by updating the symbolic link:

::

  sudo rm /etc/alternatives/libglx.so
  sudo ln -s /usr/lib64/xorg/modules/extensions/xorg/xorg-libglx.so /etc/alternatives/libglx.so


AOMP RHEL 7 Install
---------------------

AOMP will install to /usr/lib/aomp. The AOMP environment variable will automatically be set to the install location. This may require a new terminal to be launched to see the change.

**The installation may need the following dependency:**

::

  sudo yum install perl-Digest-MD5


**Download and Install**

::

  wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_REDHAT_7-0.7-5.x86_64.rpm
  sudo rpm -i aomp_REDHAT_7-0.7-5.x86_64.rpm

If CUDA is not installed the installation may cancel, to bypass this:

::

  sudo rpm -i --nodeps aomp_REDHAT_7-0.7-5.x86_64.rpm

Confirm AOMP environment variable is set:

::

  echo $AOMP


**Prerequisites**

The ROCm kernel driver is required for AMD GPU support. Also, to control access to the ROCm device, a user group "video" must be created and users need to be added to this group.

**AMD KFD Driver**

::

  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm


**Install and setup Devtoolset-7**

Devtoolset-7 is recommended, follow instructions 1-3 here:
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD. https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

**Install dkms tool**

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0

**Install rock-dkms**

::

  sudo yum install rock-dkms


**Set Group Access**

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

**NVIDIA CUDA Driver**

To build AOMP with support for nvptx GPUs, you must first install CUDA 10. We recommend CUDA 10.0. CUDA 10.1 will not work until AOMP moves to the trunk development of LLVM 9. The CUDA installation is now optional.

**Download Instructions for CUDA (CentOS/RHEL 7)**

    * Go to https://developer.nvidia.com/cuda-10.0-download-archive
    * For SLES-15, select Linux®, x86_64, RHEL or CentOS, 7, rpm(local) and then click Download.
    * Navigate to the rpm in your Linux® directory and run the following commands:

::

  sudo rpm -i cuda-repo-rhel7-10-0-local-10.0.130-410.48-1.0-1.x86_64.rpm
  sudo yum clean all
  sudo yum install cuda

Install Without Root
----------------------

By default, the packages install their content to the release directory /usr/lib/aomp_0.X-Y and then a symbolic link is created at /usr/lib/aomp to the release directory. This requires root access.

Once installed go to `TESTINSTALL <https://github.com/ROCm-Developer-Tools/aomp/blob/master/docs/TESTINSTALL.md>`_ for instructions on getting started with AOMP examples.

**Debian**

To install the debian package without root access into your home directory, you can run these commands.
On Ubuntu 18.04 LTS (bionic beaver):

::

   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1804_0.7-5_amd64.deb
   dpkg -x aomp_Ubuntu1804_0.7-5_amd64.deb /tmp/temproot


On Ubuntu 16.04:

::

   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_Ubuntu1604_0.7-5_amd64.deb
   dpkg -x aomp_Ubuntu1604_0.7-5_amd64.deb /tmp/temproot

::

   mv /tmp/temproot/usr $HOME
   export PATH=$PATH:$HOME/usr/lib/aomp/bin
   export AOMP=$HOME/usr/lib/aomp

The last two commands could be put into your .bash_profile file so you can always access the compiler.

**RPM**

To install the rpm package without root access into your home directory, you can run these commands.

::

   mkdir /tmp/temproot ; cd /tmp/temproot 
   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp_SLES15_SP1-0.7-5.x86_64.rpm
   rpm2cpio aomp_SLES15_SP1-0.7-5.x86_64.rpm | cpio -idmv
   mv /tmp/temproot/usr/lib $HOME
   export PATH=$PATH:$HOME/rocm/aomp/bin
   export AOMP=$HOME/rocm/aomp

The last two commands could be put into your .bash_profile file so you can always access the compiler.

Build and Install From Release Source Tarball
------------------------------------------------

The AOMP build and install from the release source tarball can be done manually or with spack. Building from source requires a number of platform dependencies. These dependencies are not yet provided with the spack configuration file. So if you are building from source either manually or building with spack, you must install the prerequisites for the platforms listed below.

**Source Build Prerequisites**

To build AOMP from source you must: 1. install certain distribution packages, 2. ensure the KFD kernel module is installed and operating, 3. create the Unix video group, and 4. install spack if required.

**1. Required Distribution Packages**

**Debian or Ubuntu Packages**

::

   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk


**SLES-15-SP1 Packages**

::

  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel


**RHEL 7 Packages**

Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD. https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

**The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.**

::

  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel


The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin

::

  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake'


**2. Verify KFD Driver**

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.

**Debian or Ubuntu Support**

These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found `HERE <https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository>`_.

::

  wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  sudo apt install rock-dkms


**SUSE SLES-15-SP1 Support**

**Important Note:** There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:

::

  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu


**RHEL 7 Support**

::

  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm


**Install dkms tool**

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`


Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0


**Install rock-dkms**

::

  sudo yum install rock-dkms

3. Create the Unix Video Group

Regardless of Linux distribution, you must create a video group to contain the users authorized to use the GPU.

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

4. Install spack

To use spack to build and install from the release source tarball, you must install spack first. Please refer to these `install instructions for instructions <https://spack.readthedocs.io/en/latest/getting_started.html#installation>`_ on installing spack. Remember,the aomp spack configuration file is currently missing dependencies, so be sure to install the packages listed above before proceeding.

**Build AOMP manually from release source tarball**

To build and install aomp from the release source tarball run these commands:

::

   wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp-0.7-5.tar.gz
   tar -xzf aomp-0.7-5.tar.gz
   cd aomp
   nohup make &

Depending on your system, the last command could take a very long time. So it is recommended to use nohup and background the process. The simple Makefile that make will use runs build script "build_aomp.sh" and sets some flags to avoid git checks and applying ROCm patches. Here is that Makefile:

::

  AOMP ?= /usr/local/aomp
  AOMP_REPOS = $(shell pwd)
  all:
        AOMP=$(AOMP) AOMP_REPOS=$(AOMP_REPOS) AOMP_CHECK_GIT_BRANCH=0 AOMP_APPLY_ROCM_PATCHES=0 $(AOMP_REPOS)/aomp/bin/build_aomp.sh


If you set the environment variable AOMP, the Makefile will install to that directory. Otherwise, the Makefile will install into /usr/local. So you must have authorization to write into /usr/local if you do not set the environment variable AOMP. Let's assume you set the environment variable AOMP to "$HOME/rocm/aomp" in .bash_profile. The build_aomp.sh script will install into $HOME/rocm/aomp_0.7-5 and create a symbolic link from $HOME/rocm/aomp to $HOME/rocm/aomp_0.7-5. This feature allows multiple versions of AOMP to be installed concurrently. To enable a backlevel version of AOMP, simply set AOMP to $HOME/rocm/aomp_0.7-4.

**Build AOMP with spack**

Assuming your have installed the prerequisites listed above, use these commands to fetch the source and build aomp. Currently the aomp configuration is not yet in the spack git hub so you must create the spack package first.

::

   wget https://github.com/ROCm-Developer-Tools/aomp/blob/master/bin/package.py
   spack create -n aomp -t makefile --force https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_0.7-5/aomp-0.7-5.tar.gz
   spack edit aomp
   spack install aomp


The "spack create" command will download and start an editor of a newly created spack config file. With the exception of the sha256 value, copy the contents of the downloaded package.py file into into the spack configuration file. You may restart this editor with the command "spack edit aomp"

Depending on your system, the "spack install aomp" command could take a very long time. Unless you set the AOMP environment variable, AOMP will be installed in /usr/local/aomp_ with a symbolic link from /usr/local/aomp to /usr/local/aomp_. Be sure you have write access to /usr/local or set AOMP to a location where you have write access.


Source Install V 0.7-6 (DEV)
--------------------------------

Build and install from sources is possible. However, the source build for AOMP is complex for several reasons.

    * Many repos are required. The clone_aomp.sh script ensures you have all repos and the correct branch.
    * Requires that Cuda SDK 10 is installed for NVIDIA GPUs. ROCm does not need to be installed for AOMP.
    * It is a bootstrapped build. The built and installed LLVM compiler is used to build library components.
    * Additional package dependencies are required that are not required when installing the AOMP package.

**Source Build Prerequisites**

**1. Required Distribution Packages**

**Debian or Ubuntu Packages**

::

   sudo apt-get install cmake g++-5 g++ pkg-config libpci-dev libnuma-dev libelf-dev libffi-dev git python libopenmpi-dev gawk

**SLES-15-SP1 Packages**

::

  sudo zypper install -y git pciutils-devel cmake python-base libffi-devel gcc gcc-c++ libnuma-devel libelf-devel patchutils openmpi2-devel


**RHEL 7 Packages**

Building from source requires a newer gcc. Devtoolset-7 is recommended, follow instructions 1-3 here:
Note that devtoolset-7 is a Software Collections package, and it is not supported by AMD. https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

**The build_aomp.sh script will automatically enable devtoolset-7 if found in /opt/rh/devtoolset-7/enable. If you want to build an individual component you will need to manually start devtoolset-7 from the instructions above.**

::

  sudo yum install cmake3 pciutils-devel numactl-devel libffi-devel


The build scripts use cmake, so we need to link cmake --> cmake3 in /usr/bin

::

  sudo ln -s /usr/bin/cmake3 /usr/bin/cmake


**2. Verify KFD Driver**

Please verify you have the proper software installed as AOMP needs certain support to function properly, such as the KFD driver for AMD GPUs.

**Debian or Ubuntu Support**

These commands are for supported Debian-based systems and target only the rock_dkms core component. More information can be found `HERE <https://rocm.github.io/ROCmInstall.html#ubuntu-support---installing-from-a-debian-repository>`_.

::

  wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  sudo apt install rock-dkms


**SUSE SLES-15-SP1 Support**

**Important Note:** There is a conflict with the KFD when simultaneously running the GUI on SLES-15-SP1, which leads to unpredicatable behavior when offloading to the GPU. We recommend using SLES-15-SP1 in text mode to avoid running both the KFD and GUI at the same time.

SUSE SLES-15-SP1 comes with kfd support installed. To verify this:

::

  sudo dmesg | grep kfd
  sudo dmesg | grep amdgpu

**RHEL 7 Support**

::

  sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
  sudo subscription-manager repos --enable rhel-7-server-optional-rpms
  sudo subscription-manager repos --enable rhel-7-server-extras-rpms
  sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

**Install dkms tool**

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

Create a /etc/yum.repos.d/rocm.repo file with the following contents:

::

  [ROCm]
  name=ROCm
  baseurl=http://repo.radeon.com/rocm/yum/rpm
  enabled=1
  gpgcheck=0

**Install rock-dkms**

::

  sudo yum install rock-dkms

**3. Create the Unix Video Group**

::

  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
  sudo reboot
  sudo usermod -a -G video $USER

**Clone and Build AOMP**

::

   cd $HOME ; mkdir -p git/aomp ; cd git/aomp
   git clone https://github.com/rocm-developer-tools/aomp
   cd $HOME/git/aomp/aomp/bin

**Choose a Build Version (Development or Release)** The development version is the next version to be released. It is possible that the development version is broken due to regressions that often occur during development. If instead, you want to build from the sources of a previous release such as 0.7-5 that is possible as well.

**For the Development Branch:**

   git checkout master
   git pull

For the Release Branch:

   git checkout rel_0.7-5
   git pull

Clone and Build:

   ./clone_aomp.sh
   ./build_aomp.sh

Depending on your system, the last two commands could take a very long time. For more information, please refer :ref:`AOMP developers README`.

You only need to do the checkout/pull in the AOMP repository. The file "bin/aomp_common_vars" lists the branches of each repository for a particular AOMP release. In the master branch of AOMP, aomp_common_vars lists the development branches. It is a good idea to run clone_aomp.sh twice after you checkout a release to be sure you pulled all the checkouts for a particular release.

For more information on Release Packages, click `here <https://github.com/ROCm-Developer-Tools/aomp/releases>`_

Test Install
*************

**Getting Started**

The default install location is /usr/lib/aomp. To run the given examples, for example in /usr/lib/aomp/examples/openmp do the following:

**Copy the example openmp directory somewhere writable**

::

  cd /usr/lib/aomp/examples/
  cp -rp openmp /work/tmp/openmp-examples
  cd /work/tmp/openmp-examples/vmulsum

**Point to the installed AOMP by setting AOMP environment variable**

::

  export AOMP=/usr/lib/aomp

**Make Instructions**

::

  make clean
  make run


Run 'make help' for more details.

View the OpenMP Examples `README <https://github.com/ROCm-Developer-Tools/aomp/blob/master/examples/openmp>`_ for more information.

AOMP Limitations
*****************

See the `release notes <https://github.com/ROCm-Developer-Tools/aomp/releases>`_ in github. Here are some limitations.

 - Dwarf debugging is turned off for GPUs. -g will turn on host level debugging only.
 - Some simd constructs fail to vectorize on both host and GPUs.  
