.. image:: /Installation_Guide/amdblack.jpg
|
===============================
AMD ROCm Installation Guide 
===============================

-  `Deploying ROCm`_


-  `Prerequisites`_
   
   
-  `Supported Operating Systems`_
   
   
   -  `Ubuntu`_
   
   
   -  `CentOS v7.7/RHEL v7.8 and CentOS/RHEL 8.1`_
   
   
   -  `SLES 15 Service Pack 1`_
   
   
   
-  `HIP Installation Instructions`_


-  `AMD ROCm MultiVersion Installation`_


-  `ROCm Installation Known Issues and Workarounds`_


-  `Getting the ROCm Source Code`_


.. _Deploying ROCm:

Deploying ROCm
~~~~~~~~~~~~~~~~

AMD hosts both Debian and RPM repositories for the ROCm v3.x packages.

The following directions show how to install ROCm on supported Debian-based systems such as Ubuntu 18.04.x

**Note**: These directions may not work as written on unsupported Debian-based distributions. For example, newer versions of Ubuntu may not be compatible with the rock-dkms kernel driver. In this case, you can exclude the rocm-dkms and rock-dkms packages.


Prerequisites 
~~~~~~~~~~~~~~~

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
   
   
Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   

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

    wget -q -O - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -

    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list


The gpg key may change; ensure it is updated when installing a new release. If the key signature verification fails while updating, re-add the key from the ROCm apt repository.

The current rocm.gpg.key is not available in a standard key ring distribution, but has the following sha1sum hash:

::

  e85a40d1a43453fe37d63aa6899bc96e08f2817a rocm.gpg.key

3. Install the ROCm meta-package. Update the appropriate repository list and install the rocm-dkms meta-package:

::

     sudo apt update

     sudo apt install rocm-dkms && sudo reboot
    

4. Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

     groups
     

5. To add your user to the video group, use the following command with the sudo password:

::

     sudo usermod -a -G video $LOGNAME

6. By default, you must add any future users to the video group. To add future users to the video group, run the following command:

::

     echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf

     echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

7. Restart the system.

8. After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed by both commands, the installation is considered successful.

::

     /opt/rocm/bin/rocminfo
     /opt/rocm/opencl/bin/x86_64/clinfo

Note: To run the ROCm programs, add the ROCm binaries in your PATH.

::

    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh


Uninstalling ROCm Packages from Ubuntu
''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages from Ubuntu 16.04.6 or Ubuntu 18.04.4, run the following command:

::

  sudo apt autoremove rocm-opencl rocm-dkms rocm-dev rocm-utils && sudo reboot


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
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules


.. _CentOS RHEL:

CentOS v7.7/RHEL v7.8 and CentOS/RHEL 8.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section describes how to install ROCm on supported RPM-based systems such as CentOS v7.7/RHEL v7.8 and CentOS/RHEL v8.1.

Preparing RHEL for Installation
'''''''''''''''''''''''''''''''''''

RHEL is a subscription-based operating system. You must enable the external repositories to install on the devtoolset-7 environment and the dkms support files.

Note: The following steps do not apply to the CentOS installation.

1. The subscription for RHEL must be enabled and attached to a pool ID. See the Obtaining an RHEL image and license page for instructions on registering your system with the RHEL subscription server and attaching to a pool id.

2. Enable the following repositories for RHEL v7.x:

::
   
    sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms 
    sudo subscription-manager repos --enable rhel-7-server-optional-rpms
    sudo subscription-manager repos --enable rhel-7-server-extras-rpms


3. Enable additional repositories by downloading and installing the epel-release-latest-7/epel-release-latest-8 repository RPM:

::

   sudo rpm -ivh <repo>


For more details, 

* see https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm for RHEL v7.x

* see https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm for RHEL v8.x

4. Install and set up Devtoolset-7.
   
**Note**: Devtoolset is not required for CentOS/RHEL v8.x

To setup the Devtoolset-7 environment, follow the instructions on this page: https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

Note: devtoolset-7 is a software collections package and is not supported by AMD.

Installing CentOS v7.7/v8.1 for DKMS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the dkms tool to install the kernel drivers on CentOS/RHEL:

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

.. _ROCM install:

Installing ROCm
'''''''''''''''''

To install ROCm on your system, follow the instructions below:

1. Delete the previous versions of ROCm before installing the latest version.

2. Create a /etc/yum.repos.d/rocm.repo file with the following contents:

* CentOS/RHEL 7.x : http://repo.radeon.com/rocm/yum/rpm 

* CentOS/RHEL 8.x : http://repo.radeon.com/rocm/centos8/rpm

::

    [ROCm] 
    name=ROCm
    baseurl=http://repo.radeon.com/rocm/yum/rpm 
    enabled=1
    gpgcheck=0

Note: The URL of the repository must point to the location of the repositories’ repodata database.

3. Install ROCm components using the following command:

**Note**: This step is applicable only for CentOS/RHEL v8.1 and is not required for v7.8.

::

    sudo yum install rocm-dkms && sudo reboot


4. Restart the system. The rock-dkms component is installed and the /dev/kfd device is now available.

5. Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

    groups

6. To add your user to the video group, use the following command with the sudo password:

::

    sudo usermod -a -G video $LOGNAME


7.  By default, add any future users to the video group. Run the following command to add users to the video group:

::

  echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
  echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

Note:  Before updating to the latest version of the operating system, delete the ROCm packages to avoid DKMS-related issues.

8. Restart the system.

9. Test the ROCm installation.


Testing the ROCm Installation
'''''''''''''''''''''''''''''''

After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed, you are good to go!

::

  /opt/rocm/bin/rocminfo
  /opt/rocm/opencl/bin/x86_64/clinfo


**Note**: Add the ROCm binaries in your PATH for easy implementation of the ROCm programs.

::

  echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh


Compiling Applications Using HCC, HIP, and Other ROCm Software
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


To compile applications or samples, run the following command to use gcc-7.2 provided by the devtoolset-7 environment:

::

  scl enable devtoolset-7 bash


Uninstalling ROCm from CentOS/RHEL 
''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages, run the following command:

::

  sudo yum autoremove rocm-opencl rocm-dkms rock-dkms


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
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules  
  sudo reboot

**Note**: You can use this command instead of installing rocm-dkms.

**Note**: Ensure you restart the system after ROCm installation. 

.. _SLES 15 Service Pack 1:


SLES 15 Service Pack 1
^^^^^^^^^^^^^^^^^^^^^^^

The following section tells you how to perform an install and uninstall ROCm on SLES 15 SP 1. 

**Installation**


1. Install the "dkms" package.

::

	sudo SUSEConnect --product PackageHub/15.1/x86_64
	sudo zypper install dkms
	
2. Add the ROCm repo.
 
::

	sudo zypper clean –all
	sudo zypper addrepo --no-gpgcheck http://repo.radeon.com/rocm/zyp/zypper/ rocm 
	sudo zypper ref
	zypper install rocm-dkms
	sudo zypper install rocm-dkms
	sudo reboot

3. Run the following command once

::

	cat <<EOF | sudo tee /etc/modprobe.d/10-unsupported-modules.conf
	allow_unsupported_modules 1
	EOF
	sudo modprobe amdgpu

4. Verify the ROCm installation.

5. Run /opt/rocm/bin/rocminfo and /opt/rocm/opencl/bin/x86_64/clinfo commands to list the GPUs and verify that the ROCm installation is successful.

6. Set permissions. 

To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

	groups

7. To add your user to the video group, use the following command with the sudo password:
	
::

	sudo usermod -a -G video $LOGNAME
	
8. By default, add any future users to the video group. Run the following command to add users to the video group:

::

	echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
	echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

9. Restart the system.
10. Test the basic ROCm installation.
11. After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed by both commands, the installation is considered successful.

::

	/opt/rocm/bin/rocminfo
	/opt/rocm/opencl/bin/x86_64/clinfo

Note: To run the ROCm programs more efficiently, add the ROCm binaries in your PATH.


::

echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64'|sudo tee -a /etc/profile.d/rocm.sh

**Uninstallation**

To uninstall, use the following command:

::

	sudo zypper remove rocm-opencl rocm-dkms rock-dkms

Note: Ensure all other installed packages/components are removed.
Note: Ensure all the content in the /opt/rocm directory is completely removed. If the command does not remove all the ROCm components/packages, ensure you remove them individually.

Performing an OpenCL-only Installation of ROCm
''''''''''''''''''''''''''''''''''''''''''''''''

Some users may want to install a subset of the full ROCm installation. If you are trying to install on a system with a limited amount of storage space, or which will only run a small collection of known applications, you may want to install only the packages that are required to run OpenCL applications. To do that, you can run the following installation command instead of the command to install rocm-dkms.

::
  
  sudo yum install rock-dkms rocm-opencl-devel && sudo reboot
  
  


HIP Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Installing pre-built packages <#installing-pre-built-packages>`__

   -  `Prerequisites <#prerequisites>`__
   -  `HIP-hcc <#hip-hcc>`__
   -  `HIP-clang <#hip-clang>`__
   -  `HIP-nvcc <#hip-nvcc>`__
   -  `Verify your installation <#verify-your-installation>`__

-  `Building HIP from source <#building-hip-from-source>`__

   -  `HCC Options <#hcc-options>`__

      -  `Using HIP with the AMD Native-GCN
         compiler. <#using-hip-with-the-amd-native-gcn-compiler>`__

.. raw:: html

   <!-- tocstop -->

Installing pre-built packages
=============================

HIP can be easily installed using pre-built binary packages using the
package manager for your platform.

Prerequisites
-------------

HIP code can be developed either on AMD ROCm platform using hcc or clang
compiler, or a CUDA platform with nvcc installed:

HIP-hcc
-------

-  Add the ROCm package server to your system as per the OS-specific
   guide available
   `here <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`__.
-  Install the â€œhip-hccâ€ package. This will install HCC and the HIP
   porting layer.

::

   apt-get install hip-hcc

-  Default paths and environment variables:

   -  By default HIP looks for hcc in /opt/rocm/hcc (can be overridden
      by setting HCC_HOME environment variable)
   -  By default HIP looks for HSA in /opt/rocm/hsa (can be overridden
      by setting HSA_PATH environment variable)
   -  By default HIP is installed into /opt/rocm/hip (can be overridden
      by setting HIP_PATH environment variable).
   -  Optionally, consider adding /opt/rocm/bin to your PATH to make it
      easier to use the tools.

HIP-clang
---------

-  Using clang to compile HIP program for AMD GPU is under development.
   Users need to build LLVM, clang, lld, ROCm device library, and HIP
   from source.

-  Install the
   `rocm <http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/>`__
   packages. ROCm will install some of the necessary components,
   including the kernel driver, HSA runtime, etc.

-  Build HIP-Clang

::

   git clone https://github.com/llvm/llvm-project.git
   mkdir -p build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_EXTERNAL_LLD_SOURCE_DIR=../lld -DLLVM_EXTERNAL_CLANG_SOURCE_DIR=../clang ../llvm
   make -j
   sudo make install

-  Build Rocm device library

::

   export PATH=/opt/rocm/llvm/bin:$PATH
   git clone -b amd-stg-open https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git
   cd ROCm-Device-Libs
   mkdir -p build && cd build
   CC=clang CXX=clang++ cmake -DLLVM_DIR=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_WERROR=1 -DLLVM_ENABLE_ASSERTIONS=1 ..
   make -j
   sudo make install

-  Build HIP

::

   git clone -b master https://github.com/ROCm-Developer-Tools/HIP.git
   cd HIP
   mkdir -p build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/hip -DHIP_COMPILER=clang -DCMAKE_BUILD_TYPE=Release ..
   make -j
   sudo make install

-  Default paths and environment variables:

   -  By default HIP looks for HSA in /opt/rocm/hsa (can be overridden
      by setting HSA_PATH environment variable)
   -  By default HIP is installed into /opt/rocm/hip (can be overridden
      by setting HIP_PATH environment variable).
   -  By default HIP looks for clang in /opt/rocm/llvm/bin (can be
      overridden by setting HIP_CLANG_PATH environment variable)
   -  By default HIP looks for device library in /opt/rocm/lib (can be
      overriden by setting DEVICE_LIB_PATH environment variable).
   -  Optionally, consider adding /opt/rocm/bin to your PATH to make it
      easier to use the tools.
   -  Optionally, set HIPCC_VERBOSE=7 to output the command line for
      compilation to make sure clang is used instead of hcc.

HIP-nvcc
--------

-  Add the ROCm package server to your system as per the OS-specific
   guide available
   `here <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`__.
-  Install the â€œhip-nvccâ€ package. This will install CUDA SDK and the
   HIP porting layer.

::

   apt-get install hip-nvcc

-  Default paths and environment variables:

   -  By default HIP looks for CUDA SDK in /usr/local/cuda (can be
      overriden by setting CUDA_PATH env variable)
   -  By default HIP is installed into /opt/rocm/hip (can be overridden
      by setting HIP_PATH environment variable).
   -  Optionally, consider adding /opt/rocm/bin to your path to make it
      easier to use the tools.

Verify your installation
------------------------

Run hipconfig (instructions below assume default installation path) :

.. code:: shell

   /opt/rocm/bin/hipconfig --full

Compile and run the `square
sample <https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square>`__.

Building HIP from source
========================

HIP source code is available and the project can be built from source on
the HCC platform.

1. Follow the above steps to install and validate the binary packages.
2. Download HIP source code (from the `GitHub
   repot <https://github.com/ROCm-Developer-Tools/HIP>`__.)
3. Install HIP build-time dependencies using
   ``sudo apt-get install libelf-dev``.
4. Build and install HIP (This is the simple version assuming default
   paths ; see below for additional options.)

By default, HIP uses HCC to compile programs. To use HIP-Clang, add
-DHIP_COMPILER=clang to cmake command line.

::

   cd HIP
   mkdir build
   cd build
   cmake .. 
   make
   make install

-  Default paths:

   -  By default cmake looks for hcc in /opt/rocm/hcc (can be overridden
      by setting ``-DHCC_HOME=/path/to/hcc`` in the cmake step).\*
   -  By default cmake looks for HSA in /opt/rocm/hsa (can be overridden
      by setting ``-DHSA_PATH=/path/to/hsa`` in the cmake step).\*
   -  By default cmake installs HIP to /opt/rocm/hip (can be overridden
      by setting ``-DCMAKE_INSTALL_PREFIX=/where/to/install/hip`` in the
      cmake step).\*

Hereâ€™s a richer command-line that overrides the default paths:

.. code:: shell

   cd HIP
   mkdir build
   cd build
   cmake -DHSA_PATH=/path/to/hsa -DHCC_HOME=/path/to/hcc -DCMAKE_INSTALL_PREFIX=/where/to/install/hip -DCMAKE_BUILD_TYPE=Release ..
   make
   make install

-  After installation, make sure HIP_PATH is pointed to
   ``/where/to/install/hip``.

  
  

AMD ROCm MultiVersion Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install and access multiple versions of the ROCm toolkit simultaneously.

Previously, users could install only a single version of the ROCm toolkit. 

Now, users have the option to install multiple versions simultaneously and toggle to the desired version of the ROCm toolkit. From the v3.3 release, multiple versions of ROCm packages can be installed in the */opt/rocm-<version>* folder.
 
**Prerequisites**
###############################

Ensure the existing installations of ROCm, including */opt/rocm*, are completely removed before the v3.3 ROCm toolkit installation. The ROCm v3.3 package requires a clean installation.

* To install a single instance of ROCm, use the rocm-dkms or rocm-dev packages to install all the required components. This creates a symbolic link */opt/rocm* pointing to the corresponding version of ROCm installed on the system. 

* To install individual ROCm components, create the */opt/rocm* symbolic link pointing to the version of ROCm installed on the system. For example, *# ln -s /opt/rocm-3.3.0 /opt/rocm*

* To install multiple instance ROCm packages, create */opt/rocm* symbolic link pointing to the version of ROCm installed/used on the system. For example, *# ln -s /opt/rocm-3.3.0 /opt/rocm*

**Note**: The Kernel Fusion Driver (KFD) must be compatible with all versions of the ROCm software installed on the system.


Before You Begin
#################

Review the following important notes:

**Single Version Installation**

To install a single instance of the ROCm package, access the non-versioned packages. You must not install any components from the multi-instance set.

For example, 

* rocm-dkms

* rocm-dev

* hip

A fresh installation or an upgrade of the single-version installation will remove the existing version completely and install the new version in the */opt/rocm-<version>* folder.

.. image:: /Current_Release_Notes/singleinstance.png

**Multi Version Installation**

* To install a multi-instance of the ROCm package, access the versioned packages and components. 

For example,

  * rocm-dkms3.3.0

  * rocm-dev3.3.0

  * hip3.3.0

* The new multi-instance package enables you to install two versions of the ROCm toolkit simultaneously and provides the ability to toggle between the two versioned packages.

* The ROCm-DEV package does not create symlinks

* Users must create symlinks if required

* Multi-version installation with previous ROCm versions is not supported

* Kernel Fusion Driver (KFD) must be compatible with all versions of ROCm installations

.. image:: /Current_Release_Notes/MultiIns.png

**IMPORTANT**: A single instance ROCm package cannot co-exist with the multi-instance package. 

**NOTE**: The multi-instance installation applies only to ROCm v3.3 and above. This package requires a fresh installation after the complete removal of existing ROCm packages. The multi-version installation is not backward compatible. 

**Note**: If you install the multi-instance version of AMD ROCm and create a sym-link to */opt/rocm*, you must run ‘Idconfig’ to ensure the software stack functions correctly with the sym-link. 
  

ROCm Installation Known Issues and Workarounds 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows how to use the repo binary to download the ROCm source code. If you choose a directory other than ~/bin/ to install the repo, you must use that chosen directory in the code as shown below:

::

  mkdir -p ~/ROCm/
  cd ~/ROCm/
  ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.5.0
  repo sync


Note: Using this sample code will cause the repo to download the open source code associated with this ROCm release. Ensure that you have ssh-keys configured on your machine for your GitHub ID prior to the download.

Building the ROCm Source Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ROCm component repository contains directions for building that component. You can access the desired component for instructions to build the repository.


.. _Machine Learning and High Performance Computing Software Stack for AMD GPU:

===================================================================================
Machine Learning and High Performance Computing Software Stack for AMD GPU v3.5.0
===================================================================================


**ROCm Version 3.5.0**

.. _ROCm Binary Package Structure:

ROCm Binary Package Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm is a collection of software ranging from drivers and runtimes to libraries and developer tools. In AMD's package distributions, these software projects are provided as a separate packages. This allows users to install only the packages they need, if they do not wish to install all of ROCm. These packages will install most of the ROCm software into ``/opt/rocm/`` by default.

The packages for each of the major ROCm components are:

ROCm Core Components
=====================

 -   ROCk Kernel Driver: ``rock-dkms``
 -   ROCr Runtime: ``hsa-rocr-dev``, ``hsa-ext-rocr-dev``
 -   ROCt Thunk Interface: ``hsakmt-roct``, ``hsakmt-roct-dev``


ROCm Support Software
======================

 -   ROCm SMI: ``rocm-smi``
 -   ROCm cmake: ``rocm-cmake``
 -   rocminfo: ``rocminfo``
 -   ROCm Bandwidth Test: ``rocm_bandwidth_test``
     
    
ROCm Compilers
================

 -   HCC compiler: ``hcc``   (in deprecation)     
 -   HIP: ``hip_base``, ``hip_doc``, ``hip_hcc``, ``hip_samples``      
 -   ROCM Clang-OCL Kernel Compiler: ``rocm-clang-ocl``
     

ROCm Device Libraries
===========================
     
 -   ROCm Device Libraries: ``rocm-device-libs``     
 -   ROCm OpenCL: ``rocm-opencl``, ``rocm-opencl-devel`` (on RHEL/CentOS), ``rocm-opencl-dev`` (on Ubuntu)
     
     
 ROCm Development ToolChain
===========================
     
 -   Asynchronous Task and Memory Interface (ATMI): ``atmi``     
 -   ROCm Debug Agent: ``rocm_debug_agent``     
 -   ROCm Code Object Manager: ``comgr``     
 -   ROC Profiler: ``rocprofiler-dev``     
 -   ROC Tracer: ``roctracer-dev``     
 -   Radeon Compute Profiler: ``rocm-profiler``
     

ROCm Libraries
==============
 
 -  rocALUTION: ``rocalution``
 -  rocBLAS: ``rocblas``
 -  hipBLAS: ``hipblas``
 -  hipCUB: ``hipCUB``
 -  rocFFT: ``rocfft``
 -  rocRAND: ``rocrand``
 -  rocSPARSE: ``rocsparse``
 -  hipSPARSE: ``hipsparse``
 -  ROCm SMI Lib: ``rocm-smi-lib64``
 -  rocThrust: ``rocThrust``
 -  MIOpen: ``MIOpen-HIP`` (for the HIP version), ``MIOpen-OpenCL`` (for the OpenCL version)
 -  MIOpenGEMM: ``miopengemm``
 -  MIVisionX: ``mivisionx``
 -  RCCL: ``rccl``


To make it easier to install ROCm, the AMD binary repositories provide a number of meta-packages that will automatically install multiple other packages. For example, ``rocm-dkms`` is the primary meta-package that is
used to install most of the base technology needed for ROCm to operate.
It will install the ``rock-dkms`` kernel driver, and another meta-package 
 (``rocm-dev``) which installs most of the user-land ROCm core components, support software, and development tools.
 

The *rocm-utils* meta-package will install useful utilities that, while not required for ROCm to operate, may still be beneficial to have. Finally, the *rocm-libs* meta-package will install some (but not all) of the libraries that are part of ROCm.

The chain of software installed by these meta-packages is illustrated below:


::

   └── rocm-dkms
    ├── rock-dkms
    └── rocm-dev
        ├── comgr
        ├── hip-base
        ├── hip-doc
        ├── hip-rocclr
        ├── hip-samples
        ├── hsa-amd-aqlprofile
        ├── hsa-ext-rocr-dev
        ├── hsakmt-roct
        ├── hsakmt-roct-dev
        ├── hsa-rocr-dev
        ├── llvm-amdgpu
        ├── rocm-cmake
        ├── rocm-dbgapi
        ├── rocm-debug-agent
        ├── rocm-device-libs
        ├── rocm-gdb
        ├── rocm-smi
        ├── rocm-smi-lib64
        ├── rocprofiler-dev
        └── roctracer-dev
	├── rocm-utils
            │   ├── rocm-clang-ocl
            │   └── rocminfo

  rocm-libs
    |--hipblas
    |--hipcub
    |--hipsparse
    |--rocalution
    |--rocblas
    |--rocfft
    |--rocprim
    |--rocrand
    |--rocsolver
    |--rocsparse
    \--rocthrust




These meta-packages are not required but may be useful to make it easier to install ROCm on most systems.

Note: Some users may want to skip certain packages. For instance, a user that wants to use the upstream kernel drivers (rather than those supplied by AMD) may want to skip the rocm-dkms and rock-dkms packages. Instead, they could directly install rocm-dev.

Similarly, a user that only wants to install OpenCL support instead of HCC and HIP may want to skip the rocm-dkms and rocm-dev packages. Instead, they could directly install rock-dkms, rocm-opencl, and rocm-opencl-dev and their dependencies.

.. _ROCm Platform Packages:


ROCm Platform Packages
^^^^^^^^^^^^^^^^^^^^^^^

The following platform packages are for ROCm v3.5.0:

Drivers, ToolChains, Libraries, and Source Code

The latest supported version of the drivers, tools, libraries and source code for the ROCm platform have been released and are available from the following GitHub repositories:

**ROCm Core Components**

 -  `ROCk Kernel Driver`_
 -  `ROCr Runtime`_
 -  `ROCt Thunk Interface`_

**ROCm Support Software**

 -  `ROCm SMI`_
 -  `ROCm cmake`_
 -  `rocminfo`_
 -  `ROCm Bandwidth Test`_

**ROCm Compilers**

 -  `HCC compiler`_  (in deprecation)
 -  `HIP`_
 -  `ROCM Clang-OCL Kernel Compiler`_
  
 Example Applications:

 -  `HCC Examples`_ (in deprecation)
 -  `HIP Examples`_
  
**ROCm Device Libraries and Tools**
  
 -  `ROCm Device Libraries`_
 -  `ROCm OpenCL Runtime`_
 -  `ROCm LLVM OCL`_
 -  `ROCm Device Libraries OCL`_
 -  `Asynchronous Task and Memory Interface`_
 -  `ROCr Debug Agent`_
 -  `ROCm Code Object Manager`_
 -  `ROC Profiler`_
 -  `ROC Tracer`_
 -  `AOMP`_
 -  `Radeon Compute Profiler`_
 -  `ROCm Validation Suite`_



**ROCm Libraries**

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


ROCm Core Components
=====================


.. _ROCk Kernel Driver: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/rocm-3.5.0

.. _ROCr Runtime: https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/rocm-3.5.0

.. _ROCt Thunk Interface: https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/rocm-3.5.0


ROCm Support Software
======================


.. _ROCm SMI: https://github.com/RadeonOpenCompute/ROC-smi/tree/rocm-3.5.0

.. _ROCm cmake: https://github.com/RadeonOpenCompute/rocm-cmake/tree/rocm-3.5.0

.. _rocminfo: https://github.com/RadeonOpenCompute/rocminfo/tree/rocm-3.5.0

.. _ROCm Bandwidth Test: https://github.com/RadeonOpenCompute/rocm_bandwidth_test/tree/rocm-3.5.0


ROCm Compilers
================

Note: HCC Compiler is in deprecation mode. 

.. _HCC compiler: https://github.com/RadeonOpenCompute/hcc/tree/rocm-3.5.0 

.. _HIP: https://github.com/ROCm-Developer-Tools/HIP/tree/rocm-3.5.0


.. _HCC Examples: https://github.com/ROCm-Developer-Tools/HCC-Example-Application/tree/ffd65333

.. _HIP Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/rocm-3.5.0



ROCm Device Libraries and Tools
==================================


.. _ROCm Device Libraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-3.5.0

.. _ROCm OpenCL Runtime: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/tree/roc-3.5.0

.. _ROCm LLVM OCL: https://github.com/RadeonOpenCompute/llvm-project/tree/rocm-ocl-3.5.0

.. _ROCm Device Libraries OCL: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-3.5.0

.. _ROCM Clang-OCL Kernel Compiler: https://github.com/RadeonOpenCompute/clang-ocl/tree/rocm-3.5.0

.. _Asynchronous Task and Memory Interface: https://github.com/RadeonOpenCompute/atmi/tree/rocm-3.5.0

.. _ROCr Debug Agent: https://github.com/ROCm-Developer-Tools/rocr_debug_agent/tree/roc-3.5.0

.. _ROCm Code Object Manager: https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/tree/rocm-3.5.0

.. _ROC Profiler: https://github.com/ROCm-Developer-Tools/rocprofiler/tree/rocm-3.5.0

.. _ROC Tracer: https://github.com/ROCm-Developer-Tools/roctracer/tree/rocm-3.5.0

.. _AOMP: https://github.com/ROCm-Developer-Tools/aomp/tree/rocm-3.5.0

.. _Radeon Compute Profiler: https://github.com/GPUOpen-Tools/RCP/tree/3a49405

.. _ROCm Validation Suite: https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/tree/rocm-3.5.0





ROCm Libraries
===============

.. _rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/rocm-3.5.0

.. _hipBLAS: https://github.com/ROCmSoftwarePlatform/hipBLAS/tree/rocm-3.5.0

.. _rocFFT: https://github.com/ROCmSoftwarePlatform/rocFFT/tree/rocm-3.5.0

.. _rocRAND: https://github.com/ROCmSoftwarePlatform/rocRAND/tree/rocm-3.5.0

.. _rocSPARSE: https://github.com/ROCmSoftwarePlatform/rocSPARSE/tree/rocm-3.5.0

.. _hipSPARSE: https://github.com/ROCmSoftwarePlatform/hipSPARSE/tree/rocm-3.5.0

.. _rocALUTION: https://github.com/ROCmSoftwarePlatform/rocALUTION/tree/rocm-3.5.0

.. _MIOpenGEMM: https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/tree/1.1.6

.. _mi open: https://github.com/ROCmSoftwarePlatform/MIOpen/tree/rocm-3.5.0

.. _rocThrust: https://github.com/ROCmSoftwarePlatform/rocThrust/tree/rocm-3.5.0

.. _ROCm SMI Lib: https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/rocm-3.5.0

.. _RCCL: https://github.com/ROCmSoftwarePlatform/rccl/tree/rocm-3.5.0

.. _hipCUB: https://github.com/ROCmSoftwarePlatform/hipCUB/tree/rocm-3.5.0

.. _MIVisionX: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/1.7

.. _AMDMIGraphX: https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/commit/d1e945dabce0078d44c78de67b00232b856e18bc 




List of ROCm Packages for Supported Operating Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm-Libs Meta Packages
~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------+---------------------------------------------------------+
|Package                            |  Debian 	            |   RPM						      |	
+===================================+=======================+=========================================================+
| rocFFT	                    |   Yes	            |  Yes				                      |	 
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocRAND	                    |   Yes	            |  Yes 			                              | 	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocBLAS 	                    |   Yes 	            |  Yes            		                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocSPARSE    	                    |   Yes	            |  Yes			                              | 
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocALUTION  		            |   Yes	            |  Yes  			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocPRIM			    |   Yes 	            |  Yes			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocTHRUST	                    |   Yes	            |  Yes			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| rocSOLVER	                    |   Yes                 |  Yes			                              |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| hipBLAS	                    |   Yes 	            |  Yes				                      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
| hipSPARSE 			    |   Yes	            |  Yes 				                      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
| hipcub			    |   Yes 	            |  Yes				                      |
+-----------------------------------+-----------------------+---------------------------------------------------------+


All Meta Packages
~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------+---------------------------------------------------------+
|Package                            |  Debian 	            |   RPM						      |	
+===================================+=======================+=========================================================+
|ROCm Master Package 	            |   rocm 	            |  rocm-1.6.77-Linux.rpm				      |	 
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm Developer Master Package 	    |   rocm-dev 	    |  rocm-dev-1.6.77-Linux.rpm  			      | 	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm Libraries Master Package 	    |   rocm-libs 	    |  rocm-libs-1.6.77-Linux.rpm            		      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ATMI       	                    |   atmi     	    |  atmi-0.3.7-45-gde867f2-Linux.rpm			      | 
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HCC   				    |   hcc	            |  hcc-1.0.17262-Linux.rpm  			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|hcBLAS 			    |   hcblas 	            |  hcblas-master-482646f-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|hcFFT 	                            |   hcfft. 	            |  hcfft-master-1a96022-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|hcRNG 	                            |   hcrng. 	            |  hcrng-master-c2ada99-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Core 	                    |   hip_base 	    |  hip_base-1.2.17263.rpm				      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Documents 			    |   hip_doc 	    |  hip_doc-1.2.17263.rpm				      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Compiler 			    |   hip_hcc 	    |  hip_hcc-1.2.17263.rpm				      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIP Samples 			    |   hip_samples 	    |  hip_samples-1.2.17263.rpm.			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|HIPBLAS 			    |   hipblas 	    |  hipblas-0.4.0.3-Linux.rpm			      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|MIOpen OpenCL Lib 		    |   miopen-opencl. 	    |  MIOpen-OpenCL-1.0.0-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|rocBLAS 	                    |   rocblas 	    |  rocblas-0.4.2.3-Linux.rpm      		              |		 
+-----------------------------------+-----------------------+---------------------------------------------------------+ 
|rocFFT 	                    |   rocfft 	            |  rocm-device-libs-0.0.1-Linux.rpm			      |
+-----------------------------------+-----------------------+---------------------------------------------------------+        
|ROCm Device Libs 		    |   rocm-device-libs    |  rocm-device-libs-0.0.1-Linux.rpm			      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm OpenCL for Dev with CL headers|    rocm-opencl-dev    |  rocm-opencl-devel-1.2.0-1424893.x86_64.rpm	      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm GDB 	                    |   rocm-gdb 	    |  rocm-gdb-1.5.265-gc4fb045.x86_64.rpm     	      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|RCP profiler 	                    |   rocm-profiler 	    | rocm-profiler-5.1.6386-gbaddcc9.x86_64.rpm	      |	
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm SMI Tool 	                    |   rocm-smi 	    |  rocm-smi-1.0.0_24_g68893bc-1.x86_64.rpm  	      |
+-----------------------------------+-----------------------+---------------------------------------------------------+
|ROCm Utilities 	            |   rocm-utils 	    |  rocm-utils-1.0.0-Linux.rpm			      |
+-----------------------------------+-----------------------+---------------------------------------------------------+


