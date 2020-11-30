.. image:: /Installation_Guide/amdblack.jpg
|
===============================
Install AMD ROCm 
===============================

-  `Deploying ROCm`_


-  `Prerequisites`_
   
   
- `Supported Operating Systems`_
   
   
	-   `Ubuntu`_ 
   
   
	-   `CentOS RHEL`_ 
   
   
	-  `SLES 15 Service Pack 2`_  
 
 
      
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

**Note**: You must use either ROCm or the amdgpu-pro driver. Using both drivers will result in an installation error. 

**Important - Mellanox ConnectX NIC Users**: If you are using Mellanox ConnetX NIC, you must install Mellanox OFED before installing ROCm. 

For more information about installing Mellanox OFED, refer to:

https://docs.mellanox.com/display/MLNXOFEDv461000/Installing+Mellanox+OFED


Prerequisites 
~~~~~~~~~~~~~~~

In this release, AMD ROCm extends support to SLES 15 SP2

The AMD ROCm platform is designed to support the following operating systems:

* Ubuntu 20.04.1 (5.4 and 5.6-oem) and 18.04.5 (Kernel 5.4)	

**Note**: Ubuntu versions lower than 18 are no longer supported.

**Note**: AMD ROCm only supports Long Term Support (LTS) versions of Ubuntu. Versions other than LTS may work with ROCm, however, they are not officially supported. 

* CentOS 7.8 & RHEL 7.8 (Kernel 3.10.0-1127) (Using devtoolset-7 runtime support)
* CentOS 8.2 & RHEL 8.2 (Kernel 4.18.0 ) (devtoolset is not required)
* SLES 15 SP2


**FRESH INSTALLATION OF AMD ROCm V3.10 RECOMMENDED**

A fresh and clean installation of AMD ROCm v3.10 is recommended. An upgrade from previous releases to AMD ROCm v3.10 is not supported.

**Note**: AMD ROCm release v3.3 or prior releases are not fully compatible with AMD ROCm v3.5 and higher versions. You must perform a fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or older to 3.5 or higher versions and vice-versa.


* For ROCm v3.5 and releases thereafter, the *clinfo* path is changed to - */opt/rocm/opencl/bin/clinfo*.

* For ROCm v3.3 and older releases, the *clinfo* path remains unchanged - */opt/rocm/opencl/bin/x86_64/clinfo*.

**Note**: After an operating system upgrade, AMD ROCm may upgrade automatically and result in an error. This is because AMD ROCm does not support upgrades currently. You must uninstall and reinstall AMD ROCm after an operating system upgrade.


**MULTI-VERSION INSTALLATION UPDATES**

With the AMD ROCm v3.10 release, the following ROCm multi-version installation changes apply:

The meta packages rocm-dkms<version> are now deprecated for multi-version ROCm installs.  For example, rocm-dkms3.7.0, rocm-dkms3.8.0.

* Multi-version installation of ROCm should be performed by installing rocm-dev<version> using each of the desired ROCm versions. 
  For example, rocm-dev3.7.0, rocm-dev3.8.0, rocm-dev3.9.0.   

* ‘version’ files should be created for each multi-version rocm <= 3.10.0

	* command: echo <version> | sudo tee /opt/rocm-<version>/.info/version

	* example: echo 3.10.0 | sudo tee /opt/rocm-3.10.0/.info/version

* The rock-dkms loadable kernel modules should be installed using a single rock-dkms package. 

* ROCm v3.9 and above will not set any *ldconfig* entries for ROCm libraries for multi-version installation.  Users must set *LD_LIBRARY_PATH* to load the ROCm library version of choice.


**NOTE**: The single version installation of the ROCm stack remains the same. The rocm-dkms package can be used for single version installs and is not deprecated at this time.

   
Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   

.. _Ubuntu:

Ubuntu
=========

**Note**: AMD ROCm only supports Long Term Support (LTS) versions of Ubuntu. Versions other than LTS may work with ROCm, however, they are not officially supported. 

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
 
**Note**: The public key has changed to reflect the new location. You must update to the new location as the old key will be removed in a future release.

* Old Key: https://repo.radeon.com/rocm/apt/debian/rocm.gpg.key

* New Key: https://repo.radeon.com/rocm/rocm.gpg.key 


::

    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list


The gpg key may change; ensure it is updated when installing a new release. If the key signature verification fails while updating, re-add the key from the ROCm apt repository.

The current rocm.gpg.key is not available in a standard key ring distribution, but has the following sha1sum hash:

::

  e85a40d1a43453fe37d63aa6899bc96e08f2817a rocm.gpg.key

3. Install the ROCm meta-package. Update the appropriate repository list and install the rocm-dkms meta-package:

::

     sudo apt update

     sudo apt install rocm-dkms && sudo reboot
    

4. Set permissions. To access the GPU, you must be a user in the video and render groups. Ensure your user account is a member of the video and render groups prior to using ROCm. To identify the groups you are a member of, use the following command:

::

     groups
     

5. To add your user to the video and render groups, use the following command with the sudo password:

**Note**: *render group* is required only for Ubuntu v20.04. For all other ROCm supported operating systems, continue to use *video group*.

::

     sudo usermod -a -G video $LOGNAME

     sudo usermod -a -G render $LOGNAME

6. By default, you must add any future users to the video and render groups. To add future users to the video and render groups, run the following command:

::

     echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf

     echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

     echo 'EXTRA_GROUPS=render' | sudo tee -a /etc/adduser.conf

7. Restart the system.

8. After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed by both commands, the installation is considered successful.

::

     /opt/rocm/bin/rocminfo
     /opt/rocm/opencl/bin/clinfo

Note: To run the ROCm programs, add the ROCm binaries in your PATH.

::

    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh


Uninstalling ROCm Packages from Ubuntu
''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages from Ubuntu 20.04 or Ubuntu 18.04.5, run the following command:

::

  sudo apt autoremove rocm-opencl rocm-dkms rocm-dev rocm-utils && sudo reboot


Installing Development Packages for Cross Compilation
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

It is recommended that you develop and test development packages on different systems. For example, some development or build systems may not have an AMD GPU installed. In this scenario, you must avoid installing the ROCk kernel driver on the development system.

Instead, install the following development subset of packages:

::

  sudo apt update
  sudo apt install rocm-dev


**Note**: To execute ROCm enabled applications, you must install the full ROCm driver stack on your system.

Using Debian-based ROCm with Upstream Kernel Drivers
''''''''''''''''''''''''''''''''''''''''''''''''''''''

You can install the ROCm user-level software without installing the AMD's custom ROCk kernel driver. To use the upstream kernels, run the following commands instead of installing rocm-dkms:

::

  sudo apt update	
  sudo apt install rocm-dev	
  echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules


.. _CentOS RHEL:

CentOS RHEL
============

CentOS/RHEL v7.8 and CentOS/RHEL 8.2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section describes how to install ROCm on supported RPM-based systems such as CentOS/RHEL v7.8 and CentOS/RHEL v8.2.

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


Installing CentOS v7.8/v8.2 for DKMS
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

* CentOS/RHEL 7.x : https://repo.radeon.com/rocm/yum/rpm 

* CentOS/RHEL 8.x : https://repo.radeon.com/rocm/centos8/rpm

::

    [ROCm] 
    name=ROCm
    baseurl=https://repo.radeon.com/rocm/yum/rpm
    enabled=1
    gpgcheck=1
    gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key


**Note:** The URL of the repository must point to the location of the repositories’ repodata database.

3. Install ROCm components using the following command:


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
  /opt/rocm/opencl/bin/clinfo


**Note**: Add the ROCm binaries in your PATH for easy implementation of the ROCm programs.

::

  echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin' | sudo tee -a /etc/profile.d/rocm.sh


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


.. _SLES 15 Service Pack 2:

SLES 15 Service Pack 2
========================

The following section tells you how to perform an install and uninstall ROCm on SLES 15 SP 2. 

**Installation**


1. Install the "dkms" package.

::

	sudo SUSEConnect --product PackageHub/15.1/x86_64
	sudo zypper install dkms
	
2. Add the ROCm repo.
 
::

		sudo zypper clean –all
		sudo zypper addrepo https://repo.radeon.com/rocm/zyp/zypper/ rocm
		sudo zypper ref
		sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key
		sudo zypper --gpg-auto-import-keys install rocm-dkms
		sudo reboot

3. Run the following command once

::

	cat <<EOF | sudo tee /etc/modprobe.d/10-unsupported-modules.conf
	allow_unsupported_modules 1
	EOF
	sudo modprobe amdgpu

4. Verify the ROCm installation.

5. Run /opt/rocm/bin/rocminfo and /opt/rocm/opencl/bin/clinfo commands to list the GPUs and verify that the ROCm installation is successful.

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
	/opt/rocm/opencl/bin/clinfo

Note: To run the ROCm programs more efficiently, add the ROCm binaries in your PATH.


::

echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin'|sudo tee -a /etc/profile.d/rocm.sh


**Using ROCm with Upstream Kernel Drivers**

::

	sudo zypper install rocm-dev
	echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
	sudo reboot
	

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

HIP can be easily installed using the pre-built binary packages with the package manager for your platform.


Installing pre-built packages
=============================

HIP can be easily installed using pre-built binary packages using the package manager for your platform.

HIP Prerequisites
==================

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with NVCC installed.


AMD Platform
=============

::

   sudo apt install mesa-common-dev
   sudo apt install clang
   sudo apt install comgr
   sudo apt-get -y install rocm-dkms

HIP-Clang is the compiler for compiling HIP programs on AMD platform.

HIP-Clang can be built manually:

::

   	git clone -b rocm-3.10.x https://github.com/RadeonOpenCompute/llvm-project.git
	cd llvm-project
	mkdir -p build && cd build
	cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" - 		DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm
	make -j
	sudo make install

::

The ROCm device library can be manually built as following,

::

  	export PATH=/opt/rocm/llvm/bin:$PATH
	git clone -b rocm-3.10.x https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git
	cd ROCm-Device-Libs
	mkdir -p build && cd build
	CC=clang CXX=clang++ cmake -DLLVM_DIR=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_WERROR=1 -DLLVM_ENABLE_ASSERTIONS=1 -	DCMAKE_INSTALL_PREFIX=/opt/rocm ..
	make -j
	sudo make install
::


NVIDIA Platform
================

HIP-nvcc is the compiler for HIP program compilation on NVIDIA platform.

-  Add the ROCm package server to your system as per the OS-specific
   guide available
   `here <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`__.
-  Install the 'hip-nvcc' package. This will install CUDA SDK and the
   HIP porting layer.

::

   apt-get install hip-nvcc

-  Default paths and environment variables:

   -  By default HIP looks for CUDA SDK in /usr/local/cuda (can be
      overriden by setting CUDA_PATH env variable).
      
   -  By default HIP is installed into /opt/rocm/hip (can be overridden
      by setting HIP_PATH environment variable).
      
   -  Optionally, consider adding /opt/rocm/bin to your path to make it
      easier to use the tools.


Building HIP from Source
========================

Build ROCclr
=============

ROCclr is defined on AMD platform that HIP use Radeon Open Compute
Common Language Runtime (ROCclr), which is a virtual device interface
that HIP runtimes interact with different backends. 

See https://github.com/ROCm-Developer-Tools/ROCclr

::

   	git clone -b rocm-3.10.x https://github.com/ROCm-Developer-Tools/ROCclr.git
	export ROCclr_DIR="$(readlink -f ROCclr)"
	git clone -b rocm-3.10.x https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
	export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
	cd "$ROCclr_DIR"
	mkdir -p build;cd build
	cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr ..
	make -j
	sudo make install

::

Build HIP
===========

::

   	git clone -b rocm-3.10.x https://github.com/ROCm-Developer-Tools/HIP.git
	export HIP_DIR="$(readlink -f HIP)"
	cd "$HIP_DIR"
	mkdir -p build; cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DHIP_COMPILER=clang -DHIP_PLATFORM=rocclr -	DCMAKE_PREFIX_PATH="$ROCclr_DIR/build;/opt/rocm/" -DCMAKE_INSTALL_PREFIX=	</where/to/install/hip> ..
	make -j
	sudo make install
::


Default paths and environment variables
=========================================

-  By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by
   setting HSA_PATH environment variable).
-  By default HIP is installed into /opt/rocm/hip (can be overridden by
   setting HIP_PATH environment variable).
-  By default HIP looks for clang in /opt/rocm/llvm/bin (can be
   overridden by setting HIP_CLANG_PATH environment variable)
-  By default HIP looks for device library in /opt/rocm/lib (can be
   overridden by setting DEVICE_LIB_PATH environment variable).
-  Optionally, consider adding /opt/rocm/bin to your PATH to make it
   easier to use the tools.
-  Optionally, set HIPCC_VERBOSE=7 to output the command line for
   compilation.

After installation, make sure HIP_PATH is pointed to */where/to/install/hip*


Verify your installation
========================

Run hipconfig (instructions below assume default installation path) :

.. code:: shell

   /opt/rocm/bin/hipconfig --full

Compile and run the `square
sample <https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square>`__.




AMD ROCm Multi Version Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install and access multiple versions of the ROCm toolkit simultaneously.

Previously, users could install only a single version of the ROCm toolkit. 

Now, users have the option to install multiple versions simultaneously and toggle to the desired version of the ROCm toolkit. From the v3.3 release, multiple versions of ROCm packages can be installed in the */opt/rocm-<version>* folder.
 
**Prerequisites**
###############################

Ensure the existing installations of ROCm, including */opt/rocm*, are completely removed before the v3.10 ROCm toolkit installation. The ROCm v3.10 package requires a clean installation.

* To install a single instance of ROCm, use the rocm-dkms or rocm-dev packages to install all the required components. This creates a symbolic link */opt/rocm* pointing to the corresponding version of ROCm installed on the system. 

* To install individual ROCm components, create the */opt/rocm* symbolic link pointing to the version of ROCm installed on the system. For example, *# ln -s /opt/rocm-3.10.0 /opt/rocm*

* To install multiple instance ROCm packages, create */opt/rocm* symbolic link pointing to the version of ROCm installed/used on the system. For example, *# ln -s /opt/rocm-3.10.0 /opt/rocm*

**Note**: The Kernel Fusion Driver (KFD) must be compatible with all versions of the ROCm software installed on the system.


Before You Begin
#################

Review the following important notes:

**Single Version Installation**

To install a single instance of the ROCm package, access the non-versioned packages. 

**Note**: You must not install any components from the multi-instance set.

For example, 

* rocm-dkms

* rocm-dev

* hip

A fresh installation of single-version installation will install the new version in the */opt/rocm-<version>* folder.

.. image:: /Current_Release_Notes/singleinstance.png

**Multi Version Installation**

* To install a multi-instance of the ROCm package, access the versioned packages and components. 

For example,

  * rocm-dev3.10.0

  * hip3.10.0

* kernel/firmware package doesn't have multi version so it should be installed using "apt/yum/zypper install rock-dkms".

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
  ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.10.x
  repo sync


**Note**: Using this sample code will cause the repo to download the open source code associated with this ROCm release. Ensure that you have ssh-keys configured on your machine for your GitHub ID prior to the download.



.. _Machine Learning and High Performance Computing Software Stack for AMD GPU:

============================
Software Stack for AMD GPU
============================

Machine Learning and High Performance Computing Software Stack for AMD GPU v3.10.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. _ROCm Binary Package Structure:

ROCm Binary Package Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm is a collection of software ranging from drivers and runtimes to libraries and developer tools. In AMD's package distributions, these software projects are provided as a separate packages. This allows users to install only the packages they need, if they do not wish to install all of ROCm. These packages will install most of the ROCm software into ``/opt/rocm/`` by default.

The packages for each of the major ROCm components are:

ROCm Core Components
=====================

 -   ROCk Kernel Driver: ``rock-dkms rock-dkms-firmware``
 -   ROCr Runtime: ``hsa-rocr-dev``
 -   ROCt Thunk Interface: ``hsakmt-roct``, ``hsakmt-roct-dev``


ROCm Support Software
======================

 -   ROCm SMI: ``rocm-smi``
 -   ROCm cmake: ``rocm-cmake``
 -   rocminfo: ``rocminfo``
 -   ROCm Bandwidth Test: ``rocm_bandwidth_test``
     
    
ROCm Compilers
================

 -   Clang compiler: ``llvm-amdgpu``
 -   HIP: ``hip_base``, ``hip_doc``, ``hip_rocclr``, ``hip_samples``     
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

The following platform packages are for ROCm v3.10.0:

Drivers, ToolChains, Libraries, and Source Code
==================================================

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

 -  `HIP`_
 -  `ROCM Clang-OCL Kernel Compiler`_
  
 Example Applications:

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


..  ROCm Core Components

.. _ROCk Kernel Driver: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/rocm-3.10.0

.. _ROCr Runtime: https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/rocm-3.10.0

.. _ROCt Thunk Interface: https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/rocm-3.10.0


.. ROCm Support Software

.. _ROCm SMI: https://github.com/RadeonOpenCompute/ROC-smi/tree/rocm-3.10.0

.. _ROCm cmake: https://github.com/RadeonOpenCompute/rocm-cmake/tree/rocm-3.10.0

.. _rocminfo: https://github.com/RadeonOpenCompute/rocminfo/tree/rocm-3.10.0

.. _ROCm Bandwidth Test: https://github.com/RadeonOpenCompute/rocm_bandwidth_test/tree/rocm-3.10.0


.. ROCm Compilers

.. _HIP: https://github.com/ROCm-Developer-Tools/HIP/tree/rocm-3.10.0

.. _HIP Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/rocm-3.10.0



.. ROCm Device Libraries and Tools

.. _ROCm Device Libraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-3.10.0

.. _ROCm OpenCL Runtime: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/tree/rocm-3.10.0

.. _ROCm LLVM OCL: https://github.com/RadeonOpenCompute/llvm-project/tree/rocm-ocl-3.10.0

.. _ROCm Device Libraries OCL: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-3.10.0

.. _ROCM Clang-OCL Kernel Compiler: https://github.com/RadeonOpenCompute/clang-ocl/tree/rocm-3.10.0

.. _Asynchronous Task and Memory Interface: https://github.com/RadeonOpenCompute/atmi/tree/rocm-3.10.0

.. _ROCr Debug Agent: https://github.com/ROCm-Developer-Tools/rocr_debug_agent/tree/rocm-3.10.0

.. _ROCm Code Object Manager: https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/tree/rocm-3.10.0

.. _ROC Profiler: https://github.com/ROCm-Developer-Tools/rocprofiler/tree/rocm-3.10.0

.. _ROC Tracer: https://github.com/ROCm-Developer-Tools/roctracer/tree/rocm-3.10.0

.. _AOMP: https://github.com/ROCm-Developer-Tools/aomp/tree/rocm-3.10.0

.. _Radeon Compute Profiler: https://github.com/GPUOpen-Tools/RCP/tree/3a49405

.. _ROCm Validation Suite: https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/tree/rocm-3.10.0


.. ROCm Libraries

.. _rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/rocm-3.10.0

.. _hipBLAS: https://github.com/ROCmSoftwarePlatform/hipBLAS/tree/rocm-3.10.0

.. _rocFFT: https://github.com/ROCmSoftwarePlatform/rocFFT/tree/rocm-3.10.0

.. _rocRAND: https://github.com/ROCmSoftwarePlatform/rocRAND/tree/rocm-3.10.0

.. _rocSPARSE: https://github.com/ROCmSoftwarePlatform/rocSPARSE/tree/rocm-3.10.0

.. _hipSPARSE: https://github.com/ROCmSoftwarePlatform/hipSPARSE/tree/rocm-3.10.0

.. _rocALUTION: https://github.com/ROCmSoftwarePlatform/rocALUTION/tree/rocm-3.10.0

.. _MIOpenGEMM: https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/tree/rocm-3.10.0

.. _mi open: https://github.com/ROCmSoftwarePlatform/MIOpen/tree/rocm-3.10.0

.. _rocThrust: https://github.com/ROCmSoftwarePlatform/rocThrust/tree/rocm-3.10.0

.. _ROCm SMI Lib: https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/rocm-3.10.0

.. _RCCL: https://github.com/ROCmSoftwarePlatform/rccl/tree/rocm-3.10.0

.. _hipCUB: https://github.com/ROCmSoftwarePlatform/hipCUB/tree/rocm-3.10.0

.. _MIVisionX: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/rocm-3.10.0

.. _AMDMIGraphX: https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/rocm-3.10.0




List of ROCm Packages for Supported Operating Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm-Library Meta Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Meta Packages
~~~~~~~~~~~~~~~~~

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


============================================
Hardware and Software Support Information
============================================

 
-  `Hardware and Software Support <https://github.com/RadeonOpenCompute/ROCm#Hardware-and-Software-Support>`__

- `Radeon Instinct™ GPU-Powered HPC Solutions <https://www.amd.com/en/graphics/servers-radeon-instinct-mi-powered-servers>`__
