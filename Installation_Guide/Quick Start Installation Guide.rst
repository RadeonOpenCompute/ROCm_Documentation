.. image:: /Installation_Guide/AMD1.png

======================================
AMD ROCm Installation Guide v3.1.0
======================================

|
-  `Deploying ROCm`_

   -  `Ubuntu`_
   -  `Centos RHEL v7.7`_
   -  `SLES 15 Service Pack 1`_
 
- `Multi\-Verion Installation`_

-  `ROCm Installation Known Issues and Workarounds`_
   
-  `Getting the ROCm Source Code`_

|


.. _Deploying ROCm:

Deploying ROCm
~~~~~~~~~~~~~~~~

AMD hosts both Debian and RPM repositories for the ROCm v3.0x packages.

The following directions show how to install ROCm on supported Debian-based systems such as Ubuntu 18.04.x

**Note**: These directions may not work as written on unsupported Debian-based distributions. For example, newer versions of Ubuntu may not be compatible with the rock-dkms kernel driver. In this case, you can exclude the rocm-dkms and rock-dkms packages.

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

    wget -q -O - https://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -

    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list


The gpg key may change; ensure it is updated when installing a new release. If the key signature verification fails while updating, re-add the key from the ROCm apt repository.

The current rocm.gpg.key is not available in a standard key ring distribution, but has the following sha1sum hash:

::

  e85a40d1a43453fe37d63aa6899bc96e08f2817a rocm.gpg.key
|
3. Install the ROCm meta-package. Update the appropriate repository list and install the rocm-dkms meta-package:

::

     sudo apt update

     sudo apt install rocm-dkms
|
4. Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

     groups
     

5. To add your user to the video group, use the following command for the sudo password:

::

     sudo usermod -a -G video $LOGNAME
|
6. By default, add any future users to the video group. Run the following command to add users to the video group:

::

     echo 'ADD_EXTRA_GROUPS=1' 
     sudo tee -a /etc/adduser.conf

     echo 'EXTRA_GROUPS=video'
     sudo tee -a /etc/adduser.conf
|
7. Restart the system.

|
8. Test the basic ROCm installation.

|
9. After restarting the system, run the following commands to verify that the ROCm installation is successful. If you see your GPUs listed by both commands, the installation is considered successful.

::

     /opt/rocm/bin/rocminfo
     /opt/rocm/opencl/bin/x86_64/clinfo

Note: To run the ROCm programs more efficiently, add the ROCm binaries in your PATH.

::

	echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | 
	sudo tee -a /etc/profile.d/rocm.sh


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

CentOS RHEL v7.7
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the dkms tool to install the kernel drivers on CentOS/RHEL v7.7:

::

  sudo yum install -y epel-release
  sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

.. _ROCM install:

Installing ROCm
'''''''''''''''''

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

5. Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

    groups

6. To add your user to the video group, use the following command for the sudo password:

::

    sudo usermod -a -G video $LOGNAME


7.  By default, add any future users to the video group. Run the following command to add users to the video group:

::

  echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
  echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

Note: The current release supports CentOS/RHEL v7.6. Before updating to the latest version of the operating system, delete the ROCm packages to avoid DKMS-related issues.

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

  echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' |
  sudo tee -a /etc/profile.d/rocm.sh

For more information about installation issues, see: https://rocm.github.io/install_issues.html


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

**Note**: You can use this command instead of installing rocm-dkms.

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

To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using 	 ROCm. To identify the groups you are a member of, use the following command:

::

	groups

7. To add your user to the video group, use the following command for the sudo password:
	
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
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | 

::

	sudo tee -a /etc/profile.d/rocm.sh

**Uninstallation**

To uninstall, use the following command:

::

	sudo zypper remove rocm-dkms rock-dkms

Note: Ensure all other installed packages/components are removed.
Note: Ensure all the content in the /opt/rocm directory is completely removed.

Performing an OpenCL-only Installation of ROCm
''''''''''''''''''''''''''''''''''''''''''''''''

Some users may want to install a subset of the full ROCm installation. If you are trying to install on a system with a limited amount of storage space, or which will only run a small collection of known applications, you may want to install only the packages that are required to run OpenCL applications. To do that, you can run the following installation command instead of the command to install rocm-dkms.

::

  sudo yum install rock-dkms rocm-opencl-devel
  
**Multi\-Version Installation**
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
  

ROCm Installation Known Issues and Workarounds 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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



