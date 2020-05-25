.. image:: /Installation_Guide/amdblack.jpg
|
==============================================
AMD ROCm Installation Guide v3.3.0
==============================================

-  `Deploying ROCm`_

   -  `Ubuntu`_
   
   -  `Centos RHEL v7.7`_
   
   -  `SLES 15 Service Pack 1`_
   
-  `MultiVersion Installation`_

-  `ROCm Installation Known Issues and Workarounds`_


-  `Getting the ROCm Source Code`_


.. _Deploying ROCm:

Deploying ROCm
~~~~~~~~~~~~~~~~

AMD hosts both Debian and RPM repositories for the ROCm v3.x packages.

The following directions show how to install ROCm on supported Debian-based systems such as Ubuntu 18.04.x

**Note**: These directions may not work as written on unsupported Debian-based distributions. For example, newer versions of Ubuntu may not be compatible with the rock-dkms kernel driver. In this case, you can exclude the rocm-dkms and rock-dkms packages.

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

     sudo apt install rocm-dkms

4. Set permissions. To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

::

     groups
     

5. To add your user to the video group, use the following command for the sudo password:

::

     sudo usermod -a -G video $LOGNAME

6. By default, add any future users to the video group. Run the following command to add users to the video group:

::

     echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf

     echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

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


Uninstalling ROCm Packages from Ubuntu
''''''''''''''''''''''''''''''''''''''''

To uninstall the ROCm packages from Ubuntu 16.04.6 or Ubuntu 18.04.4, run the following command:

::

  sudo apt autoremove rocm-opencl rocm-dkms rocm-dev rocm-utils


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

   sudo rpm -ivh <repo>


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

Note: The current release supports CentOS/RHEL v7.7. Before updating to the latest version of the operating system, delete the ROCm packages to avoid DKMS-related issues.

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


Compiling Applications Using HCC, HIP, and Other ROCm Software
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


To compile applications or samples, run the following command to use gcc-7.2 provided by the devtoolset-7 environment:

::

  scl enable devtoolset-7 bash


Uninstalling ROCm from CentOS/RHEL v7.7
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

To access the GPU, you must be a user in the video group. Ensure your user account is a member of the video group prior to using ROCm. To identify the groups you are a member of, use the following command:

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

  sudo yum install rock-dkms rocm-opencl-devel
  
  |
 

 **MultiVersion Installation**
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
  ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.3.0
  repo sync


Note: Using this sample code will cause the repo to download the open source code associated with this ROCm release. Ensure that you have ssh-keys configured on your machine for your GitHub ID prior to the download.

Building the ROCm Source Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ROCm component repository contains directions for building that component. You can access the desired component for instructions to build the repository.


.. _Machine Learning and High Performance Computing Software Stack for AMD GPU:

===================================================================================
Machine Learning and High Performance Computing Software Stack for AMD GPU v3.3.0
===================================================================================

Machine Learning and High Performance Computing Software Stack for AMD GPU v3.3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ROCm Version 3.3.0**

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
     
    
ROCm Development ToolChain
===========================

     -   HCC compiler: ``hcc``
     
     -   HIP: ``hip_base``, ``hip_doc``, ``hip_hcc``, ``hip_samples``
     
     -   ROCm Device Libraries: ``rocm-device-libs``
     
     -   ROCm OpenCL: ``rocm-opencl``, ``rocm-opencl-devel`` (on RHEL/CentOS), ``rocm-opencl-dev`` (on Ubuntu)
     
     -   ROCM Clang-OCL Kernel Compiler: ``rocm-clang-ocl``
     
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

   rocm-dkms
    |--rock-dkms
    \--rocm-dev
       |--comgr
       |--hcc
       |--hip-base
       |--hip-doc
       |--hip-hcc
       |--hip-samples
       |--hsakmt-roct
       |--hsakmt-roct-dev
       |--hsa-amd-aqlprofile
       |--hsa-ext-rocr-dev
       |--hsa-rocr-dev
       |--rocm-cmake
       |--rocm-device-libs
       |--rocm-smi-lib64
       |--rocprofiler-dev
       |--rocm-debug-agent
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

Note: Some users may want to skip certain packages. For instance, a user that wants to use the upstream kernel drivers (rather than those supplied by AMD) may want to skip the rocm-dkms and rock-dkms packages. Instead, they could directly install rocm-dev.

Similarly, a user that only wants to install OpenCL support instead of HCC and HIP may want to skip the rocm-dkms and rocm-dev packages. Instead, they could directly install rock-dkms, rocm-opencl, and rocm-opencl-dev and their dependencies.

.. _ROCm Platform Packages:


ROCm Platform Packages
^^^^^^^^^^^^^^^^^^^^^^^

The following platform packages are for ROCm v3.3.0:

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

**ROCm Development ToolChain**

  -  `HCC compiler`_
  -  `HIP`_
  -  `ROCm Device Libraries`_

 -  `ROCm OpenCL Runtime`_
 -  `ROCm LLVM OCL`_
 -  `ROCm Device Libraries OCL`_
         
 -  `ROCM Clang-OCL Kernel Compiler`_
 -  `Asynchronous Task and Memory Interface`_
 -  `ROCr Debug Agent`_
 -  `ROCm Code Object Manager`_
 -  `ROC Profiler`_
 -  `ROC Tracer`_
 -  `AOMP`_
 -  `Radeon Compute Profiler`_
 -  `ROCm Validation Suite`_

 Example Applications:

  -  `HCC Examples`_
  -  `HIP Examples`_

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


.. _ROCk Kernel Driver: https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/roc-3.3.0

.. _ROCr Runtime: https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/rocm-3.3.0

.. _ROCt Thunk Interface: https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/roc-3.3.0


ROCm Support Software
======================


.. _ROCm SMI: https://github.com/RadeonOpenCompute/ROC-smi/tree/roc-3.3.0

.. _ROCm cmake: https://github.com/RadeonOpenCompute/rocm-cmake/tree/rocm-3.3.0

.. _rocminfo: https://github.com/RadeonOpenCompute/rocminfo/tree/rocm-3.3.0

.. _ROCm Bandwidth Test: https://github.com/RadeonOpenCompute/rocm_bandwidth_test/tree/rocm-3.3.0


ROCm Development ToolChain
============================


.. _HCC compiler: https://github.com/RadeonOpenCompute/hcc/tree/rocm-3.3.0 

.. _HIP: https://github.com/ROCm-Developer-Tools/HIP/tree/rocm-3.3.0

.. _ROCm Device Libraries: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-ocl-3.3.0

.. _ROCm OpenCL Runtime: http://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/tree/roc-3.3.0

.. _ROCm LLVM OCL: https://github.com/RadeonOpenCompute/llvm-project/tree/rocm-ocl-3.3.0

.. _ROCm Device Libraries OCL: https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/rocm-ocl-3.3.0

.. _ROCM Clang-OCL Kernel Compiler: https://github.com/RadeonOpenCompute/clang-ocl/tree/rocm-3.3.0

.. _Asynchronous Task and Memory Interface: https://github.com/RadeonOpenCompute/atmi/tree/rocm-3.3.0

.. _ROCr Debug Agent: https://github.com/ROCm-Developer-Tools/rocr_debug_agent/tree/roc-3.3.0

.. _ROCm Code Object Manager: https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/tree/rocm-3.3.0

.. _ROC Profiler: https://github.com/ROCm-Developer-Tools/rocprofiler/tree/roc-3.3.0

.. _ROC Tracer: https://github.com/ROCm-Developer-Tools/roctracer/tree/roc-3.3.0

.. _AOMP: https://github.com/ROCm-Developer-Tools/aomp/tree/roc-3.3.0

.. _Radeon Compute Profiler: https://github.com/GPUOpen-Tools/RCP/tree/3a49405

.. _ROCm Validation Suite: https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/tree/roc-3.3.0

.. _HCC Examples: https://github.com/ROCm-Developer-Tools/HCC-Example-Application/tree/ffd65333

.. _HIP Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/rocm-3.3.0



ROCm Libraries
===============

.. _rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/rocm-3.3.0

.. _hipBLAS: https://github.com/ROCmSoftwarePlatform/hipBLAS/tree/rocm-3.3.0

.. _rocFFT: https://github.com/ROCmSoftwarePlatform/rocFFT/tree/rocm-3.3

.. _rocRAND: https://github.com/ROCmSoftwarePlatform/rocRAND/tree/rocm-3.3.0

.. _rocSPARSE: https://github.com/ROCmSoftwarePlatform/rocSPARSE/tree/rocm-3.3.0

.. _hipSPARSE: https://github.com/ROCmSoftwarePlatform/hipSPARSE/tree/rocm-3.3.0

.. _rocALUTION: https://github.com/ROCmSoftwarePlatform/rocALUTION/tree/rocm-3.3.0

.. _MIOpenGEMM: https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/tree/6275a879

.. _mi open: https://github.com/ROCmSoftwarePlatform/MIOpen/tree/roc-3.3.0

.. _rocThrust: https://github.com/ROCmSoftwarePlatform/rocThrust/tree/rocm-3.3.0

.. _ROCm SMI Lib: https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/rocm-3.3.0

.. _RCCL: https://github.com/ROCmSoftwarePlatform/rccl/tree/rocm-3.3.0

.. _hipCUB: https://github.com/ROCmSoftwarePlatform/hipCUB/tree/rocm-3.3.0

.. _MIVisionX: https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/1.7

.. _AMDMIGraphX: https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/commit/d1e945dabce0078d44c78de67b00232b856e18bc 




Features and enhancements introduced in previous versions of ROCm can be found in :ref:`Current-Release-Notes`.

=========================
AMD ROCm Version History
=========================

This file contains archived version history information for the ROCm project

New features and enhancements in ROCm v3.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This release was not productized.


New features and enhancements in ROCm v3.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'Release Notes: https://github.com/RadeonOpenCompute/ROCm/tree/roc-3.1.0

**Change in ROCm Installation Directory Structure**

A fresh installation of the ROCm toolkit installs the packages in the /opt/rocm-<version> folder. 
Previously, ROCm toolkit packages were installed in the /opt/rocm folder.

**Reliability, Accessibility, and Serviceability Support for Vega 7nm**

The Reliability, Accessibility, and Serviceability (RAS) support for Vega7nm is now available. 

**SLURM Support for AMD GPU**

SLURM (Simple Linux Utility for Resource Management) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. 


New features and enhancements in ROCm v3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release Notes: https://github.com/RadeonOpenCompute/ROCm/tree/roc-3.0.0

* Support for CentOS RHEL v7.7
* Support is extended for CentOS/RHEL v7.7 in the ROCm v3.0 release. For more information about the CentOS/RHEL v7.7 release, see:

* CentOS/RHEL

* Initial distribution of AOMP 0.7-5 in ROCm v3.0
The code base for this release of AOMP is the Clang/LLVM 9.0 sources as of October 8th, 2019. The LLVM-project branch used to build this release is AOMP-191008. It is now locked. With this release, an artifact tarball of the entire source tree is created. This tree includes a Makefile in the root directory used to build AOMP from the release tarball. You can use Spack to build AOMP from this source tarball or build manually without Spack.

* Fast Fourier Transform Updates
The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform. Fast Fourier transforms are used in signal processing, image processing, and many other areas. The following real FFT performance change is made in the ROCm v3.0 release:

* Implement efficient real/complex 2D transforms for even lengths.

Other improvements:

• More 2D test coverage sizes.

• Fix buffer allocation error for large 1D transforms.

• C++ compatibility improvements.

MemCopy Enhancement for rocProf
In the v3.0 release, the rocProf tool is enhanced with an additional capability to dump asynchronous GPU memcopy information into a .csv file. You can use the '-hsa-trace' option to create the results_mcopy.csv file. Future enhancements will include column labels.

New features and enhancements in ROCm v2.10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rocBLAS Support for Complex GEMM

The rocBLAS library is a gpu-accelerated implementation of the standard Basic Linear Algebra Subroutines (BLAS). rocBLAS is designed to enable you to develop algorithms, including high performance computing, image analysis, and machine learning.

In the AMD ROCm release v2.10, support is extended to the General Matrix Multiply (GEMM) routine for multiple small matrices processed simultaneously for rocBLAS in AMD Radeon Instinct MI50. Both single and double precision, CGEMM and ZGEMM, are now supported in rocBLAS.

Support for SLES 15 SP1

In the AMD ROCm v2.10 release, support is added for SUSE Linux® Enterprise Server (SLES) 15 SP1. SLES is a modular operating system for both multimodal and traditional IT.

Code Marker Support for rocProfiler and rocTracer Libraries

Code markers provide the external correlation ID for the calling thread. This function indicates that the calling thread is entering and leaving an external API region.

New features and enhancements in ROCm 2.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Initial release for Radeon Augmentation Library(RALI)

The AMD Radeon Augmentation Library (RALI) is designed to efficiently decode and process images from a variety of storage formats and modify them through a processing graph programmable by the user. RALI currently provides C API.

Quantization in MIGraphX v0.4

MIGraphX 0.4 introduces support for fp16 and int8 quantization. For additional details, as well as other new MIGraphX features, see MIGraphX documentation.

rocSparse csrgemm

csrgemm enables the user to perform matrix-matrix multiplication with two sparse matrices in CSR format.

Singularity Support

ROCm 2.9 adds support for Singularity container version 2.5.2.

Initial release of rocTX

ROCm 2.9 introduces rocTX, which provides a C API for code markup for performance profiling. This initial release of rocTX supports annotation of code ranges and ASCII markers. 

* Added support for Ubuntu 18.04.3
* Ubuntu 18.04.3 is now supported in ROCm 2.9.

New features and enhancements in ROCm 2.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for NCCL2.4.8 API

Implements ncclCommAbort() and ncclCommGetAsyncError() to match the NCCL 2.4.x API

New features and enhancements in ROCm 2.7.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This release is a hotfix for ROCm release 2.7.

Issues fixed in ROCm 2.7.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A defect in upgrades from older ROCm releases has been fixed.
* rocprofiler --hiptrace and --hsatrace fails to load roctracer library
* In ROCm 2.7.2, rocprofiler --hiptrace and --hsatrace fails to load roctracer library defect has been fixed.
* To generate traces, please provide directory path also using the parameter: -d <$directoryPath> for example:

/opt/rocm/bin/rocprof  --hsa-trace -d $PWD/traces /opt/rocm/hip/samples/0_Intro/bit_extract/bit_extract
All traces and results will be saved under $PWD/traces path

Upgrading from ROCm 2.7 to 2.7.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To upgrade, please remove 2.7 completely as specified for ubuntu or for centos/rhel, and install 2.7.2 as per instructions install instructions

Other notes
To use rocprofiler features, the following steps need to be completed before using rocprofiler:

Step-1: Install roctracer
Ubuntu 16.04 or Ubuntu 18.04:
sudo apt install roctracer-dev
CentOS/RHEL 7.6:
sudo yum install roctracer-dev

Step-2: Add /opt/rocm/roctracer/lib to LD_LIBRARY_PATH
New features and enhancements in ROCm 2.7
[rocFFT] Real FFT Functional
Improved real/complex 1D even-length transforms of unit stride. Performance improvements of up to 4.5x are observed. Large problem sizes should see approximately 2x.

rocRand Enhancements and Optimizations

Added support for new datatypes: uchar, ushort, half.

Improved performance on "Vega 7nm" chips, such as on the Radeon Instinct MI50

mtgp32 uniform double performance changes due generation algorithm standardization. Better quality random numbers now generated with 30% decrease in performance

Up to 5% performance improvements for other algorithms

RAS

Added support for RAS on Radeon Instinct MI50, including:

* Memory error detection
* Memory error detection counter
* ROCm-SMI enhancements
* Added ROCm-SMI CLI and LIB support for FW version, compute running processes, utilization rates, utilization counter, link error counter, and unique ID.

New features and enhancements in ROCm 2.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCmInfo enhancements

ROCmInfo was extended to do the following: For ROCr API call errors including initialization determine if the error could be explained by:

ROCk (driver) is not loaded / available
User does not have membership in appropriate group - "video"
If not above print the error string that is mapped to the returned error code
If no error string is available, print the error code in hex
Thrust - Functional Support on Vega20

ROCm2.6 contains the first official release of rocThrust and hipCUB. rocThrust is a port of thrust, a parallel algorithm library. hipCUB is a port of CUB, a reusable software component library. Thrust/CUB has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: rocThrust and hipCUB library replaces https://github.com/ROCmSoftwarePlatform/thrust (hip-thrust), i.e. hip-thrust has been separated into two libraries, rocThrust and hipCUB. Existing hip-thrust users are encouraged to port their code to rocThrust and/or hipCUB. Hip-thrust will be removed from official distribution later this year.

MIGraphX v0.3

MIGraphX optimizer adds support to read models frozen from Tensorflow framework. Further details and an example usage at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.3

MIOpen 2.0

This release contains several new features including an immediate mode for selecting convolutions, bfloat16 support, new layers, modes, and algorithms.

MIOpenDriver, a tool for benchmarking and developing kernels is now shipped with MIOpen. BFloat16 now supported in HIP requires an updated rocBLAS as a GEMM backend.

Immediate mode API now provides the ability to quickly obtain a convolution kernel.

MIOpen now contains HIP source kernels and implements the ImplicitGEMM kernels. This is a new feature and is currently disabled by default. Use the environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation this feature. ImplicitGEMM requires an up to date HIP version of at least 1.5.9211.

A new "loss" catagory of layers has been added, of which, CTC loss is the first. See the API reference for more details. 2.0 is the last release of active support for gfx803 architectures. In future releases, MIOpen will not actively debug and develop new features specifically for gfx803.

System Find-Db in memory cache is disabled by default. Please see build instructions to enable this feature. Additional documentation can be found here: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/

Bloat16 software support in rocBLAS/Tensile

Added mixed precision bfloat16/IEEE f32 to gemm_ex. The input and output matrices are bfloat16. All arithmetic is in IEEE f32.

AMD Infinity Fabric™ Link enablement

The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in two hives or two Radeon Instinct MI60 or Radeon Instinct MI50 boards in four hives via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

ROCm-smi features and bug fixes

mGPU & Vendor check

Fix clock printout if DPM is disabled

Fix finding marketing info on CentOS

Clarify some error messages

ROCm-smi-lib enhancements

Documentation updates

Improvements to *name_get functions

RCCL2 Enablement

RCCL2 supports collectives intranode communication using PCIe, Infinity Fabric™, and pinned host memory, as well as internode communication using Ethernet (TCP/IP sockets) and Infiniband/RoCE (Infiniband Verbs). Note: For Infiniband/RoCE, RDMA is not currently supported.

rocFFT enhancements

Added: Debian package with FFT test, benchmark, and sample programs
Improved: hipFFT interfaces
Improved: rocFFT CPU reference code, plan generation code and logging code

New features and enhancements in ROCm 2.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UCX 1.6 support

Support for UCX version 1.6 has been added.

BFloat16 GEMM in rocBLAS/Tensile

Software support for BFloat16 on Radeon Instinct MI50, MI60 has been added. This includes:

Mixed precision GEMM with BFloat16 input and output matrices, and all arithmetic in IEEE32 bit

Input matrix values are converted from BFloat16 to IEEE32 bit, all arithmetic and accumulation is IEEE32 bit. Output values are rounded from IEEE32 bit to BFloat16

Accuracy should be correct to 0.5 ULP

ROCm-SMI enhancements

CLI support for querying the memory size, driver version, and firmware version has been added to ROCm-smi.

[PyTorch] multi-GPU functional support (CPU aggregation/Data Parallel)

Multi-GPU support is enabled in PyTorch using Dataparallel path for versions of PyTorch built using the 06c8aa7a3bbd91cda2fd6255ec82aad21fa1c0d5 commit or later.

rocSparse optimization on Radeon Instinct MI50 and MI60

This release includes performance optimizations for csrsv routines in the rocSparse library.

[Thrust] Preview

Preview release for early adopters. rocThrust is a port of thrust, a parallel algorithm library. Thrust has been ported to the HIP/ROCm platform to use the rocPRIM library. The HIP ported library works on HIP/ROCm platforms.

Note: This library will replace https://github.com/ROCmSoftwarePlatform/thrust in a future release. The package for rocThrust (this library) currently conflicts with version 2.5 package of thrust. They should not be installed together.

Support overlapping kernel execution in same HIP stream

HIP API has been enhanced to allow independent kernels to run in parallel on the same stream.

AMD Infinity Fabric™ Link enablement

The ability to connect four Radeon Instinct MI60 or Radeon Instinct MI50 boards in one hive via AMD Infinity Fabric™ Link GPU interconnect technology has been added.

New features and enhancements in ROCm 2.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow 2.0 support

ROCm 2.4 includes the enhanced compilation toolchain and a set of bug fixes to support TensorFlow 2.0 features natively

AMD Infinity Fabric™ Link enablement

ROCm 2.4 adds support to connect two Radeon Instinct MI60 or Radeon Instinct MI50 boards via AMD Infinity Fabric™ Link GPU interconnect technology.

New features and enhancements in ROCm 2.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mem usage per GPU

Per GPU memory usage is added to rocm-smi. Display information regarding used/total bytes for VRAM, visible VRAM and GTT, via the --showmeminfo flag

MIVisionX, v1.1 - ONNX

ONNX parser changes to adjust to new file formats

MIGraphX, v0.2

MIGraphX 0.2 supports the following new features:

New Python API

* Support for additional ONNX operators and fixes that now enable a large set of Imagenet models
* Support for RNN Operators
* Support for multi-stream Execution
* [Experimental] Support for Tensorflow frozen protobuf files

See: Getting-started:-using-the-new-features-of-MIGraphX-0.2 for more details

MIOpen, v1.8 - 3d convolutions and int8

This release contains full 3-D convolution support and int8 support for inference.
Additionally, there are major updates in the performance database for major models including those found in Torchvision.
See: MIOpen releases

Caffe2 - mGPU support

Multi-gpu support is enabled for Caffe2.

rocTracer library, ROCm tracing API for collecting runtimes API and asynchronous GPU activity traces
HIP/HCC domains support is introduced in rocTracer library.

BLAS - Int8 GEMM performance, Int8 functional and performance
Introduces support and performance optimizations for Int8 GEMM, implements TRSV support, and includes improvements and optimizations with Tensile.

Prioritized L1/L2/L3 BLAS (functional)
Functional implementation of BLAS L1/L2/L3 functions

BLAS - tensile optimization
Improvements and optimizations with tensile

MIOpen Int8 support
Support for int8

New features and enhancements in ROCm 2.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rocSparse Optimization on Vega20
Cache usage optimizations for csrsv (sparse triangular solve), coomv (SpMV in COO format) and ellmv (SpMV in ELL format) are available.

DGEMM and DTRSM Optimization
Improved DGEMM performance for reduced matrix sizes (k=384, k=256)

Caffe2
Added support for multi-GPU training

New features and enhancements in ROCm 2.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RocTracer v1.0 preview release – 'rocprof' HSA runtime tracing and statistics support -
Supports HSA API tracing and HSA asynchronous GPU activity including kernels execution and memory copy

Improvements to ROCM-SMI tool -
Added support to show real-time PCIe bandwidth usage via the -b/--showbw flag

DGEMM Optimizations -
Improved DGEMM performance for large square and reduced matrix sizes (k=384, k=256)

New features and enhancements in ROCm 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds support for RHEL 7.6 / CentOS 7.6 and Ubuntu 18.04.1

Adds support for Vega 7nm, Polaris 12 GPUs

Introduces MIVisionX
A comprehensive computer vision and machine intelligence libraries, utilities and applications bundled into a single toolkit.
Improvements to ROCm Libraries
rocSPARSE & hipSPARSE
rocBLAS with improved DGEMM efficiency on Vega 7nm

MIOpen
This release contains general bug fixes and an updated performance database
Group convolutions backwards weights performance has been improved

RNNs now support fp16
Tensorflow multi-gpu and Tensorflow FP16 support for Vega 7nm
TensorFlow v1.12 is enabled with fp16 support
PyTorch/Caffe2 with Vega 7nm Support

fp16 support is enabled

Several bug fixes and performance enhancements

Known Issue: breaking changes are introduced in ROCm 2.0 which are not addressed upstream yet. Meanwhile, please continue to use ROCm fork at https://github.com/ROCmSoftwarePlatform/pytorch

Improvements to ROCProfiler tool

Support for Vega 7nm

Support for hipStreamCreateWithPriority

Creates a stream with the specified priority. It creates a stream on which enqueued kernels have a different priority for execution compared to kernels enqueued on normal priority streams. The priority could be higher or lower than normal priority streams.

OpenCL 2.0 support

ROCm 2.0 introduces full support for kernels written in the OpenCL 2.0 C language on certain devices and systems.  Applications can detect this support by calling the “clGetDeviceInfo” query function with “parame_name” argument set to “CL_DEVICE_OPENCL_C_VERSION”.  

In order to make use of OpenCL 2.0 C language features, the application must include the option “-cl-std=CL2.0” in options passed to the runtime API calls responsible for compiling or building device programs.  The complete specification for the OpenCL 2.0 C language can be obtained using the following link: https://www.khronos.org/registry/OpenCL/specs/opencl-2.0-openclc.pdf

Improved Virtual Addressing (48 bit VA) management for Vega 10 and later GPUs

Fixes Clang AddressSanitizer and potentially other 3rd-party memory debugging tools with ROCm

Small performance improvement on workloads that do a lot of memory management

Removes virtual address space limitations on systems with more VRAM than system memory
Kubernetes support

New features and enhancements in ROCm 1.9.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RDMA(MPI) support on Vega 7nm

Support ROCnRDMA based on Mellanox InfiniBand

Improvements to HCC

Improved link time optimization

Improvements to ROCProfiler tool

General bug fixes and implemented versioning APIs

New features and enhancements in ROCm 1.9.2

RDMA(MPI) support on Vega 7nm

Support ROCnRDMA based on Mellanox InfiniBand

Improvements to HCC

Improved link time optimization

Improvements to ROCProfiler tool

General bug fixes and implemented versioning APIs

Critical bug fixes

New features and enhancements in ROCm 1.9.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added DPM support to Vega 7nm

Dynamic Power Management feature is enabled on Vega 7nm.

Fix for 'ROCm profiling' that used to fail with a “Version mismatch between HSA runtime and libhsa-runtime-tools64.so.1” error

New features and enhancements in ROCm 1.9.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preview for Vega 7nm
Enables developer preview support for Vega 7nm

System Management Interface
Adds support for the ROCm SMI (System Management Interface) library, which provides monitoring and management capabilities for AMD GPUs.

Improvements to HIP/HCC
Support for gfx906

Added deprecation warning for C++AMP. This will be the last version of HCC supporting C++AMP.

Improved optimization for global address space pointers passing into a GPU kernel

Fixed several race conditions in the HCC runtime

Performance tuning to the unpinned copy engine

Several codegen enhancement fixes in the compiler backend

Preview for rocprof Profiling Tool

Developer preview (alpha) of profiling tool rocProfiler. It includes a command-line front-end, rpl_run.sh, which enables:

Cmd-line tool for dumping public per kernel perf-counters/metrics and kernel timestamps

Input file with counters list and kernels selecting parameters

Multiple counters groups and app runs supported

Output results in CSV format

The tool can be installed from the rocprofiler-dev package. It will be installed into: /opt/rocm/bin/rpl_run.sh

Preview for rocr Debug Agent rocr_debug_agent

The ROCr Debug Agent is a library that can be loaded by ROCm Platform Runtime to provide the following functionality:

Print the state for wavefronts that report memory violation or upon executing a "s_trap 2" instruction.
Allows SIGINT (ctrl c) or SIGTERM (kill -15) to print wavefront state of aborted GPU dispatches.
It is enabled on Vega10 GPUs on ROCm1.9.
The ROCm1.9 release will install the ROCr Debug Agent library at /opt/rocm/lib/librocr_debug_agent64.so

New distribution support
Binary package support for Ubuntu 18.04
ROCm 1.9 is ABI compatible with KFD in upstream Linux kernels.
Upstream Linux kernels support the following GPUs in these releases: 4.17: Fiji, Polaris 10, Polaris 11 4.18: Fiji, Polaris 10, Polaris 11, Vega10

Some ROCm features are not available in the upstream KFD:

More system memory available to ROCm applications
Interoperability between graphics and compute
RDMA
IPC
To try ROCm with an upstream kernel, install ROCm as normal, but do not install the rock-dkms package. Also add a udev rule to control /dev/kfd permissions:

    echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="video"' | sudo tee /etc/udev/rules.d/70-kfd.rules
    
New features as of ROCm 1.8.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm 1.8.3 is a minor update meant to fix compatibility issues on Ubuntu releases running kernel 4.15.0-33

New features as of ROCm 1.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DKMS driver installation

Debian packages are provided for DKMS on Ubuntu

RPM packages are provided for CentOS/RHEL 7.4 and 7.5 support

See the ROCT-Thunk-Interface and ROCK-Kernel-Driver for additional documentation on driver setup

New distribution support

Binary package support for Ubuntu 16.04 and 18.04

Binary package support for CentOS 7.4 and 7.5

Binary package support for RHEL 7.4 and 7.5

Improved OpenMPI via UCX support

UCX support for OpenMPI

ROCm RDMA

New Features as of ROCm 1.7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DKMS driver installation

New driver installation uses Dynamic Kernel Module Support (DKMS)

Only amdkfd and amdgpu kernel modules are installed to support AMD hardware

Currently only Debian packages are provided for DKMS (no Fedora suport available)

See the ROCT-Thunk-Interface and ROCK-Kernel-Driver for additional documentation on driver setup

New Features as of ROCm 1.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developer preview of the new OpenCL 1.2 compatible language runtime and compiler

OpenCL 2.0 compatible kernel language support with OpenCL 1.2 compatible runtime

Supports offline ahead of time compilation today; during the Beta phase we will add in-process/in-memory compilation.

Binary Package support for Ubuntu 16.04

Binary Package support for Fedora 24 is not currently available

Dropping binary package support for Ubuntu 14.04, Fedora 23

IPC support
                 
