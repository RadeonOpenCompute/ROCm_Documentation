.. image:: /Installation_Guide/amdblack.jpg

==============================================
AMD ROCm QuickStart Installation Guide v3.1.0
==============================================

-  `Deploying ROCm`_

   -  `Ubuntu`_
   -  `Centos RHEL v7.7`_
   -  `SLES 15 Service Pack 1`_

-  `ROCm Installation Known Issues and Workarounds`_

   
-  `Getting the ROCm Source Code`_


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

    wget -q0 –http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | 

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


.. _Machine Learning and High Performance Computing Software Stack for AMD GPU:

===================================================================================
Machine Learning and High Performance Computing Software Stack for AMD GPU v3.1.0
===================================================================================

For **AMD ROCm v3.1** Machine Learning and High Performance Computing Software Stack, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

|

Machine Learning and High Performance Computing Software Stack for AMD GPU v3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ROCm Version 3.0**

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

For AMD ROCm v3.1 Machine Learning and High Performance Computing Software Stack, see

https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

The following platform packages are for ROCm v3.0:

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
                 
