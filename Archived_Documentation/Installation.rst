
.. image:: /Installation_Guide/amdblack.jpg
|
==============================================
Installation Guide 
==============================================


.. contents::


Overview of ROCm Installation Methods
--------------------------------------

In addition to the installation method using the native Package Manager, AMD ROCm v4.5 introduces new methods to install ROCm. With this release, the ROCm installation uses the amdgpu-install and amdgpu-uninstall scripts.  

The amdgpu-install script streamlines the installation process by:

- Abstracting the distribution-specific package installation logic

- Performing the repository set-up

- Allowing a user to specify the use case and automating the installation of all the required packages

- Performing post-install checks to verify whether the installation was completed successfully 

- Installing the uninstallation script

The amdgpu-uninstall script allows the removal of the entire ROCm stack by using a single command.

Some of the ROCm-specific use cases that the installer currently supports are: 


- OpenCL (ROCr/KFD based) runtime

- HIP runtimes

- ROCm libraries and applications

- ROCm Compiler and device libraries

- ROCr runtime and thunk


For more information, refer to the Installation Methods section in this guide.



About This Document
====================

This document is intended for users familiar with the Linux environments and discusses the installation/uninstallation of ROCm programming models on the various flavors of Linux. 


This document also refers to Radeon™ Software for Linux® as AMDGPU stack, including the kernel-mode driver amdgpu-dkms.


The guide provides the installation instructions for the following:


- ROCm Installation

- Heterogeneous-Computing Interface for Portability (HIP) SDK

- OPENCL ™ SDK

- Kernel Mode Driver



System Requirements
======================


The system requirements for the ROCm v4.5 installation are as follows:

.. image:: SuppEnv.png
   :alt: Screenshot     

 
 
**NOTE**: Installing ROCm on Linux will require superuser privileges. For systems that have enabled sudo packages, ensure you use the sudo prefix for all required commands.
 
 
 

Prerequisite Actions
---------------------
 

You must perform the following steps before installing ROCm programming models and check if the system meets all of the requirements to proceed with the installation.
 
- Confirm the system has a supported Linux distribution version

- Confirm the system has a ROCm-capable GPU

- Confirm the system has standard compilers and tools installed



Confirm You Have a Supported Linux Distribution Version
=========================================================


The ROCm installation is supported only on specific Linux distributions and their kernel versions. 

**NOTE**: The ROCm installation is not supported on 32-bit operating systems.


How to Check Linux Distribution and Kernel Versions on Your System
*******************************************************************


Linux Distribution Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure you obtain the distribution information of the system by using the following command on your system from the Command Line Interface (CLI),

:: 
               

            $ uname -m && cat /etc/*release
            






For example, running the command above on an Ubuntu system results in the following output: 
 
::
 
            x86_64
            DISTRIB_ID=Ubuntu
            DISTRIB_RELEASE=18.04
            DISTRIB_CODENAME=bionic
            DISTRIB_DESCRIPTION="Ubuntu 18.04.5 LTS"
         



Kernel Information
^^^^^^^^^^^^^^^^^^^

Type the following command to check the kernel version of your Linux system.

::


               $ uname -srmv




The output of the command above lists the kernel version in the following format: 

::
           
           
            Linux 5.4.0-77-generic #86~18.04.5-Ubuntu SMP Fri Jun 18 01:23:22 UTC 2021 x86_64



OS and Kernel Version Match
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Confirm that the obtained Linux distribution and kernel versions match with System Requirements.


Confirm You Have a ROCm-Capable GPU
=====================================

The ROCm platform is designed to support the following list of GPUs: 


.. image:: ROCmProgMod.png
   :alt: Screenshot 
   
   
How to Verify Your System Has a ROCm-Capable GPU
**************************************************

To verify that your system has a ROCm-capable GPU, enter the following command from the Command Line Interface (CLI):

::

               $ sudo lshw -class display
               The command displays the details of detected GPUs on the system in the following format:
               *-display
               description: VGA compatible controller
               product: Vega 20
               vendor: Advanced Micro Devices, Inc. [AMD/ATI]
               physical id: 0
               bus info: pci@0000:43:00.
               version: c1
               width: 64 bits
                      clock: 33MHz
                      capabilities: vga_controller bus_master cap_list rom
                      configuration: driver=amdgpu latency=0
                      resources: irq:66 memory:80000000-8fffffff memory:90000000-901fffff ioport:2000(size=256) memory:9f600000-9f67ffff memory:c0000-dffff
                      
                      

**NOTE**: Verify from the output that the product field value matches the supported GPU Architecture in the table above.
      
      

Confirm the System Has Compiler and Tools Installed
======================================================

You must install and configure Devtoolset-7 to use RHEL/CentOS 7.9


How to Install and Configure Devtoolset-7
*******************************************

Refer to the RHEL/CentOS Environment section for more information on the steps necessary for installing and setting up Devtoolset-7. 


Meta-packages in ROCm Programming Models 
------------------------------------------

This section provides information about the required meta-packages for the following AMD ROCm™ programming models:

- Heterogeneous-Computing Interface for Portability (HIP) 

- OpenCL™


ROCm Package Naming Conventions
================================

A meta-package is a grouping of related packages and dependencies used to support a specific use-case, for example, running HIP applications. All meta-packages exist in both versioned and non-versioned forms.

- Non-versioned packages – For a single installation of the latest version of ROCm

- Versioned packages – For multiple installations of ROCm


.. image:: PackName.png
   :alt: Screenshot 


The image above demonstrates the single and multi-version ROCm packages' naming structure, including examples for various Linux distributions.


Components of ROCm Programming Models
=======================================

The following image demonstrates the high-level layered architecture of ROCm programming models and their meta-packages. All meta-packages are a combination of required packages and libraries. For example, 

- rocm-hip-runtime is used to deploy on supported machines to execute HIP applications. 

- rocm-hip-sdk contains runtime components to deploy and execute HIP applications and tools to develop the applications. 

.. image:: MetaPack.png
   :alt: Screenshot 
   
   
**NOTE**: rocm-llvm is a single package that installs the required ROCm compiler files.


.. image:: MetaPackTable.png
   :alt: Screenshot 
   
   
Packages in ROCm Programming Models
======================================

This section discusses the available meta-packages and their packages. In a ROCm programming model, packages refer to a collection of scripts, libraries, text files, a manifest, license, and other associated files that enable you to install a meta-package. 

The following image visualizes the meta-packages and their associated packages in a ROCm programming model.

.. image:: AssoPack.png
   :alt: Screenshot 
   
**NOTE**: The image above is for informational purposes only as the individual packages in a meta-package are subject to change. Users should install meta-packages, and not individual packages, to avoid conflicts.   


Installation Methods
----------------------

You may use the following installation methods to install ROCm:

- Installer Script Method

- Package Manager Method


Installer Script Method
==========================

The Installer script method automates the installation process for the AMDGPU and ROCm stack. The Installer script handles the complete installation process for ROCm, including setting up the repository, cleaning the system, updating and installing the desired drivers and meta-packages. With this approach, the system has more control over the ROCm installation process. Thus, users who are less familiar with the Linux standard commands can choose this method for ROCm installation.

For a fresh AMDGPU and ROCm installation using the Installer script method on Linux distribution, you must:

- Meet Prerequisites - Ensure the Prerequisite Actions are met before downloading and installing the installer using the Installer Script method.

- Download and Install the Installer – Ensure you download and install the installer script from the recommended URL. Note, the installer package is updated periodically to resolve known issues and add new features. The links for each Linux distribution always point to the latest available build.

- Use the Installer Script on Linux Distributions – Ensure you execute the script for installing use cases.


Downloading and Installing the Installer Script on Ubuntu
**********************************************************

Ubuntu 18.04
^^^^^^^^^^^^^^

Install the wget package on your system using the command below to download the repo installer package:

::

               $ sudo apt-get install wget
               
               

Download and install the repo installer package using the following command:

::

               $ wget http://repo.radeon.com/amdgpu-install/21.40/ubuntu/bionic/amdgpu-install-21.40.40500-1_all.deb
               
               $ sudo apt-get install ./amdgpu-install-21.40.40500-1_all.deb
 

Ubuntu 20.04
^^^^^^^^^^^^^^

Install the wget package on your system using the following command to download the repo installer package.

::

               $ sudo apt-get install wget
               
               
Download and install the repo installer package.

::

               $ wget http://repo.radeon.com/amdgpu-install/21.40/ubuntu/focal/amdgpu-install-21.40.40500-1_all.deb
               
               $ sudo apt-get install ./amdgpu-install-21.40.40500-1_all.deb


Downloading and Installing the Installer Script on RHEL/CentOS
****************************************************************

RHEL/CentOS 7.9
^^^^^^^^^^^^^^^^^

Use the following command to download and install the installer on RHEL/CentOS 7.9.

::

               $ sudo yum install http://repo.radeon.com/amdgpu-install/21.40/rhel/7.9/amdgpu-install-21.40.40500-1.noarch.rpm
               
               
               
RHEL/CentOS 8.4
^^^^^^^^^^^^^^^^

Use the following command to download and install the installer on RHEL/CentOS 8.4.

::

               $ sudo yum install http://repo.radeon.com/amdgpu-install/21.40/rhel/8.4/amdgpu-install-21.40.40500-1.noarch.rpm
               
               

Downloading and Installing the Installer Script on SLES 15
**************************************************************

SLES 15 Service Pack 3
^^^^^^^^^^^^^^^^^^^^^^^^

Use the following command to download and install the installer on SLES 

::

               $ sudo zypper install http://repo.radeon.com/amdgpu-install/21.40/sle/15/amdgpu-install-21.40.40500-1.noarch.rpm
               
 

Using the Installer Script on Linux Distributions 
***************************************************

To install use cases specific to your requirements, use the installer amdgpu-install as follows:

::

               # To install a single use case 
               $ sudo amdgpu-install --usecase=rocm
               
               
::
              
               # To install multiple use-cases 
               $ sudo amdgpu-install --usecase=hiplibsdk,rocm 
               
               
:: 

               # To display a list of available use cases. Note, the list in this section represents only a sample of available use cases for ROCm.
               $ sudo amdgpu-install --list-usecase
               If --usecase option is not present, the default selection is "graphics,opencl,hip"

               Available use cases:
               rocm(for users and developers requiring full ROCm stack)
               - OpenCL (ROCr/KFD based) runtime
               - HIP runtimes
               - Machine learning framework
               - All ROCm libraries and applications
               - ROCm Compiler and device libraries
               - ROCr runtime and thunk
               lrt(for users of applications requiring ROCm runtime)
               - ROCm Compiler and device libraries
               - ROCr runtime and thunk
               opencl(for users of applications requiring OpenCL on Vega or       
               later products)
               - ROCr based OpenCL
               - ROCm Language runtime

               openclsdk (for application developers requiring ROCr based OpenCL)
               - ROCr based OpenCL
               - ROCm Language runtime
               - development and SDK files for ROCr based OpenCL

               hip(for users of HIP runtime on AMD products)
               - HIP runtimes
               hiplibsdk for application developers requiring HIP on AMD products)
               - HIP runtimes
               - ROCm math libraries
               - HIP development libraries



**NOTE**: Adding -y as a parameter to amdgpu-install will skip user prompts (for automation). For example, 

::

               amdgpu-install -y --usecase=rocm
               
              
 

Package Manager Method
========================
 
The Package Manager method involves a manual set up of the repository, which includes cleaning up the system, updating and installing/uninstalling meta-packages using standard commands such as yum, apt, and others respective to the Linux distribution. 

**NOTE**: Users must enter the desired meta-package as the <package-name> in the command. To utilize the newly installed packages, users must install the relevant drivers and restart the system after the installation.

The typical functions of a package manager installation system include:

- Working with file archivers to extract package archives.

- Ensuring the integrity and authenticity of the package by verifying them checksums and digital certificates, respectively.

- Looking up, downloading, installing, or updating existing packages from an online repository. 

- Grouping packages by function to reduce user confusion.

- Managing dependencies to ensure a package is installed with all packages it requires, thus avoiding dependency.

**NOTE**: Users may consult the documentation for their package manager for more details.
              
               

Installing ROCm on Linux Distributions
****************************************

For a fresh ROCm installation using the Package Manager method on a Linux distribution, follow the steps below:

1.	Meet prerequisites - Ensure the Prerequisite Actions are met before the ROCm installation

2.	Install kernel headers and development packages - Ensure kernel headers and development packages are installed on the system

3.	Select the base URLs for AMDGPU and ROCm stack repository – Ensure the base URLs for AMDGPU, and ROCm stack repositories are selected

4.	Add AMDGPU stack repository – Ensure AMDGPU stack repository is added

5.	Install the kernel-mode driver and reboot the system – Ensure the kernel-mode driver is installed and the system is rebooted

6.	Add ROCm stack repository – Ensure the ROCm stack repository is added

7.	Install ROCm meta-packages – Users may install the desired meta-packages

8.	Verify installation for the applicable distributions – Verify if the installation is successful.

**NOTE**: Refer to the sections below for specific commands to install each Linux distribution's ROCm and AMDGPU stack.
