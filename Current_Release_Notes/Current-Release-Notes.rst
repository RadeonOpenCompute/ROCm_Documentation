.. image:: /Current_Release_Notes/amdblack.jpg

|

=============================================================
AMD Radeon Open Compute platforM (ROCm) Release Notes v3.1.0
=============================================================
April 1st, 2020

What Is ROCm?
==============

ROCm is designed to be a universal platform for gpu-accelerated computing. This modular design allows hardware vendors to build drivers that support the ROCm framework. ROCm is also designed to integrate multiple programming languages and makes it easy to add support for other languages. 

Note: You can also clone the source code for individual ROCm components from the GitHub repositories.

ROCm Components
~~~~~~~~~~~~~~~~

The following components for the ROCm platform are released and available for the v3.1
release:

• Drivers

• Tools

• Libraries

• Source Code

You can access the latest supported version of drivers, tools, libraries, and source code for the ROCm platform at the following location:
https://github.com/RadeonOpenCompute/ROCm


Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm v3.1.x platform is designed to support the following operating systems:


* Ubuntu 16.04.6(Kernel 4.15) and 18.04.3(Kernel 5.3)

* CentOS v7.7 (Using devtoolset-7 runtime support)

* RHEL v7.7 (Using devtoolset-7 runtime support)

* SLES 15 SP1 

Important ROCm Links
~~~~~~~~~~~~~~~~~~~~~~~

Access the following links for more information on:

* ROCm documentation, see 
https://rocm-documentation.readthedocs.io/en/latest/index.html

* ROCm binary structure, see
https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#rocm-binary-package-structure

* Common ROCm installation issues, see
https://rocm.github.io/install_issues.html

* Instructions to install PyTorch after ROCm is installed – https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#pytorch

Note: These instructions reference the rocm/pytorch:rocm3.0_ubuntu16.04_py2.7_pytorch image. However, you can substitute the Ubuntu 18.04 image listed at https://hub.docker.com/r/rocm/pytorch/tags



What\'s New in This Release
===========================

**Change in ROCm Installation Directory Structure**
###################################################

A fresh installation of the ROCm toolkit installs the packages in the */opt/rocm-\<version>* folder. 
	
Previously, ROCm toolkit packages were installed in the */opt/rocm* folder. 

.. image:: /Current_Release_Notes/Versionchange1.png


**Reliability, Accessibility, and Serviceability Support for Vega7nm**
######################################################################

The Reliability, Accessibility, and Serviceability (RAS) support for Vega7nm is now available. The support includes:

* UMC RAS – HBM ECC (uncorrectable error injection), page retirement, RAS recovery via GPU (BACO) reset
* GFX RAS – GFX, MMHUB ECC (uncorrectable error injection), RAS recovery via GPU (BACO) reset
* PCIE RAS – PCIE_BIF ECC (uncorrectable error injection), RAS recovery via GPU (BACO) reset



**SLURM Support for AMD GPU**
##############################

SLURM (Simple Linux Utility for Resource Management) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. The latest version 20.02.0 of SLURM includes AMD plugins that enable SLURM to detect and configure AMD GPUs automatically.  It also collects and reports the energy consumption of AMD GPUs.


The following webpage describes the features, fixed issues, and information about downloading and installing the ROCm software.
It also covers known issues and deprecated features in the ROCm v3.1 release.

https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

Refer the QuickStart Installation Guide.pdf for ROCm installation instructions on the following platforms:

* Ubuntu
* CentOS/RHEL
* SLES 15 Service Pack 1


