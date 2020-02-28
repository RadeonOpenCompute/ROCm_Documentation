

# AMD ROCm Release Notes v3.1.0
This page describes the features, fixed issues, and information about downloading and installing the ROCm software.
It also covers known issues and deprecated features in the ROCm v3.1 release.

- [What Is ROCm?](#What-Is-ROCm)
  * [ROCm Components](#ROCm-Components)
  * [Supported Operating Systems](#Supported-Operating-Systems)
  * [Important ROCm Links](#Important-ROCm-Links)
  
- [What\'s New in This Release](#Whats-New-in-This-Release)
  * [Change in ROCm Installation Directory Structure](#Change-in-ROCm-Installation-Directory-Structure)
  * [Reliability, Accessibility, and Serviceability Support for Vega7nm](#Reliability-Accessibility-and-Serviceability-Support-for-Vega7nm)
  * [SLURM Support for AMD GPU](#SLURM-Support-for-AMD-GPU)
  
  
- [Known Issues](#Known-Issues)
	* [MIVision MIGraphX Installation](#MIVision-MIGraphX-Installation)
	* [Using TensorFlow](#Using-TensorFlow)
	* [HIP Compiler Dependency Issue](#HIP-Compiler-Dependency-Issue)
	* [Error Running ROC Profiler](#Error-Running-ROC-Profiler)
    	
 
  
- [Deploying ROCm](#Deploying-ROCm)
  * [Ubuntu](#Ubuntu)
  * [CentOS RHEL v7](#CentOS-RHEL-v7)
  * [SLES 15 Service Pack 1](#SLES-15-Service-Pack-1)


- [Getting the ROCm Source Code](#Getting-the-ROCm-Source-Code)
- [Hardware and Software Support](#Hardware-and-Software-Support)
- [Machine Learning and High Performance Computing Software Stack for AMD GPU](#Machine-Learning-and-High-Performance-Computing-Software-Stack-for-AMD-GPU)
  * [ROCm Binary Package Structure](#ROCm-Binary-Package-Structure)
  * [ROCm Platform Packages](#ROCm-Platform-Packages)
  

## What Is ROCm?
ROCm is designed to be a universal platform for gpu-accelerated computing. This modular design allows hardware vendors to build drivers that support the ROCm framework. ROCm is also designed to integrate multiple programming languages and makes it easy to add support for other languages. 

Note: You can also clone the source code for individual ROCm components from the GitHub repositories.


### ROCm Components
The following components for the ROCm platform are released and available for the v3.1
release:

• Drivers

• Tools

• Libraries

• Source Code

You can access the latest supported version of drivers, tools, libraries, and source code for the ROCm platform at the following location:
https://github.com/RadeonOpenCompute/ROCm

### Supported Operating Systems
The ROCm v3.1.x platform is designed to support the following operating systems:


* Ubuntu 16.04.6(Kernel 4.15) and 18.04.3(Kernel 5.3)

* CentOS v7.7 (Using devtoolset-7 runtime support)

* RHEL v7.7 (Using devtoolset-7 runtime support)

* SLES 15 SP1 


For details about deploying the ROCm v3.1.0.x on these operating systems, see the Deploying ROCm section later in the document.

### Important ROCm Links

Access the following links for more information on:

* ROCm documentation, see 
https://rocm-documentation.readthedocs.io/en/latest/index.html

* ROCm binary structure, see
https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#rocm-binary-package-structure

* Common ROCm installation issues, see
https://rocm.github.io/install_issues.html

* Instructions to install PyTorch after ROCm is installed – https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#pytorch

Note: These instructions reference the rocm/pytorch:rocm3.0_ubuntu16.04_py2.7_pytorch image. However, you can substitute the Ubuntu 18.04 image listed at https://hub.docker.com/r/rocm/pytorch/tags


## What\'s New in This Release

### Change in ROCm Installation Directory Structure
A fresh installation of the ROCm toolkit installs the packages in the */opt/rocm-\<version>* folder. 
	
Previously, ROCm toolkit packages were installed in the */opt/rocm* folder. 

![ScreenShot](Versionchange1.png)

## Reliability, Accessibility, and Serviceability Support for Vega7nm
The Reliability, Accessibility, and Serviceability (RAS) support for Vega7nm is now available. The support includes:

* UMC RAS – HBM ECC (uncorrectable error injection), page retirement, RAS recovery via GPU (BACO) reset
* GFX RAS – GFX, MMHUB ECC (uncorrectable error injection), RAS recovery via GPU (BACO) reset
* PCIE RAS – PCIE_BIF ECC (uncorrectable error injection), RAS recovery via GPU (BACO) reset

## SLURM Support for AMD GPU
SLURM (Simple Linux Utility for Resource Management) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. The latest version 20.02.0 of SLURM includes AMD plugins that enable SLURM to detect and configure AMD GPUs automatically.  It also collects and reports the energy consumption of AMD GPUs.



## Known Issues 

### MIVision MIGraphX Installation

* Install and use the latest version of the MIVision/MIGraphX code.  
* Ensure the /opt/rocm symbolic link for the new version of ROCm is present and points to the right version of the ROCm toolkit. 

### Using TensorFlow
The TensorFlow build system requires the following additional changes to support the new installation path:

* Ensure the /opt/rocm symbolic link is preset and points to the right version of the ROCm toolkit.
* Modify the build configure file to include the header files from the respective ROCm version-specific folder

### HIP Compiler Dependency Issue
If the HIP compiler has a dependency on /opt/rocm, use the following workaround: 

* Ensure the /opt/rocm symbolic link points to the right version of the ROCm software
* Use the ROCM_PATH environment variable that points to the version of the ROCm software installed on the system. 
* Use the rocm-dkms package to install required ROCm components.	

### Error Running ROC Profiler
**Issue:** Running ROC profiler results in the following error -
“: hip / hsa trace due to "ImportError: No module named sqlite3" error”

**Workaround:** Export the Python version before running ROC profiler: 

	*export ROCP_PYTHON_VERSION=<python version>*
	*ex: export ROCP_PYTHON_VERSION=python3*
