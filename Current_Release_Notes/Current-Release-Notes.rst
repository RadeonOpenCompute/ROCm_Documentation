
=============================================================
AMD Radeon Open Compute platforM (ROCm) Release Notes v3.1.0
=============================================================
February 28th, 2020

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

The GitHub link also consists of ROCm installation instructions for all platforms.


