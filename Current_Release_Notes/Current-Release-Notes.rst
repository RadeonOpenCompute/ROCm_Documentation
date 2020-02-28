
=============================================================
AMD Radeon Open Compute platforM (ROCm) Release Notes v3.1.0
=============================================================

What\'s New in This Release
===========================

**Change in ROCm Installation Directory Structure**
###################################################

A fresh installation of the ROCm toolkit installs the packages in the */opt/rocm-\<version>* folder. 
	
Previously, ROCm toolkit packages were installed in the */opt/rocm* folder. 



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


New features and enhancements in ROCm 3.0
===========================================
**Support for CentOS RHEL v7.7**
Support is extended for CentOS/RHEL v7.7 in the ROCm v3.0 release. For more information about the CentOS/RHEL v7.7 release, see:
CentOS/RHEL

**Initial distribution of AOMP 0.7-5 in ROCm v3.0**
The code base for this release of AOMP is the Clang/LLVM 9.0 sources as of October 8th, 2019. The LLVM-project branch used to build this release is AOMP-191008. It is now locked. With this release, an artifact tarball of the entire source tree is created. This tree includes a Makefile in the root directory used to build AOMP from the release tarball. You can use Spack to build AOMP from this source tarball or build manually without Spack.
For more information about AOMP 0.7-5, see: AOMP

**Fast Fourier Transform Updates**
The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform. Fast Fourier transforms are used in signal processing, image processing, and many other areas. The following real FFT performance change is made in the ROCm v3.0 release:

•	Implement efficient real/complex 2D transforms for even lengths.

Other improvements:

•	More 2D test coverage sizes.

•	Fix buffer allocation error for large 1D transforms.

•	C++ compatibility improvements.

**MemCopy Enhancement for rocProf**
In the v3.0 release, the rocProf tool is enhanced with an additional capability to dump asynchronous GPU memcopy information into a .csv file. You can use the '-hsa-trace' option to create the results_mcopy.csv file. Future enhancements will include column labels.




