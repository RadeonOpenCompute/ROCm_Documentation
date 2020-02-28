

AMD ROCm Release Notes v3.1.0
==============================
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




