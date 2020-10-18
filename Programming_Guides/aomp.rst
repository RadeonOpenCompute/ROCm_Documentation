

================
OpenMP Support
================


OpenMP-Extras V12.9-0
------------------------

The openmp-extras auxiliary package supports OpenMP within the ROCm compiler, which is on llvm 12, and is independent of the aomp-amdgpu
package. It contains OpenMP specific header files, which are installed in */opt/rocm/llvm/include* as well as runtime libraries, fortran runtime
libraries, and device bitcode files in */opt/rocm/llvm/lib*. The auxiliary package also consists of examples in */opt/rocm/llvm/examples*.

OpenMP-Extras Installation
--------------------------

Openmp-extras is automatically installed as a part of the rocm-dkms or rocm-dev package. Refer to the AMD ROCm Installation Guide at

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


OpenMP-Extras Source Build
--------------------------

For instructions on building OpenMP-Extras from source, refer to `OPENMPEXTRAS_SOURCE_INSTALL.<https://github.com/ROCm-Developer-Tools/aomp/blob/rocm-3.9.x/docs/OPENMPEXTRAS_SOURCE_INSTALL.md>`__

System package dependencies can be found `here. <https://github.com/ROCm-Developer-Tools/aomp/blob/rocm-3.9.0/docs/SOURCEINSTALL.md>`__


**NOTE**: The ROCm compiler, which supports OpenMP for AMDGPU, is located in */opt/rocm/llvm/bin/clang*. The AOMP OpenMP support in ROCm
v3.9 is based on the standalone AOMP v11.9-0, with LLVM v11 as the underlying system. However, the ROCm compiler's OpenMP support is based
on LLVM v12 (upstream).


2. AOMP - V11.9-0
-----------------

Overview
~~~~~~~~

AOMP in ROCm is a scripted build of LLVM 11 and supporting software. It has support for OpenMP target offload on AMD GPUs. The build of AOMP
uses a combination of sources from ROCm along with AOMP specific repos (aomp, flang, aomp-extras, amd-llvm-project).

AOMP Installation
~~~~~~~~~~~~~~~~~

AOMP in ROCm can be installed with the optional aomp-amdgpu package after rocm-dkms is installed.

**NOTE**: The optional AOMP package will reside in */opt/rocm/aomp and clang* can be found in */opt/rocm/aomp/bin/clang*.

**AOMP Optional Package Deprecation** Before the AMD ROCm v3.9 release, the optional AOMP package provided support for OpenMP. While AOMP is
available in this release, the optional package may be deprecated from ROCm in the future. It is recommended you transition to the ROCm
compiler by using openmp-extras or AOMP standalone `releases <https://github.com/ROCM-Developer-Tools/aomp/releases>`__ for
OpenMP support.

AOMP Installation Instructions
==============================

AOMP in ROCm can be installed via your package manager using the aomp-amdgpu package.

AOMP Source Build
-----------------

See `ROCM_AOMP_SOURCE_INSTALL <https://github.com/ROCm-Developer-Tools/aomp/blob/rocm-3.9.x/docs/ROCM_AOMP_SOURCE_INSTALL.md>`__
for instructions on building AOMP from source. 

System package dependencies can be found `here. <https://github.com/ROCm-Developer-Tools/aomp/blob/rocm-3.9.0/docs/SOURCEINSTALL.md>`__



