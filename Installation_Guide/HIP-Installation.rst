
.. image:: amdblack.jpg
|
=======================
HIP Installation v4.5
=======================

HIP can be easily installed using the pre-built binary packages with the package manager for your platform.


HIP Prerequisites
====================

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with NVCC installed.



AMD Platform
===============

::

   sudo apt install mesa-common-dev
   sudo apt install clang
   sudo apt install comgr
   sudo apt-get -y install rocm-dkms
   

HIP-Clang is the compiler for compiling HIP programs on AMD platform.

HIP-Clang can be built manually:

::

   	git clone -b rocm-4.5.x https://github.com/RadeonOpenCompute/llvm-project.git
	cd llvm-project
	mkdir -p build && cd build
	cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 - 
	DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" - DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm
	make -j
	sudo make install


::

The ROCm device library can be manually built as following,

::

  	export PATH=/opt/rocm/llvm/bin:$PATH
	git clone -b rocm-4.5.x https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git
	cd ROCm-Device-Libs
	mkdir -p build && cd build
	CC=clang CXX=clang++ cmake -DLLVM_DIR=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_WERROR=1 -DLLVM_ENABLE_ASSERTIONS=1 -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
	make -j
	sudo make install

::


NVIDIA Platform
=================

HIP-nvcc is the compiler for HIP program compilation on NVIDIA platform.

-  Add the ROCm package server to your system as per the OS-specific
   guide available
   `here <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`__.
   
-  Install the "hip-runtime-nvidia" and "hip-devel" package. This will install CUDA SDK and the HIP porting layer.

::

 		apt-get install hip-runtime-nvidia hip-devel
		

-  Default paths and environment variables:

   -  By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable).
      
   -  By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
      
   -  Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.


Building HIP from Source
========================


Get HIP source code
*********************

::

		git clone -b rocm-4.5.x https://github.com/ROCm-Developer-Tools/hipamd.git
		git clone -b rocm-4.5.x https://github.com/ROCm-Developer-Tools/hip.git
		git clone -b rocm-4.5.x https://github.com/ROCm-Developer-Tools/ROCclr.git
		git clone -b rocm-4.5.x https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
		


Set the environment variables
******************************

::

		export HIPAMD_DIR="$(readlink -f hipamd)"
		export HIP_DIR="$(readlink -f hip)"
		export ROCclr_DIR="$(readlink -f ROCclr)"
		export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
		

ROCclr is defined on AMD platform that HIP use Radeon Open Compute Common Language Runtime (ROCclr), which is a virtual device interface that HIP runtimes interact with different backends. 

See https://github.com/ROCm-Developer-Tools/ROCclr

HIPAMD repository provides implementation specifically for AMD platform. See https://github.com/ROCm-Developer-Tools/hipamd


Build HIP
*********

::

		cd "$HIPAMD_DIR"
		mkdir -p build; cd build
		cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install ..
		make -j$(nproc)
		sudo make install

**Note**: If you don't specify CMAKE_INSTALL_PREFIX, hip runtime will be installed to "/opt/rocm/hip". By default, release version of AMDHIP is built.


Default paths and environment variables
******************************************

- By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable).

- By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).

- By default HIP looks for clang in /opt/rocm/llvm/bin (can be overridden by setting HIP_CLANG_PATH environment variable)

- By default HIP looks for device library in /opt/rocm/lib (can be overridden by setting DEVICE_LIB_PATH environment variable)

- Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools

- Optionally, set HIPCC_VERBOSE=7 to output the command line for compilation

After installation, make sure HIP_PATH is pointed to /where/to/install/hip


Verify your installation
**************************

Run hipconfig (instructions below assume default installation path):

::

		/opt/rocm/bin/hipconfig --full

Compile and run the square sample. You can access the square sample at,

https://github.com/ROCm-Developer-Tools/HIP/tree/main/samples/0_Intro/square
