
.. image:: amdblack.jpg
|

====================
HIP Installation 
====================

HIP can be easily installed using the pre-built binary packages with the package manager for your platform.


Installing pre-built packages
=============================

HIP can be easily installed using pre-built binary packages using the package manager for your platform.

HIP Prerequisites
==================

HIP code can be developed either on AMD ROCm platform using HIP-Clang compiler, or a CUDA platform with NVCC installed.


AMD Platform
=============

::

   sudo apt install mesa-common-dev
   sudo apt install clang
   sudo apt install comgr
   sudo apt-get -y install rocm-dkms

HIP-Clang is the compiler for compiling HIP programs on AMD platform.

HIP-Clang can be built manually:

::

   	git clone -b roc-4.0-x  https://github.com/RadeonOpenCompute/llvm-project.git
	cd llvm-project
	mkdir -p build && cd build
	cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -					DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm
	make -j
	sudo make install

::

The ROCm device library can be manually built as following,

::

  	export PATH=/opt/rocm/llvm/bin:$PATH
	git clone -b roc-4.0-x  https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git
	cd ROCm-Device-Libs
	mkdir -p build && cd build
	CC=clang CXX=clang++ cmake -DLLVM_DIR=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_WERROR=1 -DLLVM_ENABLE_ASSERTIONS=1 -	DCMAKE_INSTALL_PREFIX=/opt/rocm ..
	make -j
	sudo make install

::


NVIDIA Platform
================

HIP-nvcc is the compiler for HIP program compilation on NVIDIA platform.

-  Add the ROCm package server to your system as per the OS-specific
   guide available
   `here <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`__.
-  Install the 'hip-nvcc' package. This will install CUDA SDK and the HIP porting layer.

::

   apt-get install hip-nvcc

-  Default paths and environment variables:

   -  By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable).
      
   -  By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
      
   -  Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.


Building HIP from Source
========================

Build ROCclr
=============

ROCclr is defined on AMD platform that HIP use Radeon Open Compute
Common Language Runtime (ROCclr), which is a virtual device interface
that HIP runtimes interact with different backends. 

See https://github.com/ROCm-Developer-Tools/ROCclr

::

   	git clone -b rocm-4.0.x https://github.com/ROCm-Developer-Tools/ROCclr.git
	export ROCclr_DIR="$(readlink -f ROCclr)"
	git clone -b rocm-4.0.x https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
	export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
	cd "$ROCclr_DIR"
	mkdir -p build;cd build
	cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr ..
	make -j
	sudo make install


::

Build HIP
===========

::

   git clone -b rocm-4.0.x https://github.com/ROCm-Developer-Tools/HIP.git
	export HIP_DIR="$(readlink -f HIP)"
	cd "$HIP_DIR"
	mkdir -p build; cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DHIP_COMPILER=clang -DHIP_PLATFORM=rocclr -DCMAKE_PREFIX_PATH="$ROCclr_DIR/build;/opt/rocm/" -	DCMAKE_INSTALL_PREFIX=</where/to/install/hip> ..
	make -j
	sudo make install

::


Default paths and environment variables
=========================================

-  By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by
   setting HSA_PATH environment variable).
-  By default HIP is installed into /opt/rocm/hip (can be overridden by
   setting HIP_PATH environment variable).
-  By default HIP looks for clang in /opt/rocm/llvm/bin (can be
   overridden by setting HIP_CLANG_PATH environment variable)
-  By default HIP looks for device library in /opt/rocm/lib (can be
   overridden by setting DEVICE_LIB_PATH environment variable).
-  Optionally, consider adding /opt/rocm/bin to your PATH to make it
   easier to use the tools.
-  Optionally, set HIPCC_VERBOSE=7 to output the command line for
   compilation.

After installation, make sure HIP_PATH is pointed to */where/to/install/hip*


Verify your installation
========================

Run hipconfig (instructions below assume default installation path) :

.. code:: shell

   /opt/rocm/bin/hipconfig --full

Compile and run the `square
sample <https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square>`__.
