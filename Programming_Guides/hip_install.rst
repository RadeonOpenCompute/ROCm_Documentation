# ROCm Documentation has moved to docs.amd.com

.. meta::
   :http-equiv=Refresh: 0; url='https://docs.amd.com'
.. _HIP-IN:

Installing pre-built packages
###############################

HIP can be easily installed using pre-built binary packages using the package manager for your platform.

Prerequisites
###############

HIP code can be developed either on AMD ROCm platform using hcc or clang compiler, or a CUDA platform with nvcc installed:

AMD-hcc
#########

    * Add the ROCm package server to your system as per the OS-specific guide available here.
    * Install the "hip_hcc" package. This will install HCC and the HIP porting layer.

::

  apt-get install hip_hcc

    * Default paths and environment variables:
        * By default HIP looks for hcc in /opt/rocm/hcc (can be overridden by setting HCC_HOME environment variable)
        * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable)
        * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
        * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.

HIP-clang
###########

    * Using clang to compile HIP program for AMD GPU is under development. Users need to build LLVM, clang, lld, ROCm device library, and HIP from source.

    * Install the `rocm <http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/>`_ packages. ROCm will install some of the necessary components, including the kernel driver, HSA runtime, etc.

    * Build LLVM/clang/lld by using the following repository and branch and following the general LLVM/clang build procedure. It is recommended to use -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm with cmake so that LLVM/clang/lld are installed to the default path expected by hipcc.
        * `LLVM: <https://github.com/RadeonOpenCompute/llvm.git>`_ amd-common branch
        * `clang: <https://github.com/RadeonOpenCompute/clang>`_ amd-common branch
        * `lld: <https://github.com/RadeonOpenCompute/lld>`_ amd-common branch

    * Build Rocm device library
        * `Checkout <https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git>`_ master branch and build it with clang built from the last step.

    * Build HIP
        * `Checkout <https://github.com/ROCm-Developer-Tools/HIP.git>`_ master branch and build it with HCC installed with ROCm packages. Please use -DHIP_COMPILER=clang with cmake to enable hip-clang.

    * Default paths and environment variables:
        * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable)
        * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
        * By default HIP looks for clang in /opt/rocm/llvm/bin (can be overridden by setting HIP_CLANG_PATH environment variable)
        * By default HIP looks for device library in /opt/rocm/lib (can be overriden by setting DEVICE_LIB_PATH environment variable).
        * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.
        * Optionally, set HIPCC_VERBOSE=7 to output the command line for compilation to make sure clang is used instead of hcc.

NVIDIA-nvcc
##############

    * Add the ROCm package server to your system as per the OS-specific guide available here.
    * Install the "hip_nvcc" package. This will install CUDA SDK and the HIP porting layer.

::

  apt-get install hip_nvcc


    * Default paths and environment variables:
        * By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable)
        * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
        * Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.

Verify your installation
###########################

Run hipconfig (instructions below assume default installation path) :

::

  /opt/rocm/bin/hipconfig --full

Compile and run the `square sample <https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square>`_.

Building HIP from source
##########################

HIP source code is available and the project can be built from source on the HCC platform.

    * Follow the above steps to install and validate the binary packages.
    * Download HIP source code
    * Install HIP build-time dependencies using sudo apt-get install libelf-dev.
    * Build and install HIP (This is the simple version assuming default paths ; see below for additional options.)

By default, HIP uses HCC to compile programs. To use HIP-Clang, add -DHIP_COMPILER=clang to cmake command line.

::

  cd HIP
  mkdir build
  cd build
  cmake .. 
  make
  make install

    * Default paths:
        * By default cmake looks for hcc in /opt/rocm/hcc (can be overridden by setting -DHCC_HOME=/path/to/hcc in the cmake step).*
        * By default cmake looks for HSA in /opt/rocm/hsa (can be overridden by setting -DHSA_PATH=/path/to/hsa in the cmake step).*
        * By default cmake installs HIP to /opt/rocm/hip (can be overridden by setting -DCMAKE_INSTALL_PREFIX=/where/to/install/hip in the cmake step).*

Here's a richer command-line that overrides the default paths:

::

  cd HIP
  mkdir build
  cd build  
  cmake -DHSA_PATH=/path/to/hsa -DHCC_HOME=/path/to/hcc -DCMAKE_INSTALL_PREFIX=/where/to/install/hip -DCMAKE_BUILD_TYPE=Release ..
  make
  make install

    * After installation, make sure HIP_PATH is pointed to /where/to/install/hip.
