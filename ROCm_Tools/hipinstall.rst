.. _hipinstall:

====================
build from source
====================

Installing pre-built packages
********************************
HIP can be easily installed using pre-built binary packages using the package manager for your platform.

Prerequisites
****************
HIP code can be developed either on AMD ROCm platform using hcc compiler, or a CUDA platform with nvcc installed:

AMD-hcc
**********
 * Install the `rocm <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-    	guide-ubuntu>`_ packages. ROCm will install all of the necessary components, including the kernel driver, runtime software, HCC   	compiler, and HIP.

 * Default paths and environment variables:

     * By default HIP looks for hcc in /opt/rocm/hcc (can be overridden by setting HCC_HOME environment variable) 
     * By default HIP looks for HSA in /opt/rocm/hsa (can be overridden by setting HSA_PATH environment variable)
     * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
     * Optionally, consider adding /opt/rocm/bin to your PATH to make it easier to use the tools.

NVIDIA-nvcc
************
 * Configure the additional package server as described `here <https://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/>`_.
 * Install the "hip_nvcc" package. This will install CUDA SDK and the HIP porting layer.

:: 

  apt-get install hip_nvcc

 * Default paths and environment variables:
     * By default HIP looks for CUDA SDK in /usr/local/cuda (can be overriden by setting CUDA_PATH env variable)
     * By default HIP is installed into /opt/rocm/hip (can be overridden by setting HIP_PATH environment variable).
     * Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.

Verify your installation
**************************
Run hipconfig (instructions below assume default installation path) :

::

  /opt/rocm/bin/hipconfig --full
  Compile and run the square sample.

Building HIP from source
****************************
HIP source code is available and the project can be built from source on the HCC platform.

  1. Follow the above steps to install and validate the binary packages.
  2. Download HIP source code (from the GitHub repot.)
  3. Install HIP build-time dependencies using sudo apt-get install libelf-dev.
  4. Build and install HIP (This is the simple version assuming default paths ; see below for additional options.)

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
   * By default cmake installs HIP to /opt/rocm/hip (can be overridden by setting -DCMAKE_INSTALL_PREFIX=/where/to/install/hip in the 	   cmake step).*

Here's a richer command-line that overrides the default paths:

::

  cd HIP
  mkdir build
  cd build
  cmake -DHSA_PATH=/path/to/hsa -DHCC_HOME=/path/to/hcc -DCMAKE_INSTALL_PREFIX=/where/to/install/hip -DCMAKE_BUILD_TYPE=Release ..
  make
  make install

After installation, make sure HIP_PATH is pointed to /where/to/install/hip.
HCC Options

Using HIP with the AMD Native-GCN compiler.
*********************************************
AMD recently released a direct-to-GCN-ISA target. This compiler generates GCN ISA directly from LLVM, without going through an intermediate compiler IR such as HSAIL or PTX. The native GCN target is included with upstream LLVM, and has also been integrated with HCC compiler and can be used to compiler HIP programs for AMD. Binary packages for the direct-to-isa package are included with the rocm package. Alternatively, this sections describes how to build it from source:

 1. Install the ROCm packages as described above.
 2. Follow the instructions here :ref:`HCC-Native-GCN-ISA`
     * In the make step for HCC, we recommend setting -DCMAKE_INSTALL_PREFIX.
     * Set HCC_HOME environment variable before compiling HIP program to point to the native compiler:
    export HCC_HOME=/path/to/native/hcc
