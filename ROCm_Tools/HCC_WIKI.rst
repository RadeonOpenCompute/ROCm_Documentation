# ROCm Documentation has moved to docs.amd.com

.. meta::
   :http-equiv=Refresh: 0; url='https://docs.amd.com'
.. _HCCwiki:
 
HCC WIKI (Deprecated)
======================

HCC is an Open Source, Optimizing C++ Compiler for Heterogeneous Compute
**************************************************************************

HCC supports heterogeneous offload to AMD APUs and discrete GPUs via HSA enabled runtimes and drivers. It is an ISO compliant C++ 11/14 compiler. It is based on Clang, the LLVM Compiler Infrastructure and the “libc++” C++ standard library.

Accelerator Modes Supported
*****************************

`HC (Heterogeneous Compute) C++ API <https://scchan.github.io/hcc>`_
++++++++++++++++++++++++++++++++++++++++++

Inspired by C++ AMP and C++17, this is the default C++ compute API for the HCC compiler. HC has some important differences from C++ AMP including removing the “restrict” keyword, supporting additional data types in kernels, providing more control over synchronization and data movement, and providing pointer-based memory allocation. It is designed to expose cutting edge compute capabilities on Boltzmann and HSA devices to developers while offering the productivity and usability of C++.

`HIP <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/hip-programming-guide.html>`_
+++++++++++
HIP provides a set of tools and API for converting CUDA applications into a portable C++ API. An application using the HIP API could be compiled by hcc to target AMD GPUs. Please refer to `HIP's <https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hip-programing-guide>`_ repository for more information.

`C++ AMP <http://download.microsoft.com/download/2/2/9/22972859-15C2-4D96-97AE-93344241D56C/CppAMPOpenSpecificationV12.pdf>`_
+++++++++++++++++++++++
**NOTE** The supported for C++AMP is being deprecated. The ROCm 1.9 release is the last release of HCC supporting C++AMP.

Microsoft C++ AMP is a C++ accelerator API with support for GPU offload. This mode is compatible with Version 1.2 of the C++ AMP specification.

`C++ Parallel STL <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3960.pdf>`_
++++++++++++++++++++++++++++++++
HCC provides an initial implementation of the parallel algorithms described in the ISO C++ Extensions for Parallelism, which enables parallel acceleration for certain STL algorithms.


Platform Requirements
++++++++++++++++++++++++
Accelerated applications could be run on Radeon discrete GPUs from the Fiji family (e.g R9 Nano, R9 Fury, R9 Fury X, FirePro S9300 x2, Polaris 10, Polaris 11) and from Vega10 family(e.g. Radeon RX Vega64, RX Vega56, RX Vega FE) paired with an AMD CPU with Zen cores or newer (e.g. Ryzen, Threadripper, Epyc) or an Intel Haswell CPU or newer.

HCC currently only works on Linux and with the open source ROCK kernel driver and the ROCR runtime (see Installation for details). It will not work with the closed source AMD graphics driver.

Compiler Backends
+++++++++++++++++++

This backend compiles GPU kernels into native GCN ISA, which could be directly execute on the GPU hardware. It's being actively developed by the Radeon Technology Group in LLVM.

Installation
++++++++++++++

**Prerequisites**

Before continuing with the installation, please make sure any previously installed hcc compiler has been removed from on your system.

Install `ROCm <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide>`_ and make sure it works correctly.

Ubuntu
++++++++


**Ubuntu 16.04 & 18.04**

Follow the instruction `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide>`_ to setup the ROCm apt repository and install the rocm-dkms or the rocm-dev meta-package

**RHEL 7.4/CentOS 7**

Follow the instruction `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide>`_ to setup the ROCm yum rpm repository and install the rocm-dkms meta-package for CentOS/RHEL 7 Support.

Please follow steps to prepare `devtoolset-7 environment <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#install-and-setup-devtoolset-7>`_ which is needed for compiling HCC from source. This environment only requires to be installed once, but must enter the environment before compiling using command: scl enable devtoolset-7 bash

 Note: CentOS 7 cmake is outdated, will need to use alternate cmake3.

**openSUSE Leap 42.3**

Currently, HCC support for openSUSE is experimental and the compiler has to be built from source.

Building HCC from Source
########################
First, install the build dependencies: 
::
  
  # Ubuntu 16.04 & 18.04
  sudo apt-get install coreutils git cmake make g++  g++-multilib gcc-multilib python \
findutils libelf1 libpci3 file debianutils libunwind-dev pkg-config \
hsa-rocr-dev hsa-ext-rocr-dev hsakmt-roct-dev rocm-utils

::

  # Fedora 24
sudo dnf install coreutils git cmake make gcc-c++ python findutils elfutils-libelf \
pciutils-libs file pth rpm-build libunwind-devel hsa-rocr-dev hsa-ext-rocr-dev \
hsakmt-roct-dev pkgconfig rocm-utils

::

  # CentOS 7
  sudo yum install coreutils git cmake3 make gcc-c++ devtoolset-7-gcc-c++ python findutils \
elfutils-libelf pciutils-libs file pth rpm-build redhat-lsb-core pkgconfig \
hsa-rocr-dev hsa-ext-rocr-dev hsakmt-roct-dev rocm-utils

::

  # openSUSE Leap 42.3
  sudo zypper install coreutils git cmake make gcc-c++ python python-xml findutils elfutils pciutils-devel file rpm-build libunwind-devel pkg-config libpth-devel
   
  # install libc++ from OSB
  sudo zypper addrepo \
  -f http://download.opensuse.org/repositories/devel:/tools:/compiler/openSUSE_Leap_42.3/ devel_tools_compiler
  sudo zypper update
  sudo zypper install libc++-devel


Clone the HCC source tree: 
::
  # automatically fetches all submodules
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

Create a build directory and run cmake to configure the build: 
::
  mkdir build; cd build
  cmake ../hcc

Compile HCC: 
::
  make -j [number of threads]

Install HCC:
::
  sudo make install

Run the unit tests: 
:: 
  make test

Create an installer package (DEB or RPM file)
::
  make package

How to use HCC
##############
Here's a simple `saxpy example <https://gist.github.com/scchan/540d410456e3e2682dbf018d3c179008>`_ written with the hc API.

**Compiling Your First HCC Program**

To compile and link in a single step:
::
 # Assume HCC is installed and added to PATH
 hcc -hc saxpy.cpp -o saxpy

To build with separate compile and link steps:
::
 # Assume HCC is installed and added to PATH
 # Notice the the hcc-config command is between two backticks 
 hcc -hc saxpy.cpp -c -o saxpy.cpp.o
 hcc -hc saxpy.cpp.o -o saxpy

**Compiling for Different GPU Architectures**

By default, HCC would auto-detect all the GPUs available to run on and set the correct GPU architectures. Users could use the --amdgpu-target=<GCN Version> option to compile for a specific architecture and to disable the auto-detection. The following table shows the different versions currently supported by HCC.


============ ================== ==============================================================
GCN Version   GPU/APU Family       Examples of Radeon GPU
       
============ ================== ==============================================================

gfx803        GFX8               R9 Fury, R9 Fury X, R9 Nano, FirePro S9300 x2, Radeon RX 480,
                                 Radeon RX 470, Radeon RX 460

gfx900        GFX9                 Vega10

============ ================== ============================================================== 

Required AMDGPU Attributes

The flat work-group size is the number of work-items in the work-group size specified when the kernel is dispatched. It is the product of the sizes of the x, y, and z dimension of the work-group.

HCC supports the ``__attribute__((amdgpu_flat_work_group_size(<min>, <max>)))`` attribute for the AMDGPU target. This attribute may be attached to a kernel function definition and is an optimization hint. Currently the default behaviour is ``(128,256)``.

If you plan to use a work-group size more than 256, use this attribute to specify a new flat work-group size. For example, if your dimensions are 512, 1, 1 for x, y, z respectively, use the attribute as ``__attribute__((amdgpu_flat_work_group_size(512)))`` to specify a new maximum flat work-group size.

API documentation

`API reference of HCC <https://doc-july-11.readthedocs.io/en/latest/ROCm_API_References/HCC-API.html#hcc-api>`_
