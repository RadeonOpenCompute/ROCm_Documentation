

================
OpenMP Support
================

Overview
------------

The ROCm installation includes an LLVM-based implementation that fully supports the OpenMP 4.5 standard and a subset of the OpenMP 5.0 standard. Fortran, C/C++ compilers, and corresponding runtime libraries are included. Along with host APIs, the OpenMP compilers support offloading code and data onto GPU devices. The GPUs supported are the same as those supported by this ROCm release. This document briefly describes the installation location of the OpenMP toolchain and example usage of device offloading. 

Installation
-------------

The OpenMP toolchain is automatically installed as part of the standard ROCm installation and is available under /opt/rocm-{version}/llvm. The sub-directories are:

- bin: Compilers (flang and clang) and other binaries

- examples: How to compile and run these programs is shown in the usage section below. 

- include: Header files

- lib: Libraries including those required for target offload

- lib-debug: Debug versions of the above libraries

Usage
------

The example programs can be compiled and run by pointing the environment variable AOMP to the OpenMP install directory. For example:

::

      % export AOMP=/opt/rocm-{version}/llvm
      
      % cd $AOMP/examples/openmp/veccopy
      
      % make run



The above invocation of Make will compile and run the program. Note, the options that are required for target offload from an OpenMP program: 

::

      -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=<gpu-arch>


The value of gpu-arch can be obtained by running the following command:

::

      % /opt/rocm-{version}/bin/rocminfo | grep gfx
      

Helpful Tips
-------------

Setting the environment variable LIBOMPTARGET_KERNEL_TRACE while running an OpenMP program produces valuable information. Among other details, a value of 1 will lead the runtime to emit the number of teams and threads for every kernel run on the GPU. A value of 2 leads additionally to a trace of implementation-level APIs and corresponding timing information. 

