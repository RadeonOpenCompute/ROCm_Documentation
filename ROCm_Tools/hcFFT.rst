.. _hcFFT:

============
hcFFT
============

 * :ref:`Home`
 * :ref:`Examples`
 * :ref:`Installation`
 * :ref:`Introduction`
 * :ref:`KeyFeature`
 * :ref:`Prerequisites`
 * :ref:`TestedEnvironments`

.. _Home`:

HOME
#####

welcome to the hcFFT wiki!

.. _Examples:

Examples
#########

FFT 1D R2C example:

file: hcfft_1D_R2C.cpp

::

  #!c++
  
  #include <iostream>
  #include <cstdlib>
  #include "hcfft.h"
  #include "hc_am.hpp"
  #include "hcfftlib.h"

  int main(int argc, char* argv[]) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    // HCFFT work flow
    hcfftHandle plan;
    hcfftResult status  = hcfftPlan1d(&plan, N, HCFFT_R2C);
    assert(status == HCFFT_SUCCESS);
    int Rsize = N;
    int Csize = (N / 2) + 1;
    hcfftReal* input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
    int seed = 123456789;
    srand(seed);

    // Populate the input
    for(int i = 0; i < Rsize ; i++) {
      input[i] = rand();
    }

    hcfftComplex* output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));

    std::vector<hc::accelerator> accs = hc::accelerator::get_all();
    assert(accs.size() && "Number of Accelerators == 0!");
    hc::accelerator_view accl_view = accs[1].get_default_view();

    hcfftReal* idata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
    accl_view.copy(input, idata, sizeof(hcfftReal) * Rsize);
    hcfftComplex* odata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
    accl_view.copy(output,  odata, sizeof(hcfftComplex) * Csize);
    status = hcfftExecR2C(plan, idata, odata);
    assert(status == HCFFT_SUCCESS);
    accl_view.copy(odata, output, sizeof(hcfftComplex) * Csize);
    status =  hcfftDestroy(plan);
    assert(status == HCFFT_SUCCESS);
    free(input);
    free(output);
    hc::am_free(idata);
    hc::am_free(odata); 
  }
 
* Compiling the example code:

Assuming the library and compiler installation is followed as in installation.

/opt/rocm/hcc/bin/clang++ `/opt/rocm/hcc/bin/hcc-config --cxxflags --ldflags` -lhc_am -lhcfft -I../lib/include -L../build/lib/src hcfft_1D_R2C.cpp

.. _Installation:

Installation
##############

The following are the steps to use the library

 * ROCM 1.5 Kernel, Driver and Compiler Installation (if not done until now)
 * Library installation.

ROCM 1.5 Installation
***********************
To Know more about ROCM refer 
https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md

**a. Installing Debian ROCM repositories**

Before proceeding, make sure to completely uninstall any pre-release ROCm packages.

Refer https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages for instructions to remove pre-release ROCM packages.

Steps to install rocm package are,

::

  wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -

  sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

  sudo apt-get update

  sudo apt-get install rocm

Then, make the ROCm kernel your default kernel. If using grub2 as your bootloader, you can edit the GRUB_DEFAULT variable in the following file:


::

  sudo vi /etc/default/grub

  sudo update-grub

and Reboot the system

**b. Verifying the Installation**

Once Reboot, to verify that the ROCm stack completed successfully you can execute HSA vector_copy sample application:

   * cd /opt/rocm/hsa/sample        
   * make       
   * ./vector_copy

Library Installation
***********************
**a. Install using Prebuilt debian**

::

  wget https://github.com/ROCmSoftwarePlatform/hcFFT/blob/master/pre-builds/hcfft-master-87a37f5-Linux.deb
  sudo dpkg -i hcfft-master-87a37f5-Linux.deb

**b. Build debian from source**

::

  git clone https://github.com/ROCmSoftwarePlatform/hcFFT.git && cd hcFFT

  chmod +x build.sh && ./build.sh

build.sh execution builds the library and generates a debian under build directory.

**c. Install CPU based FFTW3 library**

::

  sudo apt-get install fftw3 fftw3-dev pkg-config

.. _Introduction:

Introduction
#############

This repository hosts the HCC based FFT Library, that targets GPU acceleration of FFT routines on AMD devices. To know what HCC compiler features, refer `here <https://github.com/RadeonOpenCompute/hcc>`_.

The following are the sub-routines that are implemented

 1. R2C : Transforms Real valued input in Time domain to Complex valued output in Frequency domain.

 2. C2R : Transforms Complex valued input in Frequency domain to Real valued output in Real domain.

 3. C2C : Transforms Complex valued input in Frequency domain to Complex valued output in Real domain or vice versa

.. _KeyFeature:

KeyFeature
############
 
 * Support 1D, 2D and 3D Fast Fourier Transforms
 * Supports R2C, C2R, C2C, D2Z, Z2D and Z2Z Transforms
 * Support Out-Of-Place data storage
 * Ability to Choose desired target accelerator
 * Single and Double precision

.. _Prerequisites:

Prerequisites
#################

This section lists the known set of hardware and software requirements to build this library

Hardware
*********

 * CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU 
 * System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
 * Hard Drive > 200GB (Better if SSD or NVMe driver for NN application over multiple GPUs)
 * Minimum GPU Memory (Global) > 2GB

GPU cards supported
*********************
 * dGPU: AMD R9 Fury X, R9 Fury, R9 Nano
 * APU: AMD Kaveri or Carrizo

AMD Driver and Runtime
**************************
 * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
 * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime

System software
*****************
 * Ubuntu 14.04 trusty and later
 * GCC 4.6 and later
 * CPP 4.6 and later (come with GCC package)
 * python 2.7 and later
 * python-pip
 * BeautifulSoup4 (installed using python-pip)
 * HCC 0.9 from here

Tools and Misc
******************
 * git 1.9 and later
 * cmake 2.6 and later (2.6 and 2.8 are tested)
 * firewall off
 * root privilege or user account in sudo group

Ubuntu Packages
****************
 * libc6-dev-i386
 * liblapack-dev
 * graphicsmagick
 * libblas-dev

.. _TestedEnvironments:

Tested Environments
######################
This sections enumerates the list of tested combinations of Hardware and system softwares.

Driver versions
******************

 * Boltzmann Early Release Driver + dGPU
      * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
      * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime
 * Traditional HSA driver + APU (Kaveri)

GPU Cards
***********
 * Radeon R9 Nano 
 * Radeon R9 FuryX
 * Radeon R9 Fury 
 * Kaveri and Carizo APU

Server System
**************
 * Supermicro SYS 2028GR-THT 6 R9 NANO
 * Supermicro SYS-1028GQ-TRT 4 R9 NANO
 * Supermicro SYS-7048GR-TR Tower 4 R9 NANO












































