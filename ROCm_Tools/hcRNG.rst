.. _hcRNG:

============
hcRNG
============

Home
#####

Welcome to the hcRNG wiki!


Examples
##########

Random number generator Mrg31k3p example:

file: Randomarray.cpp

#!c++

::

 
  //This example is a simple random array generation and it compares host output with device output
  //Random number generator Mrg31k3p
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <stdint.h>
  #include <assert.h>
  #include <hcRNG/mrg31k3p.h>
  #include <hcRNG/hcRNG.h>
  #include <hc.hpp>
  #include <hc_am.hpp>
  using namespace hc;
 
  int main()
  {
        hcrngStatus status = HCRNG_SUCCESS;
        bool ispassed = 1;
        size_t streamBufferSize;
        // Number oi streams
        size_t streamCount = 10;
        //Number of random numbers to be generated
        //numberCount must be a multiple of streamCount
        size_t numberCount = 100; 
        //Enumerate the list of accelerators
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].create_view());
        //Allocate memory for host pointers
        float *Random1 = (float*) malloc(sizeof(float) * numberCount);
        float *Random2 = (float*) malloc(sizeof(float) * numberCount);
        float *outBufferDevice = hc::am_alloc(sizeof(float) * numberCount, acc[1], 0);
 
        //Create streams
        hcrngMrg31k3pStream *streams = hcrngMrg31k3pCreateStreams(NULL, streamCount, &streamBufferSize, NULL);
        hcrngMrg31k3pStream *streams_buffer = hc::am_alloc(sizeof(hcrngMrg31k3pStream) * streamCount, acc[1], 0);
        accl_view.copy(streams, streams_buffer, streamCount* sizeof(hcrngMrg31k3pStream));
 
        //Invoke random number generators in device (here strean_length and streams_per_thread arguments are default) 
        status = hcrngMrg31k3pDeviceRandomU01Array_single(accl_view, streamCount, streams_buffer, numberCount, outBufferDevice);
 
        if(status) std::cout << "TEST FAILED" << std::endl;
        accl_view.copy(outBufferDevice, Random1, numberCount * sizeof(float));
 
        //Invoke random number generators in host
        for (size_t i = 0; i < numberCount; i++)
          Random2[i] = hcrngMrg31k3pRandomU01(&streams[i % streamCount]);   
        // Compare host and device outputs
        for(int i =0; i < numberCount; i++) {
            if (Random1[i] != Random2[i]) {
                ispassed = 0;
                std::cout <<" RANDDEVICE[" << i<< "] " << Random1[i] << "and RANDHOST[" << i <<"] mismatches"<< Random2[i] << 			std::endl;
                break;
            }
            else
                continue;
        }
        if(!ispassed) std::cout << "TEST FAILED" << std::endl;
 
        //Free host resources
        free(Random1);
        free(Random2);
        //Release device resources
        hc::am_free(outBufferDevice);
        hc::am_free(streams_buffer);
        return 0;
  }  
 

* Compiling the example code:

    /opt/hcc/bin/clang++ `/opt/hcc/bin/hcc-config --cxxflags --ldflags` -lhc_am -lhcrng Randomarray.cpp

Installation
#############

Installation steps
********************
The following are the steps to use the library

  * ROCM 2.0 Kernel, Driver and Compiler Installation (if not done until now)
  * Library installation.

ROCM 2.0 Installation
************************
To Know more about ROCM refer https://rocm-documentation.readthedocs.io/en/latest/Current_Release_Notes/Current-Release-Notes.html

**a. Installing Debian ROCM repositories**

Before proceeding, make sure to completely uninstall any pre-release ROCm packages.

Refer `Here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#removing-pre-release-packages>`_ for instructions to remove pre-release ROCM packages

Follow Steps to install rocm package
::
  wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
  sudo apt-get update
  sudo apt-get install rocm


Then, make the ROCm kernel your default kernel. If using grub2 as your bootloader, you can edit the GRUB_DEFAULT variable in the following file:
::
  sudo vi /etc/default/grub
  sudo update-grub

and **Reboot the system**

**b. Verifying the Installation**

Once Reboot, to verify that the ROCm stack completed successfully you can execute HSA vector_copy sample application:
::
  cd /opt/rocm/hsa/sample        
  make       
  ./vector_copy

Library Installation
***********************
**a. Install using Prebuilt debian**

::
  
  wget https://github.com/ROCmSoftwarePlatform/hcRNG/blob/master/pre-builds/hcrng-master-184472e-Linux.deb
  sudo dpkg -i hcrng-master-184472e-Linux.deb

**b. Build debian from source**

::
  
  git clone https://github.com/ROCmSoftwarePlatform/hcRNG.git && cd hcRNG
  chmod +x build.sh && ./build.sh

**build.sh** execution builds the library and generates a debian under build directory.

Introduction
##############

The hcRNG library is an implementation of uniform random number generators targeting the AMD heterogeneous hardware via HCC compiler runtime. The computational resources of underlying AMD heterogenous compute gets exposed and exploited through the HCC C++ frontend. Refer `here <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#hcc>`_ for more details on HCC compiler.

The following list enumerates the current set of RNG generators that are supported so far.

  * MRG31k3p
  * MRG32k3a
  * LFSR113
  * Philox-4x32-10


Key Features
##############

 * Support for 4 commonly used uniform random number generators.
 * Single and Double precision.
 * Multiple streams, created on the host and generates random numbers either on the host or on computing devices.


Prerequisites
##############

This section lists the known set of hardware and software requirements to build this library

Hardware
**********
 * CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU
 * System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
 * Hard Drive > 200GB (Better if SSD or NVMe driver for NN application over multiple GPUs)
 * Minimum GPU Memory (Global) > 2GB

GPU cards supported
*******************
 * dGPU: AMD R9 Fury X, R9 Fury, R9 Nano
 * APU: AMD Kaveri or Carrizo

AMD Driver and Runtime
************************
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
*******************
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

Tested Environments
#######################

Driver versions
******************
  * Boltzmann Early Release Driver + dGPU
      * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
      * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime
  * Traditional HSA driver + APU (Kaveri)

GPU Cards
************
  * Radeon R9 Nano
  * Radeon R9 FuryX
  * Radeon R9 Fury
  * Kaveri and Carizo APU

Server System
***************
  * Supermicro SYS 2028GR-THT 6 R9 NANO
  * Supermicro SYS-1028GQ-TRT 4 R9 NANO
  * Supermicro SYS-7048GR-TR Tower 4 R9 NANO


Unit testing
##############

a) Automated testing:
**********************
Follow these steps to start automated testing:
::
  cd ~/hcRNG/
  ./build.sh --test=on

b) Manual testing:
******************
(i) Google testing (GTEST) with Functionality check
::
  cd ~/hcRNG/build/test/unit/bin/

All functions are tested against google test.























































