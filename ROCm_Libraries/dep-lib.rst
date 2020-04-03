.. _HCRG:

**********
hcRNG
**********

hCRNG has been deprecated and has been replaced by `rocRAND <https://github.com/ROCmSoftwarePlatform/rocRAND>`_ 
#################################################################################################################

Introduction
##################

The hcRNG library is an implementation of uniform random number generators targeting the AMD heterogeneous hardware via HCC compiler runtime. The computational resources of underlying AMD heterogenous compute gets exposed and exploited through the HCC C++ frontend. Refer `here <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#hcc>`_ for more details on HCC compiler.

The following list enumerates the current set of RNG generators that are supported so far.

  * MRG31k3p
  * MRG32k3a
  * LFSR113
  * Philox-4x32-10

Examples
##########

Random number generator Mrg31k3p example:

file: Randomarray.cpp


::

  #!c++
  
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
#################

**Installation steps**

The following are the steps to use the library

  * ROCM 2.7 Kernel, Driver and Compiler Installation (if not done until now)
  * Library installation.

**ROCM 2.7 Installation**

To Know more about ROCM refer `here <https://rocm-documentation.readthedocs.io/en/latest/Current_Release_Notes/Current-Release-Notes.html>`_

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

**Library Installation**

**a. Install using Prebuilt debian**

::
  
  wget https://github.com/ROCmSoftwarePlatform/hcRNG/blob/master/pre-builds/hcrng-master-184472e-Linux.deb
  sudo dpkg -i hcrng-master-184472e-Linux.deb

**b. Build debian from source**

::
  
  git clone https://github.com/ROCmSoftwarePlatform/hcRNG.git && cd hcRNG
  chmod +x build.sh && ./build.sh

**build.sh** execution builds the library and generates a debian under build directory.


Key Features
#################

 * Support for 4 commonly used uniform random number generators.
 * Single and Double precision.
 * Multiple streams, created on the host and generates random numbers either on the host or on computing devices.


**Prerequisites**

This section lists the known set of hardware and software requirements to build this library

**Hardware**

 * CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU
 * System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
 * Hard Drive > 200GB (Better if SSD or NVMe driver for NN application over multiple GPUs)
 * Minimum GPU Memory (Global) > 2GB

**GPU cards supported**

 * dGPU: AMD R9 Fury X, R9 Fury, R9 Nano
 * APU: AMD Kaveri or Carrizo

**AMD Driver and Runtime**

  * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
  * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime

**System software**

 * Ubuntu 14.04 trusty and later
 * GCC 4.6 and later
 * CPP 4.6 and later (come with GCC package)
 * python 2.7 and later
 * python-pip
 * BeautifulSoup4 (installed using python-pip)
 * HCC 0.9 from here

**Tools and Misc**

 * git 1.9 and later
 * cmake 2.6 and later (2.6 and 2.8 are tested)
 * firewall off
 * root privilege or user account in sudo group

**Ubuntu Packages**

 * libc6-dev-i386
 * liblapack-dev
 * graphicsmagick
 * libblas-dev


Tested Environments
#######################

**Driver versions**

  * Boltzmann Early Release Driver + dGPU
      * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
      * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime
  * Traditional HSA driver + APU (Kaveri)

**GPU Cards**

  * Radeon R9 Nano
  * Radeon R9 FuryX
  * Radeon R9 Fury
  * Kaveri and Carizo APU

**Server System**

  * Supermicro SYS 2028GR-THT 6 R9 NANO
  * Supermicro SYS-1028GQ-TRT 4 R9 NANO
  * Supermicro SYS-7048GR-TR Tower 4 R9 NANO


Unit testing
#################

**a) Automated testing:**

Follow these steps to start automated testing:
::
  cd ~/hcRNG/
  ./build.sh --test=on

**b) Manual testing:**

(i) Google testing (GTEST) with Functionality check
::
  cd ~/hcRNG/build/test/unit/bin/

All functions are tested against google test.


.. _HIPE:

**************
hipeigen
**************

Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

For more information go to http://eigen.tuxfamily.org/.

Installation instructions for ROCm
#####################################
The ROCm Platform brings a rich foundation to advanced computing by seamlessly integrating the CPU and GPU with the goal of solving real-world problems.

To insatll rocm, please follow:

Installing from AMD ROCm repositories
#########################################
AMD is hosting both debian and rpm repositories for the ROCm 2.7 packages. The packages in both repositories have been signed to ensure package integrity. Directions for each repository are given below:

* Debian repository - apt-get
* Add the ROCm apt repository

Complete installation steps of ROCm can be found `Here <https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html>`_

or 

For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows:

::

  wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
  sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

The gpg key might change, so it may need to be updated when installing a new release.

**Install or Update**


Next, update the apt-get repository list and install/update the rocm package:

 .. WARNING::
       Before proceeding, make sure to completely uninstall any pre-release ROCm packages

::

  sudo apt-get update
  sudo apt-get install rocm


Then, make the ROCm kernel your default kernel. If using grub2 as your bootloader, you can edit the GRUB_DEFAULT variable in the following file:

::

  sudo vi /etc/default/grub
  sudo update-grub

Once complete, **reboot your system.**

We recommend you verify your installation to make sure everything completed successfully.

Installation instructions for Eigen
#########################################
**Explanation before starting**

Eigen consists only of header files, hence there is nothing to compile before you can use it. Moreover, these header files do not depend on your platform, they are the same for everybody.

**Method 1. Installing without using CMake**

You can use right away the headers in the Eigen/ subdirectory. In order to install, just copy this Eigen/ subdirectory to your favorite location. If you also want the unsupported features, copy the unsupported/ subdirectory too.

**Method 2. Installing using CMake**

Let's call this directory 'source_dir' (where this INSTALL file is). Before starting, create another directory which we will call 'build_dir'.

Do:
::

  cd build_dir
  cmake source_dir
  make install

The make install step may require administrator privileges.

You can adjust the installation destination (the "prefix") by passing the -DCMAKE_INSTALL_PREFIX=myprefix option to cmake, as is explained in the message that cmake prints at the end.

Build and Run hipeigen direct tests
#########################################
To build the direct tests for hipeigen:
::
  cd build_dir
  make check -j $(nproc)


Note: All direct tests should pass with ROCm 2.7


.. _CLFF:

*******
clFFT
*******

For Github Repository `clFFT <https://github.com/clMathLibraries/clFFT>`_

clFFT is a software library containing FFT functions written in OpenCL. In addition to GPU devices, the library also supports running on CPU devices to facilitate debugging and heterogeneous programming.

Pre-built binaries are available `here <https://github.com/clMathLibraries/clFFT/releases>`_.

What's New
###########

    * Support for powers of 11&13 size transforms
    * Support for 1D large size transforms with no extra memory allocation requirement with environment flag CLFFT_REQUEST_LIB_NOMEMALLOC=1 for complex FFTs of powers of 2,3,5,10 sizes

Note
-----

    * clFFT requires platform/runtime that supports OpenCL 1.2


Introduction to clFFT
############################
The FFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the FFT definition to reduce the mathematical intensity required from O(N^2) to O(N log2(N)) when the sequence length N is the product of small prime factors. Currently, there is no standard API for FFT routines. Hardware vendors usually provide a set of high-performance FFTs optimized for their systems: no two vendors employ the same interfaces for their FFT routines. clFFT provides a set of FFT routines that are optimized for AMD graphics processors, but also are functional across CPU and other compute devices.

The clFFT library is an open source OpenCL library implementation of discrete Fast Fourier Transforms. The library:

 * provides a fast and accurate platform for calculating discrete FFTs.

 * works on CPU or GPU backends.

 * supports in-place or out-of-place transforms.

 * supports 1D, 2D, and 3D transforms with a batch size that can be greater than 1.

 * supports planar (real and complex components in separate arrays) and interleaved (real and complex components as a pair contiguous in memory) formats.

 * supports dimension lengths that can be any combination of powers of 2, 3, 5, 7, 11 and 13.

 * Supports single and double precision floating point formats.

clFFT library user documentation
########################################
`Library and API documentation <http://clmathlibraries.github.io/clFFT/>`_ for developers is available online as a GitHub Pages website

API semantic versioning
############################
Good software is typically the result of the loop of feedback and iteration; software interfaces no less so. clFFT follows the `semantic <http://semver.org/>`_ versioning guidelines. The version number used is of the form MAJOR.MINOR.PATCH.

clFFT Wiki
############################
The `project wiki <https://github.com/clMathLibraries/clFFT/wiki>`_ contains helpful documentation, including a `build primer <https://github.com/clMathLibraries/clFFT/wiki/Build>`_

Contributing code
############################
Please refer to and read the `Contributing <https://github.com/clMathLibraries/clFFT/blob/master/CONTRIBUTING.md>`_ document for guidelines on how to contribute code to this open source project. The code in the /master branch is considered to be stable, and all pull-requests must be made against the /develop branch.

License
############################
The source for clFFT is licensed under the `Apache License <http://www.apache.org/licenses/LICENSE-2.0>`_ , Version 2.0

Example
############################
The following simple example shows how to use clFFT to compute a simple 1D forward transform

::

 #include <stdlib.h>

 /* No need to explicitely include the OpenCL headers */
 #include <clFFT.h>

 int main( void )
 {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufX;
	float *X;
    cl_event event = NULL;
    int ret = 0;
	size_t N = 16;

	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[1] = {N};

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Setup clFFT. */
	clfftSetupData fftSetup;
	err = clfftInitSetupData(&fftSetup);
	err = clfftSetup(&fftSetup);

	/* Allocate host & initialize data. */
	/* Only allocation shown for simplicity. */
	X = (float *)malloc(N * 2 * sizeof(*X));

    /* Prepare OpenCL memory objects and place data inside them. */
    bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(*X), NULL, &err );

    err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
	N * 2 * sizeof( *X ), X, 0, NULL, NULL );

	/* Create a default plan for a complex FFT. */
	err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

	/* Set plan parameters. */
	err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /* Bake the plan. */
	err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

	/* Execute the plan. */
	err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL);

	/* Wait for calculations to be finished. */
	err = clFinish(queue);

	/* Fetch results of calculations. */
	err = clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, N * 2 * sizeof( *X ), X, 0, NULL, NULL );

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufX );

	free(X);

	/* Release the plan. */
	err = clfftDestroyPlan( &planHandle );

    /* Release clFFT library. */
    clfftTeardown( );

    /* Release OpenCL working objects. */
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return ret;
  }

Build dependencies
############################
**Library for Windows**

To develop the clFFT library code on a Windows operating system, ensure to install the following packages on your system:

 * Windows® 7/8.1

 * Visual Studio 2012 or later

 * Latest CMake

 * An OpenCL SDK, such as APP SDK 3.0

**Library for Linux**

To develop the clFFT library code on a Linux operating system, ensure to install the following packages on your system:

 * GCC 4.6 and onwards

 * Latest CMake

 * An OpenCL SDK, such as APP SDK 3.0

**Library for Mac OSX**

To develop the clFFT library code on a Mac OS X, it is recommended to generate Unix makefiles with cmake.

**Test infrastructure**

To test the developed clFFT library code, ensure to install the following packages on your system:

 * Googletest v1.6

 * Latest FFTW
 
 * Latest Boost

Performance infrastructure
###################################
To measure the performance of the clFFT library code, ensure that the Python package is installed on your system.


.. _CLB:

*********
clBLAS
*********


For Github repository `clBLAS <https://github.com/clMathLibraries/clBLAS>`_

This repository houses the code for the OpenCL™ BLAS portion of clMath. The complete set of BLAS level 1, 2 & 3 routines is implemented. Please see Netlib BLAS for the list of supported routines. In addition to GPU devices, the library also supports running on CPU devices to facilitate debugging and multicore programming. APPML 1.12 is the most current generally available pre-packaged binary version of the library available for download for both Linux and Windows platforms.

The primary goal of clBLAS is to make it easier for developers to utilize the inherent performance and power efficiency benefits of heterogeneous computing. clBLAS interfaces do not hide nor wrap OpenCL interfaces, but rather leaves OpenCL state management to the control of the user to allow for maximum performance and flexibility. The clBLAS library does generate and enqueue optimized OpenCL kernels, relieving the user from the task of writing, optimizing and maintaining kernel code themselves.

**clBLAS update notes 01/2017**

v2.12 is a bugfix release as a rollup of all fixes in /develop branch
Thanks to @pavanky, @iotamudelta, @shahsan10, @psyhtest, @haahh, @hughperkins, @tfauck @abhiShandy, @IvanVergiliev, @zougloub, @mgates3 for contributions to clBLAS v2.12
Summary of fixes available to read on the releases tab

clBLAS library user documentation
##########################################
`Library and API documentation <http://clmathlibraries.github.io/clBLAS/>`_ for developers is available online as a GitHub Pages website

**clBLAS Wiki**

The project `wiki <https://github.com/clMathLibraries/clBLAS/wiki>`_ contains helpful documentation, including a `build primer <https://github.com/clMathLibraries/clBLAS/wiki>`_

**Contributing code**

Please refer to and read the `Contributing document <https://github.com/clMathLibraries/clBLAS/blob/master/CONTRIBUTING.md>`_ for guidelines on how to contribute code to this open source project. The code in the /master branch is considered to be stable, and all pull-requests should be made against the /develop branch.

License
##########################################
The source for clBLAS is licensed under the Apache License, Version 2.0

Example
##########################################
The simple example below shows how to use clBLAS to compute an OpenCL accelerated SGEMM

::

    #include <sys/types.h>
    #include <stdio.h>

    /* Include the clBLAS header. It includes the appropriate OpenCL headers */
    #include <clBLAS.h>

    /* This example uses predefined matrices and their characteristics for
     * simplicity purpose.
    */

    #define M  4
    #define N  3
    #define K  5

    static const cl_float alpha = 10;

    static const cl_float A[M*K] = {
    11, 12, 13, 14, 15,
    21, 22, 23, 24, 25,
    31, 32, 33, 34, 35,
    41, 42, 43, 44, 45,
    };
    static const size_t lda = K;        /* i.e. lda = K */

    static const cl_float B[K*N] = {
    11, 12, 13,
    21, 22, 23,
    31, 32, 33,
    41, 42, 43,
    51, 52, 53,
    };
    static const size_t ldb = N;        /* i.e. ldb = N */

    static const cl_float beta = 20;

    static cl_float C[M*N] = {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
    };
    static const size_t ldc = N;        /* i.e. ldc = N */

    static cl_float result[M*N];

    int main( void )
    {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Setup clBLAS */
    err = clblasSetup( );

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer( ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
                          NULL, &err );
    bufB = clCreateBuffer( ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
                          NULL, &err );
    bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                          NULL, &err );

    err = clEnqueueWriteBuffer( queue, bufA, CL_TRUE, 0,
        M * K * sizeof( *A ), A, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufB, CL_TRUE, 0,
        K * N * sizeof( *B ), B, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufC, CL_TRUE, 0,
        M * N * sizeof( *C ), C, 0, NULL, NULL );

        /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
        err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                                M, N, K,
                                alpha, bufA, 0, lda,
                                bufB, 0, ldb, beta,
                                bufC, 0, ldc,
                                1, &queue, 0, NULL, &event );

    /* Wait for calculations to be finished. */
    err = clWaitForEvents( 1, &event );

    /* Fetch results of calculations from GPU memory. */
    err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,
                                M * N * sizeof(*result),
                                result, 0, NULL, NULL );

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufC );
    clReleaseMemObject( bufB );
    clReleaseMemObject( bufA );

    /* Finalize work with clBLAS */
    clblasTeardown( );

    /* Release OpenCL working objects. */
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return ret;
    }


Build dependencies
##########################################
**Library for Windows**

 * Windows® 7/8
 * Visual Studio 2010 SP1, 2012
 * An OpenCL SDK, such as APP SDK 2.8
 * Latest CMake

**Library for Linux**

 * GCC 4.6 and onwards
 * An OpenCL SDK, such as APP SDK 2.9
 * Latest CMake

**Library for Mac OSX**

 * Recommended to generate Unix makefiles with cmake

**Test infrastructure**

 * Googletest v1.6
 * Latest Boost
 * CPU BLAS
 * Netlib CBLAS (recommended) Ubuntu: install by "apt-get install libblas-dev" Windows: download & install lapack-3.6.0 which comes 	with CBLAS
 * or ACML on windows/linux; Accelerate on Mac OSX

Performance infrastructure
##########################################
Python


.. _CLS:

**************
clSPARSE
**************
 
For Github repository `clSPARSE <https://github.com/clMathLibraries/clSPARSE>`_

an OpenCL™ library implementing Sparse linear algebra routines. This project is a result of a collaboration between `AMD Inc. <http://www.amd.com/en>`_ and `Vratis Ltd. <http://www.vratis.com/>`_.

What's new in clSPARSE v0.10.1
###################################
 * bug fix release
     * Fixes for travis builds
     * Fix to the matrix market reader in the cuSPARSE benchmark to synchronize with the regular MM reader
     * Replace cl.hpp with cl2.hpp (thanks to arrayfire)
     * Fixes for the Nvidia platform; tested 352.79
        * Fixed buffer overruns in CSR-Adaptive kernels
        * Fix invalid memory access on Nvidia GPUs in CSR-Adaptive SpMV kernel

Build Status
#############

Pre-built binaries are available on our `releases page <https://github.com/clMathLibraries/clSPARSE/releases>`_

clSPARSE features
#######################
 * Sparse Matrix - dense Vector multiply (SpM-dV)
 * Sparse Matrix - dense Matrix multiply (SpM-dM)
 * Sparse Matrix - Sparse Matrix multiply Sparse Matrix Multiply(SpGEMM) - Single Precision
 * Iterative conjugate gradient solver (CG)
 * Iterative biconjugate gradient stabilized solver (BiCGStab)
 * Dense to CSR conversions (& converse)
 * COO to CSR conversions (& converse)
 * Functions to read matrix market files in COO or CSR format
True in spirit with the other clMath libraries, clSPARSE exports a “C” interface to allow projects to build wrappers around clSPARSE in any language they need. A great deal of thought and effort went into designing the API’s to make them less ‘cluttered’ compared to the older clMath libraries. OpenCL state is not explicitly passed through the API, which enables the library to be forward compatible when users are ready to switch from OpenCL 1.2 to OpenCL 2.0 3

API semantic versioning
##############################
Good software is typically the result of iteration and feedback. clSPARSE follows the `semantic <http://semver.org/>`_ versioning guidelines, and while the major version number remains '0', the public API should not be considered stable. We release clSPARSE as beta software (0.y.z) early to the community to elicit feedback and comment. This comes with the expectation that with feedback, we may incorporate breaking changes to the API that might require early users to recompile, or rewrite portions of their code as we iterate on the design.

**clSPARSE Wiki**

The `project wiki <https://github.com/clMathLibraries/clSPARSE/wiki>`_ contains helpful documentation.
A `build primer <https://github.com/clMathLibraries/clSPARSE/wiki/Build>`_ is available, which describes how to use cmake to generate platforms specific build files

**Samples**

clSPARSE contains a directory of simple `OpenCL samples <https://github.com/clMathLibraries/clSPARSE/tree/master/samples>`_ that demonstrate the use of the API in both C and C++. The `superbuild <https://blog.kitware.com/wp-content/uploads/2016/01/kitware_quarterly1009.pdf>`_ script for clSPARSE also builds the samples as an external project, to demonstrate how an application would find and link to clSPARSE with cmake.

**clSPARSE library documentation**

API documentation is available at http://clmathlibraries.github.io/clSPARSE/. The samples give an excellent starting point to basic library operations.

**Contributing code**

Please refer to and read the `Contributing <https://github.com/clMathLibraries/clSPARSE/blob/master/CONTRIBUTING.md>`_ document for guidelines on how to contribute code to this open source project. Code in the /master branch is considered to be stable and new library releases are made when commits are merged into /master. Active development and pull-requests should be made to the develop branch.

License
#####################
clSPARSE is licensed under the `Apache License <http://www.apache.org/licenses/LICENSE-2.0>`_, Version 2.0

**Compiling for Windows**

 * Windows® 7/8
 * Visual Studio 2013 and above
 * CMake 2.8.12 (download from `Kitware <http://www.cmake.org/download/>`_)
 * Solution (.sln) or
 * Nmake makefiles
 * An OpenCL SDK, such as APP SDK 3.0

**Compiling for Linux**

 * GCC 4.8 and above
 * CMake 2.8.12 (install with distro package manager )
 * Unix makefiles or
     * KDevelop or
     * QT Creator
     * An OpenCL SDK, such as APP SDK 3.0

**Compiling for Mac OSX**

 * CMake 2.8.12 (install via brew)
 * Unix makefiles or
 * XCode
 * An OpenCL SDK (installed via xcode-select --install)

**Bench & Test infrastructure dependencies**

 * Googletest v1.7
 * Boost v1.58
 * Footnotes

[1]: Changed to reflect CppCoreGuidelines: `F.21 <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#a-namerf-out-multia-f21-to-return-multiple-out-values-prefer-returning-a-tuple-or-struct>`_

[2]: Changed to reflect CppCoreGuidelines: `NL.8 <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#a-namerl-namea-nl8-use-a-consistent-naming-style>`_

[3]: OpenCL 2.0 support is not yet fully implemented; only the interfaces have been designed



.. _CLR:

****************
clRNG
****************
 
For Github repository `clRNG <https://github.com/clMathLibraries/clRNG>`_

A library for uniform random number generation in OpenCL.

Streams of random numbers act as virtual random number generators. They can be created on the host computer in unlimited numbers, and then used either on the host or on computing devices by work items to generate random numbers. Each stream also has equally-spaced substreams, which are occasionally useful. The API is currently implemented for four different RNGs, namely the MRG31k3p, MRG32k3a, LFSR113 and Philox-4×32-10 generators.


What's New
##############
Libraries related to clRNG, for probability distributions and quasi-Monte Carlo methods, are available:

 * `clProbDist <https://github.com/umontreal-simul/clProbDist>`_
 * `clQMC <https://github.com/umontreal-simul/clQMC>`_

**Releases**

The first public version of clRNG is v1.0.0 beta. Please go to `releases <https://github.com/clMathLibraries/clRNG/releases>`_ for downloads.

Building
##############
 1. Install the runtime dependency:
      * An OpenCL SDK, such as APP SDK.
 
 2. Install the build dependencies:

     * The CMake cross-platform build system. Visual Studio users can use CMake Tools for Visual Studio.
     * A recent C compiler, such as `GCC 4.9 <https://gcc.gnu.org/>`_ , or Visual Studio 2013.

 3. Get the clRNG source code.

 4. Configure the project using `CMake <https://cmake.org/>`_ (to generate standard makefiles) or `CMake Tools for Visual Studio <https://cmaketools.codeplex.com/>`_ (to generate solution and project files).

 5. Build the project.

 6. Install the project (by default, the library will be installed in the package directory under the build directory).

 7. Point the environment variable CLRNG_ROOT to the installation directory, i.e., the directory under which include/clRNG can be   	found. This step is optional if the library is installed under /usr, which is the default.

 8. In order to execute the example programs (under the bin subdirectory of the installation directory) or to link clRNG into other 	software, the dynamic linker must be informed where to find the clRNG shared library. The name and location of the shared library 	  generally depend on the platform.

 9. Optionally run the tests.

Example Instructions for Linux
#####################################
On a 64-bit Linux platform, steps 3 through 9 from above, executed in a Bash-compatible shell, could consist of:

::

  git clone https://github.com/clMathLibraries/clRNG.git
  mkdir clRNG.build; cd clRNG.build; cmake ../clRNG/src
  make
  make install
  export CLRNG_ROOT=$PWD/package
  export LD_LIBRARY_PATH=$CLRNG_ROOT/lib64:$LD_LIBRARY_PATH
  $CLRNG_ROOT/bin/CTest
  
**Examples**

Examples can be found in src/client. The compiled client program examples can be found under the bin subdirectory of the installation package ($CLRNG_ROOT/bin under Linux). Note that the examples expect an OpenCL GPU device to be available.

**Simple example**

The simple example below shows how to use clRNG to generate random numbers by directly using device side headers (.clh) in your OpenCL kernel.

::

  #include <stdlib.h>
  #include <string.h>

  #include "clRNG/clRNG.h"
  #include "clRNG/mrg31k3p.h"

  int main( void )
  {
      cl_int err;
      cl_platform_id platform = 0;
      cl_device_id device = 0;
      cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
      cl_context ctx = 0;
      cl_command_queue queue = 0;
      cl_program program = 0;
      cl_kernel kernel = 0;
      cl_event event = 0;
      cl_mem bufIn, bufOut;
      float *out;
      char *clrng_root;
      char include_str[1024];
      char build_log[4096];
      size_t i = 0;
      size_t numWorkItems = 64;
      clrngMrg31k3pStream *streams = 0;
      size_t streamBufferSize = 0;
      size_t kernelLines = 0;

      /* Sample kernel that calls clRNG device-side interfaces to generate random numbers */
      const char *kernelSrc[] = {
      "    #define CLRNG_SINGLE_PRECISION                                   \n",
      "    #include <clRNG/mrg31k3p.clh>                                    \n",
      "                                                                     \n",
      "    __kernel void example(__global clrngMrg31k3pHostStream *streams, \n",
      "                          __global float *out)                       \n",
      "    {                                                                \n",
      "        int gid = get_global_id(0);                                  \n",
      "                                                                     \n",
      "        clrngMrg31k3pStream workItemStream;                          \n",
      "        clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream,   \n",
      "                                                     &streams[gid]); \n",
      "                                                                     \n",
      "        out[gid] = clrngMrg31k3pRandomU01(&workItemStream);          \n",
      "    }                                                                \n",
      "                                                                     \n",
      };

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Make sure CLRNG_ROOT is specified to get library path */
    clrng_root = getenv("CLRNG_ROOT");
    if(clrng_root == NULL) printf("\nSpecify environment variable CLRNG_ROOT as described\n");
    strcpy(include_str, "-I ");
    strcat(include_str, clrng_root);
    strcat(include_str, "/include");

    /* Create sample kernel */
    kernelLines = sizeof(kernelSrc) / sizeof(kernelSrc[0]);
    program = clCreateProgramWithSource(ctx, kernelLines, kernelSrc, NULL, &err);
    err = clBuildProgram(program, 1, &device, include_str, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        printf("\nclBuildProgram has failed\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 4096, build_log, NULL);
        printf("%s", build_log);
    }
    kernel = clCreateKernel(program, "example", &err);

    /* Create streams */
    streams = clrngMrg31k3pCreateStreams(NULL, numWorkItems, &streamBufferSize, (clrngStatus *)&err);

    /* Create buffers for the kernel */
    bufIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, streamBufferSize, streams, &err);
    bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, numWorkItems * sizeof(cl_float), NULL, &err);

    /* Setup the kernel */
    err = clSetKernelArg(kernel, 0, sizeof(bufIn),  &bufIn);
    err = clSetKernelArg(kernel, 1, sizeof(bufOut), &bufOut);

    /* Execute the kernel and read back results */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &numWorkItems, NULL, 0, NULL, &event);
    err = clWaitForEvents(1, &event);
    out = (float *)malloc(numWorkItems * sizeof(out[0]));
    err = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, numWorkItems * sizeof(out[0]), out, 0, NULL, NULL);

    /* Release allocated resources */
    clReleaseEvent(event);
    free(out);
    clReleaseMemObject(bufIn);
    clReleaseMemObject(bufOut);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
  }

Building the documentation manually
###########################################
The documentation can be generated by running make from within the doc directory. This requires Doxygen to be installed.


.. _HCF:

*************
hcFFT
*************

hcFFT has been deprecated and has been replaced by `rocFFT <https://github.com/ROCmSoftwarePlatform/rocFFT>`_
#################################################################################################################

Installation
###############

The following are the steps to use the library

 * ROCM 2.7 Kernel, Driver and Compiler Installation (if not done until now)
 * Library installation.

**ROCM 2.7 Installation**

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

**Library Installation**

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


Introduction
##############

This repository hosts the HCC based FFT Library, that targets GPU acceleration of FFT routines on AMD devices. To know what HCC compiler features, refer `here <https://github.com/RadeonOpenCompute/hcc>`_.

The following are the sub-routines that are implemented

 1. R2C : Transforms Real valued input in Time domain to Complex valued output in Frequency domain.

 2. C2R : Transforms Complex valued input in Frequency domain to Real valued output in Real domain.

 3. C2C : Transforms Complex valued input in Frequency domain to Complex valued output in Real domain or vice versa


KeyFeature
#############
 
 * Support 1D, 2D and 3D Fast Fourier Transforms
 * Supports R2C, C2R, C2C, D2Z, Z2D and Z2Z Transforms
 * Support Out-Of-Place data storage
 * Ability to Choose desired target accelerator
 * Single and Double precision


**Prerequisites**


This section lists the known set of hardware and software requirements to build this library

**Hardware**


 * CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU 
 * System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
 * Hard Drive > 200GB (Better if SSD or NVMe driver for NN application over multiple GPUs)
 * Minimum GPU Memory (Global) > 2GB

**GPU cards supported**

 * dGPU: AMD R9 Fury X, R9 Fury, R9 Nano
 * APU: AMD Kaveri or Carrizo

**AMD Driver and Runtime**

 * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
 * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime

**System software**

 * Ubuntu 14.04 trusty and later
 * GCC 4.6 and later
 * CPP 4.6 and later (come with GCC package)
 * python 2.7 and later
 * python-pip
 * BeautifulSoup4 (installed using python-pip)
 * HCC 0.9 from here

**Tools and Misc**

 * git 1.9 and later
 * cmake 2.6 and later (2.6 and 2.8 are tested)
 * firewall off
 * root privilege or user account in sudo group

**Ubuntu Packages**

 * libc6-dev-i386
 * liblapack-dev
 * graphicsmagick
 * libblas-dev




Examples
##########

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



Tested Environments
########################
This sections enumerates the list of tested combinations of Hardware and system softwares.

**Driver versions**


 * Boltzmann Early Release Driver + dGPU
      * Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
      * HSA runtime API and runtime for Boltzmann: https://github.com/RadeonOpenCompute/ROCR-Runtime
 * Traditional HSA driver + APU (Kaveri)

**GPU Cards**

 * Radeon R9 Nano 
 * Radeon R9 FuryX
 * Radeon R9 Fury 
 * Kaveri and Carizo APU

**Server System**

 * Supermicro SYS 2028GR-THT 6 R9 NANO
 * Supermicro SYS-1028GQ-TRT 4 R9 NANO
 * Supermicro SYS-7048GR-TR Tower 4 R9 NANO

