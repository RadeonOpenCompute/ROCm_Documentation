.. _ROCm_Libraries:

=====================
ROCm Libraries
=====================

**********
rocFFT
**********

rocFFT is a software library for computing Fast Fourier Transforms (FFT) written in HIP. It is part of AMD's software ecosystem based on ROCm. In addition to AMD GPU devices, the library can also be compiled with the CUDA compiler using HIP tools for running on Nvidia GPU devices.

API design
###############
Please refer to the :ref:`rocFFTAPI` for current documentation. Work in progress.

Installing pre-built packages
################################
Download pre-built packages either from `ROCm's package servers <https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html>`_ or by clicking the github releases tab and manually downloading, which could be newer. Release notes are available for each release on the releases tab.
::
 sudo apt update && sudo apt install rocfft

Quickstart rocFFT build
#########################
Bash helper build script (Ubuntu only)
The root of this repository has a helper bash script install.sh to build and install rocFFT on Ubuntu with a single command. It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install. A few commands in the script need sudo access, so it may prompt you for a password.
* ``./install -h`` -- shows help
* ``./install -id`` -- build library, build dependencies and install globally (-d flag only needs to be specified once on a system)
* ``./install -c --cuda`` -- build library and clients for cuda backend into a local directory
Manual build (all supported platforms)
If you use a distro other than Ubuntu, or would like more control over the build process, the `rocfft build wiki <https://github.com/ROCmSoftwarePlatform/rocFFT/wiki/Build>`_ has helpful information on how to configure cmake and manually build.

Library and API Documentation
Please refer to the Library documentation for current documentation.


Example
###############
The following is a simple example code that shows how to use rocFFT to compute a 1D single precision 16-point complex forward transform.

::

  #include <iostream>
  #include <vector>
  #include "hip/hip_runtime_api.h"
  #include "hip/hip_vector_types.h"
  #include "rocfft.h"

  int main()
  {
          // rocFFT gpu compute
          // ========================================

          size_t N = 16;
          size_t Nbytes = N * sizeof(float2);

          // Create HIP device buffer
          float2 *x;
          hipMalloc(&x, Nbytes);

          // Initialize data
          std::vector<float2> cx(N);
          for (size_t i = 0; i < N; i++)
          {
                  cx[i].x = 1;
                  cx[i].y = -1;
          }

          //  Copy data to device
          hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);

          // Create rocFFT plan
          rocfft_plan plan = NULL;
          size_t length = N;
          rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1,   			&length, 1, NULL);

          // Execute plan
          rocfft_execute(plan, (void**) &x, NULL, NULL);

          // Wait for execution to finish
          hipDeviceSynchronize();

          // Destroy plan
          rocfft_plan_destroy(plan);

          // Copy result back to host
          std::vector<float2> y(N);
          hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);
 
          // Print results
          for (size_t i = 0; i < N; i++)
          {
                  std::cout << y[i].x << ", " << y[i].y << std::endl;
          }

          // Free device buffer
          hipFree(x);

          return 0;
    }






******************
rocBLAS
******************


 * `rocBLAS Github link <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_

A BLAS implementation on top of AMD's Radeon Open Compute `ROCm <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html>`_ runtime and toolchains. rocBLAS is implemented in the `HIP <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hip-programing-guide>`_ programming language and optimized for AMD's latest discrete GPUs.

Installing pre-built packages
##############################
Download pre-built packages either from `ROCm's package servers <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-ubuntu>`_ or by clicking the github releases tab and manually downloading, which could be newer. Release notes are available for each release on the releases tab.
::
  sudo apt update && sudo apt install rocblas

Quickstart rocBLAS build
###########################

**Bash helper build script (Ubuntu only)**

The root of this repository has a helper bash script install.sh to build and install rocBLAS on Ubuntu with a single command. It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install. A few commands in the script need sudo access, so it may prompt you for a password.
::
  ./install -h -- shows help
  ./install -id -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

Manual build (all supported platforms)
#########################################
If you use a distro other than Ubuntu, or would like more control over the build process, the :ref:`rocblaswiki` has helpful information on how to configure cmake and manually build.

**Functions supported**

A list of exported functions from rocblas can be found on the wiki

rocBLAS interface examples
#############################
In general, the rocBLAS interface is compatible with CPU oriented `Netlib BLAS <http://www.netlib.org/blas/>`_ and the cuBLAS-v2 API, with the explicit exception that traditional BLAS interfaces do not accept handles. The cuBLAS' cublasHandle_t is replaced with rocblas_handle everywhere. Thus, porting a CUDA application which originally calls the cuBLAS API to a HIP application calling rocBLAS API should be relatively straightforward. For example, the rocBLAS SGEMV interface is

GEMV API
###############

::

  rocblas_status
  rocblas_sgemv(rocblas_handle handle,
                rocblas_operation trans,
                rocblas_int m, rocblas_int n,
                const float* alpha,
                const float* A, rocblas_int lda,
                const float* x, rocblas_int incx,
                const float* beta,
                float* y, rocblas_int incy);

Batched and strided GEMM API
################################
rocBLAS GEMM can process matrices in batches with regular strides. There are several permutations of these API's, the following is an example that takes everything

::

  rocblas_status
  rocblas_sgemm_strided_batched(
      rocblas_handle handle,
      rocblas_operation transa, rocblas_operation transb,
      rocblas_int m, rocblas_int n, rocblas_int k,
      const float* alpha,
      const float* A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
      const float* B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
      const float* beta,
            float* C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
      rocblas_int batch_count )

rocBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data. Users are responsible for copying data from/to the host and device memory. HIP provides memcpy style API's to facilitate data management.

Asynchronous API
###################
Except a few routines (like TRSM) having memory allocation inside preventing asynchronicity, most of the library routines (like BLAS-1 SCAL, BLAS-2 GEMV, BLAS-3 GEMM) are configured to operate in asynchronous fashion with respect to CPU, meaning these library functions return immediately.

For more information regarding rocBLAS library and corresponding API documentation, refer `rocBLAS <https://rocblas.readthedocs.io/en/latest/>`_


************
hipBLAS
************

Please Refer here for Github link `hipBLAS <https://github.com/ROCmSoftwarePlatform/hipBLAS>`_

hipBLAS is a BLAS marshalling library, with multiple supported backends. It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application. hipBLAS exports an interface that does not require the client to change, regardless of the chosen backend. Currently, hipBLAS supports :ref:`rocblas` and `cuBLAS <https://developer.nvidia.com/cublas>`_ as backends.

Installing pre-built packages
#################################

Download pre-built packages either from ROCm's package servers or by clicking the github releases tab and manually downloading, which could be newer. Release notes are available for each release on the releases tab.
::
  sudo apt update && sudo apt install hipblas

Quickstart hipBLAS build
#############################
**Bash helper build script (Ubuntu only)**

The root of this repository has a helper bash script install.sh to build and install hipBLAS on Ubuntu with a single command. It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install. A few commands in the script need sudo access, so it may prompt you for a password.
::
  ./install -h -- shows help
  ./install -id -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

**Manual build (all supported platforms)**

If you use a distro other than Ubuntu, or would like more control over the build process, the hipblas build wiki has helpful information on how to configure cmake and manually build.

**Functions supported**

A list of exported functions from hipblas can be found on the wiki

hipBLAS interface examples
######################################
The hipBLAS interface is compatible with rocBLAS and cuBLAS-v2 APIs. Porting a CUDA application which originally calls the cuBLAS API to an application calling hipBLAS API should be relatively straightforward. For example, the hipBLAS SGEMV interface is

GEMV API
######################################
::

  hipblasStatus_t
  hipblasSgemv( hipblasHandle_t handle,
               hipblasOperation_t trans,
               int m, int n, const float *alpha,
               const float *A, int lda,
               const float *x, int incx, const float *beta,
               float *y, int incy );

Batched and strided GEMM API
######################################
hipBLAS GEMM can process matrices in batches with regular strides. There are several permutations of these API's, the following is an example that takes everything

:: 

  hipblasStatus_t
  hipblasSgemmStridedBatched( hipblasHandle_t handle,
               hipblasOperation_t transa, hipblasOperation_t transb,
               int m, int n, int k, const float *alpha,
               const float *A, int lda, long long bsa,
               const float *B, int ldb, long long bsb, const float *beta,
               float *C, int ldc, long long bsc,
               int batchCount);

hipBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data. Users are responsible for copying data from/to the host and device memory.





**********
hcRNG
**********

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
#################

**Installation steps**

The following are the steps to use the library

  * ROCM 2.0 Kernel, Driver and Compiler Installation (if not done until now)
  * Library installation.

**ROCM 2.0 Installation**

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
AMD is hosting both debian and rpm repositories for the ROCm 2.0 packages. The packages in both repositories have been signed to ensure package integrity. Directions for each repository are given below:

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


Note: All direct tests should pass with ROCm2.0





*************
clFFT
*************

For Github Repository `clFFT <https://github.com/clMathLibraries/clFFT>`_

clFFT is a software library containing FFT functions written in OpenCL. In addition to GPU devices, the library also supports running on CPU devices to facilitate debugging and heterogeneous programming.

Pre-built binaries are available here.

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



***************
clBLAS
***************


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


*************
hcFFT
*************


Installation
###############

The following are the steps to use the library

 * ROCM 2.0 Kernel, Driver and Compiler Installation (if not done until now)
 * Library installation.

**ROCM 2.0 Installation**

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




*****************
Tensile
*****************

A tool for creating a benchmark-driven backend library for GEMMs, GEMM-like problems (such as batched GEMM), N-dimensional tensor contractions, and anything else that multiplies two multi-dimensional objects together on a GPU.

Overview for creating a custom TensileLib backend library for your application:

1. Install Tensile (optional), or at least install the PyYAML dependency (mandatory).
2. Create a benchmark config.yaml file.
3. Run the benchmark to produce a library logic.yaml file.
4. Add the Tensile library to your application's CMake target. The Tensile library will be written, compiled and linked to your application at application-compile-time.

    * GPU kernels, written in HIP or OpenCL.
    * Solution classes which enqueue the kernels.
    * APIs which call the fastest solution for a problem.

**Quick Example:**
****************

::

  sudo apt-get install python-yaml
  mkdir Tensile
  cd Tensile
  git clone https://github.com/RadeonOpenCompute/Tensile.git repo
  mkdir build
  cd build
  python ../repo/Tensile/Tensile.py ../repo/Tensile/Configs/sgemm_5760.yaml ./

After a while of benchmarking, Tensile will print out the path to the client you can run.

::

  ./4_LibraryClient/build/client -h
  ./4_LibraryClient/build/client --sizes 5760 5760 5760


Benchmark Config
####################

Example Benchmark config.yaml

:: 

  GlobalParameters:
    PrintLevel: 1
    ForceRedoBenchmarkProblems: False
    ForceRedoLibraryLogic: True
    ForceRedoLibraryClient: True
    CMakeBuildType: Release
    EnqueuesPerSync: 1
    SyncsPerBenchmark: 1
    LibraryPrintDebug: False
    NumElementsToValidate: 128
    ValidationMaxToPrint: 16
    ValidationPrintValids: False
    ShortNames: False
    MergeFiles: True
    PlatformIdx: 0
    DeviceIdx: 0
    DataInitTypeAB: 0

  BenchmarkProblems:
    - # sgemm NN
      - # ProblemType
        OperationType: GEMM
        DataType: s
        TransposeA: False
        TransposeB: False
        UseBeta: True
        Batched: False

      - # BenchmarkProblemSizeGroup
        InitialSolutionParameters:
        BenchmarkCommonParameters:
          - ProblemSizes:
            - Range: [ [5760], 0, 0 ]
          - LoopDoWhile: [False]
          - NumLoadsCoalescedA: [-1]
          - NumLoadsCoalescedB: [1]
          - WorkGroupMapping: [1]
        ForkParameters:
           - ThreadTile:
           - [ 8, 8 ]
           - [ 4, 8 ]
           - [ 4, 4 ]
          - WorkGroup:
            - [  8, 16,  1 ]
            - [ 16, 16,  1 ]
          - LoopTail: [False, True]
          - EdgeType: ["None", "Branch", "ShiftPtr"]
          - DepthU: [ 8, 16]
          - VectorWidth: [1, 2, 4]
        BenchmarkForkParameters:
        JoinParameters:
          - MacroTile
        BenchmarkJoinParameters:
        BenchmarkFinalParameters:
          - ProblemSizes:
            - Range: [ [5760], 0, 0 ]

  LibraryLogic:

  LibraryClient:


**Structure of config.yaml**


Top level data structure whose keys are Parameters, BenchmarkProblems, LibraryLogic and LibraryClient.

 * Parameters contains a dictionary storing global parameters used for all parts of the benchmarking.
 * BenchmarkProblems contains a list of dictionaries representing the benchmarks to conduct; each element, i.e. dictionary, in the list is for benchmarking a single ProblemType. The keys for these dictionaries are ProblemType, InitialSolutionParameters, 	     	BenchmarkCommonParameters, ForkParameters, BenchmarkForkParameters, JoinParameters, BenchmarkJoinParameters and 		     	BenchmarkFinalParameters. See Benchmark Protocol for more information on these steps.
 * LibraryLogic contains a dictionary storing parameters for analyzing the benchmark data and designing how the backend library will select which Solution for certain ProblemSizes.
 * LibraryClient contains a dictionary storing parameters for actually creating the library and creating a client which calls into the library.

**Global Parameters**


* Name: Prefix to add to API function names; typically name of device.
* MinimumRequiredVersion: Which version of Tensile is required to interpret this yaml file
* RuntimeLanguage: Use HIP or OpenCL runtime.
* KernelLanguage: For OpenCL runtime, kernel language must be set to OpenCL. For HIP runtime, kernel language can be set to HIP or assembly (gfx803, gfx900).
* PrintLevel: 0=Tensile prints nothing, 1=prints some, 2=prints a lot.
* ForceRedoBenchmarkProblems: False means don't redo a benchmark phase if results for it already exist.
* ForceRedoLibraryLogic: False means don't re-generate library logic if it already exist.
* ForceRedoLibraryClient: False means don't re-generate library client if it already exist.
* CMakeBuildType: Release or Debug
* EnqueuesPerSync: Num enqueues before syncing the queue.
* SyncsPerBenchmark: Num queue syncs for each problem size.
* LibraryPrintDebug: True means Tensile solutions will print kernel enqueue info to stdout
* NumElementsToValidate: Number of elements to validate; 0 means no validation.
* ValidationMaxToPrint: How many invalid results to print.
* ValidationPrintValids: True means print validation comparisons that are valid, not just invalids.
* ShortNames: Convert long kernel, solution and files names to short serial ids.
* MergeFiles: False means write each solution and kernel to its own file.
* PlatformIdx: OpenCL platform id.
* DeviceIdx: OpenCL or HIP device id.
* DataInitType[AB,C]: Initialize validation data with 0=0's, 1=1's, 2=serial, 3=random.
* KernelTime: Use kernel time reported from runtime rather than api times from cpu clocks to compare kernel performance.

The exhaustive list of global parameters and their defaults is stored in Common.py.

**Problem Type Parameters**

* OperationType: GEMM or TensorContraction.
* DataType: s, d, c, z, h
* UseBeta: False means library/solutions/kernel won't accept a beta parameter; thus beta=0.
* UseInitialStrides: False means data is contiguous in memory.
* HighPrecisionAccumulate: For tmpC += a*b, use twice the precision for tmpC as for DataType. Not yet implemented.
* ComplexConjugateA: True or False; ignored for real precision.
* ComplexConjugateB: True or False; ignored for real precision.

For OperationType=GEMM only:
* TransposeA: True or False.
* TransposeB: True or False.
* Batched: True or False.

For OperationType=TensorContraction only (showing batched gemm NT: C[ijk] = Sum[l] A[ilk] * B[jlk])
* IndexAssignmentsA: [0, 3, 2]
* IndexAssignmentsB: [1, 3, 2]
* NumDimensionsC: 3.

**Defaults**

Because of the flexibility / complexity of the benchmarking process and, therefore, of the config.yaml files; Tensile has a default value for every parameter. If you neglect to put LoopUnroll anywhere in your benchmark, rather than crashing or complaining, Tensile will put the default LoopUnroll options into the default phase (common, fork, join...). This guarantees ease of use and more importantly backward compatibility; every time we add a new possible solution parameter, you don't necessarily need to update your configs; we'll have a default figured out for you.

However, this may cause some confusion. If your config fork 2 parameters, but you see that 3 were forked during benchmarking, that's because you didn't specify the 3rd parameter anywhere, so Tensile stuck it in its default phase, which was forking (for example). Also, specifying ForkParameters: and leaving it empty isn't the same as leaving JoinParameter out of your config. If you leave ForkParameters out of your config, Tensile will add a ForkParameters step and put the default parameters into it (unless you put all the parameters elsewhere), but if you specify ForkParameters and leave it empty, then you won't work anything.

Therefore, it is safest to specify all parameters in your config.yaml files; that way you'll guarantee the behavior you want. See /Tensile/Common.py for the current list of parameters.

Benchmark Protocol
###############################

**Old Benchmark Architecture was Intractable**

The benchmarking strategy from version 1 was vanilla flavored brute force: 
 | ``(8 WorkGroups)* (12 ThreadTiles)* (4 NumLoadsCoalescedAs)*``
 | ``(4 NumLoadsCoalescedBs)* (3 LoopUnrolls)* (5 BranchTypes)* ...*(1024 ProblemSizes)=23,592,960`` is a multiplicative series 
which grows very quickly.Adding one more boolean parameter doubles the number of kernel enqueues of the benchmark.

**Incremental Benchmark is Faster**

Tensile version 2 allows the user to manually interrupt the multiplicative series with "additions" instead of "multiplies", i.e., 
 | ``(8 WorkGroups)* (12 ThreadTiles)+ (4 NumLoadsCoalescedAs)*``
 | ``(4 NumLoadsCoalescedBs)*(3 LoopUnrolls)+ (5 BranchTypes)* ...+(1024 ProblemSizes)=1,151``  is a dramatically smaller number of enqueues.Now, adding one more boolean parameter may only add on 2 more enqueues.

**Phases of Benchmark**

To make the Tensile's programability more manageable for the user and developer, the benchmarking protocol has been split up into several steps encoded in a config.yaml file. The below sections reference the following config.yaml. Note that this config.yaml has been created to be a simple illustration and doesn't not represent an actual good benchmark protocol. See the configs included in the repository (/Tensile/Configs) for examples of good benchmarking configs.

::

  BenchmarkProblems:
   - # sgemm
     - # Problem Type
       OperationType: GEMM
     - # Benchmark Size-Group
      InitialSolutionParameters:
        - WorkGroup: [ [ 16, 16, 1 ] ]
        - NumLoadsCoalescedA: [ 1 ]
        - NumLoadsCoalescedB: [ 1 ]
        - ThreadTile: [ [ 4, 4 ] ]

      BenchmarkCommonParameters:
        - ProblemSizes:
          - Range: [ [512], [512], [512] ]
        - EdgeType: ["Branch", "ShiftPtr"]
          PrefetchGlobalRead: [False, True]

      ForkParameters:
        - WorkGroup: [ [8, 32, 1], [16, 16, 1], [32, 8, 1] ]
          ThreadTile: [ [2, 8], [4, 4], [8, 2] ]

      BenchmarkForkParameters:
        - ProblemSizes:
          - Exact: [ 2880, 2880, 2880 ]
        - NumLoadsCoalescedA: [ 1, 2, 4, 8 ]
        - NumLoadsCoalescedB: [ 1, 2, 4, 8 ]

      JoinParameters:
        - MacroTile

      BenchmarkJoinParameters:
        - LoopUnroll: [8, 16]

      BenchmarkFinalParameters:
        - ProblemSizes:
          - Range: [ [16, 128], [16, 128], [256] ]


**Initial Solution Parameters**

A Solution is comprised of ~20 parameters, and all are needed to create a kernel. Therefore, during the first benchmark which determines which WorkGroupShape is fastest, what are the other 19 solution parameters which are used to describe the kernels that we benchmark? That's what InitialSolutionParameters are for. The solution used for benchmarking WorkGroupShape will use the parameters from InitialSolutionParameters. The user must choose good default solution parameters in order to correctly identify subsequent optimal parameters.

**Problem Sizes**

Each step of the benchmark can override what problem sizes will be benchmarked. A ProblemSizes entry of type Range is a list whose length is the number of indices in the ProblemType. A GEMM ProblemSizes must have 3 elements while a batched-GEMM ProblemSizes must have 4 elements. So, for a ProblemType of C[ij] = Sum[k] A[ik]*B[jk], the ProblemSizes elements represent [SizeI, SizeJ, SizeK]. For each index, there are 5 ways of specifying the sizes of that index:

 1.[1968]
  * Benchmark only size 1968; n = 1.
  
 2.[16, 1920]
  * Benchmark sizes 16 to 1968 using the default step size (=16); n = 123.
 
 3.[16, 32, 1968]
  * Benchmark sizes 16 to 1968 using a step size of 32; n = 61.
 
 4.[64, 32, 16, 1968]
  * Benchmark sizes from 64 to 1968 with a step size of 32. Also, increase the step size by 16 each iteration.
  * This causes fewer sizes to be benchmarked when the sizes are large, and more benchmarks where the sizes are small; this is 	      	typically desired behavior.
  * n = 16 (64, 96, 144, 208, 288, 384, 496, 624, 768, 928, 1104, 1296, 1504, 1728, 1968). The stride at the beginning is 32, but     	the stride at the end is 256.
 
 5.[0]
  * The size of this index is just whatever size index 0 is. For a 3-dimensional ProblemType, this allows benchmarking only a 2- 	      	dimensional or 1-dimensional slice of problem sizes.

Here are a few examples of valid ProblemSizes for 3D GEMMs:

Range: [ [16, 128], [16, 128], [16, 128] ] # n = 512
Range: [ [16, 128], 0, 0] # n = 8
Range: [ [16, 16, 16, 5760], 0, [1024, 1024, 4096] ] # n = 108

Benchmark Common Parameters
**************************************
During this first phase of benchmarking, we examine parameters which will be the same for all solutions for this ProblemType. During each step of benchmarking, there is only 1 winner. In the above example we are benchmarking the dictionary {EdgeType: [ Branch, ShiftPtr], PrefetchGlobalRead: [False, True]}.; therefore, this benchmark step generates 4 solution candidates, and the winner will be the fastest EdgeType/PrefetchGlobalRead combination. Assuming the winner is ET=SP and PGR=T, then all solutions for this ProblemType will have ET=SP and PGR=T. Also, once a parameter has been determined, all subsequent benchmarking steps will use this determined parameter rather than pulling values from InitialSolutionParameters. Because the common parameters will apply to all kernels, they are typically the parameters which are compiler-dependent or hardware-dependent rather than being tile-dependent.

**Fork Parameters**
****************************
If we continued to determine every parameter in the above manner, we'd end up with a single fastest solution for the specified ProblemSizes; we usually desire multiple different solutions with varying parameters which may be fastest for different groups of ProblemSizes. One simple example of this is small tiles sizes are fastest for small problem sizes, and large tiles are fastest for large tile sizes.

Therefore, we allow "forking" parameters; this means keeping multiple winners after each benchmark steps. In the above example we fork {WorkGroup: [...], ThreadTile: [...]}. This means that in subsequent benchmarking steps, rather than having one winning parameter, we'll have one winning parameter per fork permutation; we'll have 9 winners.

**Benchmark Fork Parameters**

When we benchmark the fork parameters, we retain one winner per permutation. Therefore, we first determine the fastest NumLoadsCoalescedA for each of the WG,TT permutations, then we determine the fastest NumLoadsCoalescedB for each permutation.

Join Parameters
*******************
After determining fastest parameters for all the forked solution permutations, we have the option of reducing the number of winning solutions. When a parameter is listed in the JoinParameters section, that means that of the kept winning solutions, each will have a different value for that parameter. Listing more parameters to join results in more winners being kept, while having a JoinParameters section with no parameters listed results on only 1 fastest solution.

In our example we join over the MacroTile (work-group x thread-tile). After forking tiles, there were 9 solutions that we kept. After joining MacroTile, we'll only keep six: 16x256, 32x128, 64x64, 128x32 and 256x16. The solutions that are kept are based on their performance during the last BenchmarkForkParameters benchmark, or, if there weren't any, JoinParameters will conduct a benchmark of all solution candidates then choose the fastest.

**Benchmark Join Parameters**

After narrowing the list of fastest solutions through joining, you can continue to benchmark parameters, keeping one winning parameter per solution permutation.

Benchmark Final Parameters
********************************
After all the parameter benchmarking has been completed and the final list of fastest solution has been assembled, we can benchmark all the solution over a large set of ProblemSizes. This benchmark represent the final output of benchmarking; it outputs a .csv file where the rows are all the problem sizes and the columns are all the solutions. This is the information which gets analysed to produce the library logic.



Dependencies
##################

**CMake**

  * CMake 2.8

**Python**

   * Python 2.7
   * PyYAML (Can be installed via apt, apt-get, yum, pip...; module is typically named python-yaml, pyyaml or PyYAML.)

**Compilers**

 * For Tensile_BACKEND = OpenCL1.2
      * Visual Studio 14 (2015). (VS 2012 may also be supported; c++11 should no longer be required by Tensile. Need to verify.)
      * GCC 4.8
 * For Tensile_BACKEND = HIP
      * ROCM 2.0

**Installation**


Tensile can be installed via:

1. Install directly from repo using pip:

::

   pip install git+https://github.com/RadeonOpenCompute/Tensile.git@develop
   tensile config.yaml benchmark_path


2. Download repo and install manually:

::

  git clone https://github.com/RadeonOpenCompute/Tensile.git
  cd Tensile
  sudo python setup.py install
  tensile config.yaml benchmark_path

3. Download repo and don't install; install PyYAML dependency manually and call python scripts manually:

::

   git clone https://github.com/RadeonOpenCompute/Tensile.git 
   python Tensile/Tensile/Tensile.py config.yaml benchmark_path


Kernel Parameters
#####################

Solution / Kernel Parameters
*********************************

* LoopDoWhile: True=DoWhile loop, False=While or For loop
* LoopTail: Additional loop with LoopUnroll=1.
* EdgeType: Branch, ShiftPtr or None
* WorkGroup: [dim0, dim1, LocalSplitU]
* ThreadTile: [dim0, dim1]
* GlobalSplitU: Split up summation among work-groups to create more concurrency. This option launches a kernel to handle the beta     	scaling, then a second kernel where the writes to global memory are atomic.
* PrefetchGlobalRead: True means outer loop should prefetch global data one iteration ahead.
* PrefetchLocalRead: True means inner loop should prefetch lds data one iteration ahead.
* WorkGroupMapping: In what order will work-groups compute C; affects cacheing.
* LoopUnroll: How many iterations to unroll inner loop; helps loading coalesced memory.
* MacroTile: Derrived from WorkGroup*ThreadTile.
* DepthU: Derrived from LoopUnroll*SplitU.
* NumLoadsCoalescedA,B: Number of loads from A in coalesced dimension.
* GlobalReadCoalesceGroupA,B: True means adjacent threads map to adjacent global read elements (but, if transposing data then write   	to lds is scattered).
* GlobalReadCoalesceVectorA,B: True means vector components map to adjacent global read elements (but, if transposing data then write 	to lds is scattered).
* VectorWidth: Thread tile elements are contiguous for faster memory accesses. For example VW=4 means a thread will read a float4     	 from memory rather than 4 non-contiguous floats.

The exhaustive list of solution parameters and their defaults is stored in Common.py.

Kernel Parameters Affect Performance
****************************************
The kernel parameters affect many aspects of performance. Changing a parameter may help address one performance bottleneck but worsen another. That is why searching through the parameter space is vital to discovering the fastest kernel for a given problem.



 .. image:: img1.png
     :align: center
   
**How N-Dimensional Tensor Contractions Are Mapped to Finite-Dimensional GPU Kernels**

For a traditional GEMM, the 2-dimensional output, C[i,j], is mapped to launching a 2-dimensional grid of work groups, each of which has a 2-dimensional grid of work items; one dimension belongs to i and one dimension belongs to j. The 1-dimensional summation is represented by a single loop within the kernel body.

**Special Dimensions: D0, D1 and DU**

To handle arbitrary dimensionality, Tensile begins by determining 3 special dimensions: D0, D1 and DU.

D0 and D1 are the free indices of A and B (one belongs to A and one to B) which have the shortest strides. This allows the inner-most loops to read from A and B the fastest via coalescing. In a traditional GEMM, every matrix has a dimension with a shortest stride of 1, but Tensile doesn't make that assumption. Of these two dimensions, D0 is the dimension which has the shortest tensor C stride which allows for fast writing.

DU represents the summation index with the shortest combined stride (stride in A + stride in B); it becomes the inner most loop which gets "U"nrolled. This assignment is also mean't to assure fast reading in the inner-most summation loop. There can be multiple summation indices (i.e. embedded loops) and DU will be iterated over in the inner most loop.

GPU Kernel Dimension
************************
OpenCL allows for 3-dimensional grid of work-groups, and each work-group can be a 3-dimensional grid of work-items. Tensile assigns D0 to be dimension-0 of the work-group and work-item grid; it assigns D1 to be dimension-1 of the work-group and work-item grids. All other free or batch dimensions are flattened down into the final dimension-2 of the work-group and work-item grids. Withing the GPU kernel, dimensions-2 is reconstituted back into whatever dimensions it represents.


Languages
##################

**Tensile Benchmarking is Python**

The benchmarking module, Tensile.py, is written in python. The python scripts write solution, kernels, cmake files and all other C/C++ files used for benchmarking.

**Tensile Library**

The Tensile API, Tensile.h, is confined to C89 so that it will be usable by most software. The code behind the API is allowed to be c++11.

**Device Languages**

The device languages Tensile supports for the gpu kernels is

* OpenCL 1.2
* HIP
* Assembly
   * gfx803 
   * gfx900

**Library Logic**

Running the LibraryLogic phase of benchmarking analyses the benchmark data and encodes a mapping for each problem type. For each problem type, it maps problem sizes to best solution (i.e. kernel).

When you build Tensile.lib, you point the TensileCreateLibrary function to a directory where your library logic yaml files are.

Problem Nomenclature
########################

**Example Problems**


* C[i,j] = Sum[k] A[i,k] * B[k,j] (GEMM; 2 free indices and 1 summation index)
* C[i,j,k] = Sum[l] A[i,l,k] * B[l,j,k] (batched-GEMM; 2 free indices, 1 batched index and 1 summation index)
* C[i,j] = Sum[k,l] A[i,k,l] * B[j,l,k] (2D summation)
* C[i,j,k,l,m] = Sum[n] A[i,k,m,l,n] * B[j,k,l,n,m] (GEMM with 3 batched indices)
* C[i,j,k,l,m] = Sum[n,o] A[i,k,m,o,n] * B[j,m,l,n,o] (4 free indices, 2 summation indices and 1 batched index)
* C[i,j,k,l] = Sum[m,n] A[i,j,m,n,l] * B[m,n,k,j,l] (batched image convolution mapped to 7D tensor contraction)
* and even crazier

**Nomenclature**


The indices describe the dimensionality of the problem being solved. A GEMM operation takes 2 2-dimensional matrices as input (totaling 4 input dimensions) and contracts them along one dimension (which cancels out 2 of the dimensions), resulting in a 2-dimensional result.

Whenever an index shows up in multiple tensors, those tensors must be the same size along that dimension but they may have different strides.

There are 3 categories of indices/dimensions that Tensile deals with: free, batch and bound.

**Free Indices**

Free indices are the indices of tensor C which come in pairs; one of the pair shows up in tensor A while the other shows up in tensor B. In the really crazy example above, i/j/k/l are the 4 free indices of tensor C. Indices i and k come from tensor A and indices j and l come from tensor B.

**Batch Indices**

Batch indices are the indices of tensor C which shows up in both tensor A and tensor B. For example, the difference between the GEMM example and the batched-GEMM example above is the additional index. In the batched-GEMM example, the index K is the batch index which is batching together multiple independent GEMMs.

**Bound/Summation Indices**

The final type of indices are called bound indices or summation indices. These indices do not show up in tensor C; they show up in the summation symbol (Sum[k]) and in tensors A and B. It is along these indices that we perform the inner products (pairwise multiply then sum).

**Limitations**

Problem supported by Tensile must meet the following conditions:

There must be at least one pair of free indices.

Tensile.lib
########################

After running the benchmark and generating library config files, you're ready to add Tensile.lib to your project. Tensile provides a TensileCreateLibrary function, which can be called:

::

  set(Tensile_BACKEND "HIP")
  set( Tensile_LOGIC_PATH "~/LibraryLogic" CACHE STRING "Path to Tensile logic.yaml files")
  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" OFF)
  option( Tensile_SHORT_NAMES "Tensile to use short file/function names? Use if compiler complains they're too long." OFF)
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF)

  find_package(Tensile) # use if Tensile has been installed

  TensileCreateLibrary(
    ${Tensile_LOGIC_PATH}
    ${Tensile_BACKEND}
    ${Tensile_MERGE_FILES}
    ${Tensile_SHORT_NAMES}
    ${Tensile_PRINT_DEBUG}
    Tensile_ROOT ${Tensile_ROOT} # optional; use if tensile not installed
    )
  target_link_libraries( TARGET Tensile )


**Versioning**


Tensile follows semantic versioning practices, i.e. Major.Minor.Patch, in BenchmarkConfig.yaml files, LibraryConfig.yaml files and in cmake find_package. Tensile is compatible with a "MinimumRequiredVersion" if Tensile.Major==MRV.Major and Tensile.Minor.Patch >= MRV.Minor.Patch.

* Major: Tensile increments the major version if the public API changes, or if either the benchmark.yaml or library-config.yaml files 	change format in a non-backwards-compatible manner.
* Minor: Tensile increments the minor version when new kernel, solution or benchmarking features are introduced in a backwards-	      	compatible manner.
* Patch: Bug fixes or minor improvements.

***************
rocALUTION
***************

Introduction
############

Overview
########
rocALUTION is a sparse linear algebra library with focus on exploring fine-grained parallelism, targeting modern processors and accelerators including multi/many-core CPU and GPU platforms. The main goal of this package is to provide a portable library for iterative sparse methods on state of the art hardware. rocALUTION can be seen as middle-ware between different parallel backends and application specific packages.

The major features and characteristics of the library are

* Various backends
    * Host - fallback backend, designed for CPUs
    * GPU/HIP - accelerator backend, designed for HIP capable AMD GPUs
    * OpenMP - designed for multi-core CPUs
    * MPI - designed for multi-node and multi-GPU configurations
* Easy to use
    The syntax and structure of the library provide easy learning curves. With the help of the examples, anyone can try out the library - no knowledge in HIP, OpenMP or MPI programming required.
* No special hardware requirements
    There are no hardware requirements to install and run rocALUTION. If a GPU device and HIP is available, the library will use them.
* Variety of iterative solvers
    * Fixed-Point iteration - Jacobi, Gauss-Seidel, Symmetric-Gauss Seidel, SOR and SSOR
    * Krylov subspace methods - CR, CG, BiCGStab, BiCGStab(l), GMRES, IDR, QMRCGSTAB, Flexible CG/GMRES
    * Mixed-precision defect-correction scheme
    * Chebyshev iteration
    * Multiple MultiGrid schemes, geometric and algebraic
* Various preconditioners
    * Matrix splitting - Jacobi, (Multi-colored) Gauss-Seidel, Symmetric Gauss-Seidel, SOR, SSOR
    * Factorization - ILU(0), ILU(p) (based on levels), ILU(p,q) (power(q)-pattern method), Multi-Elimination ILU (nested/recursive), ILUT (based on threshold) and IC(0)
    * Approximate Inverse - Chebyshev matrix-valued polynomial, SPAI, FSAI and TNS
    * Diagonal-based preconditioner for Saddle-point problems
    * Block-type of sub-preconditioners/solvers
    * Additive Schwarz and Restricted Additive Schwarz
    * Variable type preconditioners
* Generic and robust design
    rocALUTION is based on a generic and robust design allowing expansion in the direction of new solvers and preconditioners and support for various hardware types. Furthermore, the design of the library allows the use of all solvers as preconditioners in other solvers. For example you can easily define a CG solver with a Multi-Elimination preconditioner, where the last-block is preconditioned with another Chebyshev iteration method which is preconditioned with a multi-colored Symmetric Gauss-Seidel scheme.
* Portable code and results
    All code based on rocALUTION is portable and independent of HIP or OpenMP. The code will compile and run everywhere. All solvers and preconditioners are based on a single source code, which delivers portable results across all supported backends (variations are possible due to different rounding modes on the hardware). The only difference which you can see for a hardware change is the performance variation.
* Support for several sparse matrix formats
    Compressed Sparse Row (CSR), Modified Compressed Sparse Row (MCSR), Dense (DENSE), Coordinate (COO), ELL, Diagonal (DIA), Hybrid format of ELL and COO (HYB).

The code is open-source under MIT license and hosted on here: https://github.com/ROCmSoftwarePlatform/rocALUTION

Building and Installing
#######################

Installing from AMD ROCm repositories
#####################################
TODO, not yet available

Building rocALUTION from Open-Source repository
###############################################

Download rocALUTION
###################
The rocALUTION source code is available at the `rocALUTION github page <https://github.com/ROCmSoftwarePlatform/rocALUTION>`_.
Download the master branch using:

::

  git clone -b master https://github.com/ROCmSoftwarePlatform/rocALUTION.git
  cd rocALUTION


Note that if you want to contribute to rocALUTION, you will need to checkout the develop branch instead of the master branch. See :ref:`rocalution_contributing` for further details.
Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocALUTION using the *install.sh* script.

Using *install.sh* to build dependencies + library
##################################################
The following table lists common uses of *install.sh* to build dependencies + library. Accelerator support via HIP and OpenMP will be enabled by default, whereas MPI is disabled.

===================== ====
Command               Description
===================== ====
`./install.sh -h`     Print help information.
`./install.sh -d`     Build dependencies and library in your local directory. The `-d` flag only needs to be |br| used once. For subsequent invocations of *install.sh* it is not necessary to rebuild the |br| dependencies.
`./install.sh`        Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i`     Build library, then build and install rocALUTION package in `/opt/rocm/rocalution`. You will |br| be prompted for sudo access. This will install for all users.
`./install.sh --host` Build library in your local directory without HIP support. It is assumed dependencies |br| are available.
`./install.sh --mpi`  Build library in your local directory with HIP and MPI support. It is assumed |br| dependencies are available.
===================== ====

Using *install.sh* to build dependencies + library + client
###########################################################
The client contains example code, unit tests and benchmarks. Common uses of *install.sh* to build them are listed in the table below.

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to |br| be used once. For subsequent invocations of *install.sh* it is not necessary to rebuild the |br| dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install rocALUTION package in |br| `/opt/rocm/rocalution`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install rocALUTION package in |br| `opt/rocm/rocalution`. You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build rocALUTION
#############################################
CMake 3.5 or later is required in order to build rocALUTION.

rocALUTION can be built with cmake using the following commands:

::

  # Create and change to build directory
  mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  cmake ../.. -DSUPPORT_HIP=ON \
              -DSUPPORT_MPI=OFF \
              -DSUPPORT_OMP=ON

  # Compile rocALUTION library
  make -j$(nproc)

  # Install rocALUTION to /opt/rocm
  sudo make install

GoogleTest is required in order to build rocALUTION client.

rocALUTION with dependencies and client can be built using the following commands:

::

  # Install googletest
  mkdir -p build/release/deps ; cd build/release/deps
  cmake ../../../deps
  sudo make -j$(nproc) install

  # Change to build directory
  cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
              -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocALUTION library
  make -j$(nproc)

  # Install rocALUTION to /opt/rocm
  sudo make install

The compilation process produces a shared library file *librocalution.so* and *librocalution_hip.so* if HIP support is enabled. Ensure that the library objects can be found in your library path. If you do not copy the library to a specific location you can add the path under Linux in the *LD_LIBRARY_PATH* variable.

::

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_rocalution>

Common build problems
#####################
#. **Issue:** HIP (/opt/rocm/hip) was built using hcc 1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/bin/hcc with version 1.0.yyy-yyy-yyy-yyy from hipcc (version mismatch). Please rebuild HIP including cmake or update HCC_HOME variable.

   **Solution:** Download HIP from github and use hcc to `build from source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and then use the built HIP instead of /opt/rocm/hip.

#. **Issue:** For Carrizo - HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** Add the following to the cmake command when configuring: `-DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801"`

#. **Issue:** For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** `export HCC_AMDGPU_TARGET=gfx900`

#. **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names:
              ROCMConfig.cmake |br|
              rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_

#. **Issue:** Could not find a package configuration file provided by "ROCSPARSE" with any of the following names:
              ROCSPARSE.cmake |br|
              rocsparse-config.cmake

   **Solution:** Install `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_

#. **Issue:** Could not find a package configuration file provided by "ROCBLAS" with any of the following names:
              ROCBLAS.cmake |br|
              rocblas-config.cmake

   **Solution:** Install `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_

Simple Test
###########
You can test the installation by running a CG solver on a Laplace matrix. After compiling the library you can perform the CG solver test by executing

::

  cd rocALUTION/build/release/examples

  wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
  gzip -d gr_30_30.mtx.gz

  ./cg gr_30_30.mtx

For more information regarding rocALUTION library and corresponding API documentation, refer 
`rocALUTION <https://rocalution.readthedocs.io/en/latest/library.html>`_


****************
rocSPARSE
****************
Introduction
#############
rocSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HiP for GPU devices. It is designed to be used from C and C++ code.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocSPARSE

Device and Stream Management
############################
*hipSetDevice()* and *hipGetDevice()* are HIP device management APIs. They are NOT part of the rocSPARSE API.

Asynchronous Execution
######################
All rocSPARSE library functions, unless otherwise stated, are non blocking and executed asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, *hipDeviceSynchronize()* or *hipStreamSynchronize()* can be used. This will ensure that all previously executed rocSPARSE functions on the device / this particular stream have completed.

HIP Device Management
#####################
Before a HIP kernel invocation, users need to call *hipSetDevice()* to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call *hipSetDevice()* to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with rocSPARSE. rocSPARSE honors the approach above and assumes users have already set the device before a rocSPARSE routine call.

HIP Stream Management
#####################
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with *hipStreamCreate()*) and bind it to the rocSPARSE handle. HIP kernels are invoked in rocSPARSE routines. The rocSPARSE handle is always associated with a stream, and rocSPARSE passes its stream to the kernels inside the routine. One rocSPARSE routine only takes one stream in a single invocation. If users create a stream, they are responsible for destroying it.

Multiple Streams and Multiple Devices
#####################################
If the system under test has multiple HIP devices, users can run multiple rocSPARSE handles concurrently, but can NOT run a single rocSPARSE handle on different discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

Building and Installing
#######################

Installing from AMD ROCm repositories
#####################################
rocSPARSE can be installed from `AMD ROCm repositories <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`_ by

::

  sudo apt install rocsparse


Building rocSPARSE from Open-Source repository
##############################################

Download rocSPARSE
##################
The rocSPARSE source code is available at the `rocSPARSE github page <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_.
Download the master branch using:

::

  git clone -b master https://github.com/ROCmSoftwarePlatform/rocSPARSE.git
  cd rocSPARSE


Note that if you want to contribute to rocSPARSE, you will need to checkout the develop branch instead of the master branch.

Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocSPARSE using the *install.sh* script.

Using *install.sh* to build dependencies + library
##################################################
The following table lists common uses of *install.sh* to build dependencies + library.

================= ====
Command           Description
================= ====
`./install.sh -h` Print help information.
`./install.sh -d` Build dependencies and library in your local directory. The `-d` flag only needs to be |br| used once. For subsequent invocations of *install.sh* it is not necessary to rebuild the |br| dependencies.
`./install.sh`    Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i` Build library, then build and install rocSPARSE package in `/opt/rocm/rocsparse`. You will be |br| prompted for sudo access. This will install for all users.
================= ====

Using *install.sh* to build dependencies + library + client
###########################################################
The client contains example code, unit tests and benchmarks. Common uses of *install.sh* to build them are listed in the table below.

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to be |br| used once. For subsequent invocations of *install.sh* it is not necessary to rebuild the |br| dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install rocSPARSE package in |br| `/opt/rocm/rocsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install rocSPARSE package in `opt/rocm/rocsparse`. |br| You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build rocSPARSE
############################################
CMake 3.5 or later is required in order to build rocSPARSE.
The rocSPARSE library contains both, host and device code, therefore the HCC compiler must be specified during cmake configuration process.

rocSPARSE can be built using the following commands:

::

  # Create and change to build directory
  mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  CXX=/opt/rocm/bin/hcc cmake ../..

  # Compile rocSPARSE library
  make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  sudo make install

Boost and GoogleTest is required in order to build rocSPARSE client.

rocSPARSE with dependencies and client can be built using the following commands:

::

  # Install boost on Ubuntu
  sudo apt install libboost-program-options-dev
  # Install boost on Fedora
  sudo dnf install boost-program-options

  # Install googletest
  mkdir -p build/release/deps ; cd build/release/deps
  cmake -DBUILD_BOOST=OFF ../../../deps
  sudo make -j$(nproc) install

  # Change to build directory
  cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  CXX=/opt/rocm/bin/hcc cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                    -DBUILD_CLIENTS_BENCHMARKS=ON \
                                    -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocSPARSE library
  make -j$(nproc)

  # Install rocSPARSE to /opt/rocm
  sudo make install

Common build problems
#####################
#. **Issue:** HIP (/opt/rocm/hip) was built using hcc 1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/bin/hcc with version 1.0.yyy-yyy-yyy-yyy from hipcc (version mismatch). Please rebuild HIP including cmake or update HCC_HOME variable.

   **Solution:** Download HIP from github and use hcc to `build from source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and then use the built HIP instead of /opt/rocm/hip.

#. **Issue:** For Carrizo - HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** Add the following to the cmake command when configuring: `-DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801"`

#. **Issue:** For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Failed to find compatible kernel

   **Solution:** `export HCC_AMDGPU_TARGET=gfx900`

#. **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names:
              ROCMConfig.cmake |br|
              rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_

Regarding more information about rocSPARSE and it's functions, corresponding API's, Please refer 
`rocsparse <https://rocsparse.readthedocs.io/en/latest/library.html>`_



