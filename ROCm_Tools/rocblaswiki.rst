.. _rocblaswiki:

========================
rocblas build wiki 
========================

Home
#####

Building rocBLAS
*****************
 1. For instructions to build rocblas library and clients, see Build rocBLAS libraries and verification code.
 2. For an example using rocBLAS see Example C code calling rocBLAS function.
 3. For instructions on how to run/use the client code, see Build rocBLAS libraries, verification-code, tests and benchmarks.
    
Functionality
***************
rocBLAS exports the following BLAS-like functions at this time.

Rules for obtaining the rocBLAS API from Legacy BLAS
*******************************************************
 1. The Legacy BLAS routine name is changed to lower case, and prefixed by rocblas_.

 2. A first argument rocblas_handle handle is added to all rocBlas functions.

 3. Input arguments are declared with the const modifier.

 4. Character arguments are replaced with enumerated types defined in rocblas_types.h. They are passed by value on the host.

 5. Array arguments are passed by reference on the device.

 6. Scalar arguments are passed by value on the host with the following two exceptions:

 * Scalar values alpha and beta are passed by reference on either the host or the device. The rocBLAS functions will check to see it 	the value is on the device. If this is true, it is used, else the value on the host is used.

 * Where Legacy BLAS functions have return values, the return value is instead added as the last function argument. It is returned by 	reference on either the host or the device. The rocBLAS functions will check to see it the value is on the device. If this is true, 	it is used, else the value is returned on the host. This applies to the following functions: xDOT, xDOTU, xNRM2, xASUM, IxAMAX,     	IxAMIN.

 7. The return value of all functions is rocblas_status, defined in rocblas_types.h. It is used to check for errors.
    
Additional notes
******************
 * The rocBLAS library is LP64, so rocblas_int arguments are 32 bit and rocblas_long arguments are 64 bit.

 * rocBLAS uses column-major storage for 2D arrays, and 1 based indexing for the functions xMAX and xMIN. This is the same as Legacy 	BLAS and cuBLAS. If you need row-major and 0 based indexing (used in C language arrays) download the `CBLAS <http://www.netlib.org/blas/#_cblas>`_ file cblas.tgz. Look at 	 the CBLAS functions that provide a thin interface to Legacy BLAS. They convert from 	row-major, 0 based, to column-major, 1 based. 
   This is done by swapping the order of function arguments. It is not necessary to transpose matrices.

 * The auxiliary functions rocblas_set_pointer and rocblas_get_pointer are used to set and get the value of the state variable 	     	rocblas_pointer_mode. This variable is not used, it is added for compatibility with cuBLAS. rocBLAS will check if your scalar     	argument passed by reference is on the device. If this is true it will pass by reference on the device, else it passes by         	reference on the host.

Build
#######
Dependencies For Building Library
**********************************
**CMake 3.5 or later**

The build infrastructure for rocBLAS is based on `Cmake <https://cmake.org/>`_ v3.5. This is the version of cmake available on ROCm supported platforms. If you are on a headless machine without the x-windows system, we recommend using **ccmake**; if you have access to X-windows, we recommend using **cmake-gui**.

Install one-liners cmake:

 * Ubuntu: sudo apt install cmake-qt-gui
 * Fedora: sudo dnf install cmake-gui

**Python 2.7**

By default both python2 and python3 are on Ubuntu. You can check the installation with python -V. Python is used in Tensile, and Tensile is part of rocBLAS. To build rocBLAS the default version of Python must be Python 2.7, not Python 3.

Build Library Using Script (Ubuntu only)
*******************************************
The root of this repository has a helper bash script install.sh to build and install rocBLAS on Ubuntu with a single command. It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install. A few commands in the script need sudo access, so it may prompt you for a password.

 * ./install.sh -h -- shows help
 * ./install.sh -id -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

Build Library Using Individual Commands
****************************************
The rocBLAS library has one dependency named `Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`_, which supplies the high-performance implementation of xGEMM. Tensile is downloaded by cmake during library configuration and automatically configured as part of the build, so no further action is required by the user to set it up. Tensile is predominately written in python2.7 (not python3), so it does bring python dependencies which can easily be installed with distro package managers. The rocBLAS library contains both host and device code, so the HCC compiler must be specified during cmake configuration to properly initialize build tools. Example steps to build rocBLAS:

**(One time only)**

 * Ubuntu: sudo apt install python2.7 python-yaml
 * Fedora: sudo dnf install python PyYAML

Configure and build steps
*****************************

:: 
 
  mkdir -p [ROCBLAS_BUILD_DIR]/release
  cd [ROCBLAS_BUILD_DIR]/release
  # Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
  # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
  CXX=/opt/rocm/bin/hcc ccmake [ROCBLAS_SOURCE]
  make -j$(nproc)
  sudo make install # sudo required if installing into system directory such as /opt/rocm

Build Library + Tests + Benchmarks + Samples Using Individual Commands
*************************************************************************
The repository contains source for clients that serve as samples, tests and benchmarks. Clients source can be found in the clients subdir.

**Dependencies (only necessary for rocBLAS clients)**

The rocBLAS samples have no external dependencies, but our unit test and benchmarking applications do. These clients introduce the following dependencies:

   1. `boost <http://www.boost.org/>`_
   2. `lapack <https://github.com/Reference-LAPACK/lapack-release>`_
         * lapack itself brings a dependency on a fortran compiler
   3. `googletest <https://github.com/google/googletest>`_

Linux distros typically have an easy installation mechanism for boost through the native package manager.

  * Ubuntu: sudo apt install libboost-program-options-dev
  * Fedora: sudo dnf install boost-program-options

Unfortunately, googletest and lapack are not as easy to install. Many distros do not provide a googletest package with pre-compiled libraries, and the lapack packages do not have the necessary cmake config files for cmake to configure linking the cblas library. rocBLAS provide a cmake script that builds the above dependencies from source. This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting the CMAKE_PREFIX_PATH definition. The following is a sequence of steps to build dependencies and install them to the cmake default /usr/local.

**(optional, one time only)**

::

  mkdir -p [ROCBLAS_BUILD_DIR]/release/deps
  cd [ROCBLAS_BUILD_DIR]/release/deps
  ccmake -DBUILD_BOOST=OFF [ROCBLAS_SOURCE]/deps   # assuming boost is installed through package manager as above
  make -j$(nproc) install

Once dependencies are available on the system, it is possible to configure the clients to build. This requires a few extra cmake    flags to the library cmake configure script. If the dependencies are not installed into system defaults (like /usr/local ), you should pass the CMAKE_PREFIX_PATH to cmake to help find them.

 * -DCMAKE_PREFIX_PATH="<semicolon separated paths>"

::

  # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
  CXX=/opt/rocm/bin/hcc ccmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [ROCBLAS_SOURCE]
  make -j$(nproc)
  sudo make install   # sudo required if installing into system directory such as /opt/rocm
  CUDA build errata

rocBLAS is written with HiP kernels, so it should build and run on CUDA platforms. However, currently the cmake infrastructure is broken with a CUDA backend. However, a BLAS marshalling library that presents a common interface for both ROCm and CUDA backends can be found with :ref:`hipBLAS` .

Common build problems
************************
 * Issue: "Tensile could not be found because dependency Python Interp could not be found".

 * Solution: Due to a bug in Tensile, you may need cmake-gui 3.5 and above, though in the cmakefiles it requires 2.8.

 * Issue: HIP (/opt/rocm/hip) was built using hcc 1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/hcc/hcc with version 1.0.yyy-yyy-  	  yyy-yyy from hipcc. (version does not match) . Please rebuild HIP including cmake or update HCC_HOME variable.

 * Solution: Download HIP from github and use hcc to build from source and then use the build HIP instead of /opt/rocm/hip one or    	singly overwrite the new build HIP to this location.

 * Issue: For Carrizo - HCC RUNTIME ERROR: Fail to find compatible kernel

 * Solution: Add the following to the cmake command when configuring: -DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801"

 * Issue: For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Fail to find compatible kernel

 * Solution: export HCC_AMDGPU_TARGET=gfx900


Device and Stream management in rocBLAS
#########################################

HIP Device management
***********************
hipSetDevice() & hipGetDevice() are HIP device management APIs. They are NOT part of the rocBLAS API.

Before a HIP kernel invocation, users need to call hipSetDevice() to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call hipSetDevice() to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with rocBLAS. rocBLAS honors the approach above and assumes users have already set the device before a rocBLAS routine call.

Once users set the device, they create a handle with rocblas_status rocblas_create_handle(rocblas_handle *handle)

Subsequent rocBLAS routines take this handle as an input parameter. rocBLAS ONLY queries (by hipGetDevice) the user's device; rocBLAS but does NOT set the device for users. If rocBLAS does not see a valid device, it returns an error message to users. It is the users' responsibility to provide a valid device to rocBLAS and ensure the device safety as explained soon.

Users CANNOT switch devices between rocblas_create_handle() and rocblas_destroy_handle() (the same as cuBLAS requires). If users want to change device, they must destroy the current handle, and create another rocBLAS handle (context).

Stream management
*********************
HIP kernels are always launched in a queue (otherwise known as a stream, they are the same thing).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. Howevers, users can freely create new streams (with hipStreamCreate) and bind it to the rocBLAS handle: rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id) HIP kernels are invoked in rocBLAS routines. The rocBLAS handles are always associated with a stream, and rocBLAS passes its stream to the kernels inside the routine. One rocBLAS routine only takes one stream in a single invocation. If users create a stream, they are responsible for destroying it.

Multiple streams and multiple devices
*****************************************
If the system under test has 4 HIP devices, users can run 4 rocBLAS handles (also known as contexts) on 4 devices concurrently, but can NOT span a single rocBLAS handle on 4 discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

Example C code calling rocBLAS routine
#############################################

::
 
  #include <stdlib.h>
  #include <stdio.h>
  #include <vector>
  #include <math.h>
  #include "rocblas.h"

  using namespace std;

  int main() {

      rocblas_int N = 10240;
      float alpha = 10.0;

      vector<float> hx(N);
      vector<float> hz(N);
      float* dx;
      float tolerance = 0, error;

      rocblas_handle handle;
      rocblas_create_handle(&handle);

      // allocate memory on device
      hipMalloc(&dx, N * sizeof(float));

      // Initial Data on CPU,
      srand(1);
      for( int i = 0; i < N; ++i )
      {
        hx[i] = rand() % 10 + 1;  //generate a integer number between [1, 10]
      }

      // save a copy in hz 
      hz = hx;

      hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);

      rocblas_sscal(handle, N, &alpha, dx, 1);

      // copy output from device memory to host memory
      hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);

      // verify rocblas_scal result
      for(rocblas_int i=0;i<N;i++)
      {
          error = fabs(hz[i] * alpha - hx[i]);
          if(error > tolerance)
          {
            printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
            break;
          }
      }

      if(error > tolerance){
          printf("SCAL Failed !\n");
      }
      else{
          printf("SCAL Success !\n");
      }

      hipFree(dx);
      rocblas_destroy_handle(handle);
      return 0;
  }
================================= Compiler:

The recommend host compiler is [hipcc] (https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/) (an alias of hcc). Using gcc will lead to compilation error currently. You may need to add /opt/rocm/bin to your path with the following:

::

  export PATH=$PATH:/opt/rocm/bin

If the above code is pasted into a file rocblas_sscal_example.cpp the following makefile can be used to build an executable. You will need to give the location of the library with

::

  export LD_LIBRARY_PATH=~/repos/rocBLAS/build/library-package/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

Run the executable with the command

::

  ./rocblas_sscal_example

::

  # Makefile assumes rocBLAS is installed in $(HOME)/repos/rocBLAS/build/library-package
  ROCBLAS_INSTALL_DIR=$(HOME)/repos/rocBLAS/build/library-package
  ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
  ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
  ROCBLAS_LIB=rocblas-hcc
  HIP_INCLUDE=/opt/rocm/hip/include
  LDFLAGS=-lm -L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
  LD=hipcc
  CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
  CPP=hipcc
  OBJ=rocblas_sscal_example.o
  EXE=rocblas_sscal_example

  %.o: %.cpp
          $(CPP) -c -o $@ $< $(CFLAGS)

  $(EXE) : $(OBJ)
          $(LD) $(LDFLAGS) $(OBJ) -o $@ 

  clean:
          rm -f $(EXE) $(OBJ)


Exported functions
#######################

Exported BLAS functions
*************************
rocBLAS includes the following auxiliary functions
*********************************************************

+--------------------------+
| Function name            |
+--------------------------+
| rocblas_create_handle    |
+--------------------------+
| rocblas_destroy_handle   |
+--------------------------+
| rocblas_add_stream       |
+--------------------------+
| rocblas_set_stream       |
+--------------------------+
| rocblas_get_stream       |
+--------------------------+
| rocblas_set_pointer_mode |
+--------------------------+
| rocblas_get_pointer_mode |
+--------------------------+
| rocblas_set_vector       |
+--------------------------+
| rocblas_get_vector       |
+--------------------------+
| rocblas_set_matrix       |
+--------------------------+
| rocblas_get_matrix       |
+--------------------------+

rocBLAS includes the following Level 1, 2, and 3 functions
***********************************************************

**Level 1**

+---------------+--------+--------+----------------+----------------+------+
| Function      | single | double | single complex | double complex | half |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xscal | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xcopy | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xdot  | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xswap | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xaxpy | x      | x      | x              | x              | x    |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xasum | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xnrm2 | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xamax | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xamin | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+

**Level 2**

+---------------+--------+--------+----------------+----------------+------+
| Function      | single | double | single complex | double complex | half |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xgemv | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+
| rocblas_Xger  | x      | x      | x              | x              |      |
+---------------+--------+--------+----------------+----------------+------+


**Level 3**

+------------------------+--------+--------+----------------+----------------+------+
| Function               | single | double | single complex | double complex | half |
+------------------------+--------+--------+----------------+----------------+------+
| rocblas_Xgemm          | x      | x      |                |                |      |
+------------------------+--------+--------+----------------+----------------+------+
| rocblas_Xtrtri         | x      | x      |                |                |      |
+------------------------+--------+--------+----------------+----------------+------+
| rocblas_Xtrtri_batched | x      | x      |                |                |      |
+------------------------+--------+--------+----------------+----------------+------+
| rocblas_Xtrsm          | x      | x      |                |                |      |
+------------------------+--------+--------+----------------+----------------+------+

Numerical Stability in TRSM
##############################

Division
***********

Most BLAS routines like GEMM, GEMV, GER only perform multiplication and add. As long as there is no overflow (e.g. very big number), the GPU always achieve bit-wise consistence with the CPU results. However, the TRSM and TRTRI routine perform division. Therefore, there are some precision differences between Netlib CBLAS (which we compare to) and rocBLAS.

For example, the gtest may falsely report such failure by using gtest macro ASSERT_FLOAT_EQ in STRSM:

Expected: hCPU[i+j*lda] Which is: -0.66039091

To be equal to: hGPU[i+j*lda] Which is: -0.66039127

However, since we achieve 5 digits consistency (0.66039), we think rocBLAS result is correct. Therefore, we use gtest ASSERT_NEAR in TRSM to replace ASSERT_FLOAT_EQ which is used for other routines.

Well-conditioned Matrix
***************************
Generally, TRTRI and TRSM has matrix inversion involved (accessing half of the matrix). Matrix inversion generally is not numerical stable operation, especially when the matrix is [ill-conditioned] (https://en.wikipedia.org/wiki/Condition_number). A random matrix is very ill-conditioned typically. Such matrix will cause serious overflow.

In order to generate a well-conditioned matrix, we perform an LU factorization first on this random matrix. The factorized Lower and upper part overwrite the original matrix. The factorized matrix is a well-conditioned matrix and used as the input matrix in rocBLAS TRSM test.


Profile rocBLAS kernels
#########################
By environment variable
**************************

* In bash: "export HIP_TRACE_API=1" (reset by =0) Launch your application, then it profiles every HIP APIs, including rocBLAS kernels.

* "export HIP_LAUNCH_BLOCKING = 0": make HIP APIs host-synchronous so they are blocked until any kernel launches or data-copy 	     	commands are complete (an alias is CUDA_LAUNCH_BLOCKING)

* For more profiling tools, see Profiling and Debugging HIP Code

The IR and ISA can be dumped by setting the following environment variable before building and running the app.

export KMDUMPISA=1

export KMDUMPLLVM=1

export KMDUMPDIR=/path/to/dump

By roprof

a tool very similar to nvprof, roprof is a command line tool to profile HIP kernels, roprof is located in /opt/rocm/profiler/bin

example usage

/opt/rocm/profiler/bin/rcprof -T -a profile.atp ./your_executable

it will dump several a bunch of profile.HSA*.html files, you can view it by any internet browser.

/opt/rocm/profiler/bin/rcprof --help for more options


Running
########

Notice
**********
Before reading this Wiki, it is assumed rocBLAS with the client applications has been successfully built as described in Build rocBLAS libraries and verification code

Samples
***********

::

  cd [BUILD_DIR]/clients/staging
  ./example-sscal

Example code that calls rocBLAS you can also see the following blog on the right side Example C code calling rocBLAS routine.

Unit tests
***************
Run tests with the following:

::

  cd [BUILD_DIR]/clients/staging
  ./rocblas-test

To run specific tests, use --gtest_filter=match where match is a ':'-separated list of wildcard patterns (called the positive patterns) optionally followed by a '-' and another ':'-separated pattern list (called the negative patterns). For example, run gemv tests with the following:

::

  cd [BUILD_DIR]/clients/staging
  ./rocblas-test --gtest_filter=*gemv*

Benchmarks
***************
Run bencharmks with the following:

::

  cd [BUILD_DIR]/clients/staging
  ./rocblas-bench -h
The following are examples for running particular gemm and gemv benchmark:

::

  ./rocblas-bench -f gemm -r s -m 1024 -n 1024 -k 1024 --transposeB T
  ./rocblas-bench -f gemv -m 9216 -n 9216 --lda 9216 --transposeA T

For users' convenience, `python scripts <https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/benchmarks/perf_script>`_ are provided to perform benchmarking across a range of values.












































































