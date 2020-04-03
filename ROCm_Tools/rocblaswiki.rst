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

Download rocBLAS
-----------------

Download the master branch of rocBLAS from github using:

::

  git clone -b master https://github.com/ROCmSoftwarePlatform/rocBLAS.git
  cd rocBLAS

Note if you want to contribute to rocBLAS, you will need the develop branch, not the master branch, and you will need to read .github/CONTRIBUTING.md.

Below are steps to build either (dependencies + library) or (dependencies + library + client). You only need (dependencies + library) if you call rocBLAS from your code, or if you need to install rocBLAS for other users. The client contains the test code and examples.

It is recommended that the script install.sh be used to build rocBLAS. If you need individual commands, they are also given.

Use install.sh to build (library dependencies + library)
---------------------------------------------------------

Common uses of install.sh to build (library dependencies + library) are in the table below.

===================     ===========
install.sh_command 	description
===================     ===========
./install.sh -h 	Help information.
./install.sh -d 	Build library dependencies and library in your local directory. The -d flag only needs to be used once. For subsequent invocations of install.sh it is not necessary to rebuild the dependencies.
./install.sh 	Build library in your local directory. It is assumed dependencies have been built
./install.sh -i 	Build library, then build and install rocBLAS package in /opt/rocm/rocblas. You will be prompted for sudo access. This will install for all users. If you want to keep rocBLAS in your local directory, you do not need the -i flag.
===================     ===========


Use install.sh to build (library dependencies + client dependencies + library + client)
----------------------------------------------------------------------------------------

The client contains executables in the table below.

================        ===========
executable name 	description
================        ===========
rocblas-test 	        runs Google Tests to test the library
rocblas-bench 	        executable to benchmark or test individual functions
example-sscal 	        example C code calling rocblas_sscal function
================        ===========
	
Common uses of install.sh to build (dependencies + library + client) are in the table below.

===================     ============
install.sh_command 	description
===================     ============
./install.sh -h 	Help information.
./install.sh -dc 	Build library dependencies, client dependencies, library, and client in your local directory. The -d flag only needs to be used once. For subsequent invocations of install.sh it is not necessary to rebuild the dependencies.
./install.sh -c 	Build library and client in your local directory. It is assumed the dependencies have been built.
./install.sh -idc 	Build library dependencies, client dependencies, library, client, then build and install the rocBLAS package. You will be prompted for sudo access. It is expected that if you want to install for all users you use the -i flag. If you want to keep rocBLAS in your local directory, you do not need the -i flag.
./install.sh -ic 	Build and install rocBLAS package, and build the client. You will be prompted for sudo access. This will install for all users. If you want to keep rocBLAS in your local directory, you do not need the -i flag.
===================     ============


Build (library dependencies + library) Using Individual Commands
-----------------------------------------------------------------

Before building the library please install the library dependencies CMake, Python 2.7, and Python-yaml.

**CMake 3.5 or later**

The build infrastructure for rocBLAS is based on `Cmake <https://cmake.org/>`_ v3.5. This is the version of cmake available on ROCm supported platforms. If you are on a headless machine without the x-windows system, we recommend using **ccmake**; if you have access to X-windows, we recommend using **cmake-gui**.

Install one-liners cmake:

    * Ubuntu: sudo apt install cmake-qt-gui
    * Fedora: sudo dnf install cmake-gui

**Python 2.7**

By default both python2 and python3 are on Ubuntu. You can check the installation with python -V. Python is used in Tensile, and Tensile is part of rocBLAS. To build rocBLAS the default version of Python must be Python 2.7, not Python 3.

**Python-yaml**

PyYAML files contain training information from Tensile that is used to build gemm kernels in rocBLAS.

Install one-liners PyYAML:

    * Ubuntu: sudo apt install python2.7 python-yaml
    * Fedora: sudo dnf install python PyYAML

**Build library**

The rocBLAS library contains both host and device code, so the HCC compiler must be specified during cmake configuration to properly initialize build tools. Example steps to build rocBLAS:

::

   # after downloading and changing to rocblas directory:
   mkdir -p build/release
   cd build/release
   # Default install path is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other install path
   # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=Debug to specify Debug configuration
   CXX=/opt/rocm/bin/hcc cmake ../..
   make -j$(nproc)
   #if you want to install in /opt/rocm or the directory set in cmake with -DCMAKE_INSTALL_PREFIX
   sudo make install # sudo required if installing into system directory such as /opt/rocm


Build (library dependencies + client dependencies + library + client) using Individual Commands
-------------------------------------------------------------------------------------------------

**Additional dependencies for the rocBLAS clients**

The unit tests and benchmarking applications in the client introduce the following dependencies:

#. `boost <https://www.boost.org/>`_
#. `fortran <https://gcc.gnu.org/wiki/GFortran>`_ 
#. `lapack <https://github.com/Reference-LAPACK/lapack-release>`_
         * lapack itself brings a dependency on a fortran compiler
#.  `googletest <https://github.com/google/googletest>`_


**boost**

Linux distros typically have an easy installation mechanism for boost through the native package manager.

::

   Ubuntu: sudo apt install libboost-program-options-dev
   Fedora: sudo dnf install boost-program-options


Unfortunately, googletest and lapack are not as easy to install. Many distros do not provide a googletest package with pre-compiled libraries, and the lapack packages do not have the necessary cmake config files for cmake to configure linking the cblas library. rocBLAS provide a cmake script that builds the above dependencies from source. This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting the CMAKE_PREFIX_PATH definition. The following is a sequence of steps to build dependencies and install them to the cmake default /usr/local.

**gfortran and lapack**

LAPACK is used in the client to test rocBLAS. LAPACK is a Fortran Library, so gfortran is required for building the client.

::

   Ubuntu apt-get update

   apt-get install gfortran

   Fedora yum install gcc-gfortran

   mkdir -p build/release/deps
   cd build/release/deps
   cmake -DBUILD_BOOST=OFF ../../deps   # assuming boost is installed through package manager as above
   make -j$(nproc) install


Build Library and Client Using Individual Commands
----------------------------------------------------

Once dependencies are available on the system, it is possible to configure the clients to build. This requires a few extra cmake flags to the library cmake configure script. If the dependencies are not installed into system defaults (like /usr/local ), you should pass the CMAKE_PREFIX_PATH to cmake to help find them.

``-DCMAKE_PREFIX_PATH="<semicolon separated paths>"``

::

   # after downloading and changing to rocblas directory:
   mkdir -p build/release
   cd build/release
   # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
   CXX=/opt/rocm/bin/hcc cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON ../..
   make -j$(nproc)
   sudo make install   # sudo required if installing into system directory such as /opt/rocm


Use of Tensile
----------------

The rocBLAS library uses `Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`_, which supplies the high-performance implementation of xGEMM. Tensile is downloaded by cmake during library configuration and automatically configured as part of the build, so no further action is required by the user to set it up.

CUDA build errata
------------------

rocBLAS is written with HiP kernels, so it should build and run on CUDA platforms. However, currently the cmake infrastructure is broken with a CUDA backend. However, a BLAS marshalling library that presents a common interface for both ROCm and CUDA backends can be found with `hipBLAS <https://github.com/ROCmSoftwarePlatform/hipBLAS>`_.


Common build problems
-----------------------

    * **Issue:** "Tensile could not be found because dependency Python Interp could not be found".

      **Solution:** Due to a bug in Tensile, you may need cmake-gui 3.5 and above, though in the cmakefiles it requires 2.8.

    * **Issue:** HIP (/opt/rocm/hip) was built using hcc 1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/hcc/hcc with version 1.0.yyy-yyy-yyy-yyy from hipcc. (version does not match) . Please rebuild HIP including cmake or update HCC_HOME variable.

      **Solution:** Download HIP from github and use hcc to `build from source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and then use the build HIP instead of /opt/rocm/hip one or singly overwrite the new build HIP to this location.

    * **Issue:** For Carrizo - HCC RUNTIME ERROR: Fail to find compatible kernel

      **Solution:** Add the following to the cmake command when configuring: -DCMAKE_CXX_FLAGS="--amdgpu-target=gfx801"

    * **Issue:** For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Fail to find compatible kernel

      **Solution:** export HCC_AMDGPU_TARGET=gfx900

    * **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names:

    ROCMConfig.cmake

    rocm-config.cmake



      **Solution:** Install ROCm `cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_.


Example
########

::

  #include <stdlib.h>
  #include <stdio.h>
  #include <vector>
  #include <math.h>
  #include "rocblas.h"

  using namespace std;

  int main()
          {
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
      }

          if(error > tolerance)
          {
          printf("SCAL Failed !\n");
          }
          else
          {
          printf("SCAL Success !\n");
          }

          hipFree(dx);
          rocblas_destroy_handle(handle);
          return 0;
          }

Paste the above code into the file rocblas_sscal_example.cpp

**Use hipcc Compiler:**

The recommend host compiler is `hipcc <https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/>`_. To use hipcc you will need to add /opt/rocm/bin to your path with the following:

::

 export PATH=$PATH:/opt/rocm/bin

The following makefile can be used to build the executable.

The Makefile assumes that rocBLAS is installed in the default location /opt/rocm/rocblas. If you have rocBLAS installed in your home directory in ~/rocBLAS/build/release/rocblas-install/rocblas then edit Makefile and change /opt/rocm/rocblas to ~/rocBLAS/build/release/rocblas-install/rocblas.

You may need to give the location of the library with

::

  export LD_LIBRARY_PATH=/opt/rocm/rocblas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

Run the executable with the command

::

  ./rocblas_sscal_example

  # Makefile assumes rocBLAS is installed in /opt/rocm/rocblas

  ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
  ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
  ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
  ROCBLAS_LIB=rocblas
  HIP_INCLUDE=/opt/rocm/hip/include
  LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
  LD=hipcc
  CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
  CPP=hipcc
  OBJ=rocblas_sscal_example.o
  EXE=rocblas_sscal_example

  %.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

  $(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $@ 

  clean:
	rm -f $(EXE) $(OBJ)


**Use g++ Compiler:**

Use the Makefile below:

::

  ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
  ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
  ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
  ROCBLAS_LIB=rocblas
  ROCM_INCLUDE=/opt/rocm/include
  LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB) -L/opt/rocm/lib -lhip_hcc
  LD=g++
  CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(ROCM_INCLUDE) -D__HIP_PLATFORM_HCC__
  CPP=g++
  OBJ=rocblas_sscal_example.o
  EXE=rocblas_sscal_example

  %.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

  $(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $@

  clean:
	rm -f $(EXE) $(OBJ)

Running
#########

Notice
--------

This wiki describes running the examples, tests, and benchmarks in the client. Before reading this Wiki, it is assumed rocBLAS (dependencies + library + client) has been built as described in `Build <https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/1.Build>`_

Examples
---------

The default for [BUILD_DIR] is ~/rocblas/build.

::

  cd [BUILD_DIR]/release/clients/staging
  ./example-sscal
  ./example-scal-template
  ./example-sgemm
  ./example-sgemm-strided-batched


Code for the examples is at: `samples <https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/samples>`_

In addition see `Example <https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/2.Example>`_

Unit tests
-----------

Run tests with the following:


  cd [BUILD_DIR]/release/clients/staging
  ./rocblas-test


To run specific tests, use --gtest_filter=match where match is a ':'-separated list of wildcard patterns (called the positive patterns) optionally followed by a '-' and another ':'-separated pattern list (called the negative patterns). For example, run gemv tests with the following:


  cd [BUILD_DIR]/release/clients/staging
  ./rocblas-test --gtest_filter=*checkin*gemm*float*-*batched*:*NaN*


Benchmarks
-------------

Run bencharmks with the following:


  cd [BUILD_DIR]/release/clients/staging
  ./rocblas-bench -h


The following are examples for running particular gemm and gemv benchmark:

::

  ./rocblas-bench -f gemm -r s -m 1024 -n 1024 -k 1024 --transposeB T -v 1
  ./rocblas-bench -f gemv -m 9216 -n 9216 --lda 9216 --transposeA T


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

Rules for obtaining the rocBLAS API from Legacy BLAS
------------------------------------------------------

    * The Legacy BLAS routine name is changed to lower case, and prefixed by rocblas_.

    * A first argument rocblas_handle handle is added to all rocBlas functions.

    * Input arguments are declared with the const modifier.

    * Character arguments are replaced with enumerated types defined in rocblas_types.h. They are passed by value on the host.

    * Array arguments are passed by reference on the device.

    * Scalar arguments are passed by value on the host with the following two exceptions:

       * Scalar values alpha and beta are passed by reference on either the host or the device. The rocBLAS functions will check to see it the value is on the device. If this is true, it is used, else the value on the host is used.

       * Where Legacy BLAS functions have return values, the return value is instead added as the last function argument. It is returned by reference on either the host or the device. The rocBLAS functions will check to see it the value is on the device. If this is true, it is used, else the value is returned on the host. This applies to the following functions: xDOT, xDOTU, xNRM2, xASUM, IxAMAX, IxAMIN.

    * The return value of all functions is rocblas_status, defined in rocblas_types.h. It is used to check for errors.

LP64 interface
---------------

The rocBLAS library is LP64, so rocblas_int arguments are 32 bit and rocblas_long arguments are 64 bit.


Column-major storage and 1 based indexing
-------------------------------------------

rocBLAS uses column-major storage for 2D arrays, and 1 based indexing for the functions xMAX and xMIN. This is the same as Legacy BLAS and cuBLAS.

If you need row-major and 0 based indexing (used in C language arrays) download the `CBLAS <http://www.netlib.org/blas/#_cblas>`_ file cblas.tgz. Look at the CBLAS functions that provide a thin interface to Legacy BLAS. They convert from row-major, 0 based, to column-major, 1 based. This is done by swapping the order of function arguments. It is not necessary to transpose matrices.

Pointer mode
--------------

The auxiliary functions rocblas_set_pointer and rocblas_get_pointer are used to set and get the value of the state variable rocblas_pointer_mode. If rocblas_pointer_mode == rocblas_pointer_mode_host then scalar parameters must be allocated on the host. If rocblas_pointer_mode == rocblas_pointer_mode_device, then scalar parameters must be allocated on the device.

There are two types of scalar parameter:

    scaling parameters like alpha and beta used in functions like axpy, gemv, gemm
    scalar results from functions amax, amin, asum, dot, nrm2

For scalar parameters like alpha and beta when rocblas_pointer_mode == rocblas_pointer_mode_host they can be allocated on the host heap or stack. The kernel launch is asynchronous, and if they are on the heap they can be freed after the return from the kernel launch. When rocblas_pointer_mode == rocblas_pointer_mode_device they must not be changed till the kernel completes.

For scalar results, when rocblas_pointer_mode == rocblas_pointer_mode_host then the function blocks the CPU till the GPU has copied the result back to the host. When rocblas_pointer_mode == rocblas_pointer_mode_device the function will return after the asynchronous launch. Similarly to vector and matrix results, the scalar result is only available when the kernel has completed execution.

Asynchronous API
-------------------
Except a functions having memory allocation inside preventing asynchronicity, most of the rocBLAS functions are configured to operate in asynchronous fashion with respect to CPU, meaning these library functions return immediately

Logging
#########

Four environment variables can be set to control logging:

    * ROCBLAS_LAYER
    * ROCBLAS_LOG_TRACE_PATH
    * ROCBLAS_LOG_BENCH_PATH
    * ROCBLAS_LOG_PROFILE_PATH

ROCBLAS_LAYER is a bitwise OR of zero or more bit masks as follows:

    * If ROCBLAS_LAYER is not set, then there is no logging
    * If (ROCBLAS_LAYER & 1) != 0, then there is trace logging
    * If (ROCBLAS_LAYER & 2) != 0, then there is bench logging
    * If (ROCBLAS_LAYER & 4) != 0, then there is profile logging

Trace logging outputs a line each time a rocBLAS function is called. The line contains the function name and the values of arguments.

Bench logging outputs a line each time a rocBLAS function is called. The line can be used with the executable rocblas-bench to call the function with the same arguments.

Profile logging, at the end of program execution, outputs a YAML description of each rocBLAS function called, the values of its arguments, and the number of times it was called with those arguments.

The default stream for logging output is standard error. Three environment variables can set the full path name for a log file:

    * ROCBLAS_LOG_TRACE_PATH sets the full path name for trace logging
    * ROCBLAS_LOG_BENCH_PATH sets the full path name for bench logging
    * ROCBLAS_LOG_PROFILE_PATH sets the full path name for profile logging

If a path name cannot be opened, then the corresponding logging output is streamed to standard error.

Note that performance will degrade when logging is enabled.

When profile logging is enabled, memory usage will increase. If the program exits abnormally, then it is possible that profile logging will not be outputted before the program exits.



Train Tensile for rocBLAS
##########################


Below are 10 steps that can be used to build Tensile and rocBLAS for the sizes specified in rocblas_sgemm_asm_miopen.yaml

::
  
   git clone -b develop https://github.com/ROCmSoftwarePlatform/Tensile.git
   cd Tensile
   mkdir build
   cd build
   time python ../Tensile/Tensile.py ../Tensile/Configs/rocblas_sgemm_asm_miopen.yaml ./ 2>&1 | tee Tensile.out
   cd ../..
   git clone -b develop https://github.com/ROCmSoftwarePlatform/rocBLAS.git
   cd rocBLAS
   cp ../Tensile/build/3_LibraryLogic/*.yaml library/src/blas3/Tensile/Logic/asm_miopen
   ./install.sh -idc -l asm_miopen


Notes on each step

    * Clones the develop branch of Tensile
    * Change to Tensile directory
    * Make a directory for building Tensile
    * Change to build directory
    * Time the build of Tensile, and echo output to Tensile.out. This builds Tensile for the kernel parameters and sizes specified in rocblas_sgemm_asm_miopen.yaml. The output are the .yaml files in 3_LibraryLogic. These output files contain all the information rocBLAS needs to build the assembly files for sgemm.
    * Change directory
    * Clone the develop branch of rocBLAS
    * Change directory
    * Copy the .yaml files for rocblas_sgemm_asm_miopen.yaml from Tensile to rocBLAS. These files contain the information to build the sgemm assembly kernels in rocBLAS
    * Build and install rocBLAS using the specifications in the .yaml files in library/src/blas3/Tensile/Logic/asm_miopen. The install.sh flag d installs dependencies. You only need this the first time you run install.sh. For subsequent runs use ./install.sh -ic -l asm_miopen.


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

* For more profiling tools, see `Profiling and Debugging HIP Code <https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/blob/master/docs/markdown/hip_profiling.md#profiling-hip-apis>`_

The IR and ISA can be dumped by setting the following environment variable before building and running the app.

::

  export KMDUMPISA=1

  export KMDUMPLLVM=1

  export KMDUMPDIR=/path/to/dump


By roprof
-----------

a tool very similar to nvprof, roprof is a command line tool to profile HIP kernels, roprof is located in /opt/rocm/profiler/bin

example usage

::

  /opt/rocm/profiler/bin/rcprof -T -a profile.atp ./your_executable

it will dump several a bunch of profile.HSA*.html files, you can view it by any internet browser.

::

  /opt/rocm/profiler/bin/rcprof --help for more options






































































