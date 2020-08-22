
.. _hipsparsewiki:

Building hipSPARSE
##################

    * For instructions to build hipSPARSE library and clients, see `hipsbuild <https://sep5.readthedocs.io/en/latest/ROCm_Libraries/hipsparse_wiki.html#build>`_ hipSPARSE libraries and verification code
    * For an example using hipSPARSE see `example <https://sep5.readthedocs.io/en/latest/ROCm_Libraries/hipsparse_wiki.html#example-c-code>`_ C code.
    * For instructions on how to run/use the client code, see :ref:`Running`.

Functionality
#################

hipSPARSE exports the following :ref:`export` sparse BLAS-like functions at this time.

Platform: rocSPARSE or cuSPARSE
#################################

hipSPARSE is a marshalling library, so it runs with either `rocSPARSE <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#id51>`_ or `cuSPARSE <https://developer.nvidia.com/cusparse>`_ configured as the backend SPARSE library, chosen at cmake configure time.

**CUDA unit test failures**

There are a some library unit tests failing with cuSPARSE; we believe these failures are benign and can be ignored. Our unit tests are testing with negative sizes and edge cases which are handled differently between the two libraries, and our unit tests do not account for these differences.

.. _hipsbuild:

Build
#######

Build dependencies and library (no client) using install.sh
###############################################################

The install.sh script can be used to build the dependencies and the hipSPARSE library. Common uses of install.sh are in the table below.

====================    ============
install.sh_command 	description
====================    ============
./install.sh -h 	Help information.
./install.sh -d 	Build dependencies and library in your local directory. The -d flag only needs to be used once. For subsequent invocations of install.sh it is not necessary to rebuild the dependencies.
./install.sh -c 	Build library and client in your local directory. It is assumed the dependencies have been built.
./install.sh -i 	Build library, then build and install hipSPARSE package. You will be prompted for sudo access. It is expected that if you want to install for all users you use the -i flag. If you want to keep hipSPARSE in your local directory, you do not need the -i flag.
./install.sh --cuda 	Build library in your local directory using cuSPARSE backend. It is assumed dependencies have been built.
./install.sh -d --cuda 	Build dependencies and library in your local directory using cuSPARSE backend.
./install.sh -idc 	Build dependencies, build the library, build the client, then build and install the hipSPARSE package. You will be prompted for sudo access. It is expected that if you want to install for all users you use the -i flag. If you want to keep hipSPARSE in your local directory, you do not need the -i flag.
./install.sh -ic 	Build and install hipSPARSE package and build the client. You will be prompted for sudo access. It is expected that if you want to install for all users you use the -i flag. If you want to keep hipSPARSE in your local directory, you do not need the -i flag.
./install.sh -dc 	Build dependencies, library, and client in your local directory. The -d flag only needs to be used once. For subsequent invocations of install.sh it is not necessary to rebuild the dependencies.
./install.sh 	       Build library in your local directory. It is assumed dependencies have been built.
=====================   =============

Build library and client using install.sh
##########################################

The install.sh script can be used to build the hipSPARSE library and client. The client contains executables in the table below.

================  ============
executable name   description
================  ============
hipsparse-test 	  run Google Tests to test the library
example-coomv 	  example C code calling hipsparseDcoomv function
example-csrmv 	  example C code calling hipsparseDcsrmv function
example-ellmv 	  example C code calling hipsparseDellmv function
example-hybmv 	  example C code calling hipsparseDhybmv function
example-handle 	  example C code creating and destroying a hipSPARSE handle
================  ============


Note that the client requires additional dependencies not needed by the library. These include googletest. Below are common uses of install.sh to build the dependencies, library, and client

**Note:** Adding the --cuda flag will run the script using cuSPARSE backend.

Dependencies For Building Library
#####################################

**CMake 3.5 or later**

The build infrastructure for hipSPARSE is based on `Cmake <https://cmake.org/>`_ v3.5. This is the version of cmake available on ROCm supported platforms. If you are on a headless machine without X system, we recommend using **ccmake**; if you have access to X, we recommend using **cmake-gui**.

Install one-liners cmake:

::

  Ubuntu: sudo apt install cmake-qt-gui
  Fedora: sudo dnf install cmake-gui


**rocSPARSE (HIP)**

HIP backend of hipSPARSE is based on `rocSPARSE <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#id51>`_. Currently, while rocSPARSE is not available in the public ROCm repositories, you will have to install the package manually.

**cuSPARSE (NVCC)**

NVCC backend of hipSPARSE is based on `cuSPARSE <https://developer.nvidia.com/cusparse>`_. You can install it using

    * Ubuntu: sudo apt install cuda-cusparse-<ver>
    * Fedora: currently not supported

**Configure and build steps**

Example steps to build hipSPARSE:

::

  mkdir -p [HIPSPARSE_BUILD_DIR]/release
  cd [HIPSPARSE_BUILD_DIR]/release
  # Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
  # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
  cmake [HIPSPARSE_SOURCE]
  make -j$(nproc)
  sudo make install # sudo required if installing into system directory such as /opt/rocm

Additional dependencies only necessary for hipSPARSE clients
###############################################################

The unit tests and benchmarking applications in the client introduce the following dependencies:

    * `googletest <https://github.com/google/googletest>`_

Unfortunately, googletest is not as easy to install. Many distros do not provide a googletest package with pre-compiled libraries. hipSPARSE provides a cmake script that builds the above dependencies from source. This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting the CMAKE_PREFIX_PATH definition. The following is a sequence of steps to build dependencies and install them to the cmake default /usr/local.

**(optional, one time only)**

::

  mkdir -p [HIPSPARSE_BUILD_DIR]/release/deps
  cd [HIPSPARSE_BUILD_DIR]/release/deps
  ccmake -DBUILD_BOOST=OFF [HIPSPARSE_SOURCE]/deps
  make -j$(nproc) install

**Build Library + Tests + Benchmarks + Samples Using Individual Commands**

Once dependencies are available on the system, it is possible to configure the clients to build. This requires a few extra cmake flags to the library cmake configure script. If the dependencies are not installed into system defaults (e.g. /usr/local ), the user should pass the CMAKE_PREFIX_PATH to cmake to help finding them.

::

  -DCMAKE_PREFIX_PATH="<semicolon separated paths>"
  # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
  cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON [HIPSPARSE_SOURCE]
  make -j$(nproc)
  sudo make install   # sudo required if installing into system directory such as /opt/rocm

**Common build problems**

    * **Issue:** Could not find a package configuration file provided by "rocSPARSE" with any of the following names:

    ROCSPARSEConfig.cmake

    rocsparse-config.cmake

    **Solution:** Install `rocSPARSE <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#id51>`_

    **Issue:** Could not find a package configuration file provided by "ROCM" with any of the following names:

    ROCMConfig.cmake

    rocm-config.cmake

    **Solution:** Install `ROCm cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_

.. _exampleh:
Example C code
###############

::

  #include <stdlib.h>
  #include <stdio.h>
  #include <vector>
  #include <math.h>
  #include <hipsparse.h>

  using namespace std;

  int main()
    {
      int N           = 10240;
      int nnz         = 256;
      float alpha     = 10.0f;
      float tolerance = 1e-8f;

      vector<int> hx_ind(nnz);
      vector<float> hx_val(nnz);
      vector<float> hy(N);

      // Allocate memory on the device
      int* dx_ind;
      float* dx_val;
      float* dy;

      hipMalloc(&dx_ind, nnz * sizeof(int));
      hipMalloc(&dx_val, nnz * sizeof(float));
      hipMalloc(&dy, N * sizeof(float));

      // Initial Data on CPU,
      srand(1);

      for(int i = 0; i < nnz; ++i)
        {
          hx_ind[i] = i * 40;
          hx_val[i] = rand() % 10 + 1; // Generate an integer number between [1, 10]
        }

      for(int i = 0; i < N; ++i)
        {
          hy[i] = rand() % 10 + 1; // Generate an integer number between [1, 10]
        }

       // Copy data to device
       hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
      hipMemcpy(dx_val, hx_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
      hipMemcpy(dy, hy.data(), sizeof(float) * N, hipMemcpyHostToDevice);

      // Initialize rocSPARSE
      hipsparseHandle_t handle;
      hipsparseCreate(&handle);

      // Run saxpyi on device
      hipsparseSaxpyi(handle, nnz, &alpha, dx_val, dx_ind, dy, HIPSPARSE_INDEX_BASE_ZERO);

      // Copy output from device memory to host memory
      vector<float> result(N);
      hipMemcpy(result.data(), dy, sizeof(float) * N, hipMemcpyDeviceToHost);

      // Verify hipsparseSaxpyi result
      for(int i = 0; i < nnz; ++i)
        {
          hy[hx_ind[i]] += alpha * hx_val[i];
        }

      float error;
      for(int i = 0; i < N; ++i)
        {
          error = fabs(hy[i] - result[i]);
          if(error > tolerance)
           {
             fprintf(stderr, "Error in element %d: CPU=%f, GPU=%f\n", i, hy[i], result[i]);
             break;
           }
        }

      if(error > tolerance)
        {
         printf("axpyi test failed!\n");
        }
      else
        {
        printf("axpyi test passed!\n");
        }

      hipFree(dx_ind);
      hipFree(dx_val);
      hipFree(dy);

      hipsparseDestroy(handle);

      return 0;
     }

Compiling hipSPARSE example
############################

First, paste above code into a file hipsparseSaxpyi_example.cpp. To compile hipsparseSaxpyi_example.cpp, a standard C++ compiler can be used (e.g. g++):

::

  g++ -O3 -o hipsparseSaxpyi_example hipsparseSaxpyi_example.cpp -D__HIP_PLATFORM_HCC__ -I/opt/rocm/include -L/opt/rocm/lib -lhipsparse -lhip_hcc

.. _export:

Exported sparse BLAS functions
################################

hipSPARSE includes the following auxiliary functions

 +------------------------------+
 |  Function name               |   
 +==============================+
 |  hipsparseCreate             |
 +------------------------------+
 |  hipsparseDestroy            |
 +------------------------------+
 |  hipsparseGetVersion         |
 +------------------------------+
 |  hipsparseSetStream          |
 +------------------------------+
 |  hipsparseGetStream          |
 +------------------------------+
 |  hipsparseSetPointerMode     |
 +------------------------------+
 |  hipsparseGetPointerMode     |
 +------------------------------+
 |  hipsparseCreateMatDescr     |
 +------------------------------+
 |  hipsparseDestroyMatDescr    |
 +------------------------------+
 |  hipsparseCopyMatDescr       |
 +------------------------------+
 |  hipsparseSetMatIndexBase    |
 +------------------------------+
 |  hipsparseGetMatIndexBase    |
 +------------------------------+
 |  hipsparseSetMatType         |
 +------------------------------+
 |  hipsparseGetMatType         |
 +------------------------------+
 |  hipsparseSetMatFillMode     |
 +------------------------------+
 |  hipsparseGetMatFillMode     |
 +------------------------------+
 |  hipsparseSetMatDiagType     |
 +------------------------------+
 |  hipsparseGetMatDiagType     |
 +------------------------------+
 |  hipsparseCreateHybMatrix    |
 +------------------------------+
 |  hipsparseDestroyHybMatrix   |
 +------------------------------+
 |  hipsparseCreateCsrsv2Info   |
 +------------------------------+
 |  hipsparseDestroyCsrsv2Info  |
 +------------------------------+
 |  hipsparseCreateCsrilu02Info |
 +------------------------------+
 |  hipsparseCreateCsrilu02Info |
 +------------------------------+
 
 
 

hipSPARSE includes the following Level 1, 2 and conversion functions
#######################################################################
 
**Level 1**

================  ==========   =========  ================  =================  ====== 
Function 	   single 	double 	   single complex    double complex 	half
================  ==========   =========  ================  =================  ======
hipsparseXaxpyi       x	           x 	         			
hipsparseXdoti 	      x	           x 			
hipsparseXgthr        x	           x 			
hipsparseXgthrz       x	           x 	 			
hipsparseXroti        x            x 			
hipsparseXsctr 	      x	           x 			
================  ==========   =========  ================  =================  ======

**Level 2**

================================  ==========   =========  ================  =================  ====== 
Function 	                    single 	double 	   single complex    double complex 	half
================================  ==========   =========  ================  =================  ======
hipsparseXcsrmv 	               x 	   x 			
hipsparseXcsrsv2_bufferSize 	       x 	   x 			
hipsparseXcsrsv2_bufferSizeExt 	       x 	   x 			
hipsparseXcsrsv2_analysis 	       x 	   x 			
hipsparseXcsrsv2_solve 	               x 	   x 			
hipsparseXhybmv 	               x 	   x 			
================================  ==========   =========  ================  =================  ======


**Level 3**

================================  ==========   =========  ================  =================  ====== 
Function 	                    single 	double 	   single complex    double complex 	half
================================  ==========   =========  ================  =================  ======
hipsparseXcsrmm 	              x 	  x 			
hipsparseXcsrmm2 	              x 	  x 			
================================  ==========   =========  ================  =================  ======

**Extra**

================================  ==========   =========  ================  =================  ====== 
Function 	                    single 	double 	   single complex    double complex 	halfy
================================  ==========   =========  ================  =================  ======
hipsparseXcsrgemmNnz	 	              
hipsparseXcsrgemm	              x 	  x 			
hipsparseXcsrgemm2_bufferSizeExt	
hipsparseXcsrgemm2Nnz
hipsparseXcsrgemm2
================================  ==========   =========  ================  =================  ======
**Preconditioners**

=================================  ==========   =========  ================  =================  ====== 
Function 	                    single 	 double     single complex     double complex 	 half
=================================  ==========   =========  ================  =================  ======
hipsparseXcsrilu02_bufferSize 	       x 	    x 			
hipsparseXcsrilu02_bufferSizeExt       x 	    x 			
hipsparseXcsrilu02_analysis 	       x 	    x 		
hipsparseXcsrilu02 		       x 	    x 	
=================================  ==========   =========  ================  =================  ======

**Conversion**

====================================  ==========   =========  ================  =================  ====== 
Function 	                        single 	     double    single complex    double complex     half
====================================  ==========   =========  ================  =================  ======
hipsparseXcsr2coo 					
hipsparseXcsr2csc 	                  x 	       x 			
hipsparseXcsr2hyb 	                  x 	       x 			
hipsparseXcoo2csr 					
hipsparseCreateIdentityPermutation 					
hipsparseXcsrsort_bufferSizeExt 					
hipsparseXcsrsort 					
hipsparseXcoosort_bufferSizeExt 					
hipsparseXcoosortByRow 					
hipsparseXcoosortByColumn 					
====================================  ==========   =========  ================  =================  ======

Additional notes
##################

    * hipSPARSE supports 0 and 1 based indexing. The index base is selected by hipsparseIndexBase_t type, which is either passed as standalone parameter or part of the hipsparseMatDescr_t type.

    * Dense vectors are represented with a 1D array stored linearly in memory.

    * Sparse vectors are represented with a 1D data array stored linearly in memory that holds all non-zero elements and a 1D indexing array stored linearly in memory that holds the positions of the corresponding non-zero elements.

    * The auxiliary functions hipsparseSetPointer and hipsparseGetPointer are used to set and get the value of the state variable hipsparsePointerMode_t. If hipsparsePointerMode_t == HIPSPARSE_POINTER_MODE_HOST, then scalar parameters must be allocated on the host. If hipsparsePointerMode_t == HIPSPARSE_POINTER_MODE_DEVICE, then scalar parameters must be allocated on the device.

    There are two types of scalar parameter:

       * Scaling parameters, such as alpha and beta used in e.g. csrmv, coomv, ...
       * Scalar results from functions such as doti, dotci, ...

    For scalar parameters such as alpha and beta, memory can be allocated on the host heap or stack, when hipsparsePointerMode_t == HIPSPARSE_POINTER_MODE_HOST. The kernel launch is asynchronous, and if the scalar parameter is on the heap, it can be freed after the return from the kernel launch. When hipsparsePointerMode_t == HIPSPARSE_POINTER_MODE_DEVICE, the scalar parameter must not be changed till the kernel completes.

    For scalar results, when hipsparsePointerMode_t == HIPSPARSE_POINTER_MODE_HOST, then the function blocks the CPU till the GPU has copied the result back to the host. Using hipsparsePointerMode_t == HIPSPARSE_POINTER_MODE_DEVICE, the function will return after the asynchronous launch. Similarly to vector and matrix results, the scalar result is only available when the kernel has completed execution.

.. _Running:

Running
########

**Notice**

Before reading this Wiki, it is assumed hipSPARSE with the client applications has been successfully built as described in Build hipSPARSE libraries and verification code

**Samples**

::

  cd [BUILD_DIR]/example
  ./example-csrmv 1000

Example code that calls hipSPARSE csrmv routine.

**Unit tests**

Run tests with the following:

::

  cd [BUILD_DIR]/clients/tests
  ./hipsparse-test

To run specific tests, use --gtest_filter=match where match is a ':'-separated list of wildcard patterns (called the positive patterns) optionally followed by a '-' and another ':'-separated pattern list (called the negative patterns). For example, run coo2csr tests with the following commands:

::

  cd [BUILD_DIR]/clients/tests
  ./hipsparse-test --gtest_filter=*coo2csr*

Please note, that tests are only supported when configured with rocSPARSE backend.


