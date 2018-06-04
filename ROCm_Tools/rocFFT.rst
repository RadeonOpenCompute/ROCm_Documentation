.. _rocFFT:

==========
rocFFT
==========

rocFFT is a software library for computing Fast Fourier Transforms (FFT) written in HIP. It is part of AMD's software ecosystem based on ROCm. In addition to AMD GPU devices, the library can also be compiled with the CUDA compiler using HIP tools for running on Nvidia GPU devices.

API design
**************
Please refer to the :ref:`rocFFTAPI` for current documentation. Work in progress.

Installing pre-built packages
*******************************
Download pre-built packages either from `ROCm's package servers <https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html>`_ or by clicking the github releases tab and manually downloading, which could be newer. Release notes are available for each release on the releases tab.
::
 sudo apt update && sudo apt install rocfft

Quickstart rocFFT build
************************
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
*********
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
