.. _mxnet:

=========
MXNet 
=========


.. image:: MXNet_image1.png
  :align: centre
  
Introduction
##############

MXNet is a deep learning framework that has been ported to the HIP port of MXNet. It works both on HIP/ROCm and HIP/CUDA platforms.
Mxnet makes use of rocBLAS,rocRAND,rocFFT and MIOpen APIs.


Installation Guide for MXNet library
#####################################

Prerequisites
++++++++++++++

`GCC 4.8 <https://gcc.gnu.org/gcc-4.8/>`_ or later to compile C++ 11.

`GNU Make <https://www.gnu.org/software/make/>`_

**ROCm installation**

`ROCm Installation Guide <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installing-from-amd-rocm-repositories>`_

**Install Dependency**
::
 sudo apt-get install rocm-dkms rocm-dev
 sudo apt-get install rocm-device-libs rocm-libs rocblas hipblas rocrand hiprand rocfft
 sudo apt-get install rocm-opencl rocm-opencl-dev rocm-utils
 sudo apt-get install miopengemm miopen-hip

**CUDA Installation(only for HIP/CUDA)**

For compilation on HIP/CUDA Path cuda has to be installed.

Install CUDA 8.0 following the NVIDIA’s `installation guide <http://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`_ to setup MXNet with GPU support

.. note:: Make sure to add CUDA install path to LD_LIBRARY_PATH. Example - export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

Build the MXNet library
*************************
**Step 1:** Install build tools and git.
::
 $ sudo apt-get update
 $ sudo apt-get install -y build-essential git
 
**Step 2:** Install OpenBLAS.

MXNet uses BLAS and LAPACK libraries for accelerated numerical computations on CPU machine. There are several flavors of BLAS/LAPACK libraries - OpenBLAS, ATLAS and MKL. In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
::
 $ sudo apt-get install -y libopenblas-dev liblapack-dev


**Step 3:** Install `OpenCV <https://opencv.org/>`_

MXNet uses OpenCV for efficient image loading and augmentation operations.
::
 $ sudo apt-get install -y libopencv-dev
 
**Step 4:**
If building on GPU:

set USE_CUDA = 1

**Step 5:**
To enable MIOpen

set USE_CUDNN=1 (NOTE: Currently this feature is under development)


**Step 6:** Download MXNet sources and build MXNet core shared library.
::
 $ git clone --recursive https://github.com/ROCmSoftwarePlatform/mxnet
 $ cd mxnet

**To compile on HCC PLATFORM(HIP/ROCm):**
::
 $ export HIP_PLATFORM=hcc
 $ make -jn (n = no of cores)

**To compile on NVCC PLATFORM(HIP/CUDA):**
::
 $ export HIP_PLATFORM=nvcc
 $ make -jn (n = no of cores) 

**Note:**

1. USE_OPENCV, USE_BLAS, USE_CUDA, USE_CUDA_PATH are make file flags to set compilation options to use OpenCV, CUDA libraries. You can explore and use more compilation options in make/config.mk. Make sure to set USE_CUDA_PATH to right CUDA installation path. In most cases it is - /usr/local/cuda.
2. MXNet uses rocBLAS, rocFFT, rocRAND , MIOpen and lapack libraries for accelerated numerical computations. 

Install the MXNet Python binding
++++++++++++++++++++++++++++++++++++++

**Step 1:** Install prerequisites - python, setup-tools, python-pip and numpy.
::
 $ sudo apt-get install -y python-dev python-setuptools python-numpy python-pip

**Step 2:** Install the MXNet Python binding.
::
 $ cd python
 $ sudo python setup.py install 

**Step 3:** Execute application
::
 $ cd example/
 $ cd application-folder/
 To run on gpu change mx.cpu() to mx.gpu() in python script(application)
 $ python application-name.py


