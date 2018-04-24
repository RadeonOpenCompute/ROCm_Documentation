.. _mxnet:

MXNET
#######

========================
MXNet for Deep Learning
========================


.. image:: MXNet_image1.png
  :align: centre
  

MXNet is a deep learning framework designed for both efficiency and flexibility. It allows you to **mix** `symbolic and imperative programming<http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts>`_ to **maximize** efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of `blue prints and guidelines <http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts>`_ for building deep learning systems, and interesting insights of DL systems for hackers.

Join the chat at https://gitter.im/dmlc/mxnet


Contents
***********

 * `Documentation and Tutorials <http://mxnet.io/>`_
 * `Design Notes <http://mxnet.io/architecture/index.html>`_
 * `Code Examples <https://github.com/ROCmSoftwarePlatform/mxnet/blob/master/example>`_
 * `Installation <http://mxnet.io/get_started/setup.html>`_
 * `Pretrained Models <https://github.com/dmlc/mxnet-model-gallery>`_
 * `Contribute to MXNet <http://mxnet.io/community/contribute.html>`_
 * `Frequent Asked Questions <http://mxnet.io/how_to/faq.html>`_

Features
**********

 * Design notes providing useful insights that can re-used by other DL projects
 * Flexible configuration for arbitrary computation graph
 * Mix and match imperative and symbolic programming to maximize flexibility and efficiency
 * Lightweight, memory efficient and portable to smart devices
 * Scales up to multi GPUs and distributed setting with auto parallelism
 * Support for Python, R, C++ and Julia
 * Cloud-friendly and directly compatible with S3, HDFS, and Azure

Installation Guide for HIP Port
###################################

Generic Installation Steps:
+++++++++++++++++++++++++++++

Install the system requirement following the :ref:`Installation-Guide`

Installation Steps on HCC and NVCC PLATFORM
+++++++++++++++++++++++++++++++++++++++++++++++

Prerequisites
+++++++++++++++++
Install CUDA 8.0 following the NVIDIA’s `installation guide <http://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`_ to setup MXNet with GPU support

.. note:: Make sure to add CUDA install path to LD_LIBRARY_PATH. Example - export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

Building MXNet from source is a 2 step process.
*************************************************
Build the MXNet core shared library, libmxnet.so, from the C++ sources.
Build the language specific bindings. Example - Python bindings, Scala bindings.

Minimum Requirements
+++++++++++++++++++++
`GCC 4.8 <https://gcc.gnu.org/gcc-4.8/>`_ or later to compile C++ 11.
`GNU Make <https://www.gnu.org/software/make/>`_

ROCm installation
*******************

**Step 1:** Add the ROCm apt repository For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows:
::
 $ wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
 $ sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

**Step 2:** Install or Update Next, update the apt-get repository list and install/update the rocm package: Warning: Before proceeding, make sure to completely `uninstall any previous ROCm package <https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages>`_
::
 $ sudo apt-get update
 $ sudo apt-get install rocm

**Step 3:** Install dependent libraries
::
 $ sudo apt-get install rocm-device-libs rocblas rocm-libs 

For detailed installation steps refer the given `installation link <https://github.com/RadeonOpenCompute/ROCm>`_

Build the MXNet core shared library
************************************
**Step 1:** Install build tools and git.
::
 $ sudo apt-get update
 $ sudo apt-get install -y build-essential git

**Step 2:** Install `OpenCV <https://opencv.org/>`_

MXNet uses OpenCV for efficient image loading and augmentation operations.
::
 $ sudo apt-get install -y libopencv-dev

**Step 3:** To build MXNet with Thrust
::
 $ git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust

Add thrust path to the Makefile,
::
 ifeq ($(HIP_PLATFORM), hcc)
                HIPINCLUDE += -I<Root path of Thrust>
                <Example: HIPINCLUDE += -I../Thrust>
 endif

**Step 4:** Download MXNet sources and build MXNet core shared library.
::
 $ git clone --recursive https://github.com/ROCmSoftwarePlatform/mxnet
 $ cd mxnet

To compile on HCC PLATFORM:
::
 $ export HIP_PLATFORM=hcc
 $ make -jn (n = no of cores)

To compile on NVCC PLATFORM:
::
 $ export HIP_PLATFORM=nvcc
 $ make -jn (n = no of cores) 

**Note:**

1. USE_OPENCV, USE_BLAS, USE_CUDA, USE_CUDA_PATH are make file flags to set compilation options to use OpenCV, CUDA libraries. You can explore and use more compilation options in make/config.mk. Make sure to set USE_CUDA_PATH to right CUDA installation path. In most cases it is - /usr/local/cuda.
2. MXNet uses rocBLAS, hcFFT, hcRNG and lapack libraries for accelerated numerical computations. cuDNN is not enabled as it is being migrated to Miopen.

Install the MXNet Python binding
++++++++++++++++++++++++++++++++++++++

**Step 1:** Install prerequisites - python, setup-tools, python-pip and numpy.
::
 $ sudo apt-get install -y python-dev python-setuptools python-numpy python-pip

**Step 2:** Install the MXNet Python binding.
::
 $ cd python
 $ sudo python setup.py install 

