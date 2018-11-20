.. _Deep-learning:

Deep Learning on ROCm
######################

ROCm Tensorflow v1.12 Release
*****************************
We are excited to announce the release of ROCm enabled TensorFlow v1.12 for AMD GPUs.

Tensorflow Installation
***********************
First, you’ll need to install the open-source ROCm 1.9.2 stack. Details can be found here: https://rocm.github.io/ROCmInstall.html 

Then, install these other relevant ROCm packages:
::
   sudo apt update
   sudo apt install rocm-libs miopen-hip cxlactivitylogger

And finally, install TensorFlow itself (via the Python Package Index):
::
   sudo apt install wget python3-pip
   # Pip3 install the whl package from PyPI
   pip3 install --user tensorflow-rocm
Now that Tensorflow v1.12 is installed!

Tensorflow More Resources
*************************
Tensorflow docker images are also publicly available, more details can be found here: https://hub.docker.com/r/rocm/tensorflow/

Please connect with us for any questions, our official github repository is here: https://github.com/ROCmSoftwarePlatform/tensorflow-upstream

ROCm MIOpen v1.6 Release
*************************
Announcing our new Foundation for Deep Learning acceleration MIOpen 1.6 which introduces support for Convolution Neural Network (CNN) acceleration — built to run on top of the ROCm software stack!

This release includes the following:
   * Training in fp16 (half precision) including mixed-precision is now fully supported
   * Batch Normalization in fp16 (half precision) including mixed-precision are now available
   * Performance improvements for 3x3 and 1x1 single-precision convolutions
   * Layer fusions for BatchNorm+Activation are now available
   * Layer fusions with convolutions now support varying strides and padding configurations
   * Support for OpenCL and HIP enabled frameworks API's
   * MIOpen Driver enables the testing of forward/backward calls of any particular layer in MIOpen.
   * Binary Package support for Ubuntu 16.04 and Fedora 24
   * Source code at https://github.com/ROCmSoftwarePlatform/MIOpen
   * Documentation
       * `MIOpen <https://rocmsoftwareplatform.github.io/MIOpen/doc/html/apireference.html>`_
       * `MIOpenGemm <https://rocmsoftwareplatform.github.io/MIOpenGEMM/doc/html/index.html>`_

Porting from cuDNN to MIOpen
****************************
The `porting guide <https://github.com/dagamayank/ROCm.github.io/blob/master/doc/miopen_porting_guide.pdf>`_ highlights the key differences between the current cuDNN and MIOpen APIs.

The ROCm 1.9.2 has prebuilt packages for MIOpen
***********************************************
Install the ROCm MIOpen implementation (assuming you already have the ‘rocm’ and ‘rocm-opencl-dev” package installed):

**For just OpenCL development**

::

  sudo apt-get install miopengemm miopen-opencl

**For HIP development**

::

  sudo apt-get install miopengemm miopen-hip

Or you can build from `source code <https://github.com/ROCmSoftwarePlatform/MIOpen>`_

Deep Learning Framework support for ROCm
*******************************************

+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Framework  | Status      | MIOpen Enabled | Upstreamed     | Current Repository                                  |
+============+=============+================+================+=====================================================+
| Caffe      | Public      | Yes            |                | https://github.com/ROCmSoftwarePlatform/hipCaffe    |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Tensorflow | Development | Yes            | CLA inProgress | Notes: Working on NCCL and XLA enablement, Running  |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Caffe2     | Upstreaming | Yes            | CLA inProgress | https://github.com/ROCmSoftwarePlatform/caffe2      |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Torch      | HIP         | Upstreaming    | Development    | https://github.com/ROCmSoftwarePlatform/cutorch_hip |
|            |             |                | inProgress     |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| HIPnn      | Upstreaming | Development    |                | https://github.com/ROCmSoftwarePlatform/cunn_hip    |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| PyTorch    | Development | Development    |                |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| MxNet      | Development | Development    |                | https://github.com/ROCmSoftwarePlatform/mxnet       |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| CNTK       | Development | Development    |                |                                                     |
|            |             |                |                |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+

Tutorials
*************
**hipCaffe**

* :ref:`caffe`
  
**MXNet**
  
* :ref:`mxnet`
 


























