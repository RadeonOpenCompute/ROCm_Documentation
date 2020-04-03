.. _Deep-learning:

=======================
Deep Learning on ROCm
=======================

***********
TensorFlow
***********

ROCm Tensorflow v1.14 Release
*****************************
We are excited to announce the release of ROCm enabled TensorFlow v1.14 for AMD GPUs.
In this release we have the following features enabled on top of upstream TF1.14 enhancements:
    * We integrated ROCm RCCL library for mGPU communication, details in `RCCL github repo <https://github.com/ROCmSoftwarePlatform/rccl>`_
    * XLA backend is enabled for AMD GPUs, the functionality is complete, performance optimization is in progress.  

ROCm Tensorflow v2.0.0-beta1 Release
*****************************
In addition to Tensorflow v1.14 release, we also enabled Tensorflow v2.0.0-beta1 for AMD GPUs. The TF-ROCm 2.0.0-beta1 release supports Tensorflow V2 API.
Both whl packages and docker containers are available below. 

Tensorflow Installation
***********************

First, you’ll need to install the open-source ROCm 3.0 stack. Details can be found `here <https://github.com/RadeonOpenCompute/ROCm>`_


Then, install these other relevant ROCm packages:
::
   sudo apt update
   sudo apt install rocm-libs miopen-hip cxlactivitylogger rccl

And finally, install TensorFlow itself (via the Python Package Index):
::
   sudo apt install wget python3-pip
   # Pip3 install the whl package from PyPI
   pip3 install --user tensorflow-rocm
Now that Tensorflow v2.0 is installed!

Tensorflow More Resources
*************************
Tensorflow docker images are also publicly available, more details can be found `here <https://hub.docker.com/r/rocm/tensorflow/>`_

The official github repository is `here <https://github.com/ROCmSoftwarePlatform/tensorflow-upstream>`_

*******
MIOpen
*******

ROCm MIOpen v2.0.1 Release
*************************
Announcing our new Foundation for Deep Learning acceleration MIOpen 2.0 which introduces support for Convolution Neural Network (CNN) acceleration — built to run on top of the ROCm software stack!

This release includes the following:
   
   * This release contains bug fixes and performance improvements.
   * Additionally, the convolution algorithm Implicit GEMM is now enabled by default
   * Known issues:
        * Backward propagation for batch normalization in fp16 mode may trigger NaN in some cases
        * Softmax Log mode may produce an incorrect result in back propagation
   * `Source code <https://github.com/ROCmSoftwarePlatform/MIOpen>`_
   * Documentation
       * `MIOpen <https://rocmsoftwareplatform.github.io/MIOpen/doc/html/apireference.html>`_
       * `MIOpenGemm <https://rocmsoftwareplatform.github.io/MIOpenGEMM/doc/html/index.html>`_

**Changes:**

   * Added Winograd multi-pass convolution kernel
   * Fixed issue with hip compiler paths
   * Fixed immediate mode behavior with auto-tuning environment variable
   * Fixed issue with system find-db in-memory cache, the fix enable the cache by default
   * Improved logging
   * Improved how symbols are hidden in the library
   * Updated default behavior to enable implicit GEMM

Porting from cuDNN to MIOpen
****************************
The `porting guide <https://github.com/dagamayank/ROCm.github.io/blob/master/doc/miopen_porting_guide.pdf>`_ highlights the key differences between the current cuDNN and MIOpen APIs.


The ROCm 3.0 has prebuilt packages for MIOpen
***********************************************
Install the ROCm MIOpen implementation (assuming you already have the ‘rocm’ and ‘rocm-opencl-dev” package installed):

MIOpen can be installed on Ubuntu using

::

  apt-get


**For just OpenCL development**

::

  sudo apt-get install miopengemm miopen-opencl

**For HIP development**

::

  sudo apt-get install miopengemm miopen-hip

Or you can build from `source code <https://github.com/ROCmSoftwarePlatform/MIOpen>`_

Currently both the backends cannot be installed on the same system simultaneously. If a different backend other than what currently exists on the system is desired, please uninstall the existing backend completely and then install the new backend.

*********
PyTorch
*********

Building PyTorch for ROCm
**************************

This is a quick guide to setup PyTorch with ROCm support inside a docker container. Assumes a .deb based system. See `ROCm install <https://github.com/RadeonOpenCompute/ROCm#supported-operating-systems---new-operating-systems-available>`_ for supported operating systems and general information on the ROCm software stack.


A ROCm install version 3.0 is required currently.

1. Install or update rocm-dev on the host system:

::

  sudo apt-get install rocm-dev
  or
  sudo apt-get update
  sudo apt-get upgrade

Recommended:Install using published PyTorch ROCm docker image:
**************************************************************

2. Obtain docker image:

::

  docker pull rocm/pytorch:rocm3.0_ubuntu16.04_py3.6_pytorch

3. Clone PyTorch repository on the host:

::

  cd ~
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git submodule init
  git submodule update

4. Start a docker container using the downloaded image:

::

  sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:rocm3.0_ubuntu16.04_py3.6_pytorch

Note: This will mount your host home directory on /data in the container.

5. Change to previous PyTorch checkout from within the running docker:

::

  cd /data/pytorch

6. Build PyTorch for ROCm:

Unless you are running a gfx900/Vega10-type GPU (MI25, Vega56, Vega64,...), explicitly export the GPU architecture to build for, e.g.:
export HCC_AMDGPU_TARGET=gfx906

then
::

  .jenkins/pytorch/build.sh

This will first hipify the PyTorch sources and then compile using 4 concurrent jobs, needing 16 GB of RAM to be available to the docker image.

7. Confirm working installation:

::

  PYTORCH_TEST_WITH_ROCM=1 python test/run_test.py --verbose

No tests will fail if the compilation and installation is correct.

8. Install torchvision:

::

  pip install torchvision

This step is optional but most PyTorch scripts will use torchvision to load models. E.g., running the pytorch examples requires torchvision.

9. Commit the container to preserve the pytorch install (from the host):

::

  sudo docker commit <container_id> -m 'pytorch installed'

Option 2: Install using PyTorch upstream docker file
****************************************************

2. Clone PyTorch repository on the host:

::

  cd ~
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git submodule init
  git submodule update

3. Build PyTorch docker image:

::
  
  cd pytorch/docker/caffe2/jenkins
  ./build.sh py2-clang7-rocmdeb-ubuntu16.04

This should complete with a message "Successfully built <image_id>"
Note here that other software versions may be chosen, such setups are currently not tested though!

4. Start a docker container using the new image:

::

  sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video <image_id>

Note: This will mount your host home directory on /data in the container.

5. Change to previous PyTorch checkout from within the running docker:

::

  cd /data/pytorch

6. Build PyTorch for ROCm:

Unless you are running a gfx900/Vega10-type GPU (MI25, Vega56, Vega64,...), explicitly export the GPU architecture to build for, e.g.:
export HCC_AMDGPU_TARGET=gfx906

then
::

  .jenkins/pytorch/build.sh

This will first hipify the PyTorch sources and then compile using 4 concurrent jobs, needing 16 GB of RAM to be available to the docker image.

7. Confirm working installation:

::

  PYTORCH_TEST_WITH_ROCM=1 python test/run_test.py --verbose

No tests will fail if the compilation and installation is correct.

8. Install torchvision:

::

  pip install torchvision

This step is optional but most PyTorch scripts will use torchvision to load models. E.g., running the pytorch examples requires torchvision.

9. Commit the container to preserve the pytorch install (from the host):

::

  sudo docker commit <container_id> -m 'pytorch installed'

Option 3: Install using minimal ROCm docker file
************************************************

2. Download pytorch dockerfile:

`Dockerfile <https://github.com/ROCmSoftwarePlatform/pytorch/wiki/Dockerfile>`_

3. Build docker image:

::

  cd pytorch_docker
  sudo docker build .

This should complete with a message "Successfully built <image_id>"

4. Start a docker container using the new image:

::

  sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video <image_id>

Note: This will mount your host home directory on /data in the container.

5. Clone pytorch master (on to the host):

::
  
  cd ~
  git clone https://github.com/pytorch/pytorch.git or git clone https://github.com/ROCmSoftwarePlatform/pytorch.git
  cd pytorch
  git submodule init
  git submodule update

6. Run "hipify" to prepare source code (in the container):

::

  cd /data/pytorch/
  python tools/amd_build/build_pytorch_amd.py
  python tools/amd_build/build_caffe2_amd.py

7. Build and install pytorch:

Unless you are running a gfx900/Vega10-type GPU (MI25, Vega56, Vega64,...), explicitly export the GPU architecture to build for, e.g.:
export HCC_AMDGPU_TARGET=gfx906

then
::

  USE_ROCM=1 MAX_JOBS=4 python setup.py install --user 

UseMAX_JOBS=n to limit peak memory usage. If building fails try falling back to fewer jobs. 4 jobs assume available main memory of 16 GB or larger.

8. Confirm working installation:

::

  PYTORCH_TEST_WITH_ROCM=1 python test/run_test.py --verbose

No tests will fail if the compilation and installation is correct.

9. Install torchvision:

::

  pip install torchvision

This step is optional but most PyTorch scripts will use torchvision to load models. E.g., running the pytorch examples requires torchvision.

10. Commit the container to preserve the pytorch install (from the host):

::

  sudo docker commit <container_id> -m 'pytorch installed'

Try PyTorch examples
*************************

1. Clone the PyTorch examples repository:

::

  git clone https://github.com/pytorch/examples.git

2. Run individual example: MNIST

::

  cd examples/mnist

Follow instructions in README.md, in this case:
::

  pip install -r requirements.txt python main.py

3. Run individual example: Try ImageNet training

::

  cd ../imagenet

Follow instructions in README.md.


*******
Caffe2
*******


Building Caffe2 for ROCm
**************************
This is a quick guide to setup Caffe2 with ROCm support inside docker container and run on AMD GPUs. Caffe2 with ROCm support offers complete functionality on a single GPU achieving great performance on AMD GPUs using both native ROCm libraries and custom hip kernels. This requires your host system to have rocm-3.0s drivers installed. Please refer to `ROCm install <https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#installing-from-amd-rocm-repositories>`_ to install ROCm software stack. If your host system doesn't have docker installed, please refer to `docker install <https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce>`_. It is recommended to add the user to the docker group to run docker as a non-root user, please refer `here <https://docs.docker.com/install/linux/linux-postinstall/>`_.

This guide provides two options to run Caffe2.
    1. Launch the docker container using a docker image with Caffe2 installed.
    2. Build Caffe2 from source inside a Caffe2 ROCm docker image.

Option 1: Docker image with Caffe2 installed:
********************************************
This option provides a docker image which has Caffe2 installed. Users can launch the docker container and train/run deep learning models directly. This docker image will run on both gfx900(Vega10-type GPU - MI25, Vega56, Vega64,...) and gfx906(Vega20-type GPU - MI50, MI60)

1.  Launch the docker container

::

  docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:rocm3.0_ubuntu16.04_py3.6_caffe2

This will automatically download the image if it does not exist on the host. You can also pass -v argument to mount any data directories on to the container.

Option 2: Install using Caffe2 ROCm docker image:
*************************************************
1.  Clone PyTorch repository on the host:
::

  cd ~
  git clone --recurse-submodules https://github.com/pytorch/pytorch.git
  cd pytorch
  git submodule update --init --recursive

2. Launch the docker container

::

  docker pull rocm/pytorch:pytorch:rocm3.0_ubuntu16.04_py3.6_caffe2
  docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $PWD:/pytorch rocm/pytorch:rocm3.0_ubuntu16.04_py3.6_caffe2

3. Build Caffe2 from source
::

  cd /pytorch
If running on gfx900/vega10-type GPU(MI25, Vega56, Vega64,...)
::

  .jenkins/caffe2/build.sh
If running on gfx906/vega20-type GPU(MI50, MI60)
::

  HCC_AMDGPU_TARGET=gfx906 .jenkins/caffe2/build.sh

Test the Caffe2 Installation
******************************
To validate Caffe2 installation, run

1. Test Command
::

  cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

2. Running unit tests in Caffe2
::

  cd /pytorch
  .jenkins/caffe2/test.sh

Run benchmarks
**************

Caffe2 benchmarking script supports the following networks MLP, AlexNet, OverFeat, VGGA, Inception

To run benchmarks for networks MLP, AlexNet, OverFeat, VGGA, Inception run the command from pytorch home directory replacing <name_of_the_network> with one of the networks.
::

  python caffe2/python/convnet_benchmarks.py --batch_size 64 --model <name_of_the_network> --engine MIOPEN

Running example scripts
************************

Please refer to the example scripts in ``caffe2/python/examples``. It currently has ``resnet50_trainer.py`` which can run ResNet's, ResNeXt's with various layer, groups, depth configurations and ``char_rnn.py`` which uses RNNs to do character level prediction.

Building own docker images
***************************

After cloning the pytorch repository, you can build your own Caffe2 ROCm docker image. Navigate to pytorch repo and run
::

  cd docker/caffe2/jenkins
  ./build.sh py2-clang7-rocmdeb-ubuntu16.04

This should complete with a message "Successfully built <image_id>" which can then be used to install Caffe2 as in Option 2 above.


*******************************************
Deep Learning Framework support for ROCm
*******************************************

+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Framework  | Status      | MIOpen Enabled | Upstreamed     | Current Repository                                  |
+============+=============+================+================+=====================================================+
| Caffe      | Public      | Yes            |                | https://github.com/ROCmSoftwarePlatform/hipCaffe    |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Tensorflow | Development | Yes            | CLA inProgress | Notes: Working on NCCL and XLA enablement, Running  |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Caffe2     | Upstreaming | Yes            | CLA inProgress |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| Torch      | HIP         | Upstreaming    | Development    | https://github.com/ROCmSoftwarePlatform/cutorch_hip |
|            |             |                | inProgress     |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| PyTorch    | Development | Development    |                |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| MxNet      | Development | Development    |                | https://github.com/ROCmSoftwarePlatform/mxnet       |
+------------+-------------+----------------+----------------+-----------------------------------------------------+
| CNTK       | Development | Development    |                |                                                     |
|            |             |                |                |                                                     |
+------------+-------------+----------------+----------------+-----------------------------------------------------+

*************
Tutorials
*************
**hipCaffe**

* :ref:`caffe`
  
**MXNet**
  
* :ref:`mxnet`
 


























