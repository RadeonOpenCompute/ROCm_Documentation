.. _Deep-learning:

Deep Learning on ROCm
######################

ROCm Tensorflow v1.12 Release
*****************************
We are excited to announce the release of ROCm enabled TensorFlow v1.12 for AMD GPUs.

Tensorflow Installation
***********************
First, you’ll need to install the open-source ROCm 2.0 stack. Details can be found here: https://rocm.github.io/ROCmInstall.html 

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

The ROCm 2.0 has prebuilt packages for MIOpen
***********************************************
Install the ROCm MIOpen implementation (assuming you already have the ‘rocm’ and ‘rocm-opencl-dev” package installed):

**For just OpenCL development**

::

  sudo apt-get install miopengemm miopen-opencl

**For HIP development**

::

  sudo apt-get install miopengemm miopen-hip

Or you can build from `source code <https://github.com/ROCmSoftwarePlatform/MIOpen>`_


Building PyTorch for ROCm
**************************

This is a quick guide to setup PyTorch with ROCm support inside a docker container. Assumes a .deb based system. See `ROCm install <https://rocm.github.io/ROCmInstall.html>`_ for supported operating systems and general information on the ROCm software stack.

A ROCm install version 2.0 is required currently.

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

  docker pull rocm/pytorch:rocm2.0

3. Clone PyTorch repository on the host:

::

  cd ~
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git submodule init
  git submodule update

4. Start a docker container using the downloaded image:

::

  sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:rocm2.0

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

Building Caffe2 for ROCm
**************************
This is a quick guide to setup Caffe2 with ROCm support inside docker container and run on AMD GPUs. Caffe2 with ROCm support offers complete functionality on a single GPU achieving great performance on AMD GPUs using both native ROCm libraries and custom hip kernels. This requires your host system to have rocm-2.0s drivers installed. Please refer to ROCm install to install ROCm software stack. If your host system doesn't have docker installed, please refer to docker install. It is recommended to add the user to the docker group to run docker as a non-root user, please refer here.

This guide provides two options to run Caffe2.
    1. Launch the docker container using a docker image with Caffe2 installed.
    2. Build Caffe2 from source inside a Caffe2 ROCm docker image.

Option 1: Docker image with Caffe2 installed:
********************************************
This option provides a docker image which has Caffe2 installed. Users can launch the docker container and train/run deep learning models directly. This docker image will run on both gfx900(Vega10-type GPU - MI25, Vega56, Vega64,...) and gfx906(Vega20-type GPU - MI50, MI60)

   * Launch the docker container

::

  docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video rocm/caffe2:238-2.0

This will automatically download the image if it does not exist on the host. You can also pass -v argument to mount any data directories on to the container.

Option 2: Install using Caffe2 ROCm docker image:
*************************************************
1.  Clone PyTorch repository on the host:
::

  cd ~
  git clone --recurse-submodules https://github.com/pytorch/pytorch.git
  cd pytorch
  sgit submodule update --init --recursive

2. Launch the docker container

::

  docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $PWD:/pytorch rocm/caffe2:unbuilt-238-2.0

3. Build Caffe2 from source

If running on gfx900/vega10-type GPU(MI25, Vega56, Vega64,...)
::

  .jenkins/caffe2/build.sh

If running on gfx906/vega20-type GPU(MI50, MI60)HCC_AMDGPU_TARGET=gfx906 
::

  .jenkins/caffe2/build.sh

Test the Caffe2 Installation
******************************
To validate Caffe2 installation, for both options, run

1. Test Command
::

  cd build_caffe2 && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

If the test fails, make sure the following environment variables are set. LD_LIBRARY_PATH=/pytorch/build_caffe2/lib
::

  PYTHONPATH=/pytorch/build_caffe2

2. Running unit tests in Caffe2
::

  .jenkins/caffe2/test.sh

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
 


























