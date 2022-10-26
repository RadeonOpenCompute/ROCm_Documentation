.. _caffe:

hipCaffe: the HIP Port of Caffe
################################

Introduction
-------------
This repository hosts the HIP port of Caffe (or hipCaffe, for short). For details on HIP, please refer here. This HIP-ported framework is able to target both AMD ROCm and Nvidia CUDA devices from the same source code. Hardware-specific optimized library calls are also supported within this codebase.

Prerequisites
--------------
Hardware Requirements
+++++++++++++++++++++++

* For ROCm hardware requirements, `see here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#system-requirement>`_ .

Software and Driver Requirements
+++++++++++++++++++++++++++++++++
* For ROCm software requirements, `see here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-ubuntu>`_

Installation
-------------

AMD ROCm Installation
+++++++++++++++++++++++

For further background information on ROCm, refer `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-ubuntu>`_.

Installing ROCm Debian packages:
::

  PKG_REPO="http://repo.radeon.com/rocm/apt/debian/"
   
  curl -fsSL /rocm.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/rocm-archive-keyring.gpg
  
  sudo sh -c "echo deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] $PKG_REPO xenial main > /etc/apt/sources.list.d/rocm.list"
 
  sudo apt-get update
  
  sudo apt-get install rocm rocm-utils rocm-opencl rocm-opencl-dev rocm-profiler cxlactivitylogger

  echo 'export PATH=/opt/rocm/bin:$PATH' >> $HOME/.bashrc
  
  echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> $HOME/.bashrc

  source $HOME/.bashrc
  
  sudo reboot
  
Then, verify the installation. Double-check your kernel (at a minimum, you should see "kfd" in the name)::
 
   uname -r
  
In addition, check that you can run the simple HSA vector_copy sample application::
  
  cd /opt/rocm/hsa/sample
  make
  ./vector_copy
  
Pre-requisites Installation
++++++++++++++++++++++++++++

Install Caffe dependencies::
 
 sudo apt-get install \
 	pkg-config \
 	protobuf-compiler \
 	libprotobuf-dev \
 	libleveldb-dev \
 	libsnappy-dev \
 	libhdf5-serial-dev \
 	libatlas-base-dev \
 	libboost-all-dev \
 	libgflags-dev \
 	libgoogle-glog-dev \
 	liblmdb-dev \
 	python-numpy python-scipy python3-dev python-yaml python-pip \
 	libopencv-dev \
 	libfftw3-dev \
 	libelf-dev
 

Install the necessary ROCm compute libraries::
 
 sudo apt-get install rocm-libs miopen-hip miopengemm

hipCaffe Build Steps
+++++++++++++++++++++
Clone hipCaffe::
 
 git clone https://github.com/ROCmSoftwarePlatform/hipCaffe.git 
 
 cd hipCaffe
 
You may need to modify the Makefile.config file for your own installation. Then, build it::
 
 cp ./Makefile.config.example ./Makefile.config
 make 

To improve build time, consider invoking parallel make with the "-j$(nproc)" flag.

Unit Testing
-------------

Run the following commands to perform unit testing of different components of Caffe.
:: 
 make test
 ./build/test/test_all.testbin

Example Workloads
------------------

MNIST training
++++++++++++++++

Steps::
 
    ./data/mnist/get_mnist.sh
    ./examples/mnist/create_mnist.sh
    ./examples/mnist/train_lenet.sh

CIFAR-10 training
++++++++++++++++++

Steps::
 
    ./data/cifar10/get_cifar10.sh
    ./examples/cifar10/create_cifar10.sh
    ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt

CaffeNet inference
+++++++++++++++++++
Steps::

   ./data/ilsvrc12/get_ilsvrc_aux.sh
   ./scripts/download_model_binary.py models/bvlc_reference_caffenet
   ./build/examples/cpp_classification/classification.bin \ models/bvlc_reference_caffenet/deploy.prototxt \models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \data/ilsvrc12/imagenet_mean.binaryproto \data/ilsvrc12/synset_words.txt \examples/images/cat.jpg

Soumith's Convnet benchmarks
+++++++++++++++++++++++++++++++

Steps:

::
  
  git clone https://github.com/soumith/convnet-benchmarks.git
  cd convnet-benchmarks/caffe



OPTIONAL: reduce the batch sizes to avoid running out of memory for GoogleNet and VGG. For example, these configs work on Fiji: sed -i 's|input_dim: 128|input_dim: 8|1' imagenet_winners/googlenet.prototxt

::

  export CAFFE_ROOT=/path/to/your/caffe/installation
  sed -i 's#./caffe/build/tools/caffe#$CAFFE_ROOT/build/tools/caffe#' ./run_imagenet.sh
  ./run_imagenet.sh

Known Issues
-------------

Temp workaround for multi-GPU data transfer error
++++++++++++++++++++++++++++++++++++++++++++++++++

Sometimes when training with multiple GPUs, we hit this type of error signature::


 *** SIGSEGV (@0x0) received by PID 57122 (TID 0x7fd841500b80) from PID 0; stack trace: ***
     @     0x7fd8409a1390 (unknown)
     @     0x7fd8400a71f7 (unknown)
     @     0x7fd840515263 (unknown)
     @     0x7fd81f5ef907 UnpinnedCopyEngine::CopyHostToDevice()
     @     0x7fd81f5d3bb9 HSACopy::syncCopyExt()
     @     0x7fd81f5d28bc Kalmar::HSAQueue::copy_ext()
     @     0x7fd8410dba5b ihipStream_t::locked_copySync()
     @     0x7fd8411030bf hipMemcpy
     @           0x6cfd43 caffe::caffe_gpu_rng_uniform()
     @           0x5a32ba caffe::DropoutLayer<>::Forward_gpu()
     @           0x430bbf caffe::Layer<>::Forward()
     @           0x6fefe7 caffe::Net<>::ForwardFromTo()
     @           0x6feeff caffe::Net<>::Forward()
     @           0x801e8c caffe::Solver<>::Step()
     @           0x8015c3 caffe::Solver<>::Solve()
     @           0x71a277 caffe::P2PSync<>::Run()
     @           0x42dcbc train()
 

See this `comment <https://github.com/ROCmSoftwarePlatform/hipCaffe/issues/11#issuecomment-318518802>`_.

In short, here's the temporary workaround::

 export HCC_UNPINNED_COPY_MODE=2

Tutorials
----------

:ref:`hipCaffe`


