
.. _hipCaffe:

hipCaffe Quickstart Guide
###########################

In this quickstart guide, we’ll walk through the steps for ROCm installation. Then, we’ll run a few training and inference experiments and check their accuracy.

Install ROCm
-------------
Here are the main ROCm components we’ll be using::

 sudo apt-get install rocm
 sudo apt-get install rocm-libs
 sudo apt-get install miopen-hip miopengemm
 
And some misc packages::
 
 sudo apt-get install -y \
      g++-multilib \
      libunwind-dev \
      git \
      cmake cmake-curses-gui \
      vim \
      emacs-nox \
      curl \
      wget \
      rpm \
      unzip \
      bc
 
Verify ROCm
------------
Test a simple HIP sample::
 
 cp -r /opt/rocm/hip/samples ~/hip-samples && cd ~/hip-samples/0_Intro/square/
 
 make
 
 ./square.hip.out
  
Install hipCaffe
----------------
Handle the Caffe dependencies first::
 
 sudo apt-get install -y \
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
      python-skimage python-opencv python-protobuf \
      libopencv-dev \
      libfftw3-dev \
      libelf-dev
 
Note that you might need minor changes to Makefile.config (system dependent)::
 
 cd ~
 
 git clone https://github.com/ROCmSoftwarePlatform/hipCaffe.git
 
 cd hipCaffe
 
 cp ./Makefile.config.example ./Makefile.config
 
 make -j$(nproc)
 

Workloads
-----------
MNIST training
+++++++++++++++

Details on MNIST training can be found at this `link <https://github.com/BVLC/caffe/blob/master/examples/mnist/readme.md>`_. 
 
Here are the basic instructions::

 ./data/mnist/get_mnist.sh
 ./examples/mnist/create_mnist.sh
 ./examples/mnist/train_lenet.sh
 
Expected result: >99% accuracy after 10000 iterations
::

 I0717 21:06:03.349702  9965 solver.cpp:279] Solving LeNet
 I0717 21:06:03.349711  9965 solver.cpp:280] Learning Rate Policy: inv
 I0717 21:06:03.351486  9965 solver.cpp:337] Iteration 0, Testing net (#0)
 I0717 21:06:05.472965  9965 solver.cpp:404]     Test net output #0: accuracy = 0.1296
 I0717 21:06:05.473023  9965 solver.cpp:404]     Test net output #1: loss = 2.49735 (* 1 = 2.49735 loss)
 I0717 21:06:08.612304  9965 solver.cpp:228] Iteration 0, loss = 2.42257
 I0717 21:06:08.612390  9965 solver.cpp:244]     Train net output #0: loss = 2.42257 (* 1 = 2.42257 loss)
 I0717 21:06:08.612417  9965 sgd_solver.cpp:106] Iteration 0, lr = 0.01
 ...
 I0717 21:06:58.502200  9965 solver.cpp:317] Iteration 10000, loss = 0.00258486
 I0717 21:06:58.502228  9965 solver.cpp:337] Iteration 10000, Testing net (#0)
 I0717 21:06:58.701591  9965 solver.cpp:404]     Test net output #0: accuracy = 0.9917
 I0717 21:06:58.701642  9965 solver.cpp:404]     Test net output #1: loss = 0.0269806 (* 1 = 0.0269806 loss)
 I0717 21:06:58.701668  9965 solver.cpp:322] Optimization Done.
  

CIFAR-10 training
++++++++++++++++++

Details on CIFAR-10 training can be found at this `link <https://github.com/BVLC/caffe/blob/master/examples/cifar10/readme.md>`_.

Here are the basic instructions::
 
 ./data/cifar10/get_cifar10.sh
 ./examples/cifar10/create_cifar10.sh
 ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt
 
Expected result: >70% accuracy after 4000 iterations
::
 
 I0727 18:29:35.248363    33 solver.cpp:279] Solving CIFAR10_quick
 I0727 18:29:35.248366    33 solver.cpp:280] Learning Rate Policy: fixed
 I0727 18:29:35.248883    33 solver.cpp:337] Iteration 0, Testing net (#0)
 I0727 18:29:37.263290    33 solver.cpp:404]     Test net output #0: accuracy = 0.0779
 I0727 18:29:37.263319    33 solver.cpp:404]     Test net output #1: loss = 2.30241 (* 1 = 2.30241 loss)
 I0727 18:29:40.074849    33 solver.cpp:228] Iteration 0, loss = 2.3028
 I0727 18:29:40.074874    33 solver.cpp:244]     Train net output #0: loss = 2.3028 (* 1 = 2.3028 loss)
 I0727 18:29:40.074894    33 sgd_solver.cpp:106] Iteration 0, lr = 0.001
 ...
 I0727 18:30:13.425905    33 solver.cpp:317] Iteration 4000, loss = 0.536751
 I0727 18:30:13.425920    33 solver.cpp:337] Iteration 4000, Testing net (#0)
 I0727 18:30:13.722070    33 solver.cpp:404]     Test net output #0: accuracy = 0.7124
 I0727 18:30:13.722090    33 solver.cpp:404]     Test net output #1: loss = 0.848089 (* 1 = 0.848089 loss)
 I0727 18:30:13.722095    33 solver.cpp:322] Optimization Done.
 

CaffeNet inference
+++++++++++++++++++

Details on CaffeNet inference can be found at this `link <https://github.com/BVLC/caffe/blob/master/examples/cpp_classification/readme.md>`_.

Here are the basic instructions::
 
 ./data/ilsvrc12/get_ilsvrc_aux.sh
 ./scripts/download_model_binary.py models/bvlc_reference_caffenet
 ./build/examples/cpp_classification/classification.bin models/bvlc_reference_caffenet/deploy.prototxt models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg
 

Expected result: (note the ordering and associated percentages)
::
 
 ---------- Prediction for examples/images/cat.jpg ----------
 0.3134 - "n02123045 tabby, tabby cat"
 0.2380 - "n02123159 tiger cat"
 0.1235 - "n02124075 Egyptian cat"
 0.1003 - "n02119022 red fox, Vulpes vulpes"
 0.0715 - "n02127052 lynx, catamount"
 

