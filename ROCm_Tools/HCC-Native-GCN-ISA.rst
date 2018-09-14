.. _HCC-Native-GCN-ISA:

==========================
HCC-Native-GCN-ISA
==========================

Hardware Requirements
***********************
See the "Target Platform Supported" section `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#system-requirement>`_.

Note that the instructions on this wiki are for installing on a system with a Fiji GPU. If you are interested in learning how to setup hcc for other supported GPU targets, please file an `Issue <https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA/issues>`_.

Installing compiler packages:
********************************
If you want to install the latest stable version of the hcc compiler with native GCN ISA support, you should follow the instructions here. If you want to build from source, follow the directions below.

Building compiler from source:
******************************
**Software Dependencies**

**Ubuntu**

Follow the instructions found `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#>`_ to setup the apt repository.

Installing on the ROC kernel, the development files of the thunk library and the runtime:

::

  sudo apt-get install rocm-kernel hsakmt-roct-dev hsa-rocr-dev

Reboot the system with the new kernel and make sure the verify the installation with the `vector copy sample <https://github.com/RadeonOpenCompute/ROCm#verify-installation>`_.

If you have previously installed any version of the hcc compiler, then it is recommended to uninstall them:

::

  sudo apt-get purge hcc_lc hcc_hsail

Then install all other dependencies in order to build HCC from source:

::

  sudo apt-get install cmake git libelf-dev libc++abi-dev libc++-dev libdwarf-dev re2c libncurses5-dev patch wget file xz-utils       	libc6- dev-i386 python build-essential
  
**CMake**

If you are using Ubuntu 14.04, you would also need to upgrade to a newer version (>=3.0) of CMake as the version distributed by the distro is old for building clang/llvm.

**Fedora**

Follow the instructions found `here <https://github.com/RadeonOpenCompute/ROCm#rpm-repository---dnf-yum>`_ to setup the rpm repository.

Installing on the ROC kernel, the development files of the thunk library and the runtime:

::

  sudo dnf install rocm-kernel hsakmt-roct-dev hsa-rocr-dev

Reboot the system with the new kernel and make sure the verify the installation with the `vector copy sample <https://github.com/RadeonOpenCompute/ROCm#verify-installation>`_.

If you have previously installed any version of the hcc compiler, thne it is recommended to uninstall them:

::

   sudo dnf remove hcc_lc hcc_hsail

Then install all other dependencies in order to build HCC from source:

::

  sudo dnf install cmake make git gcc-c++ libstdc++-devel libdwarf-devel elfutils-libelf-devel re2c ncurses-devel patch wget file tar 	xz glibc-devel.i686 python rpmdevtools

Install other development tools:

::

  sudo dnf groupinstall "Development Tools"
  
**libc++ & libc++abi**

HCC has a dependency on libc++ and libc++abi; however, Fedora/RHEL/CentOS don't provide a working binary package so you will to build them from source by following the instructions `here <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#hcc>`_

**CentOS/RHEL**

Follow the instructions found `here <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide-fedora>`_ to setup the rpm repository.

Installing on the ROC kernel, the development files of the thunk library and the runtime:

::

  sudo yum install rocm-kernel hsakmt-roct-dev hsa-rocr-dev

Reboot the system with the new kernel and make sure the verify the installation with the `vector copy sample <https://github.com/RadeonOpenCompute/ROCm#verify-installation>`_.

Then install

::

  sudo yum install cmake make git gcc-c++ libstdc++-devel libdwarf-devel elfutils-libelf-devel re2c ncurses-devel patch wget file tar 	xz glibc-devel.i686 python rpmdevtools clang
  sudo yum groupinstall "Development Tools"

**CMake**

The CMake version from CentOS 7 and RedHat 7 is too old and it doesn't meet the minimum requirement for building LLVM and Clang. You'll need to upgrade to newer verison of `CMake <https://cmake.org/>`_.

**libc++ & libc++abi**

On Ubuntu 14.04, HCC has a dependency on libc++; however, the current libc++ package from the distro has an unmet dependency on libc++abi. Users will have to build libc++ and libc++abi from source with Clang.

It is recommended to install the release_36 release of libc++ and libc++abi and here are the instructions:

::

  git clone --branch release_36 https://github.com/llvm-mirror/llvm.git llvm
  git clone --branch release_36 https://github.com/llvm-mirror/libcxx.git llvm/projects/libcxx
  git clone --branch release_36 https://github.com/llvm-mirror/libcxxabi.git llvm/projects/libcxxabi
  mkdir -p llvm/build
  cd llvm/build
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++  ..
  make
  cd projects/libcxx
  sudo make install
  cd ../libcxxabi
  sudo make install
  
Add the libc++ and libc++abi installation path to the library search paths
(i.e. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib )

Getting the HCC Source Code
*******************************
**Fetching the HCC source code with the repo tool**

**Installing the repo tool**

If you are unable to find a distribution package for repo, you can follow the installation instructions here.

**Initialize the Workspace**

Create a workspace for holding the code and all the repo metadata. Replace <_branch_> with a branch name selected from above.

::

  mkdir hcc
  cd hcc
  repo init -u https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA.git

Fetch the source code

::

  repo sync
  
**Build Instructions**

::

  mkdir hcc/build
  cd hcc/build

  ### Substitute <_distro_> with ubuntu for Ubuntu or with fedora for Fedora/CentOS/RHEL
  cmake .. \
  -DDISTRO=<_distro_>

  make

  # optional step to build binary packages for distribution
  make package

  cd ../..

**Verifying the Build**

To verify that you have set up your system correctly, run one of the C++ AMP conformance tests. If you have done things correctly, it should pass:

::

  cd hcc/build
  perl amp-conformance/test_one.pl ../amp-conformance/ ../amp-conformance/Tests/4_Basic_Data_Elmnts/4_1_index/4_1_2_c/Copy/Test.01/   	test.cpp

You could also run the HCC's sanity test

::

  make test
  
**Install the Compiler**

::

  sudo make install

Or alternatively, you could generate a .deb or .rpm package

::

  make package

