=====================
ROCmValidationSuite
=====================

The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

ROCmValidationSuite Modules
******************************

**GPU Properties – GPUP**

The GPU Properties module queries the configuration of a target device and returns the device’s static characteristics. These static values can be used to debug issues such as device support, performance and firmware problems.

**GPU Monitor – GM module**

The GPU monitor tool is capable of running on one, some or all of the GPU(s) installed and will report various information at regular intervals. The module can be configured to halt another RVS modules execution if one of the quantities exceeds a specified boundary value.

**PCI Express State Monitor – PESM module?**

The PCIe State Monitor tool is used to actively monitor the PCIe interconnect between the host platform and the GPU. The module will register a “listener” on a target GPU’s PCIe interconnect, and log a message whenever it detects a state change. The PESM will be able to detect the following state changes:

    * PCIe link speed changes
    * GPU power state changes


**PCI Express Qualification Tool – PEQT module**

The PCIe Qualification Tool consists is used to qualify the PCIe bus on which the GPU is connected. The qualification test will be capable of determining the following characteristics of the PCIe bus interconnect to a GPU:

    * Support for Gen 3 atomic completers
    * DMA transfer statistics
    * PCIe link speed
    * PCIe link width


**P2P Benchmark and Qualification Tool – PBQT module**

The P2P Benchmark and Qualification Tool is designed to provide the list of all GPUs that support P2P and characterize the P2P links between peers. In addition to testing for P2P compatibility, this test will perform a peer-to-peer throughput test between all P2P pairs for performance evaluation. The P2P Benchmark and Qualification Tool will allow users to pick a collection of two or more GPUs on which to run. The user will also be able to select whether or not they want to run the throughput test on each of the pairs.

Please see the web page “ROCm, a New Era in Open GPU Computing” to find out more about the P2P solutions available in a ROCm environment.

**PCI Express Bandwidth Benchmark – PEBB module**

The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA transfers between system memory and a target GPU card’s memory. The maximum bandwidth obtained is reported to help debug low bandwidth issues. The benchmark should be capable of targeting one, some or all of the GPUs installed in a platform, reporting individual benchmark statistics for each.

**GPU Stress Test - GST module**

The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM (Single/Double-precision General Matrix Multiplication) workload on one, some or all GPUs. The GPUs can be of the same or different types. The duration of the benchmark should be configurable, both in terms of time (how long to run) and iterations (how many times to run).

The test should be capable driving the power level equivalent to the rated TDP of the card, or levels below that. The tool must be capable of driving cards at TDP-50% to TDP-100%, in 10% incremental jumps. This should be controllable by the user.

**Input EDPp Test - IET module**

The Input EDPp Test generates EDP peak power on all input rails. This test is used to verify if the system PSU is capable of handling the worst case power spikes of the board. Peak Current at defined period = 1 minute moving average power.

Examples and about config files `link <https://github.com/ROCm-Developer-Tools/ROCmValidationSuite/blob/roc-3.0.0/doc/ugsrc/ug1main.md>`_.

Prerequisites
***************

Ubuntu :

::

    sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git

CentOS :

::

    sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ 

RHEL :

::

  sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ 
    
  wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm
    
  sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm

SLES :

::

  sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64
   
  sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64
   
  sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ 

Install ROCm stack, rocblas and rocm_smi64
*********************************************

Install ROCm stack for Ubuntu/CentOS, Refer https://github.com/RadeonOpenCompute/ROCm

Install rocBLAS and rocm_smi64 :

Ubuntu :

::

  sudo apt-get install rocblas rocm_smi64

CentOS & RHEL :

::

  sudo yum install rocblas rocm_smi64

SUSE :

::

  sudo zypper install rocblas rocm_smi64


**Note:** If rocm_smi64 is already installed but "/opt/rocm/rocm_smi/ path doesn't exist. Do below:

Ubuntu : sudo dpkg -r rocm_smi64 && sudo apt install rocm_smi64

CentOS & RHEL : sudo rpm -e rocm_smi64 && sudo yum install rocm_smi64

SUSE : sudo rpm -e rocm_smi64 && sudo zypper install rocm_smi64

Building from Source
********************** 

This section explains how to get and compile current development stream of RVS.

**Clone repository**

::

  git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git


**Configure and build RVS:**

::

  cd ROCmValidationSuite


If OS is Ubuntu and SLES, use cmake

::

  cmake ./ -B./build
 
  make -C ./build


If OS is CentOS and RHEL, use cmake3

::

  cmake3 ./ -B./build

  make -C ./build


Build package:
 
::
 
  cd ./build
 
  make package

Note:_ based on your OS, only DEB or RPM package will be built. You may ignore an error for the unrelated configuration

**Install package:**

::

  Ubuntu : sudo dpkg -i rocm-validation-suite*.deb
  CentOS & RHEL & SUSE : sudo rpm -i --replacefiles --nodeps rocm-validation-suite*.rpm


**Running RVS**

Running version built from source code:

::

  cd ./build/bin
  sudo ./rvs -d 3
  sudo ./rvsqa.new.sh  ; It will run complete rvs test suite


Regression
***********

Regression is currently implemented for PQT module only. It comes in the form of a Python script run_regression.py.

The script will first create valid configuration files on $RVS_BUILD/regression folder. It is done by invoking prq_create_conf.py script to generate valid configuration files. If you need different tests, modify the prq_create_conf.py script to generate them.

Then, it will iterate through generated files and invoke RVS to specifying also JSON output and -d 3 logging level.

Finally, it will iterate over generated JSON output files and search for ERROR string. Results are written into $RVS_BUILD/regression/regression_res file.

Results are written into $RVS_BUILD/regression/

**Environment variables**

Before running the run_regression.py you first need to set the following environment variables for location of RVS source tree and build folders (ajdust for your particular clone):

::

  export WB=/work/yourworkfolder
  export RVS=$WB/ROCmValidationSuite
  export RVS_BUILD=$RVS/../build

**Running the script**

Just do:

::

  cd $RVS/regression
  ./run_regression.py
