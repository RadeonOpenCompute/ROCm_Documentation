ROCm Installation Guide
=======================


Introduction 
--------------------------------------

The ROCm Platform brings a rich foundation to advanced computing by seamlessly integrating the CPU and GPU with the goal of solving real-world problems.

ROCm started  with just the support of AMD’s FIJI Family of dGPUs. Starting with ROCm 1.3 we further extends support to include the Polaris Family of ASICs. With ROCm 1.6 we added Vega Family of products. 

Supported CPU's
--------------------------------------

* Radeon R9 Nano, R9 Fury, R9 Fury X, FirePro S9300x2 need a CPU that support PCIe Gen 3 and PCIe Atomics,  Currently Intel Haswell or newer CPU support this fuctionality. Example Intel Xeon E5 v3, Xeon E3 v3, Core i7, Core i5, Core 3. 
* Radeon R9 290, R9 390, FirePro S9150, S9170 can support older CPU's since it does not require PCIe Gen 3 & PCIe Atomics.    Note we do not recomend PCIe Gen 2 enabled CPU since you will cap your overal bandwith but they will work with these GPU's   

Systems Requirements 
--------------------------------------

To use ROCm on your system you need the following: 
* ROCm Capable CPU and GPU 
	* PCIe Gen 3 Enabled CPU with PCIe Platform Atomics 
		* [More about how ROCm uses PCIe Atomics](https://rocm.github.io/ROCmPCIeFeatures.html)
	* ROCm enabled GPU's 
		* Radeon Instinct Family MI25, MI8, MI6 
		* Radeon Vega Frontier Edition 
		* [Broader Set of Tested Hardware](hardware.md)
* Supported Version of Linux with a specified GCC Compiler and ToolChain 


Table 1. Native Linux Distribution Support in ROCm  1.7

|Distribution	|Kernel	|GCC	|GLIBC	|
|:--------------|:------|:------|:------|
|x86_64		|	|	|       |		
|Ubuntu 16.04	|4.11	|5.40	|2.23   |

Debian Repository: apt-get
--------------------------------------

For Debian-based systems, such as Ubuntu, configure the Debian ROCm
repository as follows:

```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
```

The gpg key might change, so it may need to be updated when installing a new release. If the key signature verification is failed while update, please re-add the key from ROCm apt repository. The current rocm.gpg.key is not avialable in a standard key ring distribution, but has the following sha1sum hash:

f7f8147431c75e505c58a6f3a3548510869357a6 rocm.gpg.key

Install or Update
--------------------------------------

Next, update the apt-get repository list and install/update the ROCm
package.

>**Warning**: Before proceeding, make sure to completely
>[uninstall any previous ROCm package](https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages):

```
sudo apt-get update
sudo apt-get install rocm-dkms rocm-opencl-dev
```

With move to upstreaming the KFD driver and the support of DKMS,  for all Console aka headless user you will need  add all  your users to the  'video" group by setting the unix permisions

```
sudo usermod -a -G video <username>
```
Once complete, reboot your system.

We recommend you [verify your installation](https://github.com/RadeonOpenCompute/ROCm#verify-installation) to make sure everything completed successfully.


Once complete, reboot your system. We recommend that you [verify](#verify-installation) your
installation to ensure everything completed successfully.


Upon restart, To test your OpenCL instance
---------------------------------------------

Post Install all user need to part of the member of “video” group so set your Unix permisions for this. 

 Build and run Hello World OCL app..

HelloWorld sample:

```
 wget https://raw.githubusercontent.com/bgaster/opencl-book-samples/master/src/Chapter_2/HelloWorld/HelloWorld.cpp
 wget https://raw.githubusercontent.com/bgaster/opencl-book-samples/master/src/Chapter_2/HelloWorld/HelloWorld.cl
```

 Build it using the default ROCm OpenCL include and library locations:
 
```
g++ -I /opt/rocm/opencl/include/ ./HelloWorld.cpp -o HelloWorld -L/opt/rocm/opencl/lib/x86_64 -lOpenCL
```

 Run it:
 
 ```
 ./HelloWorld
```

Uninstall
--------------------------------------

To uninstall the entire rocm-dev development package, execute the following command:

```shell
sudo apt-get autoremove rocm-dkms
```

Installing Development Packages for Cross-Compilation
--------------------------------------

Developing and testing software on different systems is often useful.
In this scenario, you may prefer to avoid installing the ROCm kernel
on your development system. If so, install the development subset of
packages:

```
sudo apt-get update
sudo apt-get install rocm-dev
```

Note: to execute ROCm-enabled apps, you’ll need a system with the full
ROCm driver stack installed.



Closed-Source Components
--------------------------------------

The ROCm platform relies on a few closed-source components to provide
legacy functions such as HSAIL completion and debugging/profiling
support. These components are only available through the ROCm
repositories and will eventually either be deprecated or become open
source. They are available in the hsa-ext-rocr-dev packages.

Getting ROCm Source Code
--------------------------------------

Refer to the ROCm GitHub project for the latest instructions on how to
check out the code.

`ROCm on GitHub <https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md>`
