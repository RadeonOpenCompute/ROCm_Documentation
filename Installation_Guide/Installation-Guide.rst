
.. _Installation-Guide:

===================
Installation Guide
===================


System Requirement
###################

To use ROCm on your system you need the following:

 * ROCm Capable CPU and GPU
    * PCIe Gen 3 Enabled CPU with PCIe Platform Atomics
      
      * :ref:`More-about-how-ROCm-uses-PCIe-Atomics`

    * ROCm enabled GPU’s
     
      * Radeon Instinct Family MI25, MI8, MI6
      * Radeon Vega Frontier Edition

 * Supported Version of Linux with a specified GCC Compiler and ToolChain


============= ======== ====== =======
Distribution  Kernel    GCC   GLIBC
============= ======== ====== =======
x86_64
Fedora 24      4.11      5.40   2.23
Ubuntu 16.04   4.11      5.40   2.23
============= ======== ====== =======
Table 1. Native Linux Distribution Support in ROCm 1.6


Supported CPUs
**************
The ROCm Platform leverages PCIe Atomics (Fetch ADD, Compare and SWAP, Unconditional SWAP, AtomicsOpCompletion). PCIe atomics are only supported on PCIe Gen3 Enabled CPUs and PCIe Gen3 Switches like Broadcom PLX. When you install your GPUs make sure you install them in a fully PCIe Gen3 x16 or x8 slot attached either directly to the CPU’s Root I/O controller or via a PCIe switch directly attached to the CPU’s Root I/O controller. In our experience many issues stem from trying to use consumer motherboards which provide Physical x16 Connectors that are electrically connected as e.g. PCIe Gen2 x4. This typically occurs when connecting via the Southbridge PCIe I/O controller. If you motherboard is part of this category, please do not use this connector for your GPUs, if you intend to exploit ROCm.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

    * AMD Ryzen CPUs
    * AMD RYZEN Threadripper
    * AMD  EPYC Server CPU
    * Intel Xeon E5 v3 or newer CPUs
    * Intel Xeon E3 v3 or newer CPUs
    * Intel Core i7 v3, Core i5 v3, Core i3 v3 or newer CPUs (i.e. Haswell family or newer).
    

Upcoming CPUs which will support PCIe Gen3 + PCIe Atomics are:

    * AMD Naples Server CPUs;
    * Cavium Thunder X Server Processor.
    
Supported GPUs
**************
Our GFX8 GPU’s (Fiji & Polaris family) and GFX9 (VEGA).

New GPU Support for ROCm 1.6 

   * GFX8: Radeon RX 480,Radeon RX 470,Radeon RX 460,R9 Nano,Radeon R9 Fury,Radeon R9 Fury X Radeon Pro WX7100, FirePro S9300 x2
   * Radeon Instinct Family MI25, MI8, MI6 
   * Radeon Vega Frontier Edition 

Experimental support for our GFX7 GPUs Radeon R9 290, R9 390, AMD FirePro S9150, S9170 do not support or take advantage of PCIe Atomics. However, we still recommend that you use a CPU from the list provided above.

Broader Set of Tested Hardware

Not Supported or Very Limited Support Under ROCm
*************************************************
  * We do not support ROCm with PCIe Gen 2 enabled CPUs such as the AMD Opteron, Phenom, Phenom II, Athlon, Athlon X2,Athlon II and 	Older Intel Xeon and Intel Core Architecture and Pentium CPUs.
  * We also do not support AMD Carrizo and Kaveri APU as host for compliant dGPU attachments.
  * Thunderbolt 1 and 2 enabled GPU’s are not supported by ROCm. Thunderbolt 1 & 2 are PCIe Gen2 based.
  * AMD Carrizo based APUs have limited support due to OEM & ODM’s choices when it comes to some key configuration parameters. On     	  point, we have observed that Carrizo Laptops, AIOs and Desktop systems Showed inconsistencies in exposing and enabling the System 	  BIOS parameters required by the ROCm stack. Before purchasing a Carrizo system for 	ROCm,please verify that the BIOS provides    	 an option for enabling IOMMUv2. If this is the case, the final requirement is associated with correct CRAT table support - please 	  inquire with the OEM about the latter.
  * AMD Merlin/Falcon Embedded System is also not currently supported by the public Repo.
  * AMD Carrizo and Kaveri APU with external GPU Attached are not supported by ROCm

Support for future APUs
************************
We are well aware of the excitement and anticipation built around using ROCm with an APU system which fully exposes Shared Virtual Memory alongside and cache coherency between the CPU and GPU. To this end, in 2017 we plan on testing commercial AM4 motherboards for the Bristol Ridge and Raven Ridge families of APUs. Just like you, we still waiting for access to them! Once we have the first boards in the lab we will detail our experiences via our blog, as well as build a list of motherboard that are qualified for use with ROCm.

Supported Operating Systems
***************************
The ROCm platform has been tested on the following operating systems:

   * Ubuntu 16.04
   * Fedora 24 (Hawaii based GPUs, i.e. Radeon R9 290, R9 390, AMD FirePro S9150, S9170, are not supported)



Installation Guide Ubuntu
##########################

Pre Install Directions
**********************

**Verify You Have ROCm Capable GPU Installed into the System** ::
   
    lspci | grep -i AMD

**Verify You Have a Supported Version of Linux** ::

   uname -m && cat /etc/*release

**You will see some thing like for Ubuntu** ::
 
  x86_64
  DISTRIB_ID=Ubuntu 
  DISTRIB_RELEASE=16.04
  DISTRIB_CODENAME=xenial
  DISTRIB_DESCRIPTION="Ubuntu 16.04.2 LTS"


**Verify version of GCC** ::

    gcc --version 

**You will see** ::

 gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609 

Ubuntu Install
***************
**Add the Repo Server**

For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows ::
 
   wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
   sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

The gpg key might change, so it may need to be updated when installing a new release. The current rocm.gpg.key is not avialable in a standard key ring distribution, but has the following sha1sum hash::
 
 f0d739836a9094004b0a39058d046349aacc1178 rocm.gpg.key

**Install or update ROCm** ::

   sudo apt-get update
   sudo apt-get install rocm rocm-opencl-dev

Then, make the ROCm kernel your default kernel. If using grub2 as your bootloader, you can edit the GRUB_DEFAULT variable in the following file: ::

   sudo nano /etc/default/grub

set the GRUB_Default Edit: GRUB_DEFAULT=”Advanced options for Ubuntu>Ubuntu, with Linux 4.9.0-kfd-compute-rocm-rel-1.6-77” ::
 
   sudo update-grub


**To Uninstall the a Package** ::

   sudo apt-get purge libhsakmt
   sudo apt-get purge radeon-firmware
   sudo apt-get purge $(dpkg -l | grep 'kfd\|rocm' | grep linux | grep -v libc | awk '{print $2}')

:ref:`List-of-ROCm-Packages-for-Ubuntu-Fedora`


 
Installation Guide Fedora
##########################

Use the dnf (yum) repository for installation of rpm packages. To configure a system to use the ROCm rpm directory create the file /etc/yum.repos.d/rocm.repo with the following contents: ::

 [remote]

 name=ROCm Repo

 baseurl=http://repo.radeon.com/rocm/yum/rpm/

 enabled=1

 gpgcheck=0

Execute the following commands: ::
  
  sudo dnf clean all
  sudo dnf install rocm rocm-opencl-dev

Just like Ubuntu installs, the ROCm kernel must be the default kernel used at boot time.

Post Install Manual installation steps for Fedora to support HCC compiler

A fully functional Fedora installation requires a few manual steps to properly setup, including:

  * `Building compatible libc++ and libc++abi libraries for Fedora <https://github.com/RadeonOpenCompute/hcc/wiki#fedora>`_


**Post install verification**

Verify you have the correct Kernel Post install
::

   uname -r
   4.9.0-kfd-compute-rocm-rel-1.6-148

Test if OpenCL is working based on default ROCm OpenCL include and library locations:
::

   g++ -I /opt/rocm/opencl/include/ ./HelloWorld.cpp -o HelloWorld -L/opt/rocm/opencl/lib/x86_64 -lOpenCL

Run it:
:: 

  ./HelloWorld




**To Uninstall the a Package** ::
    
   sudo dnf remove ROCm 


:ref:`List-of-ROCm-Packages-for-Ubuntu-Fedora`


**Installing development packages for cross compilation**

It is often useful to develop and test on different systems. In this scenario, you may prefer to avoid installing the ROCm Kernel to your development system.

In this case, install the development subset of packages: ::

 sudo apt-get update
 sudo apt-get install rocm-dev




FAQ on Installation
*******************

:ref:`FAQ-on-Installation`


