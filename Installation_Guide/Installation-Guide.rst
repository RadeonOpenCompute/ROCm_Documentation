
.. _Installation-Guide:

===================
Installation Guide
===================
            
ROCm 1.8
###########

Supported CPUs
****************

Starting with ROCm 1.8 we have relaxed the use of PCIe Atomics and also PCIe Lane choice for Vega10/GFX9 class GPU.  So now you can support CPU with out PCIe Atomics and also use Gen2 x1 lanes. 

Currently our GFX8 GPU's (Fiji & Polaris Family)still need to use PCIe Gen 3 and PCIe Atomics, but are looking at relaxing this in future release, once we have fully tested firmware. 

Current tested CPUs which support PCIe Gen3 + PCIe Atomics are:
 * Intel Xeon E7 v3 or newer CPUs;
 * Intel Xeon E5 v3 or newer CPUs;
 * Intel Xeon E3 v3 or newer CPUs;
 * Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e. Haswell family or newer).
 * AMD Ryzen CPUs;
 * AMD EPYC CPU's
 
 Comunity tested with Vega10 and ROCm 1.8 
  * Intel Xeon E5 v2 or newer CPUs;

More information on  Fiji and Polaris Class GPU's the ROCm Platform leverages  PCIe Atomics (Fetch ADD, Compare and SWAP, Unconditional SWAP, AtomicsOpCompletion). `PCIe atomics <https://github.com/RadeonOpenCompute/RadeonOpenCompute.github.io/blob/master/ROCmPCIeFeatures.md>`_ are only supported on PCIe Gen3 Enabled CPUs and PCIe Gen3 Switches like Broadcom PLX. For Fiji and Polaris based GPU a slot attached either directly to the CPU's Root I/O controller or via a PCIe switch directly attached to the CPU's Root I/O controller. In our experience many issues stem from trying to use consumer motherboards which provide Physical x16 Connectors that are electrically connected as e.g. PCIe Gen2 x4. This typically occurs when connecting via the Southbridge PCIe I/O controller. 
 
Experimental support for our GFX7 GPUs Radeon R9 290, R9 390, AMD FirePro S9150, S9170 note they do not support or take advantage of PCIe Atomics. However, we still recommend that you use a CPU from the list provided above.

Not supported or very limited support under ROCm
**************************************************

Limited Support 
________________

* With ROCm 1.8 and Vega10 it should support  PCIe Gen 2 enabled CPUs such as the AMD Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and Older Intel Xeon and Intel Core Architecture and Pentium CPUs. But we have done very limited testing. Since our test farm today has been catering to CPU listed above.  This is where we need comunity support. 
 * Thunderbolt 1,2 &. 3 enabled breakout boxes GPU's should now be able to work with ROCm. Thunderbolt 1 & 2 are PCIe Gen2 based. But we have done no testing on this config and would need comunity support do limited access to this type of equipment 

Not Supported 
______________

* We also do not support AMD Carrizo and Kaveri APU as host for compliant dGPU attachments.
 * Thunderbolt 1 and 2 enabled GPU's are not supported by ROCm. Thunderbolt 1 & 2 are PCIe Gen2 based.
 * AMD Carrizo based APUs have limited support due to OEM & ODM's choices when it comes to some key configuration parameters. On point, we have observed that Carrizo Laptops, AIOs and Desktop systems showed inconsistencies in exposing and enabling the System BIOS parameters required by the ROCm stack. Before purchasing a Carrizo system for ROCm, please verify that the BIOS provides an option for enabling IOMMUv2. If this is the case, the final requirement is associated with correct CRAT table support - please inquire with the OEM about the latter.
 * AMD Merlin/Falcon Embedded System is also not currently supported by the public Repo.
 * AMD Raven Ridge APU are currently not supported



The latest ROCm platform - ROCm 1.8.x
************************************

The latest tested version of the drivers, tools, libraries and source code for the ROCm platform have been released and are available under the roc-1.8.x or rocm-1.8.x tag of the following GitHub repositories:

 * :ref:`OpenComute-kernel-deriver`
 * :ref:`ROCrRuntime`
 * :ref:`ROCt`
 * :ref:`ROC-smi`
 * :ref:`HCC-Compiler`
 * `compiler-runtime <https://github.com/RadeonOpenCompute/compiler-rt/tree/roc-1.8.x>`_
 * :ref:`ROCm-Developer-Tool-HIP`
 * `HIP-Examples <https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/roc-1.8.x>`_
 * :ref:`atmi`
 
Additionally, the following mirror repositories that support the HCC compiler are also available on GitHub, and frozen for the rocm-1.8.0 release:

 * `llvm <https://github.com/RadeonOpenCompute/llvm/tree/roc-1.8.x>`_
 * `ldd <https://github.com/RadeonOpenCompute/lld/tree/roc-1.8.x>`_
 * `hcc-clang-upgrade <https://github.com/RadeonOpenCompute/hcc-clang-upgrade/tree/roc-1.8.x>`_
 * `ROCm-Device-Libs <https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/roc-1.8.x>`_

Supported Operating Systems
****************************

The ROCm 1.8 platform has been tested on the following operating systems:
 * Ubuntu 16.04
 * CentOS 7.4 (Using devetoolset-7 runtime support)
 * RHEL 7.4 (Using devetoolset-7 runtime support

Installing from AMD ROCm repositories
########################################

AMD is hosting both debian and RPM repositories for the ROCm 1.8 packages at this time. The packages in the Debian repository have been signed to ensure package integrity.  The packages in the Debian repository have been signed to ensure package integrity. Directions for each repository are given below:

Installing from a debian repository
#####################################

**First make sure your system is up to date**
::
 sudo apt update
 sudo apt dist-upgrade
 sudo apt install libnuma-dev
 sudo reboot

**Optional: Upgrade to 4.13 kernel**

Although not required, it is recommended as of ROCm 1.8.x that the system's kernel is upgraded to the latest 4.13 version available:
::
 sudo apt install linux-headers-4.13.0-32-generic linux-image-4.13.0-32-generic linux-image-extra-4.13.0-32-generic linux-signed-image-4.13.0-32-generic
 sudo reboot 

Packaging server update
************************
The packaging server has been changed from the old http://packages.amd.com to the new repository site http://repo.radeon.com.

Debian repository - apt
************************
**Add the ROCm apt repository**
For Debian based systems, like Ubuntu, configure the Debian ROCm repository as follows:
::
 wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
 sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

The gpg key might change, so it may need to be updated when installing a new release. The current rocm.gpg.key is not avialable in a standard key ring distribution, but has the following sha1sum hash:
::
 f0d739836a9094004b0a39058d046349aacc1178 rocm.gpg.key

Install or Update
*******************
Next, update the apt repository list and install/update the rocm package.

.. warning:: Before proceeding, make sure to completely uninstall any previous ROCm package:

To Install the package:
::
 sudo apt update
 sudo apt install rocm-dkms

**Next set your permissions**

With move to upstreaming the KFD driver and the support of DKMS, for all Console aka headless user, you will need to add all your users to the 'video" group by setting the Unix permissions

Configure Ensure that your user account is a member of the "video" group prior to using the ROCm driver. You can find which groups you are a member of with the following command:
::
 groups

To add yourself to the video group you will need the sudo password and can use the following command:
::
 sudo usermod -a -G video $LOGNAME 

Once complete, reboot your system.

We recommend you verify your installation to make sure everything completed successfully.


**Upon restart, To test your OpenCL instance**

Build and run Hello World OCL app..

HelloWorld sample:
::
 wget https://raw.githubusercontent.com/bgaster/opencl-book-samples/master/src/Chapter_2/HelloWorld/HelloWorld.cpp
 wget https://raw.githubusercontent.com/bgaster/opencl-book-samples/master/src/Chapter_2/HelloWorld/HelloWorld.cl

Build it using the default ROCm OpenCL include and library locations:
::
 g++ -I /opt/rocm/opencl/include/ ./HelloWorld.cpp -o HelloWorld -L/opt/rocm/opencl/lib/x86_64 -lOpenCL

Run it:
::
 ./HelloWorld

**Un-install**

To un-install the entire rocm development package execute:
::
 sudo apt autoremove rocm-dkms

Installing development packages for cross compilation
*******************************************************

It is often useful to develop and test on different systems. In this scenario, you may prefer to avoid installing the ROCm Kernel to your development system.

In this case, install the development subset of packages:
::
 sudo apt update
 sudo apt install rocm-dev

.. note:: To execute ROCm enabled apps you will require a system with the full ROCm driver stack installed


Removing pre-release packages
*******************************
If you installed any of the ROCm pre-release packages from github, they will need to be manually un-installed:
::
 sudo apt purge libhsakmt
 sudo apt purge compute-firmware
 sudo apt purge $(dpkg -l | grep 'kfd\|rocm' | grep linux | grep -v libc | awk '{print $2}')

If possible, we would recommend starting with a fresh OS install.


CentOS/RHEL 7 Support
************************

Support for CentOS/RHEL 7 has been added in ROCm 1.8, but requires a special 
runtime environment provided by the RHEL Software Collections and additional
dkms support packages to properly install in run.

Preparing RHEL 7 for installation
____________________________________

RHEL is a subscription based operating system, and must enable several external
repositories to enable installation of the devtoolset-7 environment and the DKMS
support files. These steps are not required for CentOS.

First, the subscription for RHEL must be enabled and attached to a pool id. Please
see Obtaining an RHEL image and license page for instructions on registering your
system with the RHEL subscription server and attaching to a pool id.


Second, enable the following repositories:
::
 sudo subscription-manager repos --enable rhel-7-server-rhscl-rpms
 sudo subscription-manager repos --enable rhel-7-server-optional-rpms
 sudo subscription-manager repos --enable rhel-7-server-extras-rpms



Third, enable additional repositories by downloading and installing the epel-release-latest-7 repository RPM:
::
 sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm


Install and setup Devtoolset-7
__________________________________

To setup the Devtoolset-7 environment, follow the instructions on this page:

https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

Note that devtoolset-7 is a Software Collections package, and is not supported by AMD.

* Prepare CentOS/RHEL 7.4 for DKMS Install

Installing kernel drivers on CentOS/RHEL 7.4 requires dkms tool being installed:
::
 sudo yum install -y epel-release
 sudo yum install -y dkms kernel-headers-`uname -r`

You can also follow RedHats way to downgrade kerenl and redhat rlease from 7.5 to 7.4  https://access.redhat.com/solutions/186763 via there instructions. 

Current release supports up to CentOS/RHEL 7.4. If for any reason the system needs to be updated to 7.5, don’t update the kernel. Add “--exclude=kernel*” flag to yum install. For example:
::
 sudo yum update --exclude=kernel*


At this point they system can install ROCm using the DKMS drivers.

Installing ROCm on the system
_______________________________

At this point ROCm can be installed on the target system. Create a /etc/yum.repos.d/rocm.repo file with the following contents:
::
 [ROCm]
 name=ROCm
 baseurl=http://repo.radeon.com/rocm/yum/rpm
 enabled=1
 gpgcheck=0


The repo's URL should point to the location of the repositories repodata database. Install ROCm components using these commands:
::
 sudo yum install rocm-dkms


The rock-dkms component should be installed and the /dev/kfd device should be available on reboot.

Ensure that your user account is a member of the "video" or "wheel" group prior to using the ROCm driver.
You can find which groups you are a member of with the following command:
::
 groups

To add yourself to the video (or wheel) group you will need the sudo password and can use the
following command:
::
 sudo usermod -a -G video $LOGNAME 


Compiling applications using hcc, hip, etc.
__________________________________________________

To compile applications or samples, please use gcc-7.2 provided by the devtoolset-7 environment.
To do this, compile all applications after running this command: 
::
 scl enable devtoolset-7 bash

How to un-install ROCm from CentOS/RHEL 7.4
______________________________________________

To un-install the entire rocm development package execute:
::
 sudo yum autoremove rocm-dkms



Known Issues / Workarounds for Both Distros
************************************************

If you Plan to Run with X11 - we are seeing X freezes under load
_____________________________________________________________________

ROCm 1.8.x a kernel parameter noretry has been set to 1 to improve overall system performance. However it has been proven to bring instability to graphics driver shipped with Ubuntu. This is an ongoing issue and we are looking into it.

Before that, please try apply this change by changing noretry bit to 0.
::
 echo 0 | sudo tee /sys/module/amdkfd/parameters/noretry

Files under /sys won't be preserved after reboot so you'll need to do it every time.

One way to keep noretry=0 is to change /etc/modprobe.d/amdkfd.conf and make it be:

options amdkfd noretry=0

Once it's done, run sudo update-initramfs -u. Reboot and verify /sys/module/amdkfd/parameters/noretry stays as 0.

Closed source components
***************************
The ROCm platform relies on a few closed source components to provide legacy functionality like HSAIL finalization and debugging/profiling support. These components are only available through the ROCm repositories, and will either be deprecated or become open source components in the future. These components are made available in the following packages:

 * hsa-ext-rocr-dev
 
Getting ROCm source code
##########################
Modifications can be made to the ROCm 1.7 components by modifying the open source code base and rebuilding the components. Source code can be cloned from each of the GitHub repositories using git, or users can use the repo command and the ROCm 1.7 manifest file to download the entire ROCm 1.7 source code.

Installing repo
*****************
Google's repo tool allows you to manage multiple git repositories simultaneously. You can install it by executing the following commands:
::
 curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
 chmod a+x ~/bin/repo

.. note:: make sure ~/bin exists and it is part of your PATH

Cloning the code
******************

To Clone the code form ROCm, following steps can be used:
::
 mkdir ROCm && cd ROCm
 repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-1.7.2
 repo sync

These series of commands will pull all of the open source code associated with the ROCm 1.7 release. Please ensure that ssh-keys are configured for the target machine on GitHub for your GitHub ID.

 * OpenCL Runtime and Compiler will be submitted to the Khronos Group, prior to the final release, for conformance testing.

Installing ROCk-Kernel only
***********************
To Install only ROCk-kernel the following steps can be used from the link provided :ref:`kernel-installation`

FAQ on Installation
#####################
Please refer the link for FAQ on Installation.
:ref:`FAQ-on-Installation`


