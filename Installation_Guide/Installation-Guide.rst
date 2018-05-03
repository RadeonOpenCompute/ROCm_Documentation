
.. _Installation-Guide:

===================
Installation Guide
===================
            
ROCm 1.7
###########

Supported CPUs
****************
The ROCm Platform leverages PCIe Atomics (Fetch ADD, Compare and SWAP, Unconditional SWAP, AtomicsOpCompletion). `PCIe atomics <https://github.com/RadeonOpenCompute/RadeonOpenCompute.github.io/blob/master/ROCmPCIeFeatures.md>`_ are only supported on PCIe Gen3 Enabled CPUs and PCIe Gen3 Switches like Broadcom PLX. When you install your GPUs make sure you install them in a fully PCIe Gen3 x16 or x8 slot attached either directly to the CPU's Root I/O controller or via a PCIe switch directly attached to the CPU's Root I/O controller. In our experience many issues stem from trying to use consumer motherboards which provide Physical x16 Connectors that are electrically connected as e.g. PCIe Gen2 x4. This typically occurs when connecting via the Southbridge PCIe I/O controller. If you motherboard is part of this category, please do not use this connector for your GPUs, if you intend to exploit ROCm.

Our GFX8 GPU's (Fiji & Polaris Family) and GFX9 (Vega) use PCIe Gen 3 and PCIe Atomics.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

 * Intel Xeon E5 v3 or newer CPUs;
 * Intel Xeon E3 v3 or newer CPUs;
 * Intel Core i7 v4, Core i5 v4, Core i3 v4 or newer CPUs (i.e. Haswell family or newer).
 * AMD Ryzen CPUs;

Upcoming CPUs which will support PCIe Gen3 + PCIe Atomics are:

 * AMD Naples Server CPUs;
 * Cavium Thunder X Server Processor.
 
Experimental support for our GFX7 GPUs Radeon R9 290, R9 390, AMD FirePro S9150, S9170 note they do not support or take advantage of PCIe Atomics. However, we still recommend that you use a CPU from the list provided above.

Not supported or very limited support under ROCm
**************************************************
 * We do not support ROCm with PCIe Gen 2 enabled CPUs such as the AMD Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and Older Intel Xeon and Intel Core Architecture and Pentium CPUs.
 * We also do not support AMD Carrizo and Kaveri APU as host for compliant dGPU attachments.
 * Thunderbolt 1 and 2 enabled GPU's are not supported by ROCm. Thunderbolt 1 & 2 are PCIe Gen2 based.
 * AMD Carrizo based APUs have limited support due to OEM & ODM's choices when it comes to some key configuration parameters. On point, we have observed that Carrizo Laptops, AIOs and Desktop systems showed inconsistencies in exposing and enabling the System BIOS parameters required by the ROCm stack. Before purchasing a Carrizo system for ROCm, please verify that the BIOS provides an option for enabling IOMMUv2. If this is the case, the final requirement is associated with correct CRAT table support - please inquire with the OEM about the latter.
 * AMD Merlin/Falcon Embedded System is also not currently supported by the public Repo.
 * AMD Raven Ridge APU are currently not supported

**IPC support**

The latest ROCm platform - ROCm 1.7
************************************

The latest tested version of the drivers, tools, libraries and source code for the ROCm platform have been released and are available under the roc-1.7.x or rocm-1.7.x tag of the following GitHub repositories:

 * :ref:`OpenComute-kernel-deriver`
 * :ref:`ROCrRuntime`
 * :ref:`ROCt`
 * :ref:`ROC-smi`
 * :ref:`HCC-Compiler`
 * `compiler-runtime <https://github.com/RadeonOpenCompute/compiler-rt/tree/roc-1.7.x>`_
 * :ref:`ROCm-Developer-Tool-HIP`
 * `HIP-Examples <https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-Examples/tree/roc-1.7.x>`_
 * `atmi <https://github.com/RadeonOpenCompute/atmi/tree/0.3.7>`_

 
Additionally, the following mirror repositories that support the HCC compiler are also available on GitHub, and frozen for the rocm-1.7.1 release:

 * `llvm <https://github.com/RadeonOpenCompute/llvm/tree/roc-1.7.x>`_
 * `ldd <https://github.com/RadeonOpenCompute/lld/tree/roc-1.7.x>`_
 * `hcc-clang-upgrade <https://github.com/RadeonOpenCompute/hcc-clang-upgrade/tree/roc-1.7.x>`_
 * `ROCm-Device-Libs <https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/roc-1.7.x>`_

Supported Operating Systems
****************************

The ROCm 1.7 platform has been tested on the following operating systems:
 * Ubuntu 16.04

Installing from AMD ROCm repositories
########################################

AMD is hosting only debian repositories for the ROCm 1.7 packages at this time. It is expected that an rpm repository will be available in the next point release.

The packages in the Debian repository have been signed to ensure package integrity. Directions for each repository are given below:

**First make sure your system is up to date**
::
 sudo apt update
 sudo apt dist-upgrade
 sudo apt install libnuma-dev
 sudo reboot

**Optional: Upgrade to 4.13 kernel**

Although not required, it is recommended as of ROCm 1.7.1 that the system's kernel is upgraded to the latest 4.13 version available:
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

**Next set your permsions**

With move to upstreaming the KFD driver and the support of DKMS, for all Console aka headless user, you will need to add all your users to the 'video" group by setting the Unix permissions

Configure Ensure that your user account is a member of the "video" group prior to using the ROCm driver. You can find which groups you are a member of with the following command:
::
 groups

To add yourself to the video group you will need the sudo password and can use the following command:
::
 sudo usermod -a -G video $LOGNAME 

Once complete, reboot your system.

We recommend you :ref:`verify your installation` to make sure everything completed successfully.

To install ROCm with Developer Preview of OpenCL
**************************************************

**Start by following the instruction of installing ROCm with Debian repository:**

No additional steps are required. The rocm-opencl package is now installed with rocm-dkms as a dependency. This includes the development package, rocm-opencl-dev.

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

**Installing development packages for cross compilation**

It is often useful to develop and test on different systems. In this scenario, you may prefer to avoid installing the ROCm Kernel to your development system.

In this case, install the development subset of packages:
::
 sudo apt update
 sudo apt install rocm-dev

.. note:: To execute ROCm enabled apps you will require a system with the full ROCm driver stack installed

**Known Issues / Workarounds**

#If you Plan to Run with X11 - we are seeing X freezes under load

ROCm 1.7.1 a kernel parameter noretry has been set to 1 to improve overall system performance. However it has been proven to bring instability to graphics driver shipped with Ubuntu. This is an ongoing issue and we are looking into it.

Before that, please try apply this change by changing noretry bit to 0.
::
 echo 0 | sudo tee /sys/module/amdkfd/parameters/noretry

Files under /sys won't be preserved after reboot so you'll need to do it every time.

One way to keep noretry=0 is to change /etc/modprobe.d/amdkfd.conf and make it be:

options amdkfd noretry=0

Once it's done, run sudo update-initramfs -u. Reboot and verify /sys/module/amdkfd/parameters/noretry stays as 0.

Removing pre-release packages
*******************************
If you installed any of the ROCm pre-release packages from github, they will need to be manually un-installed:
::
 sudo apt purge libhsakmt
 sudo apt purge compute-firmware
 sudo apt purge $(dpkg -l | grep 'kfd\|rocm' | grep linux | grep -v libc | awk '{print $2}')

If possible, we would recommend starting with a fresh OS install.

RPM repository - dnf (yum)
***************************
A repository containing rpm packages is currently not available for the ROCm 1.7 release.

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
 repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-1.7.1
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


