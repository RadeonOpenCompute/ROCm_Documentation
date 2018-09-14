.. _FAQ-on-Installation:

====================
FAQ on Installation
====================


Determining if the video card is installed correctly
*****************************************************

The ROCm software stack has specific requirements regarding the type of GPU supported and how it is installed in the system. The card must be installed in a PCIe slot that supports the 3.0 PCIe specification and the atomics extension. Preferably the slot is x16; x8 an x4 slots will work, but data transfer rates between host memory and GPU memory will be reduced. If the card is not installed in a compatible PCIe slot applications that dispatch a compute kernel will hang waiting for a completion signal from the GPU, which is an atomic operation.

After booting the system with the new driver installed the dmesg output will indicate if there were any problems initializing the GPU. The output of the command ‘sudo dmesg | grep kfd’ will indicate if there were any initialization problems. A properly initialized system will have dmesg output similar to this
::
 dmesg | grep kfd
 [    0.000000] Linux version 4.11.0-kfd-compute-roc-master-5051 (jenkins@jenkins-raptor-5) (gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4) ) #1 SMP Thu Jun 29 21:00:37 CDT 2017
 [    0.000000] Command line: BOOT_IMAGE=/boot/vmlinuz-4.11.0-kfd-compute-roc-master-5051 root=UUID=084440bf-e6be-4175-a72c-e3cc6ae4448c ro quiet splash vt.handoff=7
 [    0.000000] Kernel command line: BOOT_IMAGE=/boot/vmlinuz-4.11.0-kfd-compute-roc-master-5051 root=UUID=084440bf-e6be-4175-a72c-e3cc6ae4448c ro quiet splash vt.handoff=7
 [    1.245721] usb usb1: Manufacturer: Linux 4.11.0-kfd-compute-roc-master-5051 xhci-hcd
 [    1.253148] usb usb2: Manufacturer: Linux 4.11.0-kfd-compute-roc-master-5051 xhci-hcd
 [    1.316964] usb usb3: Manufacturer: Linux 4.11.0-kfd-compute-roc-master-5051 xhci-hcd
 [    1.317167] usb usb4: Manufacturer: Linux 4.11.0-kfd-compute-roc-master-5051 xhci-hcd
 [    1.428356] kfd kfd: Initialized module
 [    2.379347] kfd kfd: Allocated 3969056 bytes on gart for device 1002:7300
 [    2.379452] kfd kfd: Reserved 2 pages for cwsr.
 [    2.379468] kfd kfd: added device 1002:7300

If the GPU is installed in a PCIe slot that is not supported there will be error messages indicating that the devices full capabilities are not available.

Meta package Installation issues, rpm and dpkg
***********************************************

The ROCm repository uses several “meta” packages that provide easy installation for several components of ROCm that do not have natural dependencies. The “meta” packages are empty debian or rpm files that have dependencies on several, unrelated, ROCm components. They are useful in installing or uninstalling the entire ROCm stack with one apt-get or dnf command, and also provide automatic configuration of the /dev/kfd file permissions using the udev service.

In some cases users can “break” a ROCm installation by removing one of the “meta” packages using the rpm or dpkg command directly. The rpm and dpkg commands do not resolve dependencies like the dnf and apt-get commands do, and should not be used to remove any ‘meta’ packages, or any other ROCm package. For example, a user can remove the rocm package with the command ‘sudo dpkg –r rocm’ on Ubuntu, but that will not remove any of its dependencies. This is also true for the ‘sudo apt-get remove rocm’ command which will only remove the rocm ‘meta’ package and not its dependencies. To remove a ROCm installation completely, use ‘sudo apt-get autoremove rocm’ for Ubuntu and ‘sudo dnf remove rocm’ for Fedora.

The current meta packages are: rocm – Depends on the kernel drivers, firmware and the rocm-dev packages. rocm-dev – Depends on the roct, rocr, rocr extension, hcc and hip packages. rocm-libs – Depends on the hcBLAS, hcFFT, hcRNG, rocBLAS and hipBLAS packages.

If an installation has its ‘meta’ packages removed they can be reinstall using the standard apt-get or dnf command. Reinstall the ‘meta’ packages will not reinstall already installed dependencies

Linux Kernels are not uninstalled by default
**********************************************

If ROCm is uninstalled using dnf or apt-get the kernel packages are not uninstalled by default. This is a Linux convention, and isn’t unique the ROCm stack. To remove the kernel packages, they will have to be removed explicitly:

For debian – ‘sudo apt-get autoremove ’ For RPM – ‘sudo dnf remove ’

The rpm or dpkg command can also be used, but isn’t recommended.

Updating firmware may not trigger a rebuilding of ramfs
********************************************************

If a device isn’t detected by the ROCm kernel drivers, it is possible there is an issue loading required device firmware. This can happen if the system has downlevel firmware or if the firmware is updated, but the ramfs hasn’t been initialized with the new firmware images. To see if this is a problem, check the dmesg of the system:
::
 dmesg | grep amdgpu
 [    4.434129] [drm] amdgpu kernel modesetting enabled.
 [    4.517484] amdgpu 0000:05:00.0: enabling device (0100 -> 0103)
 [    4.517690] amdgpu 0000:05:00.0: Direct firmware load for amdgpu/vega10_gpu_info.bin failed with error -2
 [    4.517692] amdgpu 0000:05:00.0: Failed to load gpu_info firmware "amdgpu/vega10_gpu_info.bin"
 [    4.517733] amdgpu 0000:05:00.0: Fatal error during GPU init
 [    4.517757] [drm] amdgpu: finishing device.
 [    4.517914] amdgpu: probe of 0000:05:00.0 failed with error -2
 
The error displayed above indicates the kernel is having trouble loading the firmware.

If the firmware version isn’t correct, please install updated firmware packages, which should be available on the repository server. If the correct firmware is installed, reinitialize the ramfs as follows:

**Ubuntu**
::
 update-initramfs -u

**Fedora**
::
 sudo dracut --regenerate-all --force

/boot filesystem too small for installation
********************************************

This problem can occur on Fedora installation if several previous kernels are currently installed. The dnf installation will fail with the following message:
::
 Error: Transaction check error:
  installing package kernel-4.9.0_kfd_compute_rocm_rel_1.6_67-2.x86_64 needs 17MB on the /boot filesystem
 Error Summary
 -------------
 Disk Requirements:
    At least 17MB more space needed on the /boot filesystem.
 

This is not an issue with the YUM repository; it is caused by the size of the /boot filesystem and the size of the kernels already installed on it. This issue can be fixed by uninstalling previous versions of the rocm Linux kernel:
::
 sudo dnf remove rocm
 rpm -qa | grep kfd | xargs sudo rpm –e
 sudo dnf install rocm
 
Installing from an archived repository
**************************************

The Radeon repo server stores several archived releases, supporting both debian and rpm repositories. These archives are located here at http://repo.radeon.com/rocm/archive. Users can install with an archive by downloading the desired archive and then updating the package configuration file to point at the localized repo.

Debian Archive Example
*********************** 
Here is an Example:
::

  cd /temp && wget http://repo.radeon.com/rocm/archive/apt_1.6.3.tar.bz2
  tar -xvf apt_1.6.3.tar.bz2
  sudo echo “deb [amd64] file://temp/apt_1.6.3 xenial main” > /etc/apt/sources.lists.d/rocm.local.list
  sudo apt-get update && sudo apt-get install rocm

Users should make sure that no other list files contain another rocm repo configuration.

RPM Archive Example
********************
Add a /etc/yum.d/rocm.local.repo file with the following contents: ::

  [remote]
  name=ROCm Repo
  baseurl=file://packages.amd.com/rocm/yum/rpm/
  enabled=1
  gpgcheck=0
  cd /temp && wget http://repo.radeon.com/rocm/archive/yum_1.6.3.tar.bz2
  tar –xvf yum_1.6.3.tar.bz2

Then execute: ::

  sudo dnf clean all
  sudo dnf install rocm


Again, users should make sure that no other repo files contain another rocm repo reference.