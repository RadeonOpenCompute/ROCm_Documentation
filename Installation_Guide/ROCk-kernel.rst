.. _kernel-installation:

===============
ROCk-Kernel
===============

The following is a sequence of commands to Install ROCk-Kernel into the system:

**# OPTIONAL :** 
upgrade your base kernel to 4.13.0-32-generic, **reboot required**
::
 sudo apt update && sudo apt install linux-headers-4.13.0-32-generic linux-image-4.13.0-32-generic linux-image-extra-4.13.0-32-generic linux-signed-image-4.13.0-32-generic
 sudo reboot 

Installation steps:
###################

Install the ROCm compute firmware and rock-dkms kernel modules, **reboot required**
::
 wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
 echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main | sudo tee /etc/apt/sources.list.d/rocm.list
 sudo apt-get update && sudo apt-get install compute-firmware rock-dkms
 sudo update-initramfs -u
 sudo reboot

**Add user to the video group**
::
 sudo adduser $LOGNAME video

Make sure to reboot the machine after installing the ROCm kernel package to force the new kernel to load on reboot. 

You can verify the ROCm kernel is loaded by typing the following command at a prompt:
::
 lsmod | grep kfd

Printed on the screen should be similar as follows:
::
 amdkfd                270336  4
 amd_iommu_v2           20480  1 amdkfd
 amdkcl                 24576  3 amdttm,amdgpu,amdkfd
 
 
