
.. _Current-Release-Notes:

=====================
Current Release Notes
=====================

ROCm 1.7 What New?
###################

DKMS driver installation
*************************
 * New driver installation uses Dynamic Kernel Module Support (DKMS)
 * Only amdkfd and amdgpu kernel modules are installed to support AMD hardware
 * Currently only Debian packages are provided for DKMS (no Fedora suport available)
 * See the `ROCT-Thunk-Interface <https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/roc-1.7.x>`_ and `ROCK-Kernel-Driver <https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/roc-1.7.x>`_ for additional documentation on driver setup

Developer preview of the new OpenCL 1.2 compatible language runtime and compiler
**********************************************************************************
 * OpenCL 2.0 compatible kernel language support with OpenCL 1.2 compatible runtime
 * Supports offline ahead of time compilation today; during the Beta phase we will add in-process/in-memory compilation.
 * Binary Package support for Ubuntu 16.04
 * Binary Package support for Fedora 24 is not currently available
 * Dropping binary package support for Ubuntu 14.04, Fedora 23


