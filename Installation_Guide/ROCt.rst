# ROCm Documentation has moved to docs.amd.com

.. meta::
   :http-equiv=Refresh: 0; url='https://docs.amd.com'
.. _ROCT:

=====================
ROCT-Thunk-Interface
=====================

ROCt Library
##############

This repository includes the user-mode API interfaces used to interact with the ROCk driver. Currently supported agents include only the AMD/ATI Fiji family of discrete GPUs.

Starting at 1.7 release, ROCt uses drm render device. This requires the user to belong to video group. Add the user account to video group with "sudo usermod -a -G video username" command if the user if not part of video group yet.

ROCk Driver
##############
The ROCt library is not a standalone product and requires that you have the correct ROCk driver set installed. We recommend reading the full compatibility and installation details which are available in the ROCk github:

https://github.com/RadeonOpenCompute/ROCK-Radeon-Open-Compute-Kernel-Driver

Building the Thunk
####################
A simple cmake-based system is available for building thunk. To build the thunk from the the ROCT-Thunk-Interface directory, execute:
::
  mkdir -p build
  cd build
  cmake ..
  make

If the hsakmt-roct and hsakmt-roct-dev packages are desired:
::
  mkdir -p build
  cd build
  cmake ..
  make package
  make package-dev

For Github repository link :  `ROCT-Thunk-Interface <https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/roc-2.4.0>`_

Disclaimer
************
The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2017 Advanced Micro Devices, Inc. All rights reserved.
