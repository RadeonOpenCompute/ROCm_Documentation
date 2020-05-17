.. image:: /Current_Release_Notes/amdblack.jpg

|

=============================================================
AMD Radeon Open Compute platforM (ROCm) Release Notes v3.5
=============================================================
May 27th, 2020

Supported Operating Systems and Documentation Updates
=======================================================

This document describes the features, fixed issues, and information about downloading and installing the AMD ROCm software.

It also covers known issues and deprecated features in the AMD ROCm v3.5 release.


Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following components for the ROCm platform are new and available for the v3.5 release:

* Support for RHEL v8.1 
* Support for CentOS v8.1 

The ROCm v3.5.x platform is designed to support the following operating systems:

* Ubuntu 16.04.6(Kernel 4.15) and 18.04.4(Kernel 5.3)
* CentOS 7.7 (Kernel 3.10-1062) and RHEL 7.8(Kernel 3.10.0-1127)(Using devtoolset-7 runtime support)
* CentOS 7.6 + FBK 5.2
* SLES 15 SP1
* CentOS and RHEL 8.1(Kernel 4.18.0-147)

NOTE: Framework support (TensorFlow, pyTorch & Caffe2) for v8.1 is not available.





What\'s New in This Release
===========================

**Multi\-Version Installation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install and access multiple versions of the ROCm toolkit simultaneously.

Previously, users could install only a single version of the ROCm toolkit. 

Now, users have the option to install multiple versions simultaneously and toggle to the desired version of the ROCm toolkit. From the v3.3 release, multiple versions of ROCm packages can be installed in the */opt/rocm-<version>* folder.
 
**Prerequisites**
###############################

Ensure the existing installations of ROCm, including */opt/rocm*, are completely removed before the v3.3 ROCm toolkit installation. The ROCm v3.3 package requires a clean installation.

* To install a single instance of ROCm, use the rocm-dkms or rocm-dev packages to install all the required components. This creates a symbolic link */opt/rocm* pointing to the corresponding version of ROCm installed on the system. 

* To install individual ROCm components, create the */opt/rocm* symbolic link pointing to the version of ROCm installed on the system. For example, *# ln -s /opt/rocm-3.3.0 /opt/rocm*

* To install multiple instance ROCm packages, create */opt/rocm* symbolic link pointing to the version of ROCm installed/used on the system. For example, *# ln -s /opt/rocm-3.3.0 /opt/rocm*

**Note**: The Kernel Fusion Driver (KFD) must be compatible with all versions of the ROCm software installed on the system.


Before You Begin
#################

Review the following important notes:

**Single Version Installation**

To install a single instance of the ROCm package, access the non-versioned packages. You must not install any components from the multi-instance set.

For example, 

* rocm-dkms

* rocm-dev

* hip

A fresh installation or an upgrade of the single-version installation will remove the existing version completely and install the new version in the */opt/rocm-<version>* folder.

.. image:: /Current_Release_Notes/singleinstance.png

**Multi Version Installation**

* To install a multi-instance of the ROCm package, access the versioned packages and components. 

For example,

  * rocm-dkms3.3.0

  * rocm-dev3.3.0

  * hip3.3.0

* The new multi-instance package enables you to install two versions of the ROCm toolkit simultaneously and provides the ability to toggle between the two versioned packages.

* The ROCm-DEV package does not create symlinks

* Users must create symlinks if required

* Multi-version installation with previous ROCm versions is not supported

* Kernel Fusion Driver (KFD) must be compatible with all versions of ROCm installations

.. image:: /Current_Release_Notes/MultiIns.png

**IMPORTANT**: A single instance ROCm package cannot co-exist with the multi-instance package. 

**NOTE**: The multi-instance installation applies only to ROCm v3.3 and above. This package requires a fresh installation after the complete removal of existing ROCm packages. The multi-version installation is not backward compatible. 


**GPU Process Information**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new functionality to display process information for GPUs is available in this release. For example,  you can view the process details to determine if the GPU(s) must be reset. 

To display the GPU process details, you can:

* Invoke the API 

or

* Use the Command Line Interface (CLI)

For more details about the API and the command instructions, see
https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/docs/ROCm_SMI_Manual.pdf


**Support for 3D Pooling Layers**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD ROCm is enhanced to include support for 3D pooling layers. The implementation of 3D pooling layers now allows users to run 3D convolutional networks, such as ResNext3D, on AMD Radeon Instinct GPUs. 


**ONNX Enhancements**
~~~~~~~~~~~~~~~~~~~~~~~~~

Open Neural Network eXchange (ONNX) is a widely-used neural net exchange format. The AMD model compiler & optimizer support the pre-trained models in ONNX, NNEF, & Caffe formats. Currently, ONNX versions 1.3 and below are supported. 

The AMD Neural Net Intermediate Representation (NNIR) is enhanced to handle the rapidly changing ONNX versions and its layers. 

.. image:: /Current_Release_Notes/onnx.png


Deprecations in the v3.3 Release
================================

Code Object Manager (Comgr) Functions
##################################

The following Code Object Manager (Comgr) functions are deprecated.

* `amd_comgr_action_info_set_options` 
* `amd_comgr_action_info_get_options` 

These functions were originally deprecated in version 1.3 of the Comgr library as they no longer supported options with embedded spaces. 

The deprecated functions are now replaced with the array-oriented options API, which include

*	`amd_comgr_action_info_set_option_list`
*	`amd_comgr_action_info_get_option_list_count`
* `amd_comgr_action_info_get_option_list_item`


Hardware and Software Support Information
==========================================

AMD ROCm is focused on using AMD GPUs to accelerate computational tasks such as machine learning, engineering workloads, and scientific computing. In order to focus our development efforts on these domains of interest, ROCm supports a targeted set of hardware configurations. 

For more information, see 

https://github.com/RadeonOpenCompute/ROCm

DISCLAIMER 
===========
The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information.  Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.  No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document.  Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. S
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

