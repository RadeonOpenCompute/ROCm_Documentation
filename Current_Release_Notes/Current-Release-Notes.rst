.. image:: /Current_Release_Notes/amdblack.jpg

|

=============================================================
AMD Radeon Open Compute platforM (ROCm) Release Notes v3.3.0
=============================================================
April 1st, 2020

What Is ROCm?
==============

ROCm is designed to be a universal platform for gpu-accelerated computing. This modular design allows hardware vendors to build drivers that support the ROCm framework. ROCm is also designed to integrate multiple programming languages and makes it easy to add support for other languages. 

Note: You can also clone the source code for individual ROCm components from the GitHub repositories.

ROCm Components
~~~~~~~~~~~~~~~~

The following components for the ROCm platform are released and available for the v3.3
release:

• Drivers

• Tools

• Libraries

• Source Code

You can access the latest supported version of drivers, tools, libraries, and source code for the ROCm platform at the following location:
https://github.com/RadeonOpenCompute/ROCm


Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROCm v3.3.x platform is designed to support the following operating systems:


* Ubuntu 16.04.6(Kernel 4.15) and 18.04.4 (Kernel 5.3)

* CentOS v7.7 (Using devtoolset-7 runtime support)

* RHEL v7.7 (Using devtoolset-7 runtime support)

* SLES 15 SP1 


What\'s New in This Release
===========================

**Multi\- Version Installation**
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

These functions were originally deprecated in version 1.3 of the Comgr library as they no longer support options with embedded spaces. 

The deprecated functions are now replaced with the array-oriented options API, which includes 

*	`amd_comgr_action_info_set_option_list`
*	`amd_comgr_action_info_get_option_list_count`
* `amd_comgr_action_info_get_option_list_item`


Hardware and Software Support Information
==========================================

AMD ROCm is focused on using AMD GPUs to accelerate computational tasks such as machine learning, engineering workloads, and scientific computing. In order to focus our development efforts on these domains of interest, ROCm supports a targeted set of hardware configurations. 

For more information, see 

https://github.com/RadeonOpenCompute/ROCm

