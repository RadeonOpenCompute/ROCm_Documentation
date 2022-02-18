.. image:: amdblack.jpg

.. _Programming-Guides:



=============================
HIP Programming Guide v4.5 
=============================

Heterogeneous-Computing Interface for Portability (HIP) is a C++ dialect designed to ease conversion of CUDA applications to portable C++ code. It provides a C-style API and a C++ kernel language. The C++ interface can use templates and classes across the host/kernel boundary.

The HIPify tool automates much of the conversion work by performing a source-to-source transformation from CUDA to HIP. HIP code can run on AMD hardware (through the HCC compiler) or NVIDIA hardware (through the NVCC compiler) with no performance loss compared with the original CUDA code.

Programmers familiar with other GPGPU languages will find HIP easy to learn and use. AMD platforms implement this language using the HC dialect providing similar low-level control over the machine.

Use HIP when converting CUDA applications to portable C++ and for new projects that require portability between AMD and NVIDIA. HIP provides a C++ development language and access to the best development tools on both platforms.

Programming Guide (PDF)
----------------------------

You can access and download the latest version of the HIP Programming Guide.  

`Download PDF <https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_HIP_Programming_Guide.pdf>`__

or

Access the following link for the guide,

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_HIP_Programming_Guide.pdf


Related Topics
----------------

HIP API Guide 
====================

You can access the Doxygen-generated HIP API Guide at the following location:

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD-HIP-API-4.5.pdf


HIP_Supported_CUDA_API_Reference_Guide
============================================

You can access and download the latest version of the HIP-Supported CUDA API Reference Guide.  

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_HIP_Supported_CUDA_API_Reference_Guide.pdf


AMD ROCm Compiler Reference Guide 
====================================

You can access and download the AMD ROCm Compiler Reference Guide at,

https://github.com/RadeonOpenCompute/ROCm/blob/rocm-4.5.2/AMD_Compiler_Reference_Guide_v4.5.pdf


HIP Installation Instructions
===============================

For HIP installation instructions, refer to

https://rocmdocs.amd.com/en/latest/Installation_Guide/HIP-Installation.html


HIP FAQ 
=========

 * :ref:`HIP-FAQ`


