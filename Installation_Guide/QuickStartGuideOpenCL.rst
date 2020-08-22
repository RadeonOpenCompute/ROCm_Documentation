.. _QuickStartGuideOpenCL:

Quick Start Guide For OpenCL
============================

* ROCm 1.7 introduces big updates to our OpenCL compiler and runtime implementation -- built on top of the ROCm software stack! 

This developer release includes the following:
------------------------------

* OpenCL 2.0 compatible kernel language support with OpenCL 1.2 compatible runtime
* OpenCL compiler also has assembler and disassembler support,  inline assembly support is now in place. 
* Big improvements in the base compiler as we roll in new optimization for application in new Native LLVM code generator. 
* We made our base compiler intrinsics source code available
  * OCML https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/master/doc/OCML.md
  * Source code for the Intrinsic https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/master/opencl/src
* Supports offline ahead of time compilation and in-process/in-memory compilation.
* Binary Package support for Ubuntu  16.04 and Fedora 24

Quickstart Instructions
------------------------------

Here's a simple workflow to get you quickly up and running with OpenCL on ROCm.

Install the ROCm OpenCL implementation (assuming you already have the 'rocm' package installed)
::
 sudo apt-get install rocm-opencl-dev


For a sample OpenCL application, let's use a simple vector-add example from the University of Bristol's very nice "Hands On OpenCL" lectures.

.. code-block:: 


 git clone https://github.com/HandsOnOpenCL/Exercises-Solutions.git

 cd Exercises-Solutions/Exercises/Exercise02/C

 make \
   CCFLAGS="-I$OPENCL_ROOT/include/opencl1.2 -O3 -DDEVICE=CL_DEVICE_TYPE_DEFAULT" \
   LIBS="-L$OPENCL_ROOT/lib/x86_64 -lOpenCL -lm"

 ./vadd


Not for all your application that supported the AMDGPU SDK for OpenCL to get the Header,  rocm-opencl-dev now included the headerfiles. 

If your built all your code with the AMDAPPSDK you do not need to download anything else,  you can just export environment variable to  /opt/rocm/opencl    

Do not install the AMDAPPSDK 3.0  on ROCm OpenCL it designed for old driver which need headers installed.  rocm-opencl-dev package does this for you. 

Example 1 for AMDAPPSDKROOT
::
 export AMDAPPSDKROOT=/opt/rocm/opencl 


Example 2 for AMDAPPSDK
::
 export AMDAPPSDK=/opt/rocm/opencl


Where is clinfo?
::
 /opt/rocm/opencl/bin/x86_64/clinfo 


* That's it!  Super easy. 

Related Resources
-----------------

ROCm Developer website will have more information: http:/rocm.github.io

University of Bristol's "Hands On OpenCL" course webpage:  https://handsonopencl.github.io/
