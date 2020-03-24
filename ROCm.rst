.. _GuidedIntro:

.. image:: amdblack.jpg
|

AMD ROCm Documentation
===================

| `Release Notes <http://rocm-documentation.readthedocs.io/en/latest/Current_Release_Notes/Current-Release-Notes.html#rocm-1-8-what-new>`_
| Release Notes for the latest version of AMD ROCm.

`Installation Guide <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installation-guide>`_

*  `AMD ROCm Repositories <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installing-from-amd-rocm-repositories>`_

*  `Ubuntu Debian Repository <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#ubuntu-support-installing-from-a-debian-repository>`_

*  `Yum Repository <https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#centos-rhel-7-both-7-4-and-7-5-support>`__

*  `Getting ROCm Source Code <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#getting-rocm-source-code>`_

*  `Installing ROCk-Kernel <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/ROCk-kernel.html#rock-kernel>`_

*  `Installation FAQ <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/FAQ-on-Installation.html#faq-on-installation>`_


`Programming Guide <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#programming-guide>`_

This guide provides documentation on the ROCm programming model and programming interface. It describes the hardware implementation and provides guidance on how to achieve maximum performance. The appendices include:

* a list of all ROCm-enabled devices
* detailed description of all extensions to the C language
* listings of supported mathematical functions
* C++ features supported in host and device code
* technical specifications of various devices
* introduction to the low-level driver API 


| -  `ROCm Languages <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#rocm-languages>`_

ROCm stack offers options for multiple programming-languages


| -  `HC Programing Guide <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hc-programing-guide>`_

Heterogeneous Compute (HC) programming installation requirements, methods to install on various platforms, and how to build it from source

| -  `HC Best Practices <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hc-best-practices>`_

Build-in Macros, HCC Profiler mode, and API Documentaion

| -  `HIP Programing Guide <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hip-programing-guide>`_

HIP programming, installation requirements, methods to install on various platfroms, and how to build it from source

| -  `HIP Best Practices <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hip-best-practices>`_

HIP Porting, Debugging, Bugs, FAQ and other aspects of HIP

| -  `OpenCL Programing Guide <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#opencl-programing-guide>`_

OpenCL Architecture, AMD Implementation, Profiling, and other aspects of OpenCL

| -  `OpenCL Best Practices <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#opencl-best-practices>`_

Performance and optimization for various device types such as GCN devices



`GCN ISA Manuals <http://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/GCN-ISA-Manuals.html#gcn-isa-manuals>`_

* `GCN 1.1 <http://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/GCN-ISA-Manuals.html#gcn-1-1>`_  - For information on ISA Manual for Hawaii (Sea Islands Series Instruction Set Architecture) 

* `GCN 2.0 <http://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/GCN-ISA-Manuals.html#gcn-2-0>`_  - For information on ISA Manual for Fiji and Polaris (AMD Accelerated Parallel Processing technology)

* `Vega <http://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/GCN-ISA-Manuals.html#vega>`_  - Provides “Vega” Instruction Set Architecture, Program Organization, Mode register and more details. 	

* `Inline GCN ISA Assembly Guide <http://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/GCN-ISA-Manuals.html#inline-gcn-isa-assembly-guide>`_ - Covers various concepts of AMDGCN Assembly, DS Permute Instructions, Parameters to a Kernel, GPR Counting.



`ROCm API References <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#rocm-api-references>`_

*  `ROCr System Runtime API <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#rocr-system-runtime-api>`_ 

* `HCC Language Runtime API <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#hcc-language-runtime-api>`_

* `HIP Language Runtime API <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#hip-language-runtime-api>`_

* `HIP Math API <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#hip-math-api>`_

* `Thrust API Documentation <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#thrust-api-documentation>`_

* `Math Library API <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#math-library-api-s>`_ - Includes HIP MAth API with hcRNG, clBLAS, clSPARSE APIs

* `Deep Learning API <http://rocm-documentation.readthedocs.io/en/latest/ROCm_API_References/ROCm-API-References.html#deep-learning-api-s>`_ - Includes MIOpen API and MIOpenGEMM APIs	



`ROCm Tools <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#rocm-tools>`_

* `Heterogeneous Compute Compiler (HCC) <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#hcc>`_


* `GCN Assembler and Disassembler <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#gcn-assembler-and-disassembler>`_

* `GCN Assembler Tools <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#gcn-assembler-tools>`_  - AMDGPU ISA Assembler 

* `ROCm-GDB <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#rocm-gdb>`_ - ROCm-GDB tool includes installtion, configuration, and working of Debugger and APIs

* `ROCm-Profiler <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#rocm-profiler>`_ - Radeon Compute Profiler- performance analysis tool

* `ROCm-Tracer <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#roc-tracer>`_ - ROCm Tracer - provides a generic independent from specific runtime profiler to trace API and asynchronous activity. Includes details on library source tree, steps to build and run the test

* `CodeXL <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#codexl>`_ 

* `GPUperfAPI <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#gpuperfapi>`_ - GPU Performance API, cloning, system requiments, and source code directory layout



`AOMP <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#aomp-v-0-7-5>`_ 

Provides details on AOMP, a scripted build of LLVM and supporting software. Supports OpenMP target offload on AMD GPUs. Since AOMP is a clang/llvm compiler, it also supports GPU offloading with HIP, CUDA, and OpenCL.


`ROCmValidationSuite <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/ROCm-Tools.html#rocmvalidationsuite>`_ 

Provides details on ROCm Validation Suite (RVS), a system administrator’s and cluster manager’s tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

|

`ROCm Libraries <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm-Libraries.html>`_

| `rocFFT <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/rocFFT.html#rocfft>`_
| This section provides details on rocFFT,it is a AMD's software library compiled with the CUDA compiler using HIP tools for running on Nvidia GPU devices.

| `rocBLAS <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/rocblas.html#rocblas>`_
| This section provides details on rocBLAS, it is a library for BLAS on ROCm.rocBLAS is implemented in the HIP programming language and optimized for AMD’s latest discrete GPUs.

| `hipBLAS <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/hipBLAS.html#hip8las>`_
| This section provides details on hipBLAS, it is a BLAS marshalling library, with multiple supported backends. hipBLAS exports an interface that does not require the client to change. Currently,it supports :ref:`rocblas` and cuBLAS as backends.

| `hcRNG <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/hcRNG.html#hcrng>`_
| This section provides details on hcRNG. It is a software library ,where uniform random number generators targeting the AMD heterogeneous hardware via HCC compiler runtime is implemented..

| `hipeigen <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/hipeigen.html#hipeigen>`_
| This section provides details on Eigen.It is a C++ template library which provides linear algebra for  matrices, vectors, numerical solvers, and related algorithms.

| `clFFT <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/clFFT.html#c1fft>`_
| This section provides details on clFFT.It is a software library which contains  FFT functions written in OpenCL,and clFFt also supports running on CPU devices to facilitate debugging and heterogeneous programming.

| `clBLAS <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/clBLA.html#clbla>`_
| This section provides details on clBLAS. It makes easier for developers to utilize the inherent performance and power efficiency benefits of heterogeneous computing.

| `clSPARSE <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/clSPARSE.html#clsparse1>`_
| This section provides details on clSPARSE, it is an OpenCL library which implements Sparse linear algebra routines. 

| `clRNG <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/clRNG.html#cl1rng>`_
| This section provides details on clRNG,This is a library  for uniform random number generation in OpenCL.

| `hcFFT <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/hcFFT.html#hcfft>`_
| This section provides details on hcFFT, it hosts the HCC based FFT Library and  targets  GPU acceleration of FFT routines on AMD devices.

| `Tensile <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Tools/tensile.html#tensile>`_
| This section provides details on Tensile. It is a tool for creating a benchmark-driven backend library for GEMMs,N-dimensional tensor contractions and  multiplies two multi-dimensional objects together on a GPU.

| `rocALUTION <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#rocalution>`_
| This section provides details on rocALUTION. It is a sparse linear algebra library with focus on exploring fine-grained parallelism, targeting modern processors and accelerators including multi/many-core CPU and GPU platforms. It can be seen as middle-ware between different parallel backends and application specific packages.

| `rocSPARSE <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#id38>`_
| This section provides details on rocSPARSE.It is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HiP for GPU devices. It is designed to be used from C and C++ code.

| `rocThrust <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#rocthrust>`_
| This section provides details on rocThrust. It is a parallel algorithmn library.  

| `hipCUB <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#hipcub>`_ This section provides details on   hipCUB. 
| It is a thin wrapper library on top of rocPRIM or CUB. It enables developers to port the project using CUB library to the HIP layer and to 
| run them on AMD hardware.

| `ROCm SMI Library <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#rocm-smi-library>`_ This section provides details on ROCm SMI library. The ROCm System Management Interface Library, or ROCm SMI library is part of the Radeon Open Compute ROCm software stack. It is a C library for linux that provides a user space interface for applications to monitor and control GPU aplications.

| `RCCL <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#rccl>`_ This section provides details on ROCm Communications Collectives Library. It is a stand alone library of standard collective communication routines for GPUS, implememting all-reduce, all gather, reduce, broadcast, and reduce scatter.

| `AMD MivisionX <https://rocm-documentation.readthedocs.io/en/latest/ROCm_Libraries/ROCm_Libraries.html#amd-migraphx>`_
This section provides information on AMD’s graph optimization engine.


`ROCm Compiler SDK <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Compiler-SDK.html#rocm-compiler-sdk>`_

| `GCN Native ISA LLVM Code Generator <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Compiler-SDK.html#gcn-native-isa-llvm-code-generator>`_
| This section provide complete description on LLVM such as introduction, Code Object, Code conventions, Source languages, etc.,

| `ROCm Code Object Format <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Compiler-SDK.html#rocm-code-object-format>`_
| This section describes about application binary interface (ABI) provided by the AMD, implementation of the HSA runtime. It also provides details on Kernel, AMD Queue and Signals.
 
| `ROCm Device Library <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Compiler-SDK.html#roc-device-library>`_
| Documentation on instruction related to ROCm Device Library overview,Building and Testing related information with respect to Device Library is provided.

| `ROCr Runtime <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Compiler-SDK.html#rocr-runtime>`_
| This section refers the user-mode API interfaces and libraries necessary for host applications to launch compute kernels to available HSA ROCm kernel agents. we can find installation details and Infrastructure details related to ROCr.

`ROCm System Management <http://rocm-documentation.readthedocs.io/en/latest/ROCm_System_Managment/ROCm-System-Managment.html#rocm-system-management>`_


| `ROCm-SMI <http://rocm-documentation.readthedocs.io/en/latest/ROCm_System_Managment/ROCm-System-Managment.html#rocm-smi>`_
| ROCm System Management Interface a complete guide to use and work with rocm-smi tool.

| `SYSFS Interface <http://rocm-documentation.readthedocs.io/en/latest/ROCm_System_Managment/ROCm-System-Managment.html#sysfs-interface>`_
| This section provides information on sysfs file structure with details related to file structure related to system are captured in sysfs.

| `KFD Topology <http://rocm-documentation.readthedocs.io/en/latest/ROCm_System_Managment/ROCm-System-Managment.html#kfd-topology>`_
| KFD Kernel Topology is the system file structure which describes about AMD GPU related information such as nodes, Memory, Cache and IO-links.

`ROCm Virtualization & Containers <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html#rocm-virtualization-containers>`_

| `PCIe Passthrough on KVM <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html#pcie-passthrough-on-kvm>`_
| Here PCIe Passthrough on KVM is described. A KVM-based instructions assume a headless host with an input/output memory management unit (IOMMU) to pass peripheral devices such as a GPU to guest virtual machines.more information can be found on the same here.

| `ROCm-Docker <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html#rocm-docker>`_
| A framework for building the software layers defined in the Radeon Open Compute Platform into portable docker images. Detailed Information related to ROCm-Docker can be found.

`Remote Device Programming <http://rocm-documentation.readthedocs.io/en/latest/Remote_Device_Programming/Remote-Device-Programming.html#remote-device-programming>`_

| `ROCnRDMA <http://rocm-documentation.readthedocs.io/en/latest/Remote_Device_Programming/Remote-Device-Programming.html#rocnrdma>`_
| ROCmRDMA is the solution designed to allow third-party kernel drivers to utilize DMA access to the GPU memory. Complete indoemation related to ROCmRDMA is Documented here.

| `UCX <http://rocm-documentation.readthedocs.io/en/latest/Remote_Device_Programming/Remote-Device-Programming.html#ucx>`_
| This section gives information related to UCX, How to install, Running UCX and much more 

| `MPI <http://rocm-documentation.readthedocs.io/en/latest/Remote_Device_Programming/Remote-Device-Programming.html#mpi>`_
| This section gives information related to MPI.

| `IPC <http://rocm-documentation.readthedocs.io/en/latest/Remote_Device_Programming/Remote-Device-Programming.html#ipc>`_
| This section gives information related to IPC.

`Deep Learning on ROCm <http://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#deep-learning-on-rocm>`_

| This section provides details on ROCm Deep Learning concepts.

| `Porting from cuDNN to MIOpen <http://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#porting-from-cudnn-to-miopen>`_
| The porting guide highlights the key differences between the current cuDNN and MIOpen APIs.

| `Deep Learning Framework support for ROCm <http://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#deep-learning-framework-support-for-rocm>`_
| This section provides detailed chart of Frameworks supported by ROCm and repository details.

| `Tutorials <http://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html#tutorials>`_
| Here Tutorials on different DeepLearning Frameworks are documented.

`System Level Debug <http://rocm-documentation.readthedocs.io/en/latest/Other_Solutions/Other-Solutions.html#system-level-debug>`_

| `ROCm Language & System Level Debug, Flags and Environment Variables <http://rocm-documentation.readthedocs.io/en/latest/Other_Solutions/Other-Solutions.html#rocm-language-system-level-debug-flags-and-environment-variables>`_
| Here in this section we have details regardinf various system related debugs and commands for isssues faced while using ROCm.

`Tutorial <http://rocm-documentation.readthedocs.io/en/latest/Tutorial/Tutorial.html#tutorial>`_

| This section Provide details related to few Concepts of HIP and other sections.

`ROCm Glossary <http://rocm-documentation.readthedocs.io/en/latest/ROCm_Glossary/ROCm-Glossary.html#rocm-glossary>`_

| ROCm Glossary gives highlight concept and their main concept of how they work.


