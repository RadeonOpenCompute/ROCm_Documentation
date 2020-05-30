
.. image:: amdblack.jpg
.. rocm documentation master file, created by
   sphinx-quickstart on Tue Jul 11 20:12:28 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
|

Welcome to AMD ROCm Platform
=============================

AMD ROCm is the first open-source software development platform for HPC/Hyperscale-class GPU computing. AMD ROCm brings the UNIX philosophy of choice, minimalism and modular software development to GPU computing. 

AMD ROCm is built for scale; it supports multi-GPU computing in and out of server-node communication through RDMA. AMD ROCm also simplifies the stack when the driver directly incorporates RDMA peer-sync support.

The AMD ROCm Programming-Language Run-Time 
############################################

The AMD ROCr System Runtime is language independent and makes heavy use of the Heterogeneous System Architecture (HSA) Runtime API. This approach provides a rich foundation to execute programming languages such as HCC C++ and HIP.


.. image:: ROCm_Stack.png
    :align: center
    

Important features include the following:

    * Multi-GPU coarse-grain shared virtual memory
    * Process concurrency and preemption
    * Large memory allocations
    * HSA signals and atomics
    * User-mode queues and DMA
    * Standardized loader and code-object format
    * Dynamic and offline-compilation support
    * Peer-to-peer multi-GPU operation with RDMA support
    * Profiler trace and event-collection API
    * Systems-management API and tools


.. image:: ROCm_Core_Stack.png
    :align: center
    
Solid Compilation Foundation and Language Support
####################################################

•	LLVM compiler foundation
•	HCC C++ and HIP for application portability
•	GCN assembler and disassembler

AMD ROCm gives developers the flexibility of choice for hardware and aids in the development of compute-intensive applications.



.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Release Documentation
  
   Release Notes
   Current_Release_Notes/Current-Release-Notes
   Installation_Guide/Installation-Guide
   
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption:  Compiler Documentation
      
   Programming_Guides/Programming-Guides
   ROCm_Compiler_SDK/ROCm-Compiler-SDK
      
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: System Management Interface
      
   ROCm_System_Managment/ROCm-System-Managment
   Other_Solutions/Other-Solutions
      
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Library Documentation 
   
   ROCm_Libraries/ROCm_Libraries
   ROCm_API_References/ROCm-API-References
   Deep_learning/Deep-learning
   
   
  
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: ROCm-Tools
      
   ROCm_Tools/ROCm-Tools
   

    
    
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Additional Documentation 
   
   GCN_ISA_Manuals/GCN-ISA-Manuals
   Remote_Device_Programming/Remote-Device-Programming
   Tutorial/Tutorial
   ROCm_Glossary/ROCm-Glossary


   
