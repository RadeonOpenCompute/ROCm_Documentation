.. image:: /Current_Release_Notes/amdblack.jpg

.. rocm documentation master file, created by
   sphinx-quickstart on Tue Jul 11 20:12:28 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
|

New AMD ROCm™ Information Portal - ROCm v4.5 and Above
-----------------------------------------------------

Beginning ROCm release v5.0, AMD ROCm documentation has a new portal at `https://docs.amd.com <https://docs.amd.com/>`__. This portal consists
of ROCm documentation v4.5 and above.

For documentation prior to ROCm v4.5, you may continue to access `http://rocmdocs.amd.com <http://rocmdocs.amd.com/>`__.


Welcome to AMD ROCm™ Platform
=============================

AMD ROCm is the first open-source software development platform for HPC/Hyperscale-class GPU computing. AMD ROCm brings the UNIX philosophy of choice, minimalism and modular software development to GPU computing. 

Since the ROCm ecosystem is comprised of open technologies: frameworks (Tensorflow / PyTorch), libraries (MIOpen / Blas / RCCL), programming model (HIP), inter-connect (OCD) and up streamed Linux® Kernel support – the platform is continually optimized for performance and extensibility.  Tools, guidance and insights are shared freely across the ROCm GitHub community and forums.

**Note:** The AMD ROCm™ open software platform is a compute stack for headless system deployments. GUI-based software applications are currently not supported.

.. image:: latestGPU.PNG
    :align: center


AMD ROCm is built for scale; it supports multi-GPU computing in and out of server-node communication through RDMA. AMD ROCm also simplifies the stack when the driver directly incorporates RDMA peer-sync support.

The AMD ROCm Programming-Language Run-Time 
############################################

The AMD ROCr System Runtime is language independent and makes heavy use of the Heterogeneous System Architecture (HSA) Runtime API. This approach provides a rich foundation to execute programming languages, such as HIP and OpenMP.


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
•	HIP for application portability
•	GCN assembler and disassembler

AMD ROCm gives developers the flexibility of choice for hardware and aids in the development of compute-intensive applications.

ROCm Learning Center
######################

https://developer.amd.com/resources/rocm-resources/rocm-learning-center/



    
    
    
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption:  Current Release Documentation
   
  
   Release Notes
   Current_Release_Notes/Current-Release-Notes
   Current_Release_Notes/Deprecation
   Current_Release_Notes/ROCm-Version-History
   Current_Release_Notes/ROCm-Learning-Center
   Current_Release_Notes/DISCLAIMER
   
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption:  Previous ROCm Release
   
    
   Archived_Documentation/README431.rst
   Installation_Guide/Installation-Guide
   Installation_Guide/Multiversion-Installation
   Installation_Guide/Using-CMake-with-AMD-ROCm
   Installation_Guide/Mesa-Multimedia-Installation
   

   
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption:  Install ROCm
   
  
   Installation_Guide/Installation_new 
   Installation_Guide/HIP-Installation
   Installation_Guide/Tools-Installation
   Installation_Guide/Software-Stack-for-AMD-GPU  
  
   
      
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption:  Tuning Documentation
   
   
   Tuning_Documentation/Tuning_Guide
   
   
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption:  Compiler Documentation
   
      
   Programming_Guides/Programming-Guides
   Programming_Guides/HIP_API_Guide
   Programming_Guides/openmp_support
  
      
   
      
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Library Documentation 
   
   ROCm_Libraries/ROCm_Libraries
   Deep_learning/Deep-learning
  
  
   
  
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: ROCm-Tools Documentation
   
      
   ROCm_Tools/ROCm-Tools
   ROCm_Tools/ROCTracer-API
   ROCm_Tools/ROCgdb.rst
   ROCm_Tools/ROCm-Data-Center
   ROCm_Tools/rocm-debug-agent
   Other_Solutions/Other-Solutions
   Other_Solutions/rocm-validation-suite
   
   
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: System Management Interface
   
      
   ROCm_System_Managment/ROCm-System-Managment
   ROCm_System_Managment/ROCm-SMI-CLI
   
   
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: ROCm Learning Center
   
    Current_Release_Notes/ROCm-Learning-Center
    
    
.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Additional Documentation 
   
   
   GCN_ISA_Manuals/GCN-ISA-Manuals
   Remote_Device_Programming/Remote-Device-Programming
  
   

.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Archived Documentation 
   
  
   Archived_Documentation/4_1_Installation_Guide.rst
  
  
   
