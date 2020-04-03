.. _ROCm-MultiGPU:

MultiGPU In-node and Out of Node P2P Solutions
###############################################

In-node
--------
* Large BAR support BAR = Base Address Register	Making the GPU memory visible :ref:`BAR 1 Region Suppoted in Radeon Instinct MI25,MI8, MI6`
* ROCr Base driver has P2P API support
* `ROCr (HSA) AGENT API with Peer to Peer support <http://www.hsafoundation.com/html_spec111/HSA_Library.htm#Runtime/Topics/02_Core/hsa_iterate_agents.htm%3FTocPath%3DHSA%2520Runtime%2520Programmer%25E2%2580%2599s%2520Reference%2520Manual%2520Version%25201.1.1%2520%7CChapter%25202.%2520HSA%2520Core%2520Programming%2520Guide%7C2.3%2520System%2520and%2520agent%2520information%7C2.3.1%2520System%2520and%2520agent%2520information%2520API%7C_____18>`_.
* `HCC Language Runtime support of P2P	ROCr Agent API <https://scchan.github.io/hcc/classhc_1_1accelerator.html#aebd49b998f9421bd032ea450cbafd247>`_.
* `HIP Language Runtime support of P2P	P2P API’s model after CUDA P2P API’s <http://rocm-developer-tools.github.io/HIP/group__PeerToPeer.html>`_.
* OpenCL Language Runtime P2P API	Peer-to-Peer API  with Autocopy support over Intel QPI bus
   * API name -  clEnqueueBufferCopyP2PAMD
   * Releasing in OpenCL with ROCm 1.6.2
* HIP based Communication Primitives Helper Library to make it easier to use P2P - In Development
* ROCr level IPC 	Inter Process Communication 	API
* IPC is Supported in HIP API

Out of Node
------------

* `Remote DMA technology  ( RDMA) Peer-to-Peer bridge driver for PeerDirect <https://github.com/RadeonOpenCompute/ROCnRDMA>`_.
* `libibverbs Linux RDMA library YES -since ROCm 1.0 <https://github.com/RadeonOpenCompute/ROCnRDMA>`_.
* `PeerDirect Mellanox Peer API for Infiniband <https://community.mellanox.com/docs/DOC-2486>`_.

Standard Frameworks for Out of Node Communication
---------------------------------------------------
* `OpenUCX UCX is a communication library implementing high-performance messaging for MPI/PGAS frameworks - In Development <http://www.openucx.org./>`_ `Source for ROCm <https://github.com/openucx/ucx/tree/master/src/uct/rocm>`_. 
* `OpenMPI Open MPI Project is an open source Message Passing Interface https://www.open-mpi.org In Development <https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX>`_.
* `MPICH MPICH is a high-performance and widely portable implementation of the Message Passing Interface (MPI) standard (MPI-1, MPI-2 and MPI-3) <https://www.mpich.org/about/overview/>`_ `In Development <https://www.mpich.org/2016/08/30/mpich-3-3a1-released/>`_.
* `OpenSHMEM	Partitioned Global Address Space & Communication Library - In Development <https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX>`_.
* `OSU benchmark to test performance <https://github.com/ROCm-Developer-Tools/OSU_Microbenchmarks>`_.
