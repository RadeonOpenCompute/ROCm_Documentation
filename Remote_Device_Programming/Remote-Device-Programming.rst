
.. _Remote-Device-Programming:

==========================
Remote Device Programming
==========================

ROCnRDMA
=========
ROCmRDMA is the solution designed to allow third-party kernel drivers to utilize DMA access to the GPU  memory. It allows direct path for data exchange (peer-to-peer) using the standard features of PCI Express. 

Currently ROCmRDMA provides the following benefits:

 * Direct access to ROCm memory for 3rd party PCIe devices
 * Support for PeerDirect(c) interface to offloads the CPU when dealing 
   with ROCm memory for RDMA network stacks;

Restrictions and limitations
*****************************
To fully utilize ROCmRDMA  the number of limitation could apply impacting either performance or functionality in the whole:

 * It is recommended that devices utilizing ROCmRDMA share the same upstream PCI Express root complex. Such limitation depends on    	PCIe chipset manufacturses and outside of GPU controls;
 * To provide peer-to-peer DMA access all GPU local memory must be exposed via PCI memory BARs (so called large-BAR configuration);
 * It is recommended to have IOMMU support disabled or configured in pass-through mode due to limitation in Linux kernel to support  	local PCIe device memory for any form transition others then 1:1 mapping.

ROCmRDMA interface specification
*********************************
The implementation of ROCmRDMA interface could be found in `[amd_rdma.h] <https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/blob/roc-1.4.0/include/drm/amd_rdma.h>`_ file.

Data structures
*************** 

:: 
   
  
   /**
    * Structure describing information needed to P2P access from another device
    * to specific location of GPU memory
    */
   struct amd_p2p_info {
  	   uint64_t	   va;		   /**< Specify user virt. address
					     * which this page table described
					     */
	 
	   uint64_t	   size;	   /**< Specify total size of
					     * allocation
					     */
	  
	  struct pid	   *pid;	   /**< Specify process pid to which
					     * virtual address belongs
					     */
	 
	  struct sg_table *pages;	   /**< Specify DMA/Bus addresses */
	
	   void		*priv;		   /**< Pointer set by AMD kernel
					      * driver
					      */
     };

::

  /**
   * Structure providing function pointers to support rdma/p2p requirements.
   * to specific location of GPU memory
   */
   
   struct amd_rdma_interface {
  	  int (*get_pages)(uint64_t address, uint64_t length, struct pid *pid,
				  struct amd_p2p_info  **amd_p2p_data,
				  void  (*free_callback)(void *client_priv),
				  void  *client_priv);
	  int (*put_pages)(struct amd_p2p_info **amd_p2p_data);
	  int (*is_gpu_address)(uint64_t address, struct pid *pid);
	  int (*get_page_size)(uint64_t address, uint64_t length, struct pid *pid,
				  unsigned long *page_size);
  };
 
The function to query ROCmRDMA interface
****************************************

::

  
   /**
    * amdkfd_query_rdma_interface - Return interface (function pointers table) for
    *				 rdma interface
    *
    *
    *    \param interace     - OUT: Pointer to interface
    *    \return 0 if operation was successful.
    */
   int amdkfd_query_rdma_interface(const struct amd_rdma_interface **rdma);
   

The function to query ROCmRDMA interface
****************************************

::

   
   /**
    * amdkfd_query_rdma_interface - Return interface (function pointers table) for rdma interface
    * \param interace     - OUT: Pointer to interface
    * \return 0 if operation was successful.
    */
    int amdkfd_query_rdma_interface(const struct amd_rdma_interface **rdma);
   

ROCmRDMA interface functions description
*****************************************

:: 

   
   /**
    * This function makes the pages underlying a range of GPU virtual memory
    * accessible for DMA operations from another PCIe device
    *
    * \param   address       - The start address in the Unified Virtual Address
    *			       space in the specified process
    * \param   length        - The length of requested mapping
    * \param   pid           - Pointer to structure pid to which address belongs.
    *			       Could be NULL for current process address space.
    * \param   p2p_data      - On return: Pointer to structure describing
    *			       underlying pages/locations
    * \param   free_callback - Pointer to callback which will be called when access
    *			       to such memory must be stopped immediately: Memory
    *			       was freed, GECC events, etc.
    *			       Client should  immediately stop any transfer
    *			       operations and returned as soon as possible.
    *			       After return all resources associated with address
    *			       will be release and no access will be allowed.
    * \param   client_priv   - Pointer to be passed as parameter on
    *			       'free_callback;
    *
    * \return  0 if operation was successful
    */
    int get_pages(uint64_t address, uint64_t length, struct pid *pid,
		    struct amd_p2p_info **amd_p2p_data,
		    void  (*free_callback)(void *client_priv),
		    void  *client_priv);
::

   /**
    * This function release resources previously allocated by get_pages() call.
    * \param   p_p2p_data - A pointer to pointer to amd_p2p_info entries
    * 			allocated by get_pages() call.
    * \return  0 if operation was successful
    */
    int put_pages(struct amd_p2p_info **p_p2p_data)

::
   
  /**
    * Check if given address belongs to GPU address space.
    * \param   address - Address to check
    * \param   pid     - Process to which given address belongs.
    *		         Could be NULL if current one.
    * \return  0       - This is not GPU address managed by AMD driver
    *	       1       - This is GPU address managed by AMD driver
    */
    int is_gpu_address(uint64_t address, struct pid *pid);


::

   /**
    * Return the single page size to be used when building scatter/gather table
    * for given range.
    * \param   address   - Address
    * \param   length    - Range length
    * \param   pid       - Process id structure. Could be NULL if current one.
    * \param   page_size - On return: Page size
    * \return  0 if operation was successful
    */
    int get_page_size(uint64_t address, uint64_t length, struct pid *pid,
	         		unsigned long *page_size);



UCX
====

MPI
====

IPC
====

## MultiGPU In-node and Out of Node P2P Solutions

In-node			
========

* Large BAR support BAR = Base Address Register	Making the GPU memory visible [BAR 1 Region Suppoted in Radeon Instinct MI25,MI8, MI6](https://rocm.github.io/ROCmPCIeFeatures.html)
* ROCr Base driver has P2P API support	
 * [ROCr (HSA) AGENT API with Peer to Peer support](http://www.hsafoundation.com/html_spec111/HSA_Library.htm#Runtime/Topics/02_Core/hsa_iterate_agents.htm%3FTocPath%3DHSA%2520Runtime%2520Programmer%25E2%2580%2599s%2520Reference%2520Manual%2520Version%25201.1.1%2520%7CChapter%25202.%2520HSA%2520Core%2520Programming%2520Guide%7C2.3%2520System%2520and%2520agent%2520information%7C2.3.1%2520System%2520and%2520agent%2520information%2520API%7C_____18)
* [HCC Language Runtime support of P2P	 ROCr Agent API](https://scchan.github.io/hcc/classhc_1_1accelerator.html#aebd49b998f9421bd032ea450cbafd247)
* [HIP Language Runtime support of P2P	P2P API's model after CUDA P2P API's](http://rocm-developer-tools.github.io/HIP/group__PeerToPeer.html)
* OpenCL Language Runtime P2P API	Peer-to-Peer API  with Autocopy support over Intel QPI bus 
  * API name -  clEnqueueBufferCopyP2PAMD 
  * Releasing in OpenCL with ROCm 1.6.2	
* HIP based Communication Primitives Helper Library to make it easier to use P2P - In Development	
* ROCr level IPC 	Inter Process Communication 	API 
 * IPC is Supported in HIP API 

Out of Node
===========

* [Remote DMA technology  ( RDMA) 	Peer-to-Peer bridge driver for PeerDirect](https://github.com/RadeonOpenCompute/ROCnRDMA)
* [libibverbs	Linux RDMA library	YES -since ROCm 1.0](https://github.com/RadeonOpenCompute/ROCnRDMA)
* [PeerDirect	Mellanox Peer API for Infiniband](https://community.mellanox.com/docs/DOC-2486)

Standard Frameworks for Out of Node Communication
=================================================

* [OpenUCX	UCX is a communication library implementing high-performance messaging for MPI/PGAS frameworks - 	In Development](http://www.openucx.org.) [Source for ROCm](https://github.com/openucx/ucx/tree/master/src/uct/rocm)
* [OpenMPI	Open MPI Project is an open source Message Passing Interface https://www.open-mpi.org	In Development](https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX)
* [ MPICH	MPICH is a high-performance and widely portable implementation of the Message Passing Interface (MPI) standard (MPI-1, MPI-2 and MPI-3)](https://www.mpich.org/about/overview/)	 [In Development](https://www.mpich.org/2016/08/30/mpich-3-3a1-released/)
* [OpenSHMEM	Partitioned Global Address Space & Communication Library - 	In Development](https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX)
* [OSU benchmark to test performance](https://github.com/ROCm-Developer-Tools/OSU_Microbenchmarks)
