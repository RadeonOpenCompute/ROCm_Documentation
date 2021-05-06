
.. _Remote-Device-Programming:

==========================
Remote Device Programming
==========================

ROCmRDMA
=========
**Peer-to-Peer bridge driver for PeerDirect - Deprecated Repo**

This is now included as part of the ROCK `Kernel Driver <https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver>`_
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
The implementation of ROCmRDMA interface can be found in `[amd_rdma.h] <https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/blob/master/include/drm/amd_rdma.h>`_ file.

API versions
************

ROCm up to and including v4.1 supported RDMA version 1.0. 

ROCm 4.2 has enhanced the API version to 2.0, introduced the following definitions to allow users to detect the API version, and apply conditional compilation as needed:

::

   /* API versions:
    * 1.0 Original API until ROCm 4.1, AMD_RDMA_MAJOR/MINOR undefined
    * 2.0 Added IOMMU (dma-mapping) support, removed p2p_info.kfd_proc
    *     Introduced AMD_RDMA_MAJOR/MINOR version definition
    */
   #define AMD_RDMA_MAJOR 2
   #define AMD_RDMA_MINOR 0

Data structures
*************** 

:: 
   
   /**
    * Structure describing information needed to P2P access from another device
    * to specific location of GPU memory
    */
   struct amd_p2p_info {
           uint64_t        va;             /**< Specify user virt. address
                                             * which this page table
                                             * described
                                             */
           uint64_t        size;           /**< Specify total size of
                                             * allocation
                                             */
           struct pid      *pid;           /**< Specify process pid to which
                                             * virtual address belongs
                                             */
           struct sg_table *pages;         /**< Specify DMA/Bus addresses */
           void            *priv;          /**< Pointer set by AMD kernel
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
                            struct device *dma_dev,
                            struct amd_p2p_info **amd_p2p_data,
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
    * \param   dma_dev       - Device that will need a DMA mapping of the memory
    * \param   amd_p2p_data  - On return: Pointer to structure describing
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
                  struct device *dma_dev, struct amd_p2p_info **amd_p2p_data,
                  void (*free_callback)(void *client_priv),
                  void *client_priv);
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

**What is UCX ?**

Unified Communication X (UCX) is a communication library for building Message Passing (MPI), PGAS/OpenSHMEM libraries and RPC/data-centric applications. UCX utilizes high-speed networks for inter-node and shared memory mechanisms for intra-node communication. For more information, visit http://openucx.github.io/ucx/

**How to install UCX with ROCm ?**

See `How to install UCX and OpenMPI <https://github.com/openucx/ucx/wiki/Build-and-run-ROCM-UCX-OpenMPI>`_


**How to enable ROCm transport during configuration and runtime**

Access the following links to enable ROCm transport during configuration and runtime:


* For release builds: ./contrib/configure-release --prefix=/path/to/install --with-rocm=/path/to/rocm

* For debug builds: ./contrib/configure-devel --prefix=/path/to/install --with-rocm=/path/to/rocm


OpenMPI
=========

**OpenMPI and OpenSHMEM installation**

1. Get latest-and-gratest OpenMPI version:
::
  $ git clone https://github.com/open-mpi/ompi.git

2. Autogen:
::
  $ cd ompi
  $ ./autogen.pl

3. Configure with UCX
::
  $ mkdir build
  $ cd build
  ../configure --prefix=/your_install_path/ --with-ucx=/path_to_ucx_installation

4. Build:
::
  $ make
  $ make install

**Running Open MPI with UCX**

Example of the command line (for InfiniBand RC + shared memory):

::
  
  $ mpirun -np 2 -mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,sm ./app


**Open MPI runtime optimizations for UCX**

* By default OpenMPI enables build-in transports (BTLs), which may result in additional software overheads in the OpenMPI progress function. In order to workaround this issue you may try to disable certain BTLs.

::

  $ mpirun -np 2 -mca pml ucx --mca btl ^vader,tcp,openib -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,sm ./app

* OpenMPI version https://github.com/open-mpi/ompi/commit/066370202dcad8e302f2baf8921e9efd0f1f7dfc leverages more efficient timer mechanism and there fore reduces software overheads in OpenMPI progress

**MPI and OpenSHMEM release versions tested with UCX master**

 1. UCX current tarball: https://github.com/openucx/ucx/archive/master.zip

 2. The table of MPI and OpenSHMEM distributions that are tested with the HEAD of UCX master

================ ===========
MPI/OpenSHMEM     project	
OpenMPI/OSHMEM     2.1.0
MPICH		   Latest
================ ===========



IPC API
========

New Datatypes
**************

::
 
 hsa_amd_ipc_memory_handle_t
 
 /** IPC memory handle to by passed from one process to another */
 typedef struct  hsa_amd_ipc_memory_handle_s {
       uint64_t handle;
 } hsa_amd_ipc_memory_handle_t;
  
 hsa_amd_ipc_signal_handle_t
  
 /** IPC signal  handle to by passed from one process to another */
 typedef struct  hsa_amd_ipc_signal_handle_s {
      uint64_t handle;
 } hsa_amd_ipc_signal_handle_t;

  
**Memory sharing API**

Allows sharing of HSA allocated memory between different processes.

| hsa_amd_ipc_get_memory_handle
| The purpose of this API is to get / export an IPC handle for an existing allocation from pool.

**hsa_status_t HSA_API**

| hsa_amd_ipc_get_memory_handle(void *ptr, hsa_amd_ipc_memory_handle_t *ipc_handle);
| where:
|     IN:    ptr - Pointer to memory previously allocated via hsa_amd_memory_pool_allocate() call
|     OUT:   ipc_handle - Unique IPC handle to be used in IPC. 
|                         Application must pass this handle to another process.      
| 
| hsa_amd_ipc_close_memory_handle
| Close IPC memory handle previously received via "hsa_amd_ipc_get_memory_handle()" call .

**hsa_status_t HSA_API**

| hsa_amd_ipc_close_memory_handle(hsa_amd_ipc_memory_handle_t ipc_handle);
| where:
|    IN: ipc_handle - IPC Handle to close
|
| 
| hsa_amd_ipc_open_memory_handle
| Open / import an IPC memory handle exported from another process and return address to be used in the current process.

**hsa_status_t HSA_API**

| hsa_amd_ipc_open_memory_handle(hsa_amd_ipc_memory_handle_t ipc_handle, void **ptr);
| where:
|     IN:   ipc_handle - IPC Handle
|     OUT:  ptr- Address which could be used in the given process for access to the memory
|
| Client should call hsa_amd_memory_pool_free() when access to this resource is not needed any more.

**Signal sharing API**

| Allows sharing of HSA signals  between different processes.
|
| hsa_amd_ipc_get_signal_handle
| The purpose of this API is to get / export an IPC handle for an existing signal.

**hsa_status_t HSA_API**

| hsa_amd_ipc_get_signal_handle(hsa_signal_t signal, hsa_amd_ipc_signal_handle_t *ipc_handle);
| where:
|     IN:    signal     - Signal handle created as the result of hsa_signal_create() call.
|     OUT:   ipc_handle - Unique IPC handle to be used in IPC. 
|                         Application must pass this handle to another process.      
| 
| hsa_amd_ipc_close_signal_handle
| Close IPC signal handle previously received via "hsa_amd_ipc_get_signal_handle()" call .

**hsa_status_t HSA_API**

| hsa_amd_ipc_close_signal_handle(hsa_amd_ipc_signal_handle_t ipc_handle);
| where:
|     IN: ipc_handle - IPC Handle to close

| hsa_amd_ipc_open_signal_handle
| Open / import an IPC signal handle exported from another process and return address to be used in the current process.

**hsa_status_t HSA_API**

| hsa_amd_ipc_open_signal_handle(hsa_amd_ipc_signal_handle_t ipc_handle, hsa_signal_t &signal);
| where:
|     IN:   ipc_handle - IPC Handle
|     OUT:  signal     - Signal handle to be used in the current process

Client should call hsa_signal_destroy() when access to this resource is not needed any more.

**Query API**

| Query memory information

Allows query information about memory resource based on address. It is partially overlapped with the following requirement Memory info interface so it may be possible to merge those two interfaces.
::
 typedef enum hsa_amd_address_info_s {
     
     /* Return uint32_t  / boolean if address was allocated via  HSA stack */
     HSA_AMD_ADDRESS_HSA_ALLOCATED = 0x1,
 
     /** Return agent where such memory was allocated */
     HSA_AMD_ADDRESS_AGENT = 0x2,
 
     /** Return pool from which this address was allocated  */
     HSA_AMD_ADDRESS_POOL = 0x3,
 
     /** Return size of allocation   */
     HSA_AMD_ADDRESS_ALLOC_SIZE = 0x4
 
  } hsa_amd_address_info_t;


**hsa_status_t HSA_API**

| hsa_amd_get_address_info(void *ptr,  hsa_amd_address_info_t attribute,   void* value);
| where: 
|      ptr         - Address information about which to query
|      attribute   - Attribute to query


MPICH
=======

MPICH is a high-performance and widely portable implementation of the MPI-3.1 standard.  

For more information about MPICH, refer to https://www.mpich.org/


Building and Installing MPICH
******************************

To build and install MPICH with UCX and ROCm support, see the instructions below.

::
	
	git clone https://github.com/pmodels/mpich.git
	cd mpich
	git checkout v3.4
	git submodule update --init --recursive
	./autogen.sh
	./configure --prefix=</mpich/install/location> --with-device=ch4:ucx --with-ucx=</ucx/install/location>
	make -j && make install
