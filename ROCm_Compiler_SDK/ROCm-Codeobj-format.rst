
.. _ROCm-Codeobj-format:

ROCm Code Object Format
========================


   *  :ref:`Introduction`
   *  :ref:`Finalize`
   *  :ref:`Kernel-dispatch`
   *  :ref:`Hardware-registers-setup`
   *  :ref:`Initia-kernel-register-state`
   *  :ref:`Kernel-prolog-code`
   *  :ref:`Global-Readonly-Kernarg`
   *  :ref:`Scratch-memory-swizzling`
   *  :ref:`Flat-addressing`
   *  :ref:`Flat-scratch`
   *  :ref:`M0-Register`
   *  :ref:`Dynamic-call-stack`
   *  :ref:`Memory-model`
       * :ref:`Memory-model-overview`
       * :ref:`Memory-operation-constraints-global`
       * :ref:`Memory-operation-constraints-group`
       * :ref:`Memory-fence-constraints`
   * :ref:`Instruction-set-architecture`
   * :ref:`AMD-Kernel-Code`
       * :ref:`AMD-Kernel-Code-Object`
       * :ref:`Compute-shader-settings-1`
       * :ref:`Compute-shader-settings-2`
       * :ref:`AMD-Machine-Kind`
       * :ref:`Float-Round-Mode`
       * :ref:`Denorm-Mode`
   * :ref:`PCIe-Gen3-Atomic`
   * :ref:`AMD-Queue`
       * :ref:`HSA-AQL`
       * :ref:`AMD-AQL`
       * :ref:`Queue-operations`
   * :ref:`Signals`
       * :ref:`Signals-overview`
       * :ref:`Signal-kind`
       * :ref:`Signal-object`
       * :ref:`Signal-kernel-machine-code`
   * :ref:`Debugtrap`
   * :ref:`References`

.. _Introduction:

Introduction
#############
This specification defines the application binary interface (ABI) provided by the AMD implementation of the HSA runtime for AMD GPU architecture agents. The AMD GPU architecture is a family of GPU agents which differ in machine code encoding and functionality.


.. _Finalize:

Finalizer, Code Object, Executable and Loader
###############################################
Finalizer, Code Object, Executable and Loader are defined in "HSA Programmer Reference Manual Specification". AMD Code Object uses ELF format. In this document, Finalizer is any compiler producing code object, including kernel machine code.

.. _Kernel-dispatch:

Kernel dispatch
###################
The HSA Architected Queuing Language (AQL) defines a user space memory interface, an AQL Queue, to an agent that can be used to control the dispatch of kernels, using AQL Packets, in an agent independent way. All AQL packets are 64 bytes and are defined in "HSA Platform System Architecture Specification". The packet processor of a kernel agent is responsible for detecting and dispatching kernels from the AQL Queues associated with it. For AMD GPUs the packet processor is implemented by the Command Processor (CP).

The AMD HSA runtime allocates the AQL Queue object. It uses the AMD Kernel Fusion Driver (KFD) to initialize and register the AQL Queue with CP. Refer to "AMD Queue" for more information.

A kernel dispatch is initiated with the following sequence defined in "HSA System Architecture Specification" (it may occur on CPU host agent from a host program, or from an HSA kernel agent such as a GPU from another kernel):

   * A pointer to an AQL Queue for the kernel agent on which the kernel is to be executed is obtained.
   * A pointer to the amd_kernel_code_t object of the kernel to execute is obtained. It must be for a kernel that was loaded on the 	 kernel agent with which the AQL Queue is associated.
   * Space is allocated for the kernel arguments using the HSA runtime allocator for a memory region with the kernarg property for 	the kernel agent that will execute the kernel, and the values of the kernel arguments are assigned. This memory corresponds to 	 the backing memory for the kernarg segment within the kernel being called. Its layout is defined in "HSA Programmer Reference 	Manual Specification". For AMD the kernel execution directly uses the backing memory for the kernarg segment as the kernarg    	 segment.
   * Queue operations is used to reserve space in the AQL queue for the packet.
   * The packet contents are set up, including information about the actual dispatch, such as grid and work-group size, together with 	   information from the code object about the kernel, such as segment sizes.
   * The packet is assigned to packet processor by changing format field from INVALID to KERNEL_DISPATCH. Atomic memory operation  	must be used.
   * A doorbell signal for the queue is signaled to notify packet processor.

At some point, CP performs actual kernel execution:

   * CP detects a packet on AQL queue.
   * CP executes micro-code for setting up the GPU and wavefronts for a kernel dispatch.
   * CP ensures that when a wavefront starts executing the kernel machine code, the scalar general purpose registers (SGPR) and    	vector general purpose registers (VGPR) are set up based on flags in amd_kernel_code_t (see "Initial kernel register state").
   * When a wavefront start executing the kernel machine code, the prolog (see "Kernel prolog code") sets up the machine state as    	  necessary.
   * When the kernel dispatch has completed execution, CP signals the completion signal specified in the kernel dispatch packet if 	not 0.



.. _Hardware-registers-setup:

Hardware registers setup
######################################
SH_MEM_CONFIG register:

  * DEFAULT_MTYPE = 1 (MTYPE_NC)
  * ALIGNMENT_MODE = 3 (SH_MEM_ALIGNMENT_MODE_UNALIGNED)
  * PTR32 = 1 in 32-bit mode and 0 in 64-bit mode


.. _Initia-kernel-register-state:

Initial kernel register state
######################################
Prior to start of every wavefront execution, CP/SPI sets up the register state based on enable_sgpr_* and enable_vgpr_* flags in amd_kernel_code_t object:

   * SGPRs before the Work-Group Ids are set by CP using the 16 User Data registers.
   * Work-group Id registers X, Y, Z are set by SPI which supports any combination including none.
   * Scratch Wave Offset is also set by SPI which is why its value cannot be added into the value Flat Scratch Offset (which would 	avoid the Finalizer generated prolog having to do the add).
   * The VGPRs are set by SPI which only supports specifying either (X), (X, Y) or (X, Y, Z).

SGPR register numbers used for enabled registers are dense starting at SGPR0: the first enabled register is SGPR0, the next enabled register is SGPR1 etc.; disabled registers do not have an SGPR number. Because of hardware constraints, the initial SGPRs comprise up to 16 User SRGPs that are set up by CP and apply to all waves of the grid. It is possible to specify more than 16 User SGPRs using the enable_sgpr_* bit fields, in which case only the first 16 are actually initialized. These are then immediately followed by the System SGPRs that are set up by ADC/SPI and can have different values for each wave of the grid dispatch.

The number of enabled registers must match value in compute_pgm_rsrc2.user_sgpr (the total count of SGPR user data registers enabled). The enableGridWorkGroupCount* is currently not implemented in CP and should always be 0.

The following table defines SGPR registers that can be enabled and their order.

============ ============== ======================================= ==================================================================
SGPR Order    Number 
	      of Registers 		Name 					Description
============ ============== ======================================= ==================================================================
First 		  4 	       Private Segment Buffer 		     V# that can be used, together with Scratch Wave Offset as an
			      (enable_sgpr_private_segment_buffer)   offset, to access the Private/Spill/Arg segments using a segment 							     		     address. CP uses the value from  
								     amd_queue_t.scratch_resource_descriptor.	

then 		  2 		Dispatch Ptr			     64 bit address of AQL dispatch packet for kernel actually
				(enable_sgpr_dispatch_ptr) 	     executing. 									    

then 		  2 		Queue Ptr 			     64 bit address of amd_queue_t object for AQL queue on which the
				(enable_sgpr_queue_ptr) 	     dispatch packet was queued.
 									    
then 		  2 		Kernarg Segment Ptr 		     64 bit address of Kernarg segment. This is directly copied 				(enable_sgpr_kernarg_segment_ptr)    from the kernarg_address in the kernel dispatch packet. Having 									     CP load it once avoids loading it at the beginning of 									     every  wavefront.

then 		  2 		Dispatch Id 			     64 bit Dispatch ID of the dispatch packet being executed.
				(enable_sgpr_dispatch_id) 	     

then 		  2 		Flat Scratch Init		     Value used for FLAT_SCRATCH register initialization. Refer to
				(enable_sgpr_flat_scratch_init)       Flat scratch for more information.
 									     
then 		 1 		Private Segment Size 		     The 32 bit byte size of a single work-items scratch memory
				(enable_sgpr_private_segment_size)   allocation. This is the value from the kernel dispatch packet 									     Private Segment Byte Size rounded up by CP to a multiple of 									     WORD. Having CP load it once avoids loading it at the beginning 									     of every wavefront. Not used for GFX7/GFX8 since it is the same 									     value as the second SGPR of Flat Scratch Init.

then 		 1 		Grid Work-Group Count X 		32 bit count of the number of work-groups in the X dimension 					(enable_sgpr_grid_workgroup_count_X)    for the grid being executed. Computed from the fields in the 										kernel dispatch packet as ((grid_size.x + workgroup_size.x - 										1) /workgroup_size.x).

then 		 1 		Grid Work-Group Count Y 		32 bit count of the number of work-groups in the Y dimension 					(enable_sgpr_grid_workgroup_count_Y 	for the grid being executed. Computed from the fields in the 					&& less than 16 previous SGPRs) 	kernel dispatch packet as ((grid_size.y + workgroup_size.1) / 										workgroupSize.y). Only initialized if <16 previous SGPRs 										initialized.

then 		 1 		Grid Work-Group Count Z 		32 bit count of the number of work-groups in the Z dimension
				(enable_sgpr_grid_workgroup_count_Z 	for the grid being executed. Computed from the fields in the
				&& less than 16 previous SGPRs) 	kernel dispatch packet as ((grid_size.z + workgroup_size.z - 										1) / workgroupSize.z). Only initialized if <16 previous SGPRs 										initialized.
 										 										
then 		 1 		Work-Group Id X 			32 bit work group id in X dimension of grid for wavefront.
				(enable_sgpr_workgroup_id_X) 		Always present.

		
then 		 1 		Work-Group Id Y 			32 bit work group id in Y dimension of grid for wavefront.
				(enable_sgpr_workgroup_id_Y) 	

then 		 1 		Work-Group Id Z 
				(enable_sgpr_workgroup_id_Z) 		32 bit work group id in Z dimension of grid for wavefront. If 										present then Work-group Id Y will also be present.

then 		 1 		Work-Group Info 			{first_wave, 14b0000, ordered_append_term[10:0],
				(enable_sgpr_workgroup_info) 	 	threadgroup_size_in_waves[5:0]}

then 		 1 	     |  Private Segment Wave Byte Offset 	32 bit byte offset from base of scratch base of queue the  			     	     |  (enable_sgpr_private_segment_wave	executing  kernel dispatch. Must be used as an offset with 				     |  _byte_offset)			      Private/Spill/Arg  segment address when using Scratch Segment
				    				      Buffer. It must be added to Flat Scratch Offset if setting up 									      FLAT SCRATCH for flat addressing.
		            		 
============ ============== ======================================= ==================================================================

VGPR register numbers used for enabled registers are dense starting at VGPR0: the first enabled register is VGPR0, the next enabled register is VGPR1 etc.; disabled registers do not have a VGPR number.

The following table defines VGPR registers that can be enabled and their order.

========== ================ ============================================= ============================================================
VGPR Order  No.Registers 	           Name 					     Description
========== ================ ============================================= ============================================================
First 		1 	     Work-Item Id X (Always initialized)  	   32 bit work item id in X dimension of work-group for 									   wavefront lane.

then 		1 	     Work-Item Id Y (enable_vgpr_workitem_id > 0)  32 bit work item id in Y dimension of work-group for 									   wavefront lane.

then 		1 	     Work-Item Id Z (enable_vgpr_workitem_id > 1)  32 bit work item id in Z dimension of work-group for 									   wavefront lane.
========== ================ ============================================= ============================================================

.. _Kernel-prolog-code:

Kernel prolog code
######################################
For certain features, kernel is expected to perform initialization actions, normally done in kernel prologue. This is only needed if kernel uses those features.

.. _Global-Readonly-Kernarg:

Global/Readonly/Kernarg segments
######################################
Global segment can be accessed either using flat or buffer operations. Buffer operations cannot be used for large machine model for GFX7 and later as V# support for 64 bit addressing is not available.

If buffer operations are used then the Global Buffer used to access Global/Readonly/Kernarg (combined) segments using a segment address is not passed into the kernel code by CP since its base address is always 0. The prolog code initializes 4 SGPRs with a V# that has the following properties, and then uses that in the buffer instructions:

  * base address of 0
  * no swizzle
  * ATC: 1 if IOMMU present (such as APU)
  * MTYPE set to support memory coherence specified in amd_kernel_code_t.global_memory_coherence

If buffer operations are used to access Kernarg segment, Kernarg address must be added. It is available in dispatch packet (kernarg_address field) or as Kernarg Segment Ptr SGPR. Alternatively, scalar loads can be used if the kernarg offset is uniform, as the kernarg segment is constant for the duration of the kernel dispatch execution.

For GFX9, global segment can be accessed with new GLOBAL_* instructions.


.. _Scratch-memory-swizzling:

Scratch memory swizzling
######################################
Scratch memory may be used for private/spill/stack segment. Hardware will interleave (swizzle) scratch accesses of each lane of a wavefront by interleave (swizzle) element size to ensure each work-item gets a distinct memory location. Interleave size must be 2, 4, 8 or 16. The value used must match the value that the runtime configures the GPU flat scratch (SH_STATIC_MEM_CONFIG.ELEMENT_SIZE).

For GFX8 and earlier, all load and store operations done to scratch buffer must not exceed this size. For example, if the element size is 4 (32-bits or dword) and a 64-bit value must be loaded, it must be split into two 32-bit loads. This ensures that the interleaving will get the work-item specific dword for both halves of the 64-bit value. If it just did a 64-bit load then it would get one dword which belonged to its own work-item, but the second dword would belong to the adjacent lane work-item since the interleaving is in dwords.

AMD HSA Runtime Finalizer uses value 4.

.. _Flat-addressing:

Flat addressing
###################
Flat address can be used in FLAT instructions and can access global, private (scratch) and group (lds) memory.

Flat access to scratch requires hardware aperture setup and setup in kernel prologue (see Flat scratch).

For GFX7/GFX8, flat access to lds requires hardware aperture setup and M0 register setup (see M0 register).

Address operations for group/private segment may use fields from amd_queue_t, the address of which can be obtained with Queue Ptr SGPR.

To obtain null address value for a segment (nullptr HSAIL instruction),
   * For global, readonly and flat segment use value 0.
   * For group, private and kernarg segments, use value -1 (32-bit).

To convert segment address to flat address (stof HSAIL instruction),
   * For global segment, use the same value.
   * For kernarg segment, add Kernarg Segment Ptr. For small model, this is a 32-bit add. For large model, this is 32-bit add to    	 64-bit base address.
   * For group segment,
       * for large model, combine group_segment_aperture_base_hi (upper half) and segment address (lower half),
       * for small model, add group_segment_aperture_base_hi and segment address.
   * For private/spill/arg segment,
       * for large model, combine private_segment_aperture_base_hi (upper half) and segment address (lower half),
       * for small model, add private_segment_aperture_base_hi and segment address.
   * If flat address may be null, kernarg, group and private/spill arg segment machine code must have additional sequence (use V_CMP 	  and V_CNDMASK).

To convert flat address to segment address (ftos HSAIL instruction),

  * For global segment, use the same value.
  * For kernarg segment, subtract Kernarg Segment Ptr. For small model, this is a 32-bit subtract. For large model, this is 32-bit  	subtract from lower half of the 64-bit flat address.
  * For group segment,
      * for large model, use low half of the flat address,
      * for small model, subtract group_segment_aperture_base_hi.
  * For private/spill/arg segment,
      * for large model, use low half of the flat address,
      * for small model, subtract private_segment_aperture_base_hi.
  * If segment address may be null, kernarg, group and private/spill arg segment machine code must have additional sequence (use    	V_CMP and V_CNDMASK).

To determine if given flat address lies within a segment (segmentp HSAIL instruction),

  * For global segment, check that address does not lie in group/private segments
  * For group segment, check if address lies in group segment aperture
      *  for large model, check that upper half of 64-bit address == group_segment_aperture_base_hi,
      *  for small model, check that most significant 16 bits of 32-bit address (address & ~0xFFFF) == group_segment_aperture_base_hi.
  * For private segment, check if address lies in private segment aperture
      * for large model, check that upper half of 64-bit address == private_segment_aperture_base_hi,
      * for small model, check that most significant 16 bits of 32-bit address (address & ~0xFFFF) == group_segment_aperture_base_hi.
  * If flat address may be null, machine code must have additional sequence (use V_CMP).



.. _Flat-scratch:

Flat scratch
###################
If kernel may use flat operations to access scratch memory, the prolog code must set up FLAT_SCRATCH register pair (FLAT_SCRATCH_LO/FLAT_SCRATCH_HI or SGPRn-4/SGPRn-3).

For GFX7/GFX8, initialization uses Flat Scratch Init and Scratch Wave Offset sgpr registers (see Initial kernel register state):

 * The low word of Flat Scratch Init is 32 bit byte offset from SH_HIDDEN_PRIVATE_BASE_VIMID to base of memory for scratch for the   	queue executing the kernel dispatch. This is the lower 32 bits of amd_queue_t.scratch_backing_memory_location and is the same     	offset used in computing the Scratch Segment Buffer base address. The prolog must add the value of Scratch Wave Offset to it,     	shift right by 8 (offset is in 256-byte units) and move to FLAT_SCRATCH_LO for use as the FLAT SCRATCH BASE in flat memory 	     	instructions.
 * The second word of Flat Scratch Init is 32 bit byte size of a single work-items scratch memory usage. This is directly loaded from 	 the kernel dispatch packet Private Segment Byte Size and rounded up to a multiple of DWORD. Having CP load it once avoids loading 	it at the beginning of every wavefront. The prolog must move it to FLAT_SCRATCH_LO for use as FLAT SCRATCH SIZE.

For GFX9, Flat Scrath Init contains 64-bit address of scratch backing memory. The initialization sequence for FLAT_SCRATCH does 64-bit add of Flat Scratch Init and Scratch Wave Offset.

.. _M0-Register:

M0 Register
###################
For GF7/GFX8, M0 register must be initialized with total LDS size if kernel may access LDS via DS or flat operations. Total LDS size is available in dispatch packet. For M0, it is also possible to use maximum possible value of LDS for given target.

.. _Dynamic-call-stack:

Dynamic call stack
###################
In certain cases, Finalizer cannot compute the total private segment size at compile time. This can happen if calls are implemented using a call stack and recursion, alloca or calls to indirect functions are present. In this case, workitem_private_segment_byte_size field in code object only specifies the statically known private segment size. When performing actual kernel dispatch, private_segment_size_bytes field in dispatch packet will contain static private segment size plus additional space for the call stack.


.. _Memory-model:

Memory model
###################

.. _Memory-model-overview:

Memory model Overview
#######################
A memory model describes the interactions of threads through memory and their shared use of the data. Many modern programming languages implement a memory model. This section describes the mapping of common memory model constructs onto AMD GPU architecture.

Through this section, definitions and constraints from "HSA Platform System Architecture Specification 1.0" are used as reference, although similar notions exist elsewhere (for example, in C99 or C++ 11).

The following memory scopes are defined:

  * Work-item (wi)
  * Wavefront (wave)
  * Work-group (wg)
  * Agent (agent)
  * System (system)

The following memory orders are defined:

  * scacq: sequentially consistent acquire
  * screl: sequentially consistent release
  * scar: sequentially consistent acquire and release
  * rlx: relaxed

The following memory operations are defined:

  * Ordinary Load/Store (non-synchronizing operations)
  * Atomic Load/Atomic Store (synchronizing operations)
  * Atomic RMW (Read-Modify-Write: add, sub, max, min, and, or, xor, wrapinc, wrapdec, exch, cas (synchronizing operations)
  * Memory Fence (synchronizing operation)

Sometimes derived notation is used. For example, agent+ means agent and system scopes, wg- means work-group, wavefront and work-item scopes.

In the following sections, a combination of memory segment, operation, order and scope is assigned a machine code sequence. Note that if s_waitcnt vmcnt(0) is used to enforce a completion of earlier memory operations in same workitem, it can be omitted if it is also enforced using some other mechanism or proven by compiler (for example, if there are no preceding synchronizing memory operations). Similiarily, if s_waitcnt vmcnt(0) is used to enforce completion of this memory operation before the following memory operations, sometimes it can be omitted (for example, if there are no following synchronizing memory operations).

For a flat memory operation, if it may affect either global or group segment, group constraints must be applied to flat operations as well.


.. _Memory-operation-constraints-global:

Memory operation constraints for global segment
#########################################################
For global segment, the following machine code instructions may be used (see Global/Readonly/Kernarg segments):

  * Ordinary Load/Store: BUFFER_LOAD/BUFFER_STORE or FLAT_LOAD/FLAT_STORE
  * Atomic Load/Store: BUFFER_LOAD/BUFFER_STORE or FLAT_LOAD/FLAT_STORE
  * Atomic RMW: BUFFER_ATOMIC or FLAT_ATOMIC

+----------------+------------------------+--------------+------------------------------------------------------------------+
| Operation      | Memory order           | Memory scope | Machine code sequence                                            |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Ordinary Load  | -                      | -            | load with glc=0                                                  |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic Load    | rlx,scacq              | wg-          | load with glc=0                                                  |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic Load    | rlx                    | agent+       | load with glc=1                                                  |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic Load    | scacq                  | agent+       | load with glc=1; s_waitcnt vmcnt(0); buffer_wbinv_vol            |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Ordinary Store | -                      | -            | store with glc=0                                                 |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic Store   | rlx,screl              | wg-          | store with glc=0                                                 |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic Store   | rlx                    | agent+       | store with glc=0                                                 |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic Store   | screl                  | agent+       | s_waitcnt vmcnt(0); store with glc=0; s_waitcnt vmcnt(0)         |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic RMW     | rlx,scacq, screl, scar | wg-          | atomic                                                           |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic RMW     | rlx                    | agent+       | atomic                                                           |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic RMW     | scacq                  | agent+       | atomic; s_waitcnt vmcnt(0); buffer_wbinv_vol                     |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic RMW     | screl                  | agent+       | s_waitcnt vmcnt(0); atomic                                       |
+----------------+------------------------+--------------+------------------------------------------------------------------+
| Atomic RMW     | scar                   | agent+       | s_waitcnt vmcnt(0); atomic; s_waitcnt vmcnt(0); buffer_wbinv_vol |
+----------------+------------------------+--------------+------------------------------------------------------------------+

.. _Memory-operation-constraints-group:

Memory operation constraints for group segment
#########################################################
For group segment, the following machine code instructions are used:

  * Ordinary Load/Store: DS_READ/DS_WRITE
  * Atomic Load/Store: DS_READ/DS_WRITE
  * Atomic RMW: DS_ADD, DS_SUB, DS_MAX, DS_MIN, DS_AND, DS_OR, DS_XOR, DS_INC, DS_DEC, DS_WRXCHG, DS_CMPST (and corresponding RTN   	variants)

AMD LDS hardware is sequentially consistent. This means that it is not necessary to use lgkmcnt to enforce ordering in single work-item for group segment synchronization. s_waitcnt lgkmcnt(0) should still be used to enforce data dependencies, for example, after a load into a register and before use of that register (same applies to non-synchronizing operations).

The current model (and HSA) requires that global and group segments are coherent. This is why synchronizing group segment operations and memfence also use s_waitcnt vmcnt(0).

+----------------+--------------+--------------+------------------------------------------------+
| Operation      | Memory order | Memory scope | Machine code sequence                          |
+----------------+--------------+--------------+------------------------------------------------+
| Ordinary Load  | -            | -            | load                                           |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic Load    | rlx          | wg-          | load                                           |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic Load    | scacq        | wg-          | s_waitcnt vmcnt(0); load; buffer_wbinvl1_vol   |
+----------------+--------------+--------------+------------------------------------------------+
| Ordinary Store | -            | -            | store                                          |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic Store   | rlx          | wg-          | store                                          |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic Store   | screl        | wg-          | s_waitcnt vmcnt(0); store                      |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic RMW     | scacq        | wg-          | s_waitcnt vmcnt(0); atomic; buffer_wbinvl1_vol |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic RMW     | screl        | wg-          | s_waitcnt vmcnt(0); atomic                     |
+----------------+--------------+--------------+------------------------------------------------+
| Atomic RMW     | scacq        | wg-          | s_waitcnt vmcnt(0); atomic; buffer_wbinvl1_vol |
+----------------+--------------+--------------+------------------------------------------------+



.. _Memory-fence-constraints:

Memory fence constraints
######################################
Memory fence is currently applied to all segments (cross-segment synchronization). In machine code, memory fence does not have separate instruction and maps to s_waitcnt and buffer_wbinvl1_vol instructions. In addition, memory fence must not be moved in machine code with respect to other synchronizing operations. In the following table, 'memfence' refers to conceptual memory fence location.

============== ==================== ================= ========================================================
Operation 	Memory order 	     Memory scope 	Machine code sequence
============== ==================== ================= ========================================================
Memory Fence 	scacq,screl,scar 	wg- 		memfence (no additional constraints)
Memory Fence 	scacq 			agent+ 		memfence; s_waitcnt 0; buffer_wbinvl1_vol
Memory Fence 	screl 			agent+ 		s_waitcnt 0; memfence
Memory Fence 	scar 			agent + 	memfence; s_waitcnt 0; buffer_wbinvl1_vol
============== ==================== ================= ========================================================
	
.. _Instruction-set-architecture:

Instruction set architecture
######################################
AMDGPU ISA specifies instruction set architecture and capabilities used by machine code. It consists of several fields:

   * Vendor ("AMD")
   * Architecture ("AMDGPU")
   * Major (GFXIP), minor and stepping versions

These fields may be combined to form one defining string, for example, "AMD:AMDGPU:8:0:1".

======= ============== ======= ======= ========== ============================== =====================================================
Vendor 	Architecture 	Major 	Minor 	Stepping 	Comments 				Products
======= ============== ======= ======= ========== ============================== =====================================================
AMD 	  AMDGPU 	  7 	  0 	   0 	   Legacy, GFX7, 1/16 double FP 	A10-7400 series APU

AMD 	  AMDGPU 	  7 	  0 	   1 	   GFX7, 1/2 double FP 	 	  FirePro W8100, W9100, S9150, S9170; Radeon R9 290, 											  R9 290x, R390,R390x

AMD 	  AMDGPU 	  8 	  0 	   0 	   Legacy, GFX8, SPI register     XNACK FirePro S7150, S7100, W7100; Radeon R285,
						   limitation,  		  R9 380, R9 385; Mobile FirePro M7170

AMD 	  AMDGPU 	  8 	  0 	   1 	   GFX8, XNACK enabled 	           A10-8700 series APU

AMD 	  AMDGPU 	  8 	  0 	   2       GFX8, SPI register limitation  FirePro S7150, S7100, W7100; Radeon R285, R9 380,
						   XNACK disabled,		    R9 385; Mobile FirePro M7170
						   PCIe Gen3 atomics 	

AMD 	  AMDGPU 	  8 	 0 	   3 	   GFX8, XNACK disabled,	  Radeon R9 Nano, R9 Fury, R9 FuryX, Pro Duo, RX 460,
						    PCIe Gen3 atomics 		  RX 470, RX 480; FirePro S9300x2

AMD 	  AMDGPU 	  8 	 0 	   4 	   GFX8, -XNACK Legacy, 	   Radeon R9 Nano, R9 Fury, R9 FuryX, Pro Duo,
										    RX 460, RX 470, RX 480; FirePro S9300x2

AMD 	  AMDGPU 	  9 	 0 	   0 	   GFX9, -XNACK 	

AMD 	  AMDGPU 	  9 	 0 	   1 	   GFX9, +XNACK
======= ============== ======= ======= ========== ============================== =====================================================

.. _AMD-Kernel-Code:

AMD Kernel Code
###################
AMD Kernel Code object is used by AMD GPU CP to set up the hardware to execute a kernel dispatch and consists of the meta data needed to initiate the execution of a kernel, including the entry point address of the machine code that implements 



.. _AMD-Kernel-Code-Object:

AMD Kernel Code Object amd_kernel_code_t
#########################################################

+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Bits      | Size      | Field Name                                  | Description                                                                                                                                                                                                                                                      |
+===========+===========+=============================================+==================================================================================================================================================================================================================================================================+
| 31:0      | 4 bytes   | amd_code_version_major                      | The AMD major version. Must be the value AMD_KERNEL_CODE_VERSION_MAJOR. Major versions are not backwards compatible.                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 63:32     | 4 bytes   | amd_code_version_minor                      | The AMD minor version. Must be the value AMD_CODE_VERSION_MINOR. Minor versions with the same major version must be backward compatible.                                                                                                                         |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 79:64     | 2 bytes   | amd_machine_kind                            | Machine kind.                                                                                                                                                                                                                                                    |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 95:80     | 2 bytes   | amd_machine_version_major                   | Instruction set architecture: major                                                                                                                                                                                                                              |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 111:96    | 2 bytes   | amd_machine_version_minor                   | Instruction set architecture: minor                                                                                                                                                                                                                              |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 127:112   | 2 bytes   | amd_machine_version_stepping                | Instruction set architecture: stepping                                                                                                                                                                                                                           |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 191:128   | 8 bytes   | kernel_code_entry_byte_offset               | Byte offset (possibly negative) from start of amd_kernel_code_t object to kernel's entry point instruction. The actual code for the kernel is required to be 256 byte aligned to match hardware requirements (SQ cache line is 16;                               |
|           |           |                                             | entry point config register only holds bits 47:8 of the address). The Finalizer should endeavor to allocate all kernel machine code in contiguous memory pages so that a device pre-fetcher will tend to only pre-fetch Kernel Code objects,                     |
|           |           |                                             | improving cache performance. The AMD HA Runtime Finalizer generates position independent code (PIC) to avoid using relocation records and give runtime more flexibility in copying code to discrete GPU device memory.                                           |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 255:192   | 8 bytes   | kernel_code_prefetch_byte_offset            | Range of bytes to consider prefetching expressed as a signed offset and unsigned size. The (possibly negative) offset is from the start of amd_kernel_code_t object.                                                                                             |
|           |           |                                             | Set both to 0 if no prefetch information is available.                                                                                                                                                                                                           |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 319:256   | 8 bytes   | kernel_code_prefetch_byte_size              |                                                                                                                                                                                                                                                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 383:320   | 8 bytes   | max_scratch_backing_memory_byte_size        | Number of bytes of scratch backing memory required for full occupancy of target chip. This takes into account the number of bytes of scratch per work-item, the wavefront size, the maximum number of wavefronts per CU, and the number of CUs.                  |
|           |           |                                             | This is an upper limit on scratch. If the grid being dispatched is small it may only need less than this. If the kernel uses no scratch, or the Finalizer has not computed this value, it must be 0.                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 415:384   | 4 bytes   | compute_pgm_rsrc1                           | Compute Shader (CS) program settings 1 amd_compute_pgm_rsrc1                                                                                                                                                                                                     |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 447:416   | 4 bytes   | compute_pgm_rsrc2                           | Compute Shader (CS) program settings 2 amd_compute_pgm_rsrc2                                                                                                                                                                                                     |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 448       | 1 bit     | enable_sgpr_private_segment_buffer          | Enable the setup of Private Segment Buffer                                                                                                                                                                                                                       |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 449       | 1 bit     | enable_sgpr_dispatch_ptr                    | Enable the setup of Dispatch Ptr                                                                                                                                                                                                                                 |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 450       | 1 bit     | enable_sgpr_queue_ptr                       | Enable the setup of Queue Ptr                                                                                                                                                                                                                                    |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 451       | 1 bit     | enable_sgpr_kernarg_segment_ptr             | Enable the setup of Kernarg Segment Ptr                                                                                                                                                                                                                          |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 452       | 1 bit     | enable_sgpr_dispatch_id                     | Enable the setup of Dispatch Id                                                                                                                                                                                                                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 453       | 1 bit     | enable_sgpr_flat_scratch_init               | Enable the setup of Flat Scratch Init                                                                                                                                                                                                                            |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 454       | 1 bit     | enable_sgpr_private_segment_size            | Enable the setup of Private Segment Size                                                                                                                                                                                                                         |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 455       | 1 bit     | enable_sgpr_grid_workgroup_count_X          | Enable the setup of Grid Work-Group Count X                                                                                                                                                                                                                      |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 456       | 1 bit     | enable_sgpr_grid_workgroup_count_Y          | Enable the setup of Grid Work-Group Count Y                                                                                                                                                                                                                      |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 457       | 1 bit     | enable_sgpr_grid_workgroup_count_Z          | Enable the setup of Grid Work-Group Count Z                                                                                                                                                                                                                      |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 463:458   | 6 bits    |                                             | Reserved. Must be 0.                                                                                                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 464       | 1 bit     | enable_ordered_append_gds                   | Control wave ID base counter for GDS ordered-append. Used to set COMPUTE_DISPATCH_INITIATOR.ORDERED_APPEND_ENBL.                                                                                                                                                 |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 466:465   | 2 bits    | private_element_size                        | Interleave (swizzle) element size in bytes.                                                                                                                                                                                                                      |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 467       | 1 bit     | is_ptr64                                    | 1 if global memory addresses are 64 bits, otherwise 0. Must match SH_MEM_CONFIG.PTR32 (GFX7), SH_MEM_CONFIG.ADDRESS_MODE (GFX8+).                                                                                                                                |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 468       | 1 bit     | is_dynamic_call_stack                       | Indicates if the generated machine code is using dynamic call stack.                                                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 469       | 1 bit     | is_debug_enabled                            | Indicates if the generated machine code includes code required by the debugger.                                                                                                                                                                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 470       | 1 bit     | is_xnack_enabled                            | Indicates if the generated machine code uses conservative XNACK register allocation.                                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 479:471   | 9 bits    | reserved                                    | Reserved. Must be 0.                                                                                                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 511:480   | 4 bytes   | workitem_private_segment_byte_size          | The amount of memory required for the static combined private, spill and arg segments for a work-item in bytes.                                                                                                                                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 543:512   | 4 bytes   | workgroup_group_segment_byte_size           | The amount of group segment memory required by a work-group in bytes. This does not include any dynamically allocated group segment memory that may be added when the kernel is dispatched.                                                                      |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 575:544   | 4 bytes   | gds_segment_byte_size                       | Number of byte of GDS required by kernel dispatch. Must be 0 if not using GDS.                                                                                                                                                                                   |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 639:576   | 8 bytes   | kernarg_segment_byte_size                   | The size in bytes of the kernarg segment that holds the values of the arguments to the kernel. This could be used by CP to prefetch the kernarg segment pointed to by the kernel dispatch packet.                                                                |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 671:640   | 4 bytes   | workgroup_fbarrier_count                    | Number of fbarrier's used in the kernel and all functions it calls. If the implementation uses group memory to allocate the fbarriers then that amount must already be included in the workgroup_group_segment_byte_size total.                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 687:672   | 2 bytes   | wavefront_sgpr_count                        | Number of scalar registers used by a wavefront. This includes the special SGPRs for VCC, Flat Scratch (Base, Size) and XNACK (for GFX8 (VI)+).                                                                                                                   |
|           |           |                                             |  It does not include the 16 SGPR added if a trap handler is enabled. Must match compute_pgm_rsrc1.sgprs used to set COMPUTE_PGM_RSRC1.SGPRS.                                                                                                                     |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 703:688   | 2 bytes   | workitem_vgpr_count                         | Number of vector registers used by each work-item. Must match compute_pgm_rsrc1.vgprs used to set COMPUTE_PGM_RSRC1.VGPRS.                                                                                                                                       |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 719:704   | 2 bytes   | reserved_vgpr_first                         | If reserved_vgpr_count is 0 then must be 0. Otherwise, this is the first fixed VGPR number reserved.                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 735:720   | 2 bytes   | reserved_vgpr_count                         | The number of consecutive VGPRs reserved by the client. If is_debug_supported then this count includes VGPRs reserved for debugger use.                                                                                                                          |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 751:736   | 2 bytes   | reserved_sgpr_first                         | If reserved_sgpr_count is 0 then must be 0. Otherwise, this is the first fixed SGPR number reserved.                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 767:752   | 2 bytes   | reserved_sgpr_count                         | The number of consecutive SGPRs reserved by the client.                                                                                                                                                                                                          |
|           |           |                                             |  If is_debug_supported then this count includes SGPRs reserved for debugger use.                                                                                                                                                                                 |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 783:768   | 2 bytes   | debug_wavefront_private_segment_offset_sgpr | If is_debug_supported is 0 then must be 0. Otherwise, this is the fixed SGPR number used to hold the wave scratch offset for the entire kernel execution, or uint16_t(-1) if the register is not used or not known.                                              |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 799:784   | 2 bytes   | debug_private_segment_buffer_sgpr           | If is_debug_supported is 0 then must be 0. Otherwise, this is the fixed SGPR number of the first of 4 SGPRs used to hold the scratch V# used for the entire kernel execution, or uint16_t(-1) if the registers are not used or not known.                        |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 807:800   | 1 byte    | kernarg_segment_alignment                   | The maximum byte alignment of variables used by the kernel in the specified memory segment. Expressed as a power of two as defined in Table 37. Must be at least HSA_POWERTWO_16.                                                                                |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 815:808   | 1 byte    | group_segment_alignment                     |                                                                                                                                                                                                                                                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 823:816   | 1 byte    | private_segment_alignment                   |                                                                                                                                                                                                                                                                  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 831:824   | 1 byte    | wavefront_size                              | Wavefront size expressed as a power of two. Must be a power of 2 in range 1..256 inclusive. Used to support runtime query that obtains wavefront size, which may be used by application to allocated dynamic group memory and set the dispatch work-group size.  |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 863:832   | 4 bytes   | call_convention                             | Call convention used to produce the machine code for the kernel. This specifies the function call convention ABI used for indirect functions.                                                                                                                    |
|           |           |                                             | If the application specified that the Finalizer should select the call convention, then this value must be the value selected, not the -1 specified to the Finalizer. If the code object does not support indirect functions, then the value must be 0xffffffff. |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 960:864   | 12 bytes  |                                             | Reserved. Must be 0.                                                                                                                                                                                                                                             |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 1023:960  | 8 bytes   | runtime_loader_kernel_symbol                | A pointer to the loaded kernel symbol. This field must be 0 when amd_kernel_code_t is created. The HSA Runtime loader initializes this field once the code object is loaded to reference the loader symbol for the kernel.                                       |
|           |           |                                             | This field is used to allow the debugger to locate the debug information for the kernel. The definition of the loaded kernel symbol is located in hsa/runtime/executable.hpp.                                                                                    |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 2047:1024 | 128 bytes | control_directive                           | Control directives for this kernel used in generating the machine code. The values are intended to reflect the constraints that the code actually requires to correctly execute, not the values that were actually specified at finalize time.                   |
|           |           |                                             | If the finalizer chooses to ignore a control directive, and not generate constrained code, then the control directive should not be marked as enabled.                                                                                                           |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 2048      |           |                                             | Total size 256 bytes.                                                                                                                                                                                                                                            |
+-----------+-----------+---------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


.. _Compute-shader-settings-1:

Compute shader program settings 1 amd_compute_pgm_rsrc1_t
############################################################################
======== ======= ============================================= =======================================================================
Bits 	 Size 			Field Name 						Description
======== ======= ============================================= =======================================================================
5:0 	 6 bits  granulated_workitem_vgpr_count 	        | Granulated number of vector registers used by each work-item minus 									| 1(i.e. if granulated number of vector registers is 2, then 1 is 									| stored in this field). Granularity is device specific.

9:6 	 4 bits  granulated_wavefront_sgpr_count 		| Granulated number of scalar registers used by a wavefront minus 1 									| (i.e. if granulated number of scalar registers is 4, then 3 is 									| stored in this field). Granularity is device specific. This 									| includes the special SGPRs for VCC, Flat Scratch (Base, and Size) 									| and XNACK (for GFX8 (VI)+). It does not include the 16 SGPR added 									| if a trap handler is enabled.

11:10 	 2 bits  priority 					 Drives spi_priority in spi_sq newWave cmd.

13:12 	 2 bits  float_mode_round_32 				 Wavefront initial float round mode for single precision floats (32 								          bit).

15:14 	 2 bits  float_mode_round_16_64 			| Wavefront initial float round mode for double/half precision floats 									| (64/16 bit).

17:16 	 2 bits  float_mode_denorm_32 				| Wavefront initial denorm mode for single precision floats (32 bit).

19:18 	 2 bits  float_mode_denorm_16_64 			| Wavefront initial denorm mode for double/half precision floats 									| (64/16 bit).

20 	 1 bit   priv 						| Drives priv in spi_sq newWave cmd. This field is set to 0 by the 			 						| Finalizer and must be filled in by CP.

21 	 1 bit 	 enable_dx10_clamp 				| Wavefront starts execution with DX10 clamp mode enabled. Used by 									| the vector ALU to force DX-10 style treatment of NaN's (when set, 									| clamp NaN to zero, otherwise pass NaN through). Used by CP to set 									| up COMPUTE_PGM_RSRC1.DX10_CLAMP.

22 	 1 bit   debug_mode 				        | Drives debug in spi_sq newWave cmd. This field is set to 0 by the 			 						| Finalizer and must be filled in by CP.

23 	 1 bit 	 enable_ieee_mode 				| Wavefront starts execution with IEEE mode enabled. Floating point 									| opcodes that support exception flag gathering will quiet and 									| propagate signaling-NaN inputs per IEEE 754-2008. Min_dx10 and 									| max_dx10 become IEEE 754-2008 compliant due to signaling-NaN 									| propagation and quieting. Used by CP to set up 									| COMPUTE_PGM_RSRC1.IEEE_MODE.

24 	 1 bit 	  bulky 					| Only one such work-group is allowed to be active on any given 								| Compute Unit. Only one such work-group is allowed to be active on 									| any given CU. This field is set to 0 by the Finalizer and must be 									| filled in by CP.

25 	 1 bit 	  cdbg_user 					| This field is set to 0 by the Finalizer and must be filled in by CP.

31:26 	 6 bits   reserved 					 Reserved. Must be 0.

32 								Total size 4 bytes.
======== ======= ============================================= =======================================================================

.. _Compute-shader-settings-2:

Compute shader program settings 2 amd_compute_pgm_rsrc2_t
############################################################################


======== ====== ============================================= ========================================================================
Bits 	  Size 	           Field Name 	                                           Description
======== ====== ============================================= ========================================================================
0 	 1 bit 	enable_sgpr_private_segment_wave_byte_offset 	Enable the setup of the SGPR wave scratch offset system register (see 									2.9.8). Used by CP to set up COMPUTE_PGM_RSRC2.SCRATCH_EN.
5:1 	 5 bit  user_sgpr_count 				The total number of SGPR user data registers requested. This number 									must match the number of user data registers enabled.
6 	 1 bit  enable_trap_handler 				Code contains a TRAP instruction which requires a trap hander to be 									enabled. Used by CP to set up COMPUTE_PGM_RSRC2.TRAP_PRESENT. Note 									that CP shuld set COMPUTE_PGM_RSRC2.TRAP_PRESENT if either this field 									is 1 or if amd_queue.enable_trap_handler is 1 for the queue executing 									the kernel dispatch.
7 	 1 bit 	enable_sgpr_workgroup_id_x 			Enable the setup of Work-Group Id X. Also used by CP to set up 									COMPUTE_PGM_RSRC2.TGID_X_EN.
8 	 1 bit 	enable_sgpr_workgroup_id_y 			Enable the setup of Work-Group Id Y. Also used by CP to set up 									COMPUTE_PGM_RSRC2.TGID_Y_EN, TGID_Z_EN.
9 	 1 bit 	enable_sgpr_workgroup_id_z 			Enable the setup of Work-Group Id Z. Also used by CP to set up 									COMPUTE_PGM_RSRC2. TGID_Z_EN.
10 	 1 bit 	enable_sgpr_workgroup_info 			Enable the setup of Work-Group Info.
12:11 	 2 bit  enable_vgpr_workitem_id 			Enable the setup of Work-Item Id X, Y, Z. Also used by CP to 									set up COMPUTE_PGM_RSRC2.TIDIG_CMP_CNT.
13 	 1 bit 	enable_exception_address_watch 			Wavefront starts execution with specified exceptions enabled. Used by 									CP to set up COMPUTE_PGM_RSRC2.EXCP_EN_MSB (composed from following 									bits). Address Watch - TC (L1) has witnessed a thread access an 								"address of interest".
14 	 1 bit 	enable_exception_memory_violation 		Memory Violation - a memory violation has occurred for this wave from 									L1 or LDS (write-to-read-only-memory, mis-aligned atomic, LDS address 									out of range, illegal address, etc.).
23:15 	 9bits 	granulated_lds_size 				Amount of group segment (LDS) to allocate for each work-group. 									Granularity is device specific. CP should use the rounded value from 									the dispatch packet, not this value, as the dispatch may contain 									dynamically allocated group segment memory. This field is set to 0 by 									the Finalizer and CP will write directly to 									COMPUTE_PGM_RSRC2.LDS_SIZE.
24 	 1 bit 	enable_exception_ieee_754_fp_invalid_ 		Enable IEEE 754 FP Invalid Operation exception at start of wavefront 			operation					execution. enable_exception flags are used by CP to set up 									COMPUTE_PGM_RSRC2.EXCP_EN (set from bits 0..6), EXCP_EN_MSB (set from 									bits 7..8).
25 	 1 bit 	enable_exception_fp_denormal_source 		Enable FP Denormal exception at start of wavefront execution.
26 	 1 bit 	enable_exception_ieee_754_fp_division_by_zero 	Enable IEEE 754 FP Division by Zero exception at start of wavefront 									execution.
27 	 1 bit 	enable_exception_ieee_754_fp_overflow 		Enable IEEE 754 FP FP Overflow exception at start of wavefront 									execution.
28 	 1 bit 	enable_exception_ieee_754_fp_underflow 		Enable IEEE 754 FP Underflow exception at start of wavefront 									execution.
29 	 1 bit 	enable_exception_ieee_754_fp_inexact 		Enable IEEE 754 FP Inexact exception at start of wavefront execution.
30 	 1 bit 	enable_exception_int_divide_by_zero 		Enable Integer Division by Zero (rcp_iflag_f32 instruction only) 									exception at start of wavefront execution.
31 	 1 bit 							Reserved. Must be 0.
32 								Total size 4 bytes.
======== ====== ============================================= ========================================================================




.. _AMD-Machine-Kind:

AMD Machine Kind amd_machine_kind_t
######################################

============================== ============ ==============================================================================
Enumeration Name 		 Value 			Description
============================== ============ ==============================================================================
AMD_MACHINE_KIND_UNDEFINED 	   0 	      Machine kind is undefined.
AMD_MACHINE_KIND_AMDGPU 	   1 	      Machine kind is AMD GPU. Corresponds to AMD GPU ISA architecture of AMDGPU.
============================== ============ ==============================================================================

.. _Float-Round-Mode:

Float Round Mode amd_float_round_mode_t
#########################################################

====================================== ========= =====================================================================
Enumeration Name 			 Value 		Description
====================================== ========= =====================================================================
AMD_FLOAT_ROUND_MODE_NEAR_EVEN		0 		Round Ties To Even
AMD_FLOAT_ROUND_MODE_PLUS_INFINITY 	1 		Round Toward +infinity
AMD_FLOAT_ROUND_MODE_MINUS_INFINITY 	2 		Round Toward -infinity
AMD_FLOAT_ROUND_MODE_ZERO 		3 		Round Toward 0
====================================== ========= =====================================================================

.. _Denorm-Mode:
   
Denorm Mode amd_float_denorm_mode_t
######################################

====================================== ======== ============================================
Enumeration Name 			Value 	 	Description
====================================== ======== ============================================
AMD_FLOAT_DENORM_MODE_FLUSH_SRC_DST 	 0 	Flush Source and Destination Denorms
AMD_FLOAT_DENORM_MODE_FLUSH_DST 	 1 	Flush Output Denorms
AMD_FLOAT_DENORM_MODE_FLUSH_SRC 	 2 	Flush Source Denorms
AMD_FLOAT_DENORM_MODE_FLUSH_NONE 	 3 	No Flush
====================================== ======== ============================================
.. _PCIe-Gen3-Atomic:

PCIe Gen3 Atomic Operations
######################################

PCI Express Gen3 defines 3 PCIe transactions, each of which carries out a specific Atomic Operation:

   * FetchAdd (Fetch and Add)
   * Swap (Unconditional Swap)
   * CAS (Compare and Swap)

For compute capabilities supporting PCIe Gen3 atomics, system scope atomic operations use the following sequences:

   * Atomic Load/Store: FLAT_LOAD_DWORD* / FLAT_STORE_DWORD* / TLP MRd / MWr
   * Atomic add: FLAT_ATOMIC_ADD / TLP FetchAdd
   * Atomic sub: FLAT_ATOMIC_ADD + negate/ TLP FetchAdd
   * Atomic swap: FLAT_ATOMIC_SWAP / TLP Swap
   * Atomic compare-and-swap: FLAT_ATOMIC_CMPSWAP / TLP CAS
   * Other Atomic RMW operations: (max, min, and, or, xor, wrapinc, wrapdec): CAS loop

PCIe Gen3 atomics are only supported on certain hardware configurations, for example, Haswell system.

.. _AMD-Queue:

AMD Queue
###################

.. _HSA-AQL:

HSA AQL Queue Object hsa_queue_t
######################################

HSA Queue Object is defined in "HSA Platform System Architecture Specification". AMD HSA Queue handle is a pointer to amd_queue_t.


.. _AMD-AQL:

AMD AQL Queue Object amd_queue_t
######################################

The AMD HSA Runtime implementation uses the AMD Queue object (amd_queue_t) to implement AQL queues. It begins with the HSA Queue object, and then has additional information contiguously afterwards that is AMD device specific. The AMD device specific information is accessible by the AMD HSA Runtime, CP and kernel machine code.

The AMD Queue object must be allocated on 64 byte alignment. This allows CP microcode to fetch fields using cache line addresses. The entire AMD Queue object must not span a 4GiB boundary. This allows CP to save a few instructions when calculating the base address of amd_queue_t from &(amd_queue_t.read_dispatch_id) and amd_queue_t.read_dispatch_id_field_base_offset.

For GFX8 and earlier systems, only HSA Queue type SINGLE is supported.

+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
| Bits	 | Size	    | Name	          		     |   Description                                                         |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|319:0 	 | 40 bytes | 	hsa_queue 			     |	HSA Queue object						     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|447:320 | 16 bytes |  					     | | Unused. Allows hsa_queue_t to expand but still keeps      	     |
|        |          |                                        | | write_dispatch_id, which is written by the producer		     |
|        |          |                                        | | often the host CPU), in the same cache line. Must be 0.             |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|511:448 | 8 bytes  |	write_dispatch_id        	     | | 64-bit index of the next packet to be allocated by     	     |
|        |          |                                        | | application or user-level runtime. Initialized to 0 at              |
|        |          |                                        | | queue creation time.                                                |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|512 	 |          | 					     | 	Start of cache line for fields accessed by kernel machine code isa.  |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|543:512 | 4 bytes  |  group_segment_aperture_base_hi        | | For HSA64, the most significant 32 bits of the 64 bit group	     |
|        |          |                                        | | segment flat address aperture base. This is the same value          |
|        |          |                                        | | {SH_MEM_BASES:PRIVATE_BASE[15:13],For HSA32, the 32 bits of the 32  |
|	 | 	    | 					     | | bit group segment flat address aperture.This is the same value as   |
|        |          |                                        | | {SH_MEM_BASES:SHARED_BASE[15:0], 16b0}.                             |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|575:544 | 4 bytes  | private_segment_aperture_base_hi	     | | For HSA64, the most significant 32 bits of the 64 bit private       |
|	 |  	    |					     | | segment flat address aperture base.This is the same value as        |
|	 |  	    |					     | | {SH_MEM_BASES:PRIVATE_BASE[15:13], 28b0, 1b1}  For HSA32,           |
| 	 |	    |					     | |  the 32 bits of the 32 bit private segment flat address aperture    |
|	 |	    |					     | | base This is the same value as {SH_MEM_BASES:PRIVATE_BASE[15:0],    |
|	 |	    |					     | |  16b0}.							     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|607:576 | 4 bytes  | max_cu_id				     | | The number of compute units on the agent to which the queue is      |
|	 |          |                                        | | associated.							     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|639:608 | 4 bytes  | max_wave_id 	 		     | | The number of wavefronts that can be executed on a single compute   |
|	 |          |					     | | unit of the device to which the queue is associated.                |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|703:640 | 8 bytes  | max_legacy_doorbell_dispatch_id_plus_1 | | For AMD_SIGNAL_KIND_LEGACY_DOORBELL maximum value of                |
|        |          |                                        | | write_dispatch_id signaled for the queue. the This value is always  |
|        |          |                                        | | 64-bit and never decreases.		                             |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|735:704 | 4 bytes  | legacy_doorbell_lock 		     | | For AMD_SIGNAL_KIND_LEGACY_DOORBELL, atomic variable used to        |
|        |          |                                        | | protect critical section which updates the doorbell related fields  |
|        |          |                                        | | Initialized to 0, and set to 1 to lock the critical section         |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|1023:736|36 bytes  |					     | 	Padding to next cache line. Unused and must be 0. 		     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
| 1024   |          |                                        | | Start of cache line for fields accessed by the packet processor (CP |
|	 |	    | 					     | | micro code).							     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|1087:   |8 bytes   | read_dispatch_id 			     | | 64-bit index of the next packet to be consumed by compute unit      |
|1024    |          |                                        | | hardware. Initialized to 0 at queue creation time.Queue operations  |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|1119:   |4 bytes   |read_dispatch_id_field_base_byte_offset | | Byte offset from the base of hsa_queue_t to the read_dispatch_id    |
|1088	 |          |  					     | | field.when amd_kernel_code_t.enable_sgpr_dispatch_ptr is set.       |
|        |          |                                        | | This field must immediately follow read_dispatch_id.This allows the |
| 	 |          |                                        | | layout above the read_dispatch_id  field to change, and still be    |
|        | 	    | 					     | | able to get the base of the hsa_queue_t, which is needed to return  |
|	 |          | 					     | | if amd_kernel_code_t.enable_sgpr_queue_ptr is requested. These      |
|	 |  	    |					     | | fields are defined by HSA Foundation and so could change. CP only   |
|	 |	    |					     | | uses fields below read_dispatch_id which are defined by AMD.	     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|1536    |          |                                        | | Start of next cache line for fields not accessed under normal       |
|        |          |					     | | conditions by the packet processor (CP micro code). These are kept  |
|	 | 	    |				             | | in a single cache line to minimize memory accesses performed by CP  |
|	 |	    |					     | | micro code.							     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+
|2048    |          |                                        | | Total size 256 bytes.						     |
+--------+----------+----------------------------------------+-----------------------------------------------------------------------+


.. _Queue-operations:

Queue-operations
###################

A queue has an associated set of high-level operations defined in "HSA Runtime Specification" (API functions in host code) and "HSA Programmer Reference Manual Specification" (kernel code).

The following is informal description of AMD implementation of queue operations (all use memory scope system, memory order applies):

  * Load Queue Write Index: Atomic load of read_dispatch_id field
  * Store Queue Write Index: Atomic store of read_dispatch_id field
  * Load Queue Read Index: Atomic load of write_dispatch_id field
  * Store Queue Read Index: Atomic store of read_dispatch_id field
  * Add Queue Write Index: Atomic add of write_dispatch_id field
  * Compare-And-Swap Queue Write Index: Atomic CAS of write_dispatch_id field

.. _Signals:

Signals
###################

.. _Signals-overview:

Signals overview
###################
Signal handle is 8 bytes. AMD signal handle is a pointer to AMD Signal Object (amd_signal_t).

The following operations are defined on HSA Signals:

   * Signal Load
       * Read the of the current value of the signal
       * Optional acquire semantics on the signal value
   * Signal Wait on a condition
       * Blocks the thread until the requested condition on the signal value is observed
       * Condition: equals, not-equals, greater, greater-equals, lesser, lesser-equals
       * Optional acquire semantics on the signal value
       * Returns the value of the signal that caused it to wake
   * Signal Store
       * Optional release semantics on the signal value
   * Signal Read-Modify-Write Atomics (add, sub, increment, decrement, min, max, and, or, xor, exch, cas)
       * These happen immediately and atomically
       * Optional acquire-release semantics on the signal value

.. _Signal-kind:

Signal kind amd_signal_kind_t
######################################

+----+---------------------------------+-----------------------------------------------------------+
| ID | Name                            | Description                                               |
+----+---------------------------------+-----------------------------------------------------------+
| 0  | AMD_SIGNAL_KIND_INVALID         | An invalid signal.                                        |
+----+---------------------------------+-----------------------------------------------------------+
| 1  | AMD_SIGNAL_KIND_USER            | A regular signal                                          |
+----+---------------------------------+-----------------------------------------------------------+
| -1 | AMD_SIGNAL_KIND_DOORBELL        | Doorbell signal with hardware support                     |
+----+---------------------------------+-----------------------------------------------------------+
| -2 | AMD_SIGNAL_KIND_LEGACY_DOORBELL | Doorbell signal with hardware support, legacy (GFX7/GFX8) |
+----+---------------------------------+-----------------------------------------------------------+

.. _Signal-object:

Signal object amd_signal_t
######################################

========= ========== =========================================== =====================================================================
Bits 	    Size 	Name 							Description
========= ========== =========================================== =====================================================================
63:0 	   8 bytes 	kind 						Signal kind
127:64 	   8 bytes 	value					  For AMD_SIGNAL_KIND_USER: signal payload value. In small machine 									  model only the lower 32 bits is used, in large machine model all 64 									  bits are used.
127:64     8 bytes 	legacy_hardware_doorbell_ptr 		  For AMD_SIGNAL_KIND_LEGACY_DOORBELL: pointer to the doorbell IOMMU 									  memory (write-only). Used for hardware notification in Signal Store.
127:64 	   8 bytes 	hardware_doorbell_ptr 			  For AMD_SIGNAL_KIND_DOORBELL: pointer to the doorbell IOMMU 									  memory (write-only). Used for hardware notification in Signal Store.
191:128    8 bytes 	event_mailbox_ptr 			  For AMD_SIGNAL_KIND_USER: mailbox address for event notification in 								  	  Signal operations.
223:192    4 bytes 	event_id 				  For AMD_SIGNAL_KIND_USER: event id for event notification in Signal 									  operations.
255:224    4 bytes 					          Padding. Must be 0.
319:256    8 bytes 	start_ts 				  Start of the AQL packet timestamp, when profiled.
383:320    8 bytes 	end_ts 	 		 	  	  End of the AQL packet timestamp, when profiled.
448:384    8 bytes 	queue_ptr 				  For AMD_SIGNAL_KIND_*DOORBELL: the address of the associated 									  amd_queue_t, otherwise reserved and must be 0.
511:448    8 bytes 					  	  Padding to 64 byte size. Must be 0.
512 								  Total size 64 bytes
========= ========== =========================================== =====================================================================

.. _Signal-kernel-machine-code:

Signal kernel machine code
######################################
As signal kind is determined by kind field of amd_signal_t, instruction sequence for signal operation must branch on signal kind.

The following is informal description of signal operations:

 * For AMD_SIGNAL_KIND_USER kind:
       * Signal Load uses atomic load from value field of corresponding amd_signal_t (memory order applies, memory scope system).
       * Signal Wait
            * Uses poll loop on signal value.
            * s_sleep ISA instruction provides hint to the SQ to not to schedule the wave for a specified time.
            * s_memtime/s_memrealtime instruction is used to measure time (as signal wait is required to time out in reasonable time 		      interval even if condition is not met).
       * Signal Store/Signal Atomic uses the following sequence:
            * Corresponding atomic operation on signal value (memory scope system, memory order applies).
            * Load mailbox address from event_mailbox_ptr field.
            * If mailbox address is not zero:
                * load event id from event_id field.
                * atomic store of event id to mailbox address (memory scope system, memory order release).
                * s_sendmsg with argument equal to lower 8 bits of event_id.
 * For AMD_SIGNAL_KIND_LEGACY_DOORBELL:
       * Signal Store uses the following sequence:
           * Load queue address from queue_ptr field
           * Acquire spinlock protecting the legacy doorbell of the queue.
              *  Load address of the spinlock from legacy_doorbell_lock field of amd_queue_t.
              *  Compare-and-swap atomic loop, previous value 0, value to set 1 (memory order acquire, memory scope system).
              *  s_sleep ISA instruction provides hint to the SQ to not to schedule the wave for a specified time.
           * Use value+1 as next packet index and initial value for legacy dispatch id. GFX7/GFX8 hardware expects packet index to 		     point beyond the last packet to be processed.
           * Atomic store of next packet index (value+1) to max_legacy_doorbell_dispatch_id_plus_1 field (memory order relaxed, 	     memory scope system).
           * For small machine model:
               * legacy_dispatch_id = min(write_dispatch_id, read_dispatch_id + hsa_queue.size)
           * For GFX7:
              * Load queue size from hsa_queue.size field of amd_queue_t.
              *  Wrap packet index to a point within the ring buffer (ring buffer size is twice the size of the HSA queue).
              *  Convert legacy_dispatch_id to DWORD count by multiplying by 64/4 = 16.
              *  legacy_dispatch_id = (legacy_dispatch_id & ((hsa_queue.size << 1)-1)) << 4;
           * Store legacy dispatch id to the hardware MMIO doorbell.
              * Address of the doorbell is in legacy_hardware_doorbell_ptr field of amd_signal_t.
           * Release spinlock protecting the legacy doorbell of the queue. Atomic store of value 0.
       * Signal Load/Signal Wait/Signal Read-Modify-Write Atomics are not supported. Instruction sequence for these operations and 		 this signal kind is empty.
  *  For AMD_SIGNAL_KIND_DOORBELL:
       * Signal Store uses the following sequence:
           * Atomic store of value to the hardware MMIO doorbell.
       * Signal Load/Signal Wait/Signal Read-Modify-Write Atomics are not supported. Instruction sequence for these operations and 		 this signal kind is empty.

.. _Debugtrap:

Debugtrap
###################
Debugtrap halts execution of the wavefront and generates debug exception. For more information, refer to "HSA Programmer Reference Manual Specification". debugtrap accepts 32-bit unsigned value as an argument.

The following is a description of debugtrap sequence:

   * v0 contains 32-bit argument of debugtrap
   * s[0:1] contains Queue Ptr for the dispatch
   * s_trap 0x1

.. _References:

References
###########

   * `HSA Standards and Specifications <http://www.hsafoundation.com/standards/>`_
   * `HSA Platform System Architecture Specification 1.0 <www.hsafoundation.com/?ddownload=4944>`_
   * `HSA Programmer Reference Manual Specification 1.01 <www.hsafoundation.com/?ddownload=4945>`_
   * `HSA Runtime Specification 1.0 <www.hsafoundation.com/?ddownload=4946>`_
   * AMD ISA Documents
       * `AMD GCN3 Instruction Set Architecture (2015) <https://github.com/tpn/pdfs/blob/master/AMD%20-%20GCN3%20Instruction%20Set%20Architecture%20-%20Graphics%20Core%20Next%20Architecture%2C%20Generation%203%20(Revision%201.0%2C%20March%202015).pdf>`_.
       * `AMD_Southern_Islands_Instruction_Set_Architecture <http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/07/AMD_Southern_Islands_Instruction_Set_Architecture1.pdf>`_
   * `ROCR Runtime sources <https://github.com/RadeonOpenCompute/ROCR-Runtime>`_
       * `amd_hsa_kernel_code.h <https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/master/src/inc/amd_hsa_kernel_code.h>`_
       * `amd_hsa_queue.h <https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/master/src/inc/amd_hsa_queue.h>`_ 
       * `amd_hsa_signal.h <https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/master/src/inc/amd_hsa_signal.h>`_
       * `amd_hsa_common.h <https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/master/src/inc/amd_hsa_common.h>`_
   * `PCI Express Atomic Operations <https://pcisig.com/specifications/pciexpress/specifications/ECN_Atomic_Ops_080417.pdf>`_


