
.. _sysfsclasskfdtopologynodes0:

sysfs-class-kfd-topology-nodes-N
----------------------------------

|  What:            /sys/class/kfd/topology/nodes/N/cpu_cores_count
|  Date:            May 2018
|  KernelVersion:   4.13
|  Description:     This field gives information about Number of latency (= CPU) cores present on this HSA node. This value is 0 for a node with no such cores, e.g a "discrete HSA GPU".

|  What:		/sys/class/kfd/topology/nodes/N/simd_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Here the number of smid (Single Instruction Multiple Data architecture) processes count is registered
 
|  What:		/sys/class/kfd/topology/nodes/N/mem_banks_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives the Number of discoverable memory bank affinity properties on this "H-NUMA" node
 
|  What:		/sys/class/kfd/topology/nodes/N/caches_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Gives the Number of discoverable cache affinity properties on the "H-NUMA" node.
 
|  What:		/sys/class/kfd/topology/nodes/N/io_links_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives the number of discoverable IO link affinity properties of this node connecting to other nodes.
 
|  What:		/sys/class/kfd/topology/nodes/N/cpu_cores_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Gives the CPU core id details corresponding to core count
 
|  What:		/sys/class/kfd/topology/nodes/N/simd_id_base
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives simd id value.
 
|  What:		/sys/class/kfd/topology/nodes/N/max_waves_per_simd
|  Date:		May 2018 
|  KernelVersion:	4.13
|  Description:	This identifies the maximum number of launched waves per SIMD. If NUmSIMDCores is 0, this value is ignored
 
|  What:		/sys/class/kfd/topology/nodes/N/gds_size_in_kb
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives the size of Global Data Store in Kilobytes shared across SIMD Wavefronts, typically 32 or 64
 
|  What:		/sys/class/kfd/topology/nodes/N/wave_front_size
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	wavefront is group of threads (work-item) that execute together for executing kernels and this field gives the size of the wavefront used. Usually 64or 32 or a different value for some HSA based architectures
 
|  What:		/sys/class/kfd/topology/nodes/N/array_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field give Number of SIMD Arrays per Engine
 
|  What:		/sys/class/kfd/topology/nodes/N/simd_arrays_per_engine
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	It gives the simd array count for every compute unite (stream engine)
| 
|  What:		/sys/class/kfd/topology/nodes/N/cu_per_simd_array
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Gives the Number of Compute Units (CU) per SIMD Array
| 
|  What:		/sys/class/kfd/topology/nodes/N/simd_per_cu
|  Date:		May 2018
|  KernelVersion:	4.13 
|  Description:	Number of SIMD representing a Compute Unit (CU)
| 
|  What:		/sys/class/kfd/topology/nodes/N/max_slots_scratch_cu
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Bitmask of available CU slots, used for CU mask setup for the queues if assignment is desired by application necessary.
| 
|  What:		/sys/class/kfd/topology/nodes/N/vendor_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field contains the GPU vendor id; 0 on CPU-only nodes
|  
|  What:		/sys/class/kfd/topology/nodes/N/device_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field contains the  GPU device id; 0 on CPU-only nodes
| 
|  What:		/sys/class/kfd/topology/nodes/N/location_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	LocationId, 32bit value, equivalent to BDF_ID used by Linux tools especially (identifies device in the overall
system)
| 
|  What:		/sys/class/kfd/topology/nodes/N/drm_render_minor
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	drm (Direct Rendering Manager) render data count is shown  
| 
|  What:		/sys/class/kfd/topology/nodes/N/max_engine_clk_ccompute
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Maximum engine clock speed of the CPU
| 

