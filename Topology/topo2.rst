
.. _sysfsclasskfdtopologynodes0:

sysfs-class-kfd-topology-nodes-0
----------------------------------

|  What:            /sys/class/kfd/topology/nodes/0/cpu_cores_count
|  Date:            May 2018
|  KernelVersion:   4.13
|  Description:     This field gives information about number of active CPU cores available for computation

|  What:		/sys/class/kfd/topology/nodes/0/simd_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Here the number of smid (Single Instruction Multiple Data architecture) processes count is registered
 
|  What:		/sys/class/kfd/topology/nodes/0/mem_banks_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives the memory banks counts registered.
 
|  What:		/sys/class/kfd/topology/nodes/0/caches_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Gives the caches registered numbers
 
|  What:		/sys/class/kfd/topology/nodes/0/io_links_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives the number of Inout Output bus links registered.
 
|  What:		/sys/class/kfd/topology/nodes/0/cpu_cores_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Gives the CPU core id details corresponding to core count
 
|  What:		/sys/class/kfd/topology/nodes/0/simd_id_base
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	
 
|  What:		/sys/class/kfd/topology/nodes/0/max_waves_per_simd
|  Date:		May 2018 
|  KernelVersion:	4.13
|  Description:	This identifies the maximum number of launched waves per SIMD.
 
|  What:		/sys/class/kfd/topology/nodes/0/gds_size_in_kb
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field gives the gds(Global Data Share) size used as software data cache for compute kernels
 
|  What:		/sys/class/kfd/topology/nodes/0/wave_front_size
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	wavefront is group of threads (work-item) that execute together for executing kernels and this field gives the size of the wavefront used. 
                    64, may be 32 for some FSA based architectures
 
|  What:		/sys/class/kfd/topology/nodes/0/array_count
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field give Number of SIMD Arrays per Engine
 
|  What:		/sys/class/kfd/topology/nodes/0/simd_arrays_per_engine
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	It gives the simd array count for every compute unite (stream engine)
| 
|  What:		/sys/class/kfd/topology/nodes/0/cu_per_simd_array
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Gives the number of Compute Units (CU) per SIMD Array value
| 
|  What:		/sys/class/kfd/topology/nodes/0/simd_per_cu
|  Date:		May 2018
|  KernelVersion:	4.13 
|  Description:	Number of SIMD representing a Compute Unit
| 
|  What:		/sys/class/kfd/topology/nodes/0/max_slots_scratch_cu
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Maximum Number of temporary memory ("scratch") wave slots
| 
|  What:		/sys/class/kfd/topology/nodes/0/vendor_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field contains the vendor identity number
|  
|  What:		/sys/class/kfd/topology/nodes/0/device_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	This field contains the device identity number
| 
|  What:		/sys/class/kfd/topology/nodes/0/location_id
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Provides the physical location of the Memory Array
| 
|  What:		/sys/class/kfd/topology/nodes/0/drm_render_minor
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	here drm (Direct Rendering Manager) render data count is shown  
| 
|  What:		/sys/class/kfd/topology/nodes/0/max_engine_clk_ccompute
|  Date:		May 2018
|  KernelVersion:	4.13
|  Description:	Maximum clock speed to compute kernel
| 

