
.. _Other-Solutions:


System Level Debug
=====================

ROCm Language & System Level Debug, Flags and Environment Variables 
#####################################################################

| Kernel options to avoid Ethernet port getting renamed every time you change graphics cards
| net.ifnames=0 biosdevname=0

ROCr Error Code
******************

* 2  Invalid Dimension
* 4 Invalid Group Memory 
* 8 Invalid (or Null) Code 
* 32 Invalid Format </li>
* 64 Group is too large 
* 128 Out of VGPR’s 
* 0x80000000  Debug Trap 

Command to dump firmware version and get Linux Kernel version 
*****************************************************************

* sudo cat /sys/kernel/debug/dri/1/amdgpu_firmware_info 
* uname -a  

Debug Flags 
***************

Debug messages when developing/debugging base ROCm dirver. You could enable the printing from libhsakmt.so by setting an environment variable, HSAKMT_DEBUG_LEVEL. Available debug levels are 3~7. The higher level you set, the more messages will print.

* export HSAKMT_DEBUG_LEVEL=3 : only pr_err() will print.
* export HSAKMT_DEBUG_LEVEL=4 : pr_err() and pr_warn() will print.
* export HSAKMT_DEBUG_LEVEL=5 : We currently don’t implement “notice”. Setting to 5 is same as setting to 4.
* export HSAKMT_DEBUG_LEVEL=6 : pr_err(), pr_warn(), and pr_info will print.
* export HSAKMT_DEBUG_LEVEL=7 : Everything including pr_debug will print.


ROCr level env variable for debug 
************************************

* HSA_ENABLE_SDMA=0
* HSA_ENABLE_INTERRUPT=0
* HSA_SVM_GUARD_PAGES=0
* HSA_DISABLE_CACHE=1

Turn Off Page Retry on GFX9/Vega devices
**********************

  * sudo –s
  * echo 1 > /sys/module/amdkfd/parameters/noretry
  


HCC Debug Enviroment Varibles
********************************

+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_PRINT_ENV=1                     | will print usage and current values for the HCC and HIP env variables.                       |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_PRINT_ENV = 1                   | Print values of HCC environment variables                                                    |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_SERIALIZE_KERNEL= 0             | | 0x1=pre-serialize before each kernel launch, 0x2=post-serialize after each kernel launch,} |
|				      | | 0x3=both									             |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_SERIALIZE_COPY= 0               | 0x1=pre-serialize before each data copy, 0x2=post-serialize after each data copy, 0x3=both   |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_DB = 0                          | Enable HCC trace debug                                                                       |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_OPT_FLUSH = 1                   | | Perform system-scope acquire/release only at CPU sync boundaries (rather than after each   |
|                                     | |  kernel)                                                                                   |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_MAX_QUEUES= 20                  | | Set max number of HSA queues this process will use.  accelerator_views will share the      |
|				      | | allotted queues and steal from each other as necessary                                     |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_UNPINNED_COPY_MODE = 2          | | Select algorithm for unpinned copies. 0=ChooseBest(see thresholds), 1=PinInPlace,          |
|                                     | | 2=StagingBuffer,3=Memcpy                                                                   |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_CHECK_COPY = 0                  | Check dst == src after each copy operation.  Only works on large-bar systems.                |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_H2D_STAGING_THRESHOLD = 64      | | Min size (in KB) to use staging buffer algorithm for H2D copy if ChooseBest algorithm      |
|                                     | | selected                                                                                   |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_H2D_PININPLACE_THRESHOLD = 4096 | Min size (in KB) to use pin-in-place algorithm for H2D copy if ChooseBest algorithm selected |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_D2H_PININPLACE_THRESHOLD = 1024 | Min size (in KB) to use pin-in-place for D2H copy if ChooseBest algorithm selected           |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_PROFILE = 0                     | Enable HCC kernel and data profiling.  1=summary, 2=trace                                    |
+-------------------------------------+----------------------------------------------------------------------------------------------+
| HCC_PROFILE_VERBOSE  = 31           | Bitmark to control profile verbosity and format. 0x1=default, 0x2=show begin/end, 0x4=show   |
|                                     | barrier                                                                                      |
+-------------------------------------+----------------------------------------------------------------------------------------------+


HIP Environment Varibles
*************************

+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_PRINT_ENV=1              | Print HIP environment variables.                                                                    |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_LAUNCH_BLOCKING=0        || Make HIP kernel launches 'host-synchronous', so they block until any kernel launches. Alias:       |
|			       || CUDA_LAUNCH_BLOCKING								                     |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_LAUNCH_BLOCKING_KERNELS= | Comma-separated list of kernel names to make host-synchronous, so they block until completed.       |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_API_BLOCKING= 0          || Make HIP APIs 'host-synchronous', so they block until completed. Impacts hipMemcpyAsync,           |
|			       || hipMemsetAsync							                             |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_HIDDEN_FREE_MEM= 256     || Amount of memory to hide from the free memory reported by hipMemGetInfo, specified in MB.Impacts   |
| 			       || hipMemGetInfo										             |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_DB = 0                   | Print debug info.  Bitmask (HIP_DB=0xff) or flags separated by '+' (HIP_DB=api+sync+mem+copy)       |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_TRACE_API=0              | Trace each HIP API call.  Print function name and return code to stderr as program executes.        |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_TRACE_API_COLOR= green   | Color to use for HIP_API.  None/Red/Green/Yellow/Blue/Magenta/Cyan/White                            |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_PROFILE_API =  0         || Add HIP API markers to ATP file generated with CodeXL. 0x1=short API name, 0x2=full API name       |
| 			       || including args                                                                                     |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_DB_START_API =           | Comma-separated list of tid.api_seq_num for when to start debug and profiling.                      |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_DB_STOP_API =            | Comma-separated list of tid.api_seq_num for when to stop debug and profiling.                       |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_VISIBLE_DEVICES = 0      || Only devices whose index is present in the sequence are visible to HIP applications and they are   |
|			       || enumerated in the order of sequence 							    	     |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_WAIT_MODE =  0           | Force synchronization mode. 1= force yield, 2=force spin, 0=defaults specified in application       |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_FORCE_P2P_HOST =  0      || Force use of host/staging copy for peer-to-peer copies.1=always use copies, 2=always return false  |
|			       || for hipDeviceCanAccessPeer								             |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_FORCE_SYNC_COPY =  0     | Force all copies (even hipMemcpyAsync) to use sync copies                                           |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_FAIL_SOC =  0            || Fault on Sub-Optimal-Copy, rather than use a slower but functional implementation.Bit 0x1=Fail on  |
|			       || async copy with unpinned memory.  Bit 0x2=Fail peer copy rather than use staging buffer copy       |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_SYNC_HOST_ALLOC =  1     | Sync before and after all host memory allocations.  May help stability                              |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_SYNC_NULL_STREAM =  0    | Synchronize on host for null stream submissions                                                     |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_HOST_COHERENT =  1       || If set, all host memory will be allocated as fine-grained system memory.This allows                |
|			       || threadfence_system to work but prevents host memory from being cached on GPU which may have        |
|			       || performance impact.									             |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HCC_OPT_FLUSH =  1           || When set, use agent-scope fence operations rather than system-scope fence operationsflush when     |
|			       || possible. This flag controls both HIP and HCC behavior                                             |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_EVENT_SYS_RELEASE =  0   || If set, event are created with hipEventReleaseToSystem by default.  If 0, events are created with  |
|			       || hipEventReleaseToDevice by default.  The defaults can be overridden by specifying                  |
|			       || hipEventReleaseToSystem or hipEventReleaseToDevice flag when creating the event.                   |
+------------------------------+-----------------------------------------------------------------------------------------------------+

OpenCL Debug Flags
********************

* AMD_OCL_WAIT_COMMAND=1  (0 = OFF, 1 = On)

PCIe-Debug
*************

Refer here for :ref:`PCIe-Debug`

**There’s some more information here on how to debug and profile HIP applications**

* `HIP-Debugging <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP_Debugging.html#hip-debugging>`_
* `HIP-Profiling <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/hip_profiling.html#hip-profiling>`_


