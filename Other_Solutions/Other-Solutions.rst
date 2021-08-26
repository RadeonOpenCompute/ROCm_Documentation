
.. _Other-Solutions:

For the latest HIP Programming Guide documentation, refer to the PDF version at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_Programming_Guide_v4.3.pdf


System Level Debug
=====================

ROCm Language & System Level Debug, Flags, and Environment Variables 
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
  


HCC Debug Enviroment Variables
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


HIP Environment Variables
*************************



+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_HIDDEN_FREE_MEM= 256     || Amount of memory to hide from the free memory reported by hipMemGetInfo, specified in MB.Impacts   |
| 			       || hipMemGetInfo										                                                                                               |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_DB_START_API =           | Comma-separated list of tid.api_seq_num for when to start debug and profiling.                      |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_VISIBLE_DEVICES = 0      || Only devices whose index is present in the sequence are visible to HIP applications and they are   |
|			       || enumerated in the order of sequence 							    	                                                                       |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_FORCE_SYNC_COPY =  0     | Force all copies (even hipMemcpyAsync) to use sync copies                                           |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_HOST_COHERENT =  1       || If set, all host memory will be allocated as fine-grained system memory.This allows                |
|			       || threadfence_system to work but prevents host memory from being cached on GPU which may have                            |
|			       || performance impact.									                                                                                           |
+------------------------------+-----------------------------------------------------------------------------------------------------+
|                                                                                                                                    |
+------------------------------+-----------------------------------------------------------------------------------------------------+
|       |
+------------------------------+-----------------------------------------------------------------------------------------------------+

OpenCL Debug Flags
********************

* AMD_OCL_WAIT_COMMAND=1  (0 = OFF, 1 = On)

PCIe-Debug
*************

Refer here for :ref:`PCIe-Debug`

**More information here on how to debug and profile HIP applications**

* `HIP-Debugging <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/HIP_Debugging.html#hip-debugging>`_
* `HIP-Profiling <http://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/hip_profiling.html#hip-profiling>`_


