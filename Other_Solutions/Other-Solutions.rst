
.. _Other-Solutions:

For the latest HIP Programming Guide documentation and environment variables, refer to the PDF version of the HIP Programming Guide v4.5 at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_HIP_Programming_Guide.pdf

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
  

HIP Environment Variables 3.x
*******************************

+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_HIDDEN_FREE_MEM= 256     ||Amount of memory to hide from the free memory reported by hipMemGetInfo, specified in MB.Impacts    |
| 			                          || hipMemGetInfo										                                                                            |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_DB_START_API =           || Comma-separated list of tid.api_seq_num for when to start debug and profiling.                      |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_VISIBLE_DEVICES = 0      || Only devices whose index is present in the sequence are visible to HIP applications and they are   |
|			                           || enumerated in the order of sequence 							    	                                                   |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_FORCE_SYNC_COPY =  0     || Force all copies (even hipMemcpyAsync) to use sync copies                                           |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| HIP_HOST_COHERENT =  1       || If set, all host memory will be allocated as fine-grained system memory.This allows                |
|			                           || threadfence_system to work but prevents host memory from being cached on GPU which may have        |
|			                           || performance impact.									                                                                       |
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


