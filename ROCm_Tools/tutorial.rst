.. _tutorial:

tutorial
==========
 
How do I debug my GPU application?
************************************
You can start your program in rocm-gdb just like you would any application under gdb

   * rocm-gdb MatrixMul
   * You should now be in the gdb prompt and can start execution of the application
   * (ROCm-gdb) start

How do I view the list of all ROCm gdb commands?
**************************************************
To view the list of all rocm related gdb commands, you can type :: help rocm.

::

 (ROCm-gdb) help rocm
 ROCm specific features in ROCm-gdb.
 --------------------------------------------------------------------------
 ROCm focus thread command:
 rocm thread wg:<x,y,z> wi:<x,y,z>  Switch focus to a specific active GPU work-item
 --------------------------------------------------------------------------
 ROCm breakpoint commands:
 break rocm                         Break on every GPU dispatch
 break rocm:<kernel_name>           Break when kernel <kernel_name> is about to begin execution
 break rocm:<line_number>           Break when execution hits line <line_number> in temp_source
 --------------------------------------------------------------------------
 ROCm info commands:
 info rocm devices 		   Print all available GPU devices
 info rocm kernels                  Print all GPU kernel dispatches
 info rocm kernel <kernel_name>     Print all GPU kernel dispatches with a specific <kernel_name>
 info rocm [work-groups|wgs]        Print all GPU work-group items
 info rocm [work-group|wg] [<flattened_id>|<x,y,z>]  Print a specific GPU work-group item
 info rocm [work-item|wi|work-items|wis]             Print the focus GPU work-item
 info rocm [work-item|wi] <x,y,z>   Print a specific GPU work-item
 --------------------------------------------------------------------------
 ROCm specific configuration commands:
 set rocm trace [on|off]            Enable/Disable tracing of GPU dispatches
 set rocm trace <filename>          Save GPU dispatch trace to <filename>
 set rocm logging [on|off]          Enable/Disable internal logging
 set rocm show-isa [on|off]         Enable/Disable saving ISA to a temp_isa file when in GPU dispatches
 --------------------------------------------------------------------------
 ROCm variable print commands:
 print rocm:<variable>              Print value of <variable> for the focus work-item
 --------------------------------------------------------------------------
 To disassemble a GPU kernel:
 disassemble                        Show the GPU ISA disassembly text when at a GPU breakpoint
 --------------------------------------------------------------------------

How do I set breakpoints in my GPU application?
**************************************************
To set breakpoints in GPU kernels, rocm-gdb defines

   * GPU kernel function breakpoint: Similar to a gdb function breakpoint, allows you stop the application just before a specific GPU dispatch 	    starts
   * Generic GPU kernel breakpoint: Stop the application before any GPU dispatch starts
   * Source line breakpoint: A breakpoint that is set on a particular line of GPU kernel source

Setting GPU function breakpoints
********************************
The gdb :: break command has been extended to break rocm in order to set GPU breakpoints. To set a specific GPU kernel function breakpoints:

   * ``break rocm:<kernel_name>``

For matrix multiplication, you can specify the kernel name

   *  ``(ROCm-gdb) break rocm:&__OpenCL_matrixMul_kernel``

This will stop the application's execution just before the GPU kernel (in this case, the matrix multiplication kernel) begins executing on the device.

To set a general GPU kernel function breakpoint, use either of the following command:

   * ``(ROCm-gdb) break rocm``
   * ``(ROCm-gdb) break rocm:*``

This will stop the application just before every dispatch begins executing on the device.

Setting GPU kernel source breakpoints
**************************************
In order to break into GPU kernels, you need to set GPU source breakpoints. ROCm-gdb saves the kernel source for the present dispatch to a temporary file called temp_source. GPU source breakpoints can be set by specifying the line number from the temp_source GPU kernel source file. The temp_source file is overwritten by rocm-gdb on every GPU dispatch.

Once you hit a kernel function breakpoint, you can view the temp_source file and choose a line number. You can set the source breakpoint using the syntax

   * ``break rocm:<line_number>``

For example, this will set a breakpoint at line 150 in the temp_source

::

   (ROCm-gdb) b rocm:150
   GPU breakpoint 1 (PC:0x08d0 mad_u32 $s0, $s1, $s0, $s3; temp_source@line 150)

When you continue the program's execution, the application will stop when any work-item reaches line 150 in temp_source.

Setting Conditional GPU kernel source breakpoints
***************************************************
Conditional GPU breakpoints allow you to stop the application only when a particular work-item hits a breakpoint. You can set a conditional source breakpoint by specifying the a work-item using the syntax:

   * break rocm:line_number if wg:x,y,z wi:x,y,z For example, this will set a breakpoint at line 150 and only stop the application if the work-item in workgroup 2,0,0 and local work-item 1,0,0

::

   (ROCm-gdb) b rocm:150 if wg:2,0,0 wi:16,0,0
   GPU breakpoint 1 (PC:0x08d0 mad_u32 $s0, $s1, $s0, $s3; temp_source@line 150)

When the application is executed, the dispatch will stop when line 150 is executed for the above work-item as shown below:

::

   [ROCm-gdb]: Switching to work-group (2,0,0) and work-item (1,0,0)
   [ROCm-gdb]: Condition: active work-group: 2, 0, 0 @ work-item: 1, 0, 0
   [ROCm-gdb]: Breakpoint 2 at mad_u32 $s0, $s1, $s0, $s3; temp_source@line 150
   Stopped on GPU breakpoint

Managing GPU breakpoints
*************************
  *  You can use the same gdb commands such as info bre to view information about the active GPU and host breakpoints The command info bre 	  shows multiple GPU kernel source breakpoints, an GPU function breakpoint and a host breakpoint

::

   (ROCm-gdb) info bre
   Num     Type             Disp Enb Address            What
   1       GPU breakpoint   keep y   ---                Every GPU dispatch(*)
   breakpoint already hit 2 times
   4       GPU breakpoint   keep y   PC:0x06d8          add_u32 $s3, $s3, 1; temp_source@line 150
   breakpoint already hit 320 times
   5       GPU breakpoint   keep y   ---                &__Gdt_vectoradd_kernel
   6       breakpoint       keep y   0x0000000000407105 in RunTest() at MultiKernelDispatch.cpp:100

   * You can also delete GPU breakpoints using the same command as GDB's host breakpoints del breakpoint_number

How do I single step in a GPU kernel?
**************************************
You can single step in a GPU dispatch using the conventional step command. Only a single step is supported at a time.

The following shows how rocm-gdb steps 4 source lines after hitting a kernel source breakpoint

::

   (ROCm-gdb) b rocm:64
   GPU breakpoint 2 (PC:0x02a0 workitemabsid_u32   $s0, 0; temp_source@line 64)
   (ROCm-gdb) c
   Continuing.
   [New Thread 0x7fffef286700 (LWP 2776)]
   [New Thread 0x7fffeea85700 (LWP 2777)]
   Waiting for completion...
   [Switching to Thread 0x7fffeea85700 (LWP 2777)]
   [ROCm-gdb]: Switching to work-group (5,4,0) and work-item (0,8,0)
   [ROCm-gdb]: Breakpoint 2 at PC:0x02a0 workitemabsid_u32   $s0, 0; temp_source@line 64
   Stopped on GPU breakpoint

   (ROCm-gdb) step
   [ROCm-gdb]: PC:0x02ac 	 cvt_u64_u32 $d5, $s0; temp_source@line 65
   Stopped on GPU breakpoint

   (ROCm-gdb) s
   [ROCm-gdb]: PC:0x02d0 	 workitemabsid_u32   $s0, 1; temp_source@line 66
   Stopped on GPU breakpoint

   (ROCm-gdb) s
   [ROCm-gdb]: PC:0x02dc 	 ld_kernarg_align(8)_width(all)_u64  $d6, [%__global_offset_0]; temp_source@line 67
   Stopped on GPU breakpoint

   (ROCm-gdb) s
   [ROCm-gdb]: PC:0x0304 	 add_u64 $d0, $d5, $d6; temp_source@line 68
   Stopped on GPU breakpoint

   (ROCm-gdb) c
   Continuing.

How do I print GPU registers?
******************************
To print registers in a GPU kernel, the gdb print command has been extended. To print GPU registers.

   * ``print rocm:$register_name``

This will print the value $register_name for a single work-item. For example, printing GPU register $s0 will provide the value of register $s0

::

   (ROCm-gdb) print rocm:$s0
   $4 = 0

To view the data of a different work-item, you need switch focus between different work-items. The rocm thread command allows you to set the focus on a different work-item by specifying its work-item and work-group ID. It should be noted that you cannot switch focus to work-items not scheduled on the device.

Switching the focus to another work-item and printing $s0 allows us to view data for the other work-item.

::

   (ROCm-gdb) rocm thread wg:0,0,0 wi:1,0,0
   [ROCm-gdb]: Switching to work-group (0,0,0) and work-item (1,0,0)  
   (ROCm-gdb) print rocm:$s0
    $3 = 1

How do I view the GPU ISA disassembly?
***************************************
To view the GPU ISA disassembly, you can use the standard gdb disassemble command while gdb stops at the GPU function breakpoint or GPU kernel source breakpoint.

While gdb stops at a GPU kernel source breakpoint, the program counter of the focus wave is shown with a (=>) prefix and some ISA instructions above and below the program counter are shown.

::

  [ROCm-gdb]: Breakpoint 1 at GPU Kernel, &ZZ4mainEN3_EC__219__cxxamp_trampolineEPfiiiiiiifS0_iiiiiii()
  GPU kernel saved to temp_source
  Stopped on GPU breakpoint

  (ROCm-gdb) s
  [New Thread 0x7fffee0e9700 (LWP 3190)]
  [ROCm-gdb]: Switching to work-group (486,0,0) and work-item (256,0,0)
  [ROCm-gdb]: Breakpoint:  at line 24
  Stopped on GPU breakpoint
  [Switching to Thread 0x7fffee0e9700 (LWP 3190)]

  (ROCm-gdb) disassemble
  Disassembly:
     s_lshr_b32    s16, s16, 16                            // 000000000144: 8F109010
     s_mul_i32     s18, s12, s13                           // 000000000148: 92120D0C
     s_mul_i32     s20, s5, s15                            // 00000000014C: 92140F05
     s_movk_i32    s19, 0x0000                             // 000000000150: B0130000
     s_movk_i32    s21, 0x0000                             // 000000000154: B0150000
     s_add_u32     s18, s18, s20                           // 000000000158: 80121412
     s_addc_u32    s19, s19, s21                           // 00000000015C: 82131513
     s_movk_i32    s17, 0x0000                             // 000000000160: B0110000
     s_add_u32     s16, s16, s18                           // 000000000164: 80101210
     s_addc_u32    s17, s17, s19                           // 000000000168: 82111311
     s_lshr_b64    s[16:17], s[16:17], 16                  // 00000000016C: 8F909010
     s_mul_i32     s5, s5, s13                             // 000000000170: 92050D05
     s_add_u32     s5, s5, s16                             // 000000000174: 80051005
     s_mul_i32     s4, s4, s8                              // 000000000178: 92040804
     v_add_u32     v3, vcc, s4, v0                         // 00000000017C: 32060004
  =>   s_nop         0x0000                                  // 000000000180: BF800000
       s_load_dword  s4, s[6:7], 0x18                        // 000000000184: C0020103 00000018
       s_nop         0x0000                                  // 00000000018C: BF800000
       s_load_dword  s5, s[6:7], 0x40                        // 000000000190: C0020143 00000040
       s_nop         0x0000                                  // 000000000198: BF800000
       s_load_dword  s12, s[6:7], 0x20                       // 00000000019C: C0020303 00000020
       s_nop         0x0000                                  // 0000000001A4: BF800000
       s_load_dword  s13, s[6:7], 0x48                       // 0000000001A8: C0020343 00000048
       s_waitcnt     lgkmcnt(0)                              // 0000000001B0: BF8C007F
       s_nop         0x0000                                  // 0000000001B4: BF800000
       v_add_u32     v9, vcc, s4, v3                         // 0000000001B8: 32120604
       s_nop         0x0000                                  // 0000000001BC: BF800000
       v_add_u32     v13, vcc, s5, v3                        // 0000000001C0: 321A0605
       v_mov_b32     v5, s8                                  // 0000000001C4: 7E0A0208
       s_nop         0x0000                                  // 0000000001C8: BF800000
       v_ashrrev_i32  v10, 31, v9                            // 0000000001CC: 2214129F
   ...
   ...
   Remaining GPU ISA saved to temp_isa

If you want to view the complete ISA for the GPU kernel, the ISA is saved to temp_isa. The temp_isa also provides important information about the kernel such as the number of registers used, compiler flags used and the GPU ISA version.

An alternative method of viewing the ISA for every kernel is the set rocm option to save the ISA to a file whenever a GPU kernel is active. This can be enabled using the set rocm show-isa as shown below.

::

   (ROCm-gdb) set rocm show-isa on

With this option, ROCm-gdb saves the ISA for the active kernel to temp_isa whenever a GPU kernel is active.

How do I view GPU dispatch info?
*********************************
The info command has been extended to info rocm. The info rocm command allows you to view the present state of the GPU dispatch and also allows you to view information about the GPU dispatches that have executed over the lifetime of the application.

  ``(ROCm-gdb) info rocm``

This will print all the possible options for info rocm. The info rocm command allows you to view information about the active dispatch, active work-groups and active work-items on the device. The possible inputs to info rocm are below

::

   info rocm kernels                                   Print all GPU kernel dispatches
   info rocm kernel <kernel_name>                      Print all GPU kernel dispatches with a specific <kernel_name>
   info rocm [work-groups|wgs]                         Print all GPU work-group items
   info rocm [work-group|wg] [<flattened_id>|<x,y,z>]  Print a specific GPU work-group item
   info rocm [work-item|wi|work-items|wis]             Print the focus GPU work-item
   info rocm [work-item|wi] <x,y,z>                    Print a specific GPU work-item

For example, info rocm kernels on an application that dispatches two kernels shows

::

   (ROCm-gdb) info rocm kernels
   Kernels info
   Index                    KernelName  DispatchCount  # of Work-groups  Work-group Dimensions
       0       &__Gdt_vectoradd_kernel              1             1,1,1                 64,1,1
      *1    &__OpenCL_matrixMul_kernel              1             8,5,1                16,16,1

The info rocm work-groups command will show the active work-groups for the active dispatch

::

  (ROCm-gdb) info rocm work-groups
   Index            Work-group ID   Flattened Work-group ID
     *0                    0,0,0                         0
      1                    1,0,0                         1
      2                    2,0,0                         2

The info rocm wg 0 command will show the information of work-group 0 for the active dispatch

:: 

  Information for Work-group 0
  Index     Wave ID {SE,SH,CU,SIMD,Wave}            Work-item ID        Abs Work-item ID        PC            Source line
     0  0x408001c0 { 0, 0, 1,   0,   0}    [0,12, 0 - 15,15, 0]    [0,12, 0 - 15,15, 0]     0x2a8    temp_source@line 64
     1  0x408001d0 { 0, 0, 1,   1,   0}    [0, 4, 0 - 15, 7, 0]    [0, 4, 0 - 15, 7, 0]     0x2a8    temp_source@line 64
     2  0x408001e0 { 0, 0, 1,   2,   0}    [0, 0, 0 - 15, 3, 0]    [0, 0, 0 - 15, 3, 0]     0x2a8    temp_source@line 64
     3  0x408001f0 { 0, 0, 1,   3,   0}    [0, 8, 0 - 15,11, 0]    [0, 8, 0 - 15,11, 0]     0x2a8    temp_source@line 64

Wave ID contains the hardware slot ids where SE is the Shader Engine id, SH is the shader array id, CU is the Compute Unit id, SIMD is the SIMD id, and Wave is the wave slot id.

The info rocm work-item command will show the focus work-item for the active dispatch

::

  (ROCm-gdb) info rocm wi
  Information for Work-item
  Index     Wave ID {SE,SH,CU,SIMD,Wave}            Work-item ID        Abs Work-item ID        PC            Source line
     *0  0x408002d0 { 0, 0, 2,   1,   0}               [0, 0, 0]              [16, 0, 0]      0x68    temp_source@line 150

The info rocm devices command will show the available ROCm devices in the system and the device presently executing a dispatch.

::

  (ROCm-gdb) info rocm  devices
  Devices info
  Index                          Name      ChipID         CUs    Waves/CU  EngineFreq  MemoryFreq
     *0                    AMD gfx803      0x7300          64          40        1000         500

How do I view a trace of GPU dispatches
****************************************
ROCm-gdb helps developers to view information about kernels that have been launched on the GPU using the rocm trace commands. ROCm-gdb can save a trace of all the GPU kernel launches to a Comma Separated Value (CSV) file using the set rocm trace command. The following commands enable tracing GPU kernel launches to mytrace.csv.

::

  (ROCm-gdb) set rocm trace mytrace.csv
  (ROCm-gdb) set rocm trace on

You can now execute and debug the application within ROCm-gdb. Anytime during the application’s execution you can view my_trace.csv to see the kernels have been dispatched. A sample trace for an application that dispatches a vector add kernel followed by a matrix multiplication kernel in a loop is shown below.
		   		&__OpenCL_matrixMul_kernel 	
====== =========== =========== ============================= ======= ======= ================ =========== ========== ====================== 
index 	queue_id    packet_id 	  kernel_name 	              header  setup   workgroup_size   reserved0  grid_size   private_segment_size 
====== =========== =========== ============================= ======= ======= ================ =========== ========== ====================== 
	group_segment_size 	kernel_object 	kernarg_address 	reserved2 	completion_signal
0 	380095252 	0 	&__Gdt_vectoradd_kernel 	5122 	1 	{64 1 1} 	0 	{64 1 1} 	0 	0 	140737353981952 	0x713000 	0 	7513216
1 	380095252 	1 	&__OpenCL_matrixMul_kernel 	5122 	2 	{16 16 1} 	0 	{128 80 1} 	0 	0 	140737353983488 	0x6ca000 	0 	7910848
2 	380095252 	2 	&__Gdt_vectoradd_kernel 	5122 	1 	{64 1 1} 	0 	{64 1 1} 	0 	0 	140737353977856 	0x6e2000 	0 	7858432
3 	380095252 	3 	&__OpenCL_matrixMul_kernel 	5122 	2 	{16 16 1} 	0 	{128 80 1} 	0 	0 	140737353979392 	0x6a3000 	0 	7177152
4 	380095252 	4 	&__Gdt_vectoradd_kernel 	5122 	1 	{64 1 1} 	0 	{64 1 1} 	0 	0 	140737353973760 	0x666000 	0 	7981376
5 	380095252 	5 	&__OpenCL_matrixMul_kernel 	5122 	2 	{16 16 1} 	0 	{128 80 1} 	0 	0 	140737353975296 	0x7a3000 	0 	7192640
6 	380095252 	6 	&__Gdt_vectoradd_kernel 	5122 	1 	{64 1 1} 	0 	{64 1 1} 	0 	0 	140737353969664 	0x7a3000 	0 	7940224
7 	380095252 	7 	&__OpenCL_matrixMul_kernel 	5122 	2 	{16 16 1} 	0 	{128 80 1} 	0 	0 	140737353971200 	0x697000 	0 	7765760
8 	380095252 	8 	&__Gdt_vectoradd_kernel 	5122 	1 	{64 1 1} 	0 	{64 1 1} 	0 	0 	140737353965568 	0x70f000 	0 	6968192
9 	380095252 	9 	&__OpenCL_matrixMul_kernel 	5122 	2 	{16 16 1} 	0 	{128 80 1} 	0 	0 	140737353967104 	0x708000 	0 	7081216

How do I compile GPU kernels for debug?
*****************************************
To debug GPU kernels that target ROCm, you need to compile the kernels for debug and embed the HSAIL kernel source in the resulting code object. Debug flags can be passed to high level compiler and the finalizer using environment variables. To simplify this process, the rocm-gdb-debug-flags.sh script is included in the /opt/rocm/gpudebugsdk/bin directory.

It should be noted that the rocm-gdb-debug-flags.sh should be called as source rocm-gdb-debug-flags.sh and not executed as ./rocm-gdb-debug-flags.sh since the script sets environment variables and the variables need to be visible for the subsequent build commands.

  *  For applications using libHSAIL to compile their GPU kernels source rocm-gdb-debug-flags.sh should be called when the application is   	  compiled.
  *  For SNACK applications, you can call source rocm-gdb-debug-flags.sh before calling the buildrun.sh script for the SNACK applications.

Note that kernel debugging is not yet supported with applications compiled using HCC-LC.

Once the application has been built using the environment variables specified in rocm-gdb-debug-flags.sh, you can debug libHSAIL applications as described in this tutorial.

Generating logs for reporting issues in rocm-gdb
****************************************************
Additional log files can be generated by rocm-gdb. These log files should be sent to the rocm-gdb developers to allow them to diagnose issues. Logging is enabled with the ROCM_GDB_ENABLE_LOG environment variable as shown below

::

   export ROCM_GDB_ENABLE_LOG='DebugLogs'
   rocm-gdb MatrixMul

The environment variable enables logging and provides a prefix for the log file names. As the MatrixMul application executes, log files with the prefix DebugLogs_ will be generated. The log files generated include logs from GDB, the HSA Debug Agent and the HSA code objects used in the applications. Each debug session's log file's name will include a unique SessionID.
Others

A useful tutorial on how to use GDB can be found on `RMS's site <http://www.unknownroad.com/rtfm/gdbtut/>`_.
