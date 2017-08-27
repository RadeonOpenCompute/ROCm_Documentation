
.. _Other-Solutions:

================
Other Solutions
================


ROCr Error Codes
================


ROCm PCIe Feature and Overview BAR Memory
================

ROCm is an extension of  HSA platform architecture, so it shares the queueing model, memory model, signaling and synchronization protocols. Platform atomics are integral to perform queuing and signaling memory operations where there may be multiple-writers across CPU and GPU agents. 

The full list of HSA system architecture platform requirements are here: http://www.hsafoundation.com/html/HSA_Library.htm#SysArch/Topics/01_Overview/list_of_requirements.htm

For ROCm the Platform atomics are used in ROCm in the following ways:

- Update HSA queue’s read_dispatch_id: 64bit atomic add used by the command processor on the GPU agent to update the packet ID it processed.
- Update HSA queue’s write_dispatch_id: 64bit atomic add used by the CPU and GPU agent to support multi-writer queue insertions.
- Update HSA Signals – 64bit atomic ops are used for CPU & GPU synchronization.

The PCIe Platform Atomics are  CAS, FetchADD, SWAP

Here is document on PCIe Atomics https://pcisig.com/sites/default/files/specification_documents/ECN_Atomic_Ops_080417.pdf

In ROCm, we also take advantage of PCIe ID based ordering technology for P2P when the GPU originates two writes to two different targets:  

1. write to another GPU memory, 
2. then write to system memory to indicate transfer complete. 

They are routed off to different ends of the computer but we want to make sure the write to system memory to indicate transfer complete occurs AFTER P2P write to GPU has complete. 

BAR Memory Overview
================

On a Xeon E5 based system in the BIOS  we can turn on above 4GB PCIe addressing, if so he need to set MMIO Base address ( MMIOH Base) and Range ( MMIO High Size)  in the BIOS.
 
In SuperMicro system in the system bios you need to see the following

- Advanced->PCIe/PCI/PnP configuration-> Above 4G Decoding = Enabled
 
- Advanced->PCIe/PCI/PnP Configuration->MMIOH Base = 512G
 
- Advanced->PCIe/PCI/PnP Configuration->MMIO High Size = 256G
 
When we support Large Bar Capbility there is a  Large Bar Vbios which also disable the IO bar.  


For GFX9 and Vega10 which have Physical Address up 44 bit and 48 bit Virtual address.
 
- BAR0-1 registers: 64bit, prefetchable, GPU memory. 8GB or 16GB depending on Vega10 SKU. Must be placed < 2^44 to support P2P access from other Vega10.
- BAR2-3 registers: 64bit, prefetchable, Doorbell. Must be placed < 2^44 to support P2P access from other Vega10.
- BAR4 register: Optional, not a boot device.
- BAR5 register: 32bit, non-prefetchable, MMIO. Must be placed < 4GB.
 
 
Here is how our BAR works on GFX 8 GPU's with 40 bit Physical Address Limit
 
      11:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Fiji [Radeon R9 FURY / NANO Series] (rev c1)

      Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0b35
        
      Flags: bus master, fast devsel, latency 0, IRQ 119
        
      Memory at bf40000000 (64-bit, prefetchable) [size=256M]
       
      Memory at bf50000000 (64-bit, prefetchable) [size=2M]
       
      I/O ports at 3000 [size=256]
       
      Memory at c7400000 (32-bit, non-prefetchable) [size=256K]
       
      Expansion ROM at c7440000 [disabled] [size=128K]
 
Legend: 

___1___ : GPU Frame Buffer BAR – In this example it happens to be 256M, but typically this will be size of the GPU memory (typically 4GB+). This BAR has to be placed < 2^40 to allow peer-to-peer access from other GFX8 AMD GPUs. For GFX9 (Vega GPU) the BAR has to be placed < 2^44 to allow peer-to-peer access from other GFX9 AMD GPUs. 
 
___2___ : Doorbell BAR – The size of the BAR is typically will be < 10MB (currently fixed at 2MB) for this generation GPUs. This BAR has to be placed < 2^40 to allow peer-to-peer access from other current generation AMD GPUs.
 
___3___ : IO BAR - This is for legacy VGA and boot device support, but since this the GPUs in this project are not VGA devices (headless), this is not a concern even if the SBIOS does not setup. 
 
___4___ : MMIO BAR – This is required for the AMD Driver SW to access the configuration registers. Since the reminder of the BAR available is only 1 DWORD (32bit), this is placed < 4GB. This is fixed at 256KB.
 
___5___ : Expansion ROM – This is required for the AMD Driver SW to access the GPU’s video-bios. This is currently fixed at 128KB.
 
-------------------------------------------------------------------------------------------------
Excepts form Overview of Changes to PCI Express 3.0
================

By Mike Jackson, Senior Staff Architect, MindShare, Inc.

Atomic Operations – Goal: 
================

Support SMP-type operations across a PCIe network to allow for things like offloading tasks between CPU cores and accelerators like a GPU. The spec says this enables advanced synchronization mechanisms that are particularly useful with multiple producers or consumers that need to be synchronized in a non-blocking fashion. Three new atomic non-posted requests were added, plus the corresponding completion (the address must be naturally aligned with the operand size or the TLP is malformed):

- Fetch and Add – uses one operand as the “add” value. Reads the target location, adds the operand, and then writes the result back to the original location.

- Unconditional Swap – uses one operand as the “swap” value. Reads the target location and then writes the swap value to it.

- Compare and Swap – uses 2 operands: first data is compare value, second is swap value. Reads the target location, checks it against the compare value and, if equal, writes the swap value to the target location.

- AtomicOpCompletion – new completion to give the result so far atomic request and indicate that the atomicity of the transaction has been maintained.

Since AtomicOps are not locked they don’t have the performance downsides of the PCI locked protocol. Compared to locked cycles, they provide “lower latency, higher scalability, advanced synchronization algorithms, and dramatically lower impact on other PCIe traffic.” The lock mechanism can still be used across a bridge to PCI or PCI-X to achieve the desired operation.

AtomicOps can go from device to device, device to host, or host to device. Each completer indicates whether it supports this capability and guarantees atomic access if it does. The ability to route AtomicOps is also indicated in the registers for a given port.

ID-based Ordering – Goal: 
================

Improve performance by avoiding stalls caused by ordering rules. For example, posted writes are never normally allowed to pass each other in a queue, but if they are requested by different functions, we can have some confidence that the requests are not dependent on each other. The previously reserved Attribute bit [2] is now combined with the RO bit to indicate ID ordering with or without relaxed ordering. 

This only has meaning for memory requests, and is reserved for Configuration or IO requests. Completers are not required to copy this bit into a completion, and only use the bit if their enable bit is set for this operation.

To read more on PCIe Gen 3 new options http://www.mindshare.com/files/resources/PCIe%203-0.pdf 
-------------------------------------------------------------------------------------------------


