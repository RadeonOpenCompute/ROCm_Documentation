.. _PCIe-Debug:

ROCm PCIe Debug
=================

lspci helpfull options to help you debug ROCm install issue 
**************************************************************

**To find if the Linux Kerenl is seeing your GPU and to get the the slot number of the device part number you want to look at**

::

  ~$ lspci |grep ATI
  06:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
  23:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
  43:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
  63:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860


**Show Device Slot** 

lspci -s _slot number_

::

  ~$ lspci -s 43:00.0
  43:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860


**If you want to see the capabilites of the device**

lspci -vs _slot number_

Example

::

  ~$ sudo lspci -vs 63:00.0
  [sudo] password for rocm: 
  63:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860 (prog-if 00 [VGA controller])
  	 Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0c35
	 Flags: bus master, fast devsel, latency 0, IRQ 412
	 Memory at 16ff0000000 (64-bit, prefetchable) [size=256M]
	 Memory at 17000000000 (64-bit, prefetchable) [size=2M]
	 I/O ports at f000 [size=256]
	 Memory at e7100000 (32-bit, non-prefetchable) [size=512K]
	 Expansion ROM at e7180000 [disabled] [size=128K]
	 Capabilities: [48] Vendor Specific Information: Len=08 <?>
	 Capabilities: [50] Power Management version 3
	 Capabilities: [64] Express Legacy Endpoint, MSI 00
	 Capabilities: [a0] MSI: Enable+ Count=1/1 Maskable- 64bit+
	 Capabilities: [100] Vendor Specific Information: ID=0001 Rev=1 Len=010 <?>
	 Capabilities: [150] Advanced Error Reporting
	 Capabilities: [200] #15
	 Capabilities: [270] #19
	 Capabilities: [2a0] Access Control Services
	 Capabilities: [2b0] Address Translation Service (ATS)
	 Capabilities: [2c0] #13
	 Capabilities: [2d0] #1b
	 Capabilities: [320] Latency Tolerance Reporting
	 Kernel driver in use: amdgpu
	 Kernel modules: amdgpu


**Display Vendor and Device Codes and numbers** 

lspci -nvmms _slot number_

::

   ~$ lspci -nvmms 43:00.0
   Slot:	43:00.0
   Class:	0300
   Vendor:	1002
   Device:	6860
   SVendor:	1002
   SDevice:	0c35 

  
**To show kernel module running on device** 
 
 lspci -ks _slot number_

::

   ~$ lspci -ks 63:00.0
   63:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
	Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0c35
	Kernel driver in use: amdgpu
	Kernel modules: amdgpu

**When you need more information on the device** 

sudo lspci -vvvs _slot number_

Example 

::

  ~$ sudo lspci -vvvs 43:00.0
  43:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 6860 (prog-if 00 [VGA controller])
	Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0c35
	Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
	Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
	Latency: 0, Cache Line Size: 64 bytes
	Interrupt: pin A routed to IRQ 411
	Region 0: Memory at 19ff0000000 (64-bit, prefetchable) [size=256M]
	Region 2: Memory at 1a000000000 (64-bit, prefetchable) [size=2M]
	Region 4: I/O ports at b000 [size=256]
	Region 5: Memory at e9700000 (32-bit, non-prefetchable) [size=512K]
	Expansion ROM at e9780000 [disabled] [size=128K]
	Capabilities: [48] Vendor Specific Information: Len=08 <?>
	Capabilities: [50] Power Management version 3
		Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0-,D1+,D2+,D3hot+,D3cold+)
		Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
	Capabilities: [64] Express (v2) Legacy Endpoint, MSI 00
		DevCap:	MaxPayload 256 bytes, PhantFunc 0, Latency L0s <4us, L1 unlimited
			ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset-
		DevCtl:	Report errors: Correctable- Non-Fatal- Fatal- Unsupported-
			RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+
			MaxPayload 256 bytes, MaxReadReq 512 bytes
		DevSta:	CorrErr- UncorrErr- FatalErr- UnsuppReq- AuxPwr- TransPend-
		LnkCap:	Port #0, Speed 8GT/s, Width x16, ASPM L0s L1, Exit Latency L0s <64ns, L1 <1us
			ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
		LnkCtl:	ASPM L0s L1 Enabled; RCB 64 bytes Disabled- CommClk+
			ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
		LnkSta:	Speed 8GT/s, Width x16, TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
		DevCap2: Completion Timeout: Not Supported, TimeoutDis-, LTR+, OBFF Not Supported
		DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled
		LnkCtl2: Target Link Speed: 8GT/s, EnterCompliance- SpeedDis-
			 Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
			 Compliance De-emphasis: -6dB
		LnkSta2: Current De-emphasis Level: -3.5dB, EqualizationComplete+, EqualizationPhase1+
			 EqualizationPhase2+, EqualizationPhase3+, LinkEqualizationRequest-
	Capabilities: [a0] MSI: Enable+ Count=1/1 Maskable- 64bit+
		Address: 00000000fee20000  Data: 4021
	Capabilities: [100 v1] Vendor Specific Information: ID=0001 Rev=1 Len=010 <?>
	Capabilities: [150 v2] Advanced Error Reporting
		UESta:	DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
		UEMsk:	DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
		UESvrt:	DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
		CESta:	RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
		CEMsk:	RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr+
		AERCap:	First Error Pointer: 00, GenCap+ CGenEn- ChkCap+ ChkEn-
	Capabilities: [200 v1] #15
	Capabilities: [270 v1] #19
	Capabilities: [2a0 v1] Access Control Services
		ACSCap:	SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
		ACSCtl:	SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
	Capabilities: [2b0 v1] Address Translation Service (ATS)
		ATSCap:	Invalidate Queue Depth: 00
		ATSCtl:	Enable-, Smallest Translation Unit: 00
	Capabilities: [2c0 v1] #13
	Capabilities: [2d0 v1] #1b
	Capabilities: [320 v1] Latency Tolerance Reporting
		Max snoop latency: 0ns
		Max no snoop latency: 0ns
	Kernel driver in use: amdgpu
	Kernel modules: amdgpu

  
**To print PCIe root tree**
 
::

   ~$ lspci -tv
 -+-[0000:60]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1450
  |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-01.1-[61-63]----00.0-[62-63]----00.0-[63]----00.0  Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
  |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-03.1-[64]--+-00.0  Mellanox Technologies Device 1019
  |           |            \-00.1  Mellanox Technologies Device 1019
  |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-07.1-[65]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 145a
  |           |            \-00.2  Advanced Micro Devices, Inc. [AMD] Device 1456
  |           +-08.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           \-08.1-[66]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1455
  |                        \-00.1  Advanced Micro Devices, Inc. [AMD] Device 1468
  +-[0000:40]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1450
  |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-03.1-[41-43]----00.0-[42-43]----00.0-[43]----00.0  Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
  |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-07.1-[44]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 145a
  |           |            \-00.2  Advanced Micro Devices, Inc. [AMD] Device 1456
  |           +-08.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           \-08.1-[45]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1455
  |                        +-00.1  Advanced Micro Devices, Inc. [AMD] Device 1468
  |                        \-00.2  Advanced Micro Devices, Inc. [AMD] FCH SATA Controller [AHCI mode]
  +-[0000:20]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1450
  |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-01.1-[21-23]----00.0-[22-23]----00.0-[23]----00.0  Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
  |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           +-07.1-[24]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 145a
  |           |            +-00.2  Advanced Micro Devices, Inc. [AMD] Device 1456
  |           |            \-00.3  Advanced Micro Devices, Inc. [AMD] Device 145f
  |           +-08.0  Advanced Micro Devices, Inc. [AMD] Device 1452
  |           \-08.1-[25]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1455
  |                        \-00.1  Advanced Micro Devices, Inc. [AMD] Device 1468
  \-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1450
             +-01.0  Advanced Micro Devices, Inc. [AMD] Device 1452
             +-01.2-[01-02]----00.0-[02]----00.0  ASPEED Technology, Inc. ASPEED Graphics Family
             +-01.3-[03]----00.0  Device 1987:5007
             +-02.0  Advanced Micro Devices, Inc. [AMD] Device 1452
             +-03.0  Advanced Micro Devices, Inc. [AMD] Device 1452
             +-03.1-[04-06]----00.0-[05-06]----00.0-[06]----00.0  Advanced Micro Devices, Inc. [AMD/ATI] Device 6860
             +-04.0  Advanced Micro Devices, Inc. [AMD] Device 1452
             +-07.0  Advanced Micro Devices, Inc. [AMD] Device 1452
             +-07.1-[07]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 145a
             |            +-00.2  Advanced Micro Devices, Inc. [AMD] Device 1456
             |            \-00.3  Advanced Micro Devices, Inc. [AMD] Device 145f
             +-08.0  Advanced Micro Devices, Inc. [AMD] Device 1452
             +-08.1-[08]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 1455
             |            \-00.1  Advanced Micro Devices, Inc. [AMD] Device 1468
             +-14.0  Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller
             +-14.3  Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge
             +-18.0  Advanced Micro Devices, Inc. [AMD] Device 1460
             +-18.1  Advanced Micro Devices, Inc. [AMD] Device 1461
             +-18.2  Advanced Micro Devices, Inc. [AMD] Device 1462
             +-18.3  Advanced Micro Devices, Inc. [AMD] Device 1463
             +-18.4  Advanced Micro Devices, Inc. [AMD] Device 1464
             +-18.5  Advanced Micro Devices, Inc. [AMD] Device 1465
             +-18.6  Advanced Micro Devices, Inc. [AMD] Device 1466
             +-18.7  Advanced Micro Devices, Inc. [AMD] Device 1467
             +-19.0  Advanced Micro Devices, Inc. [AMD] Device 1460
             +-19.1  Advanced Micro Devices, Inc. [AMD] Device 1461
             +-19.2  Advanced Micro Devices, Inc. [AMD] Device 1462
             +-19.3  Advanced Micro Devices, Inc. [AMD] Device 1463
             +-19.4  Advanced Micro Devices, Inc. [AMD] Device 1464
             +-19.5  Advanced Micro Devices, Inc. [AMD] Device 1465
             +-19.6  Advanced Micro Devices, Inc. [AMD] Device 1466
             +-19.7  Advanced Micro Devices, Inc. [AMD] Device 1467
             +-1a.0  Advanced Micro Devices, Inc. [AMD] Device 1460
             +-1a.1  Advanced Micro Devices, Inc. [AMD] Device 1461
             +-1a.2  Advanced Micro Devices, Inc. [AMD] Device 1462
             +-1.3  Advanced Micro Devices, Inc. [AMD] Device 1463
             +-1a.4  Advanced Micro Devices, Inc. [AMD] Device 1464
             +-1a.5  Advanced Micro Devices, Inc. [AMD] Device 1465
             +-1a.6  Advanced Micro Devices, Inc. [AMD] Device 1466
             +-1a.7  Advanced Micro Devices, Inc. [AMD] Device 1467
             +-1b.0  Advanced Micro Devices, Inc. [AMD] Device 1460
             +-1b.1  Advanced Micro Devices, Inc. [AMD] Device 1461
             +-1b.2  Advanced Micro Devices, Inc. [AMD] Device 1462
             +-1b.3  Advanced Micro Devices, Inc. [AMD] Device 1463
             +-1b.4  Advanced Micro Devices, Inc. [AMD] Device 1464
             +-1b.5  Advanced Micro Devices, Inc. [AMD] Device 1465
             +-1b.6  Advanced Micro Devices, Inc. [AMD] Device 1466
             \-1b.7  Advanced Micro Devices, Inc. [AMD] Device 1467



