KFD Topology
==============

Application software needs to understand the properties of the underlying hardware to leverage the performance capabilities of the platform for feature utilization and task scheduling. The sysfs topology exposes this information in a loosely hierarchal order. The information is populated by the KFD driver is gathered from ACPI (CRAT) and AMDGPU base driver.

| The sysfs topology is arranged hierarchically as following. The root directory of the topology is 
| **/sys/devices/virtual/kfd/kfd/topology/nodes/**

Based on the platform inside this directory there will be sub-directories corresponding to each HSA Agent. A system with N HSA Agents will have N directories as shown below.

| /sys/devices/virtual/kfd/kfd/topology/nodes/0/
| /sys/devices/virtual/kfd/kfd/topology/nodes/1/
| .
| .
| /sys/devices/virtual/kfd/kfd/topology/nodes/N-1/

HSA Agent Information
**********************
The HSA Agent directory and the sub-directories inside that contains all the information about that agent. The following are the main information available.

Node Information
******************
This is available in the root directory of the HSA agent. This provides information about the compute capabilities of the agent which includes number of cores or compute units, SIMD count and clock speed.

Memory
********
The memory bank information attached to this agent is populated in “mem_banks” subdirectory.
/sys/devices/virtual/kfd/kfd/topology/nodes/N/mem_banks

Cache
*******
The caches available for this agent is populated in “cache” subdirectory
/sys/devices/virtual/kfd/kfd/topology/nodes/N/cache

IO-LINKS
**********
The IO links provides HSA agent interconnect information with latency (cost) between agents. This is useful for peer-to-peer transfers.


How to use topology information
*********************************
The information provided in sysfs should not be directly used by application software. Application software should always use Thunk library API (libhsakmt) to access topology information. Please refer to Thunk API for more information.

The data are associated with a node ID, forming a per-node element list which references the elements contained at relative offsets within that list. A node associates with a kernel agent or agent. Node ID’s should be 0-based, with the “0” ID representing the primary elements of the system (e.g., “boot cores”, memory) if applicable. The enumeration order and—if applicable—values of the ID should match other information reported through mechanisms outside of the scope of the requirements;

For example, the data and enumeration order contained in the ACPI SRAT table on some systems should match the memory order and properties reported through HSA. Further detail is out of the scope of the System Architecture and outlined in the Runtime API specification.

.. image:: simple_platform.png

Each of these nodes is interconnected with other nodes in more advanced systems to the level necessary to adequately describe the topology.

.. image:: More_advanced_topology.png

Where applicable, the node grouping of physical memory follows NUMA principles to leverage memory locality in software when multiple physical memory blocks are available in the system and agents have a different “access cost” (e.g., bandwidth/latency) to that memory.

**KFD Topology structure for AMDGPU :**

| :ref:`sysfsclasskfd`
| :ref:`sysfsclasskfdtopology`
| :ref:`sysfsclasskfdtopologynodes0`
| :ref:`sysfsclasskfdtopologynodes0iolinks01`
| :ref:`sysfsclasskfdtopologynodes0membanks0`
| sysfs-class-kfd-topology-nodes-N-caches

