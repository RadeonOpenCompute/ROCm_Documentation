

ROCm™ Data Center Tool
-----------------------

The ROCm™ Data Center Tool™ simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter environments. The main features are:

- GPU telemetry 

- GPU statistics for jobs

- Integration with third-party tools

- Open source

The tool can be used in stand-alone mode if all components are installed. However, the existing management tools can use the same set of features available in a library format. 

Refer Starting RDC in the ROCm Data Center Tool User Guide for details on different modes of operation.

Objective
============

This user guide is intended to:

•	Provide an overview of the ROCm Data Center Tool features
•	Describe how system administrators and Data Center (or HPC) users can administer and configure AMD GPUs
•	Describe the components 
•	Provide an overview of the open source developer handbook
1.1.2	Terminology
Term	Description
RDC	ROCmTM Data Center Tool
Compute node (CN)	One of many nodes containing one or more GPUs in the Data Center on which compute jobs are run
Management node (MN) or Main console	A machine running system administration applications to administer and manage the Data Center
GPU Groups	Logical grouping of one or more GPUs in a compute node
Fields	A metric that can be monitored by the RDC, such as GPU temperature, memory usage, and power usage

Field Groups	Logical grouping of multiple fields
Job	A workload that is submitted to one or more compute nodes
1.1.3	Target Audience
 The audience for the AMD ROCm Data Center™ tool consists of: 
•	Administrators: The tool will provide cluster administrator with the capability of monitoring, validating, and configuring policies. 
•	HPC Users: Provides GPU centric feedback for their workload submissions
•	OEM: Add GPU information to their existing cluster management software
•	Open Source Contributors: RDC is open source and will accept contributions from the community






https://github.com/RadeonOpenCompute/ROCm/blob/master/AMD_ROCm_DataCenter_Tool_User_Guide_v4.1.pdf
