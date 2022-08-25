
.. image:: amdblack.jpg

==========================================================
AMD Instinct™ High Performance Computing and Tuning Guide
==========================================================

HPC workloads have unique requirements. The default hardware and BIOS configurations for OEM platforms may not provide optimal performance for HPC workloads. To help enable optimal HPC settings on a per-platform and workload level, this guide calls out:

- BIOS settings that can impact performance 

- hardware configuration best practices

- supported versions of operating systems

- workload-specific recommendations for optimal BIOS and operating system settings

There is also a discussion on the AMD Instinct™ software development environment, including information on how to install and run the DGEMM and STREAM benchmarks as well as GROMACS. This guidance provides a good starting point but is not exhaustively tested across all compilers.

Prerequisites to understanding this document and to perform tuning of HPC applications include:

- Experience configuring servers 

- Administrative access to the Server's Management Interface (BMC)

- Administrative access to the operating system 

- Familiarity with OEMs Server's Management Interface (BMC) is strongly recommended

- Familiarity with the OS specific tools for configuration, monitoring and troubleshooting is strongly recommended

This document provides guidance on tuning systems with AMD Instinct™ accelerators for High Performance Computing (HPC) workloads. This document is not an all-inclusive guide, and some items referred to may have similar, but different, names in various OEM systems (for example, OEM-specific BIOS settings). This document also provides suggestions on items that should be the initial focus of additional, application-specific tuning. 

This document is based on the AMD EPYC™ 7002 series processor family (former codename “Rome”). One can expect very similar results for the AMD EYPC™ 7003 series processor family (former codename “Milan”). Specific differences in the configuration options or performance obtained will be explicitly called out through the document where needed.

While this guide is a good starting point, developers are encouraged to perform their own performance testing for additional tuning.

For more details, refer to the `AMD Instinct™ High Performance Computing and Tuning Guide <https://github.com/RadeonOpenCompute/ROCm/blob/roc-4.5.x/AMD%20Instinct%E2%84%A2High%20Performance%20Computing%20and%20Tuning%20Guide.pdf>`__





