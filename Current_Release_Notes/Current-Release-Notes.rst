.. image:: /Current_Release_Notes/amdblack.jpg

|

================================
AMD ROCm™ Release Notes v3.7.0
================================
August, 2020

This page describes the features, fixed issues, and information about downloading and installing the ROCm software. It also covers known issues in the ROCm v3.7.0 release.

`Download AMD ROCm v3.7.0 Release Notes PDF <https://github.com/RadeonOpenCompute/ROCm>`__



-  `Supported Operating Systems and Documentation
   Updates <#Supported-Operating-Systems-and-Documentation-Updates>`__

   -  `Supported Operating Systems <#Supported-Operating-Systems>`__
   -  `AMD ROCm Documentation
      Updates <#AMD-ROCm-Documentation-Updates>`__

-  `What's New in This Release <#Whats-New-in-This-Release>`__

   -  `AOMP Enhancements <#AOMP-Enhancements>`__
   -  `Compatibility with NVIDIA Communications Collective Library v2.7
      API <#Compatibility-with-NVIDIA-Communications-Collective-Library-v27-API>`__
   -  `Singular Value Decomposition of Bi-diagonal
      Matrices <#Singular-Value-Decomposition-of-Bi-diagonal-Matrices>`__
   -  `rocSPARSE_gemmi() Operations for Sparse
      Matrices <#rocSPARSE_gemmi-Operations-for-Sparse-Matrices>`__

-  `Known Issues <#Known-Issues>`__


Supported Operating Systems
===========================

.. _supported-operating-systems-1:

Supported Operating Systems
---------------------------

The AMD ROCm v3.7.x platform is designed to support the following
operating systems:

-  Ubuntu 20.04 and 18.04.4 (Kernel 5.3)
-  CentOS 7.8 & RHEL 7.8 (Kernel 3.10.0-1127) (Using devtoolset-7
   runtime support)
-  CentOS 8.2 & RHEL 8.2 (Kernel 4.18.0 ) (devtoolset is not required)
-  SLES 15 SP1

Fresh Installation of AMD ROCm v3.7 Recommended
-----------------------------------------------

A fresh and clean installation of AMD ROCm v3.7 is recommended. An
upgrade from previous releases to AMD ROCm v3.7 is not supported.

For more information, refer to the AMD ROCm Installation Guide at:
https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

**Note**: AMD ROCm release v3.3 or prior releases are not fully
compatible with AMD ROCm v3.5 and higher versions. You must perform a
fresh ROCm installation if you want to upgrade from AMD ROCm v3.3 or
older to 3.5 or higher versions and vice-versa.

AMD ROCm Documentation Updates
==============================

AMD ROCm Installation Guide
---------------------------

The AMD ROCm Installation Guide in this release includes:

-  Updated Supported Environments
-  HIP Installation Instructions

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

AMD ROCm - HIP Documentation Updates
------------------------------------

Texture and Surface Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation for Texture and Surface functions is updated and
available at:

https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html

Warp Shuffle Functions
~~~~~~~~~~~~~~~~~~~~~~

The documentation for Warp Shuffle functions is updated and available
at:

https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html

Compiler Defines and Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation for the updated HIP Porting Guide is available at:

https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html#hip-porting-guide

AMD ROCm Debug Agent
--------------------

ROCm Debug Agent Library

https://rocmdocs.amd.com/en/latest/ROCm_Tools/rocm-debug-agent.html

General AMD ROCm Documentatin Links
-----------------------------------

Access the following links for more information:

-  For AMD ROCm documentation, see

   https://rocmdocs.amd.com/en/latest/

-  For installation instructions on supped platforms, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

-  For AMD ROCm binary structure, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#build-amd-rocm

-  For AMD ROCm Release History, see

   https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#amd-rocm-version-history

What's New in This Release
==========================

AOMP ENHANCEMENTS
-----------------

AOMP is a scripted build of LLVM. It supports OpenMP target offload on AMD GPUs. Since AOMP is a Clang/LLVM compiler, it also supports GPU offloading with HIP, CUDA, and OpenCL.

The following enhancements are made for AOMP in this release: 

•	OpenMP 5.0 is enabled by default. You can use -fopenmp-version=45 for OpenMP 4.5 compliance

•	Restructured to include the ROCm compiler

•	B=Bitcode search path using hip policy HIP_DEVICE_LIB_PATH and hip-devic-lib command line option to enable global_free for kmpc_impl_free

Restructured hostrpc, including:

•	Replaced hostcall register functions with handlePayload(service, payload). Note, handlPayload has a simple switch to call the correct service handler function.

•	Removed the WITH_HSA macro

•	Moved the hostrpc stubs and host fallback functions into a single library and the include file. This enables the stubs openmp cpp source instead of hip and reorganizes the directory openmp/libomptarget/hostrpc.

•	Moved hostrpc_invoke.cl to DeviceRTLs/amdgcn.

•	Generalized the vargs processing in printf to work for any vargs function to execute on the host, including a vargs function that uses a function pointer.

•	Reorganized files, added global_allocate and global_free.

•	Fixed llvm TypeID enum to match the current upstream llvm TypeID.

•	Moved strlen_max function inside the declare target #ifdef _DEVICE_GPU in hostrpc.cpp to resolve linker failure seen in pfspecifier_str smoke test.

•	Fixed AOMP_GIT_CHECK_BRANCH in aomp_common_vars to not block builds in Red Hat if the repository is on a specific commit hash.

•	Simplified and reduced the size of openmp host runtime.

•	Switched to default OpenMP 5.0

For more information, see https://github.com/ROCm-Developer-Tools/aomp


ROCm COMMUNICATIONS COLLECTIVE LIBRARY
--------------------------------------

Compatibility with NVIDIA Communications Collective Library v2.7 API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROCm Communications Collective Library (RCCL) is now compatible with the NVIDIA Communications Collective Library (NCCL) v2.7 API.

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is also initial support for direct GPU-to-GPU send and receive operations. It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.
The collective operations are implemented using ring and tree algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

For more information about RCCL APIs and compatibility with NCCL v2.7, see
https://rccl.readthedocs.io/en/develop/index.html


Singular Value Decomposition of Bi-diagonal Matrices
----------------------------------------------------

Rocsolver_bdsqr now computes the Singular Value Decomposition (SVD) of bi-diagonal matrices. It is an auxiliary function for the SVD of general matrices (function rocsolver_gesvd). 

BDSQR computes the singular value decomposition (SVD) of a n-by-n bidiagonal matrix B.

The SVD of B has the following form:
B = Ub * S * Vb'
where 
•	S is the n-by-n diagonal matrix of singular values of B
•	the columns of Ub are the left singular vectors of B
•	the columns of Vb are its right singular vectors

The computation of the singular vectors is optional; this function accepts input matrices U (of size nu-by-n) and V (of size n-by-nv) that are overwritten with U*Ub and Vb’*V. If nu = 0 no left vectors are computed; if nv = 0 no right vectors are computed.

Optionally, this function can also compute Ub’*C for a given n-by-nc input matrix C.

PARAMETERS

•	[in] handle: rocblas_handle.
•	[in] uplo: rocblas_fill.
Specifies whether B is upper or lower bidiagonal.

•	[in] n: rocblas_int. n >= 0.
The number of rows and columns of matrix B.

•	[in] nv: rocblas_int. nv >= 0.
The number of columns of matrix V.

•	[in] nu: rocblas_int. nu >= 0.
The number of rows of matrix U.

•	[in] nc: rocblas_int. nu >= 0.
The number of columns of matrix C.

•	[inout] D: pointer to real type. Array on the GPU of dimension n.
On entry, the diagonal elements of B. On exit, if info = 0, the singular values of B in decreasing order; if info > 0, the diagonal elements of a bidiagonal matrix orthogonally equivalent to B.

•	[inout] E: pointer to real type. Array on the GPU of dimension n-1.
On entry, the off-diagonal elements of B. On exit, if info > 0, the off-diagonal elements of a bidiagonal matrix orthogonally equivalent to B (if info = 0 this matrix converges to zero).

•	[inout] V: pointer to type. Array on the GPU of dimension ldv*nv.
On entry, the matrix V. On exit, it is overwritten with Vb’*V. (Not referenced if nv = 0).

•	[in] ldv: rocblas_int. ldv >= n if nv > 0, or ldv >=1 if nv = 0.
Specifies the leading dimension of V.

•	[inout] U: pointer to type. Array on the GPU of dimension ldu*n.
On entry, the matrix U. On exit, it is overwritten with U*Ub. (Not referenced if nu = 0).

•	[in] ldu: rocblas_int. ldu >= nu.
Specifies the leading dimension of U.

•	[inout] C: pointer to type. Array on the GPU of dimension ldc*nc.
On entry, the matrix C. On exit, it is overwritten with Ub’*C. (Not referenced if nc = 0).

•	[in] ldc: rocblas_int. ldc >= n if nc > 0, or ldc >=1 if nc = 0.
Specifies the leading dimension of C.

•	[out] info: pointer to a rocblas_int on the GPU.
If info = 0, successful exit. If info = i > 0, i elements of E have not converged to zero.


For more information, see
https://rocsolver.readthedocs.io/en/latest/userguide_api.html#rocsolver-type-bdsqr


rocSPARSE_gemmi() Operations for Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This enhancement provides a dense matrix sparse matrix multiplication using the CSR storage format.
rocsparse_gemmi multiplies the scalar αα with a dense m×km×k matrix AA and the sparse k×nk×n matrix BB defined in the CSR storage format, and adds the result to the dense m×nm×n matrix CC that is multiplied by the scalar ββ, such that

C:=α⋅op(A)⋅op(B)+β⋅CC:=α⋅op(A)⋅op(B)+β⋅C

with

op(A)=⎧⎩⎨⎪⎪A,AT,AH,if trans_A == rocsparse_operation_noneif trans_A == rocsparse_operation_transposeif trans_A == rocsparse_operation_conjugate_transposeop(A)={A,if trans_A == rocsparse_operation_noneAT,if trans_A == rocsparse_operation_transposeAH,if trans_A == rocsparse_operation_conjugate_transpose

and

op(B)=⎧⎩⎨⎪⎪B,BT,BH,if trans_B == rocsparse_operation_noneif trans_B == rocsparse_operation_transposeif trans_B == rocsparse_operation_conjugate_transposeop(B)={B,if trans_B == rocsparse_operation_noneBT,if trans_B == rocsparse_operation_transposeBH,if trans_B == rocsparse_operation_conjugate_transpose
Note: This function is non-blocking and executed asynchronously with the host. It may return before the actual computation has finished.

For more information and examples, see
https://rocsparse.readthedocs.io/en/master/usermanual.html#rocsparse-gemmi


Known Issues
============

The following are the known issues in this release.

(AOMP) '˜Undefined Hidden Symbol' Linker Error Causes Compilation Failure in HIP
----------------------------------------------------------------------------------

The HIP example device_lib fails to compile due to unreferenced symbols
with Link Time Optimization resulting in '˜undefined hidden symbol'
errors.

This issue is under investigation and there is no known workaround at
this time.

MIGraphX Fails for fp16 Datatype
--------------------------------

The MIGraphX functionality does not work for the fp16 datatype.

The following workaround is recommended:

Use the AMD ROCm v3.3 of MIGraphX

Or

Build MIGraphX v3.7 from the source using AMD ROCm v3.3

Missing Google Test Installation May Cause RCCL Unit Test Compilation Failure
-----------------------------------------------------------------------------

Users of the RCCL install.sh script may encounter an RCCL unit test
compilation error. It is recommended to use CMAKE directly instead of
install.sh to compile RCCL. Ensure Google Test 1.10+ is available in the
CMAKE search path.

As a workaround, use the latest RCCL from the GitHub development branch
at: https://github.com/ROCmSoftwarePlatform/rccl/pull/237

Issue with Peer-to-Peer Transfers
---------------------------------

Using peer-to-peer (P2P) transfers on systems without the hardware P2P
assistance may produce incorrect results.

Ensure the hardware supports peer-to-peer transfers and enable the
peer-to-peer setting in the hardware to resolve this issue.

Partial Loss of Tracing Events for Large Applications
-----------------------------------------------------

An internal tracing buffer allocation issue can cause a partial loss of
some tracing events for large applications.

As a workaround, rebuild the roctracer/rocprofiler libraries from the
GitHub ˜roc-3.7" branch at: 

https://github.com/ROCm-Developer-Tools/rocprofiler â€¢
https://github.com/ROCm-Developer-Tools/roctracer

GPU Kernel C++ Names Not Demangled
----------------------------------

GPU kernel C++ names in the profiling traces and stats produced by ‘—hsa-trace’ option are not demangled.
As a workaround, users may choose to demangle the GPU kernel C++ names as required.

As a workaround, users may choose to demangle the GPU kernel C++ names
as required.

‘rocprof’ option ‘--parallel-kernels’ Not Supported in This Release
----------------------------------------------------------------------

‘rocprof’ option ‘--parallel-kernels’ is available in the options list, however,  it is not fully validated and supported in this release.


Deploying ROCm
==============

AMD hosts both Debian and RPM repositories for the ROCm v3.7.x packages.

For more information on ROCM installation on all platforms, see

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html




DISCLAIMER 
===========
The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information.  Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.  No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document.  Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. S
AMD, the AMD Arrow logo, Radeon, Ryzen, Epyc, and combinations thereof are trademarks of Advanced Micro Devices, Inc.  
Google®  is a registered trademark of Google LLC.
PCIe® is a registered trademark of PCI-SIG Corporation.
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
Ubuntu and the Ubuntu logo are registered trademarks of Canonical Ltd.
Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

