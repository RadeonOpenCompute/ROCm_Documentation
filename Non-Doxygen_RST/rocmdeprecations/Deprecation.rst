

===========================
Deprecations and Warnings 
============================

ROCm Release 5.0
-------------------


ROCM LIBRARIES CHANGES – DEPRECATIONS AND DEPRECATION REMOVAL
===============================================================

* The hipFFT.h header is now provided only by the hipFFT package.  Up to ROCm 5.0, users would get hipFFT.h in the rocFFT package too.
* The GlobalPairwiseAMG class is now entirely removed, users should use the PairwiseAMG class instead.
* The rocsparse_spmm signature in 5.0 was changed to match that of rocsparse_spmm_ex.  In 5.0, rocsparse_spmm_ex is still present, but deprecated.  Signature diff for    rocsparse_spmm

rocsparse_spmm in 5.0

::

          
          rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                          rocsparse_operation         trans_A,
                                          rocsparse_operation         trans_B,
                                          const void*                 alpha,
                                          const rocsparse_spmat_descr mat_A,
                                          const rocsparse_dnmat_descr mat_B,
                                          const void*                 beta,
                                          const rocsparse_dnmat_descr mat_C,
                                          rocsparse_datatype          compute_type,
                                          rocsparse_spmm_alg          alg,
                                          rocsparse_spmm_stage        stage,
                                          size_t*                     buffer_size,
                                          void*                       temp_buffer);
                                          
                                          




rocSPARSE_spmm in 4.0

::


          rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                          rocsparse_operation         trans_A,
                                          rocsparse_operation         trans_B,
                                          const void*                 alpha,
                                          const rocsparse_spmat_descr mat_A,
                                          const rocsparse_dnmat_descr mat_B,
                                          const void*                 beta,
                                          const rocsparse_dnmat_descr mat_C,
                                          rocsparse_datatype          compute_type,
                                          rocsparse_spmm_alg          alg,
                                          size_t*                     buffer_size,
                                          void*                       temp_buffer); 





HIP API DEPRECATIONS AND WARNINGS
====================================

**Warning - Arithmetic Operators of HIP Complex and Vector Types**

In this release, arithmetic operators of HIP complex and vector types are deprecated. 

* As alternatives to arithmetic operators of HIP complex types, users can use arithmetic operators of std::complex types. 

* As alternatives to arithmetic operators of HIP vector types, users can use the operators of the native clang vector type associated with the data member of HIP vector types.

During the deprecation, two macros_HIP_ENABLE_COMPLEX_OPERATORS and_HIP_ENABLE_VECTOR_OPERATORS are provided to allow users to conditionally enable arithmetic operators of HIP complex or vector types. 

Note, the two macros are mutually exclusive and, by default, set to Off. 

The arithmetic operators of HIP complex and vector types will be removed in a future release.

Refer to the HIP API Guide for more information.  



**Refactor of HIPCC/HIPCONFIG**

In prior ROCm releases, by default, the hipcc/hipconfig Perl scripts were used to identify and set target compiler options, target platform, compiler, and runtime appropriately.

In ROCm v5.0, hipcc.bin and hipconfig.bin have been added as the compiled binary implementations of the hipcc and hipconfig. These new binaries are currently a work-in-progress, considered, and marked as experimental. ROCm plans to fully transition to hipcc.bin and hipconfig.bin in the a future ROCm release. The existing hipcc and hipconfig Perl scripts are renamed to hipcc.pl and hipconfig.pl respectively. New top-level hipcc and hipconfig Perl scripts are created, which can switch between the Perl script or the compiled binary based on the environment variable HIPCC_USE_PERL_SCRIPT. 

In ROCm 5.0, by default, this environment variable is set to use hipcc and hipconfig through the Perl scripts.

Subsequently, Perl scripts will no longer be available in ROCm in a future release.


Warning - Compiler-Generated Code Object Version 4 Deprecation
================================================================

Support for loading compiler-generated code object version 4 will be deprecated in a future release with no release announcement and replaced with code object 5 as the default version. 

The current default is code object version 4.


MIOpenTensile
===============

MIOpenTensile is now deprecated in ROCm.


ROCm Release v4.5
-------------------

AMD Instinct MI25 End of Life
===============================

ROCm release v4.5 is the final release to support AMD Instinct MI25. AMD Instinct MI25 has reached End of Life (EOL). ROCm 4.5 represents the last certified release for software and driver support. AMD will continue to provide technical support and issue resolution for AMD Instinct MI25 on ROCm v4.5 for a period of 12 months from the software GA date.

Planned Deprecation of Code Object Versions 2 AND 3
========================================================

With the ROCm v4.5 release, the generation of code object versions 2 and 3 is being deprecated and may be removed in a future release. This deprecation notice does not impact support for the execution of AMD GPU code object versions.

The -mcode-object-version Clang option can be used to instruct the compiler to generate a specific AMD GPU code object version. In ROCm v4.5, the compiler can generate AMD GPU code object version 2, 3, and 4, with version 4 being the default if not specified.


ROCm Release v4.1
--------------------

COMPILER-GENERATED CODE OBJECT VERSION 2 DEPRECATION 
=======================================================

Compiler-generated code object version 2 is no longer supported and has been completely removed. 

Support for loading code object version 2 is also deprecated with no announced removal release.


Changed HIP Environment Variables in ROCm v4.1 Release
=======================================================

In the ROCm v3.5 release, the Heterogeneous Compute Compiler (HCC) compiler was deprecated, and the HIP-Clang compiler was introduced for compiling Heterogeneous-Compute Interface for Portability (HIP) programs. Also, the HIP runtime API was implemented on top of the Radeon Open Compute Common Language runtime (ROCclr). ROCclr is an abstraction layer that provides the ability to interact with different runtime backends such as ROCr. 

While the *HIP_PLATFORM=hcc* environment variable was functional in subsequent releases after ROCm v3.5, in the ROCm v4.1 release, changes to the following environment variables were implemented: 

* *HIP_PLATFORM=hcc was changed to HIP_PLATFORM=amd*

* *HIP_PLATFORM=nvcc was changed to HIP_PLATFORM=nvidia*

Therefore, any applications continuing to use the HIP_PLATFORM=hcc environment variable will fail.

**Workaround:**  Update the environment variables to reflect the changes mentioned above.



ROCm Release v4.0
--------------------

ROCr Runtime Deprecations
===========================

The following ROCr Runtime enumerations, functions, and structs are deprecated in the AMD ROCm v4.0 release.

Deprecated ROCr Runtime Functions

* hsa_isa_get_info

* hsa_isa_compatible

* hsa_executable_create

* hsa_executable_get_symbol

* hsa_executable_iterate_symbols

* hsa_code_object_serialize

* hsa_code_object_deserialize

* hsa_code_object_destroy

* hsa_code_object_get_info

* hsa_executable_load_code_object

* hsa_code_object_get_symbol

* hsa_code_object_get_symbol_from_name

* hsa_code_symbol_get_info

* hsa_code_object_iterate_symbols


Deprecated ROCr Runtime Enumerations
=======================================

* HSA_ISA_INFO_CALL_CONVENTION_COUNT

* HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE

* HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT

* HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH

* HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME

* HSA_EXECUTABLE_SYMBOL_INFO_AGENT

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE

* HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST

* HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION

* HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION

      * hsa_code_object_type_t
      
      * hsa_code_object_info_t
      
      * hsa_code_symbol_info_t


Deprecated ROCr Runtime Structs
================================

* hsa_code_object_t

* hsa_callback_data_t

* hsa_code_symbol_t


AOMP DEPRECATION
===================

As of AMD ROCm v4.0, AOMP (aomp-amdgpu) is deprecated. OpenMP support has moved to the openmp-extras auxiliary package, which leverages the ROCm compiler on LLVM 12.

For more information, refer to 

https://rocmdocs.amd.com/en/latest/Programming_Guides/openmp_support.html



ROCm Release v3.5
--------------------

Heterogeneous Compute Compiler
==================================

In the ROCm v3.5 release, the Heterogeneous Compute Compiler (HCC) compiler was deprecated and the HIP-Clang compiler was introduced for compiling Heterogeneous-Compute Interface for Portability (HIP) programs.

For more information, download the HIP Programming Guide at:

https://github.com/RadeonOpenCompute/ROCm
