.. image:: amdblack.jpg



===============
Deprecations
===============

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
