#!/bin/bash

doxygen Doxyfile

sed -e 's/ROCBLAS_EXPORT //g' ROCm_Libraries/rocBLAS/src/include/rocblas.h > ROCm_Libraries/rocBLAS/src/rocblas.h
sed -e 's/ROCBLAS_EXPORT //g' ROCm_Libraries/rocBLAS/src/include/rocblas-functions.h > ROCm_Libraries/rocBLAS/src/rocblas-functions.h
sed -e 's/ROCBLAS_EXPORT //g' ROCm_Libraries/rocBLAS/src/include/rocblas-types.h > ROCm_Libraries/rocBLAS/src/rocblas-types.h
sed -e 's/ROCBLAS_EXPORT //g' ROCm_Libraries/rocBLAS/src/include/rocblas-auxiliary.h > ROCm_Libraries/rocBLAS/src/rocblas-auxiliary.h

doxygen ROCm_Libraries/rocBLAS/Doxyfile

doxygen ROCm_Libraries/rocALUTION/Doxyfile

sed -e 's/ROCFFT_EXPORT //g' ROCm_Libraries/rocFFT/src/rocfft.h > ROCm_Libraries/rocFFT/src/rocfft_sed.h

doxygen ROCm_Libraries/rocFFT/Doxyfile

sed -e 's/ROCSPARSE_EXPORT//g' ROCm_Libraries/rocSPARSE/src/rocsparse-functions.h > ROCm_Libraries/rocSPARSE/src/rocsparse-functions_sed1.h
sed -e 's/\\text{if trans == rocsparse_operation_none}/if\\: trans == rocsparse\\_operation\\_none/g; s/\\text{if trans == rocsparse_operation_transpose}/if\\: trans == rocsparse\\_operation\\_transpose/g; s/\\text{if trans == rocsparse_operation_conjugate_transpose}/if\\: trans == rocsparse\\_operation\\_conjugate\\_transpose/g' ROCm_Libraries/rocSPARSE/src/rocsparse-functions_sed1.h > ROCm_Libraries/rocSPARSE/src/rocsparse-functions_sed.h

sed -e 's/ROCSPARSE_EXPORT//g' ROCm_Libraries/rocSPARSE/src/rocsparse-auxiliary.h > ROCm_Libraries/rocSPARSE/src/rocsparse-auxiliary_sed.h
sed -i 's/#include "rocsparse-export.h"//g' rocsparse-functions_sed.h
sed -i 's/#include "rocsparse-export.h"//g' rocsparse-auxiliary_sed.h

doxygen ROCm_Libraries/rocSPARSE/Doxyfile

doxygen ROCm_Libraries/rocr/Doxyfile

