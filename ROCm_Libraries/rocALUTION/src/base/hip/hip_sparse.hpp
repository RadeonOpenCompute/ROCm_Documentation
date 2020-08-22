/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_SPARSE_HPP_
#define ROCALUTION_HIP_HIP_SPARSE_HPP_

#include <rocsparse.h>

namespace rocalution {

// rocsparse csrmv analysis
template <typename ValueType>
rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          int m,
                                          int n,
                                          int nnz,
                                          const rocsparse_mat_descr descr,
                                          const ValueType* csr_val,
                                          const int* csr_row_ptr,
                                          const int* csr_col_ind,
                                          rocsparse_mat_info info);

// rocsparse csrmv
template <typename ValueType>
rocsparse_status rocsparseTcsrmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 int nnz,
                                 const ValueType* alpha,
                                 const rocsparse_mat_descr descr,
                                 const ValueType* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const ValueType* x,
                                 const ValueType* beta,
                                 ValueType* y);

// rocsparse csrsv buffer size
template <typename ValueType>
rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             int m,
                                             int nnz,
                                             const rocsparse_mat_descr descr,
                                             const ValueType* csr_val,
                                             const int* csr_row_ptr,
                                             const int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             size_t* buffer_size);

// rocsparse csrsv analysis
template <typename ValueType>
rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          int m,
                                          int nnz,
                                          const rocsparse_mat_descr descr,
                                          const ValueType* csr_val,
                                          const int* csr_row_ptr,
                                          const int* csr_col_ind,
                                          rocsparse_mat_info info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy solve,
                                          void* temp_buffer);

// rocsparse csrsv
template <typename ValueType>
rocsparse_status rocsparseTcsrsv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int nnz,
                                 const ValueType* alpha,
                                 const rocsparse_mat_descr descr,
                                 const ValueType* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const ValueType* x,
                                 ValueType* y,
                                 rocsparse_solve_policy policy,
                                 void* temp_buffer);

// rocsparse coomv
template <typename ValueType>
rocsparse_status rocsparseTcoomv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 int nnz,
                                 const ValueType* alpha,
                                 const rocsparse_mat_descr descr,
                                 const ValueType* coo_val,
                                 const int* coo_row_ind,
                                 const int* coo_col_ind,
                                 const ValueType* x,
                                 const ValueType* beta,
                                 ValueType* y);

// rocsparse ellmv
template <typename ValueType>
rocsparse_status rocsparseTellmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 const ValueType* alpha,
                                 const rocsparse_mat_descr descr,
                                 const ValueType* ell_val,
                                 const int* ell_col_ind,
                                 int ell_width,
                                 const ValueType* x,
                                 const ValueType* beta,
                                 ValueType* y);

// rocsparse csrilu0 buffer size
template <typename ValueType>
rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle handle,
                                               int m,
                                               int nnz,
                                               const rocsparse_mat_descr descr,
                                               ValueType* csr_val,
                                               const int* csr_row_ptr,
                                               const int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               size_t* buffer_size);

// rocsparse csrilu0 analysis
template <typename ValueType>
rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle handle,
                                            int m,
                                            int nnz,
                                            const rocsparse_mat_descr descr,
                                            ValueType* csr_val,
                                            const int* csr_row_ptr,
                                            const int* csr_col_ind,
                                            rocsparse_mat_info info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy solve,
                                            void* temp_buffer);

// rocsparse csrilu0
template <typename ValueType>
rocsparse_status rocsparseTcsrilu0(rocsparse_handle handle,
                                   int m,
                                   int nnz,
                                   const rocsparse_mat_descr descr,
                                   ValueType* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   rocsparse_mat_info info,
                                   rocsparse_solve_policy policy,
                                   void* temp_buffer);

// rocsparse csr2csc
template <typename ValueType>
rocsparse_status rocsparseTcsr2csc(rocsparse_handle handle,
                                   int m,
                                   int n,
                                   int nnz,
                                   const ValueType* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   ValueType* csc_val,
                                   int* csc_row_ind,
                                   int* csc_col_ptr,
                                   rocsparse_action copy_values,
                                   rocsparse_index_base idx_base,
                                   void* temp_buffer);

// rocsparse csr2ell
template <typename ValueType>
rocsparse_status rocsparseTcsr2ell(rocsparse_handle handle,
                                   int m,
                                   const rocsparse_mat_descr csr_descr,
                                   const ValueType* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   int ell_width,
                                   ValueType* ell_val,
                                   int* ell_col_ind);

// rocsparse ell2csr
template <typename ValueType>
rocsparse_status rocsparseTell2csr(rocsparse_handle handle,
                                   int m,
                                   int n,
                                   const rocsparse_mat_descr ell_descr,
                                   int ell_width,
                                   const ValueType* ell_val,
                                   const int* ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   ValueType* csr_val,
                                   const int* csr_row_ptr,
                                   int* csr_col_ind);

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_SPARSE_HPP_
