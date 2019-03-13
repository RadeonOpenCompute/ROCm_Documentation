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

#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "hip_sparse.hpp"

#include <rocsparse.h>
#include <complex>

namespace rocalution {

// rocsparse csrmv analysis
template <>
rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          int m,
                                          int n,
                                          int nnz,
                                          const rocsparse_mat_descr descr,
                                          const float* csr_val,
                                          const int* csr_row_ptr,
                                          const int* csr_col_ind,
                                          rocsparse_mat_info info)
{
    return rocsparse_scsrmv_analysis(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

template <>
rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          int m,
                                          int n,
                                          int nnz,
                                          const rocsparse_mat_descr descr,
                                          const double* csr_val,
                                          const int* csr_row_ptr,
                                          const int* csr_col_ind,
                                          rocsparse_mat_info info)
{
    return rocsparse_dcsrmv_analysis(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

// rocsparse csrmv
template <>
rocsparse_status rocsparseTcsrmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 int nnz,
                                 const float* alpha,
                                 const rocsparse_mat_descr descr,
                                 const float* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const float* x,
                                 const float* beta,
                                 float* y)
{
    return rocsparse_scsrmv(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            info,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparseTcsrmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 int nnz,
                                 const double* alpha,
                                 const rocsparse_mat_descr descr,
                                 const double* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const double* x,
                                 const double* beta,
                                 double* y)
{
    return rocsparse_dcsrmv(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            info,
                            x,
                            beta,
                            y);
}

// rocsparse csrsv buffer size
template <>
rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             int m,
                                             int nnz,
                                             const rocsparse_mat_descr descr,
                                             const float* csr_val,
                                             const int* csr_row_ptr,
                                             const int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             size_t* buffer_size)
{
    return rocsparse_scsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             int m,
                                             int nnz,
                                             const rocsparse_mat_descr descr,
                                             const double* csr_val,
                                             const int* csr_row_ptr,
                                             const int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             size_t* buffer_size)
{
    return rocsparse_dcsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

// rocsparse csrsv analysis
template <>
rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          int m,
                                          int nnz,
                                          const rocsparse_mat_descr descr,
                                          const float* csr_val,
                                          const int* csr_row_ptr,
                                          const int* csr_col_ind,
                                          rocsparse_mat_info info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy solve,
                                          void* temp_buffer)
{
    return rocsparse_scsrsv_analysis(handle,
                                     trans,
                                     m,
                                     nnz,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          int m,
                                          int nnz,
                                          const rocsparse_mat_descr descr,
                                          const double* csr_val,
                                          const int* csr_row_ptr,
                                          const int* csr_col_ind,
                                          rocsparse_mat_info info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy solve,
                                          void* temp_buffer)
{
    return rocsparse_dcsrsv_analysis(handle,
                                     trans,
                                     m,
                                     nnz,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

// rocsparse csrsv
template <>
rocsparse_status rocsparseTcsrsv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int nnz,
                                 const float* alpha,
                                 const rocsparse_mat_descr descr,
                                 const float* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const float* x,
                                 float* y,
                                 rocsparse_solve_policy policy,
                                 void* temp_buffer)
{
    return rocsparse_scsrsv_solve(handle,
                                  trans,
                                  m,
                                  nnz,
                                  alpha,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  x,
                                  y,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparseTcsrsv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int nnz,
                                 const double* alpha,
                                 const rocsparse_mat_descr descr,
                                 const double* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 rocsparse_mat_info info,
                                 const double* x,
                                 double* y,
                                 rocsparse_solve_policy policy,
                                 void* temp_buffer)
{
    return rocsparse_dcsrsv_solve(handle,
                                  trans,
                                  m,
                                  nnz,
                                  alpha,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  x,
                                  y,
                                  policy,
                                  temp_buffer);
}

// rocsparse coomv
template <>
rocsparse_status rocsparseTcoomv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 int nnz,
                                 const float* alpha,
                                 const rocsparse_mat_descr descr,
                                 const float* coo_val,
                                 const int* coo_row_ind,
                                 const int* coo_col_ind,
                                 const float* x,
                                 const float* beta,
                                 float* y)
{
    return rocsparse_scoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

template <>
rocsparse_status rocsparseTcoomv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 int nnz,
                                 const double* alpha,
                                 const rocsparse_mat_descr descr,
                                 const double* coo_val,
                                 const int* coo_row_ind,
                                 const int* coo_col_ind,
                                 const double* x,
                                 const double* beta,
                                 double* y)
{
    return rocsparse_dcoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

// rocsparse ellmv
template <>
rocsparse_status rocsparseTellmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 const float* alpha,
                                 const rocsparse_mat_descr descr,
                                 const float* ell_val,
                                 const int* ell_col_ind,
                                 int ell_width,
                                 const float* x,
                                 const float* beta,
                                 float* y)
{
    return rocsparse_sellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

template <>
rocsparse_status rocsparseTellmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 int m,
                                 int n,
                                 const double* alpha,
                                 const rocsparse_mat_descr descr,
                                 const double* ell_val,
                                 const int* ell_col_ind,
                                 int ell_width,
                                 const double* x,
                                 const double* beta,
                                 double* y)
{
    return rocsparse_dellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

// rocsparse csrilu0 buffer size
template <>
rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle handle,
                                               int m,
                                               int nnz,
                                               const rocsparse_mat_descr descr,
                                               float* csr_val,
                                               const int* csr_row_ptr,
                                               const int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               size_t* buffer_size)
{
    return rocsparse_scsrilu0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle handle,
                                               int m,
                                               int nnz,
                                               const rocsparse_mat_descr descr,
                                               double* csr_val,
                                               const int* csr_row_ptr,
                                               const int* csr_col_ind,
                                               rocsparse_mat_info info,
                                               size_t* buffer_size)
{
    return rocsparse_dcsrilu0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

// rocsparse csrilu0 analysis
template <>
rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle handle,
                                            int m,
                                            int nnz,
                                            const rocsparse_mat_descr descr,
                                            float* csr_val,
                                            const int* csr_row_ptr,
                                            const int* csr_col_ind,
                                            rocsparse_mat_info info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy solve,
                                            void* temp_buffer)
{
    return rocsparse_scsrilu0_analysis(handle,
                                       m,
                                       nnz,
                                       descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle handle,
                                            int m,
                                            int nnz,
                                            const rocsparse_mat_descr descr,
                                            double* csr_val,
                                            const int* csr_row_ptr,
                                            const int* csr_col_ind,
                                            rocsparse_mat_info info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy solve,
                                            void* temp_buffer)
{
    return rocsparse_dcsrilu0_analysis(handle,
                                       m,
                                       nnz,
                                       descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

// rocsparse csrilu0
template <>
rocsparse_status rocsparseTcsrilu0(rocsparse_handle handle,
                                   int m,
                                   int nnz,
                                   const rocsparse_mat_descr descr,
                                   float* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   rocsparse_mat_info info,
                                   rocsparse_solve_policy policy,
                                   void* temp_buffer)
{
    return rocsparse_scsrilu0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

template <>
rocsparse_status rocsparseTcsrilu0(rocsparse_handle handle,
                                   int m,
                                   int nnz,
                                   const rocsparse_mat_descr descr,
                                   double* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   rocsparse_mat_info info,
                                   rocsparse_solve_policy policy,
                                   void* temp_buffer)
{
    return rocsparse_dcsrilu0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

// rocsparse csr2csc
template <>
rocsparse_status rocsparseTcsr2csc(rocsparse_handle handle,
                                   int m,
                                   int n,
                                   int nnz,
                                   const float* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   float* csc_val,
                                   int* csc_row_ind,
                                   int* csc_col_ptr,
                                   rocsparse_action copy_values,
                                   rocsparse_index_base idx_base,
                                   void* temp_buffer)
{
    return rocsparse_scsr2csc(handle,
                              m,
                              n,
                              nnz,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              csc_val,
                              csc_row_ind,
                              csc_col_ptr,
                              copy_values,
                              idx_base,
                              temp_buffer);
}

template <>
rocsparse_status rocsparseTcsr2csc(rocsparse_handle handle,
                                   int m,
                                   int n,
                                   int nnz,
                                   const double* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   double* csc_val,
                                   int* csc_row_ind,
                                   int* csc_col_ptr,
                                   rocsparse_action copy_values,
                                   rocsparse_index_base idx_base,
                                   void* temp_buffer)
{
    return rocsparse_dcsr2csc(handle,
                              m,
                              n,
                              nnz,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              csc_val,
                              csc_row_ind,
                              csc_col_ptr,
                              copy_values,
                              idx_base,
                              temp_buffer);
}

// rocsparse csr2ell
template <>
rocsparse_status rocsparseTcsr2ell(rocsparse_handle handle,
                                   int m,
                                   const rocsparse_mat_descr csr_descr,
                                   const float* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   int ell_width,
                                   float* ell_val,
                                   int* ell_col_ind)
{
    return rocsparse_scsr2ell(handle,
                              m,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind);
}

template <>
rocsparse_status rocsparseTcsr2ell(rocsparse_handle handle,
                                   int m,
                                   const rocsparse_mat_descr csr_descr,
                                   const double* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   int ell_width,
                                   double* ell_val,
                                   int* ell_col_ind)
{
    return rocsparse_dcsr2ell(handle,
                              m,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind);
}

// rocsparse ell2csr
template <>
rocsparse_status rocsparseTell2csr(rocsparse_handle handle,
                                   int m,
                                   int n,
                                   const rocsparse_mat_descr ell_descr,
                                   int ell_width,
                                   const float* ell_val,
                                   const int* ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   float* csr_val,
                                   const int* csr_row_ptr,
                                   int* csr_col_ind)
{
    return rocsparse_sell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparseTell2csr(rocsparse_handle handle,
                                   int m,
                                   int n,
                                   const rocsparse_mat_descr ell_descr,
                                   int ell_width,
                                   const double* ell_val,
                                   const int* ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   double* csr_val,
                                   const int* csr_row_ptr,
                                   int* csr_col_ind)
{
    return rocsparse_dell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

} // namespace rocalution
