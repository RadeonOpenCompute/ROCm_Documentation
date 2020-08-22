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
#include "hip_blas.hpp"

#include <rocblas.h>
#include <complex>

namespace rocalution {

// rocblas axpy
template <>
rocblas_status rocblasTaxpy(
    rocblas_handle handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return rocblas_saxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
rocblas_status rocblasTaxpy(rocblas_handle handle,
                            int n,
                            const double* alpha,
                            const double* x,
                            int incx,
                            double* y,
                            int incy)
{
    return rocblas_daxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
rocblas_status rocblasTaxpy(rocblas_handle handle,
                            int n,
                            const std::complex<float>* alpha,
                            const std::complex<float>* x,
                            int incx,
                            std::complex<float>* y,
                            int incy)
{
    //    return rocblas_caxpy(handle, n, alpha, x, incx, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTaxpy(rocblas_handle handle,
                            int n,
                            const std::complex<double>* alpha,
                            const std::complex<double>* x,
                            int incx,
                            std::complex<double>* y,
                            int incy)
{
    //    return rocblas_zaxpy(handle, n, alpha, x, incx, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas dot
template <>
rocblas_status rocblasTdot(
    rocblas_handle handle, int n, const float* x, int incx, const float* y, int incy, float* result)
{
    return rocblas_sdot(handle, n, x, incx, y, incy, result);
}

template <>
rocblas_status rocblasTdot(rocblas_handle handle,
                           int n,
                           const double* x,
                           int incx,
                           const double* y,
                           int incy,
                           double* result)
{
    return rocblas_ddot(handle, n, x, incx, y, incy, result);
}

template <>
rocblas_status rocblasTdot(rocblas_handle handle,
                           int n,
                           const std::complex<float>* x,
                           int incx,
                           const std::complex<float>* y,
                           int incy,
                           std::complex<float>* result)
{
    //    return rocblas_cdotu(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTdot(rocblas_handle handle,
                           int n,
                           const std::complex<double>* x,
                           int incx,
                           const std::complex<double>* y,
                           int incy,
                           std::complex<double>* result)
{
    //    return rocblas_zdotu(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas dotconj
template <>
rocblas_status rocblasTdotc(
    rocblas_handle handle, int n, const float* x, int incx, const float* y, int incy, float* result)
{
    return rocblas_sdot(handle, n, x, incx, y, incy, result);
}

template <>
rocblas_status rocblasTdotc(rocblas_handle handle,
                            int n,
                            const double* x,
                            int incx,
                            const double* y,
                            int incy,
                            double* result)
{
    return rocblas_ddot(handle, n, x, incx, y, incy, result);
}

template <>
rocblas_status rocblasTdotc(rocblas_handle handle,
                            int n,
                            const std::complex<float>* x,
                            int incx,
                            const std::complex<float>* y,
                            int incy,
                            std::complex<float>* result)
{
    //    return rocblas_cdot(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTdotc(rocblas_handle handle,
                            int n,
                            const std::complex<double>* x,
                            int incx,
                            const std::complex<double>* y,
                            int incy,
                            std::complex<double>* result)
{
    //    return rocblas_zdot(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas nrm2
template <>
rocblas_status rocblasTnrm2(rocblas_handle handle, int n, const float* x, int incx, float* result)
{
    return rocblas_snrm2(handle, n, x, incx, result);
}

template <>
rocblas_status rocblasTnrm2(rocblas_handle handle, int n, const double* x, int incx, double* result)
{
    return rocblas_dnrm2(handle, n, x, incx, result);
}

template <>
rocblas_status rocblasTnrm2(rocblas_handle handle,
                            int n,
                            const std::complex<float>* x,
                            int incx,
                            std::complex<float>* result)
{
    //    return rocblas_scnrm2(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTnrm2(rocblas_handle handle,
                            int n,
                            const std::complex<double>* x,
                            int incx,
                            std::complex<double>* result)
{
    //    return rocblas_dznrm2(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas scal
template <>
rocblas_status rocblasTscal(rocblas_handle handle, int n, const float* alpha, float* x, int incx)
{
    return rocblas_sscal(handle, n, alpha, x, incx);
}

template <>
rocblas_status rocblasTscal(rocblas_handle handle, int n, const double* alpha, double* x, int incx)
{
    return rocblas_dscal(handle, n, alpha, x, incx);
}

template <>
rocblas_status rocblasTscal(rocblas_handle handle,
                            int n,
                            const std::complex<float>* alpha,
                            std::complex<float>* x,
                            int incx)
{
    //    return rocblas_cscal(handle, n, alpha, x, incx);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTscal(rocblas_handle handle,
                            int n,
                            const std::complex<double>* alpha,
                            std::complex<double>* x,
                            int incx)
{
    //    return rocblas_zscal(handle, n, alpha, x, incx);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas_amax
template <>
rocblas_status rocblasTamax(rocblas_handle handle, int n, const float* x, int incx, int* result)
{
    return rocblas_isamax(handle, n, x, incx, result);
}

template <>
rocblas_status rocblasTamax(rocblas_handle handle, int n, const double* x, int incx, int* result)
{
    return rocblas_idamax(handle, n, x, incx, result);
}

template <>
rocblas_status
rocblasTamax(rocblas_handle handle, int n, const std::complex<float>* x, int incx, int* result)
{
    //    return rocblas_iscamax(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status
rocblasTamax(rocblas_handle handle, int n, const std::complex<double>* x, int incx, int* result)
{
    //    return rocblas_idzamax(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas_asum
template <>
rocblas_status rocblasTasum(rocblas_handle handle, int n, const float* x, int incx, float* result)
{
    return rocblas_sasum(handle, n, x, incx, result);
}

template <>
rocblas_status rocblasTasum(rocblas_handle handle, int n, const double* x, int incx, double* result)
{
    return rocblas_dasum(handle, n, x, incx, result);
}

template <>
rocblas_status rocblasTasum(rocblas_handle handle,
                            int n,
                            const std::complex<float>* x,
                            int incx,
                            std::complex<float>* result)
{
    //    return rocblas_scasum(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTasum(rocblas_handle handle,
                            int n,
                            const std::complex<double>* x,
                            int incx,
                            std::complex<double>* result)
{
    //    return rocblas_dzasum(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas_gemv
template <>
rocblas_status rocblasTgemv(rocblas_handle handle,
                            rocblas_operation trans,
                            int m,
                            int n,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* x,
                            int incx,
                            const float* beta,
                            float* y,
                            int incy)
{
    return rocblas_sgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
rocblas_status rocblasTgemv(rocblas_handle handle,
                            rocblas_operation trans,
                            int m,
                            int n,
                            const double* alpha,
                            const double* A,
                            int lda,
                            const double* x,
                            int incx,
                            const double* beta,
                            double* y,
                            int incy)
{
    return rocblas_dgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
rocblas_status rocblasTgemv(rocblas_handle handle,
                            rocblas_operation trans,
                            int m,
                            int n,
                            const std::complex<float>* alpha,
                            const std::complex<float>* A,
                            int lda,
                            const std::complex<float>* x,
                            int incx,
                            const std::complex<float>* beta,
                            std::complex<float>* y,
                            int incy)
{
    //    return rocblas_cgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTgemv(rocblas_handle handle,
                            rocblas_operation trans,
                            int m,
                            int n,
                            const std::complex<double>* alpha,
                            const std::complex<double>* A,
                            int lda,
                            const std::complex<double>* x,
                            int incx,
                            const std::complex<double>* beta,
                            std::complex<double>* y,
                            int incy)
{
    //    return rocblas_zgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

// rocblas_gemm
template <>
rocblas_status rocblasTgemm(rocblas_handle handle,
                            rocblas_operation transa,
                            rocblas_operation transb,
                            int m,
                            int n,
                            int k,
                            const float* alpha,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            const float* beta,
                            float* C,
                            int ldc)
{
    return rocblas_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblasTgemm(rocblas_handle handle,
                            rocblas_operation transa,
                            rocblas_operation transb,
                            int m,
                            int n,
                            int k,
                            const double* alpha,
                            const double* A,
                            int lda,
                            const double* B,
                            int ldb,
                            const double* beta,
                            double* C,
                            int ldc)
{
    return rocblas_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status rocblasTgemm(rocblas_handle handle,
                            rocblas_operation transa,
                            rocblas_operation transb,
                            int m,
                            int n,
                            int k,
                            const std::complex<float>* alpha,
                            const std::complex<float>* A,
                            int lda,
                            const std::complex<float>* B,
                            int ldb,
                            const std::complex<float>* beta,
                            std::complex<float>* C,
                            int ldc)
{
    //    return rocblas_cgemm(handle, transa, transb, m, n, k,
    //                         alpha, A, lda, B, ldb, beta, C, ldc);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
rocblas_status rocblasTgemm(rocblas_handle handle,
                            rocblas_operation transa,
                            rocblas_operation transb,
                            int m,
                            int n,
                            int k,
                            const std::complex<double>* alpha,
                            const std::complex<double>* A,
                            int lda,
                            const std::complex<double>* B,
                            int ldb,
                            const std::complex<double>* beta,
                            std::complex<double>* C,
                            int ldc)
{
    //    return rocblas_zgemm(handle, transa, transb, m, n, k,
    //                         alpha, A, lda, B, ldb, beta, C, ldc);
    FATAL_ERROR(__FILE__, __LINE__);
}

} // namespace rocalution
