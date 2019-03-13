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

#ifndef ROCALUTION_HIP_HIP_UTILS_HPP_
#define ROCALUTION_HIP_HIP_UTILS_HPP_

#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "backend_hip.hpp"

#include <complex>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <rocsparse.h>

#define ROCBLAS_HANDLE(handle) *static_cast<rocblas_handle*>(handle)
#define ROCSPARSE_HANDLE(handle) *static_cast<rocsparse_handle*>(handle)

#define CHECK_HIP_ERROR(file, line)                              \
    {                                                            \
        hipError_t err_t;                                        \
        if((err_t = hipGetLastError()) != hipSuccess)            \
        {                                                        \
            LOG_INFO("HIP error: " << hipGetErrorString(err_t)); \
            LOG_INFO("File: " << file << "; line: " << line);    \
            exit(1);                                             \
        }                                                        \
    }

#define CHECK_ROCBLAS_ERROR(stat_t, file, line)               \
    {                                                         \
        if(stat_t != rocblas_status_success)                  \
        {                                                     \
            LOG_INFO("rocBLAS error " << stat_t);             \
            if(stat_t == rocblas_status_invalid_handle)       \
                LOG_INFO("rocblas_status_invalid_handle");    \
            if(stat_t == rocblas_status_not_implemented)      \
                LOG_INFO("rocblas_status_not_implemented");   \
            if(stat_t == rocblas_status_invalid_pointer)      \
                LOG_INFO("rocblas_status_invalid_pointer");   \
            if(stat_t == rocblas_status_invalid_size)         \
                LOG_INFO("rocblas_status_invalid_size");      \
            if(stat_t == rocblas_status_memory_error)         \
                LOG_INFO("rocblas_status_memory_error");      \
            if(stat_t == rocblas_status_internal_error)       \
                LOG_INFO("rocblas_status_internal_error");    \
            LOG_INFO("File: " << file << "; line: " << line); \
            exit(1);                                          \
        }                                                     \
    }

#define CHECK_ROCSPARSE_ERROR(status, file, line)             \
    {                                                         \
        if(status != rocsparse_status_success)                \
        {                                                     \
            LOG_INFO("rocSPARSE error " << status);           \
            if(status == rocsparse_status_invalid_handle)     \
                LOG_INFO("rocsparse_status_invalid_handle");  \
            if(status == rocsparse_status_not_implemented)    \
                LOG_INFO("rocsparse_status_not_implemented"); \
            if(status == rocsparse_status_invalid_pointer)    \
                LOG_INFO("rocsparse_status_invalid_pointer"); \
            if(status == rocsparse_status_invalid_size)       \
                LOG_INFO("rocsparse_status_invalid_size");    \
            if(status == rocsparse_status_memory_error)       \
                LOG_INFO("rocsparse_status_memory_error");    \
            if(status == rocsparse_status_internal_error)     \
                LOG_INFO("rocsparse_status_internal_error");  \
            if(status == rocsparse_status_invalid_value)      \
                LOG_INFO("rocsparse_status_invalid_value");   \
            if(status == rocsparse_status_arch_mismatch)      \
                LOG_INFO("rocsparse_status_arch_mismatch");   \
            LOG_INFO("File: " << file << "; line: " << line); \
            exit(1);                                          \
        }                                                     \
    }

namespace rocalution {

// Type traits to cast STL types to HIP types
template <typename ValueType>
struct HIPType
{
    typedef ValueType Type;
};

#ifdef SUPPORT_COMPLEX
template <>
struct HIPType<std::complex<float>>
{
    typedef hipFloatComplex Type;
};

template <>
struct HIPType<std::complex<double>>
{
    typedef hipDoubleComplex Type;
};
#endif

__device__ int __llvm_amdgcn_readlane(int index, int offset) __asm("llvm.amdgcn.readlane");

template <unsigned int WF_SIZE>
static __device__ __forceinline__ float wf_reduce_sum(float sum)
{
    typedef union flt_b32
    {
        float val;
        uint32_t b32;
    } flt_b32_t;

    flt_b32_t upper_sum;
    flt_b32_t temp_sum;
    temp_sum.val = sum;

    if(WF_SIZE > 1)
    {
        upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x80b1);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 2)
    {
        upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x804e);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 4)
    {
        upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x101f);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 8)
    {
        upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x201f);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 16)
    {
        upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x401f);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 32)
    {
        upper_sum.b32 = __llvm_amdgcn_readlane(temp_sum.b32, 32);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

template <unsigned int WF_SIZE>
static __device__ __forceinline__ double wf_reduce_sum(double sum)
{
    typedef union dbl_b32
    {
        double val;
        uint32_t b32[2];
    } dbl_b32_t;

    dbl_b32_t upper_sum;
    dbl_b32_t temp_sum;
    temp_sum.val = sum;

    if(WF_SIZE > 1)
    {
        upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x80b1);
        upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x80b1);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 2)
    {
        upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x804e);
        upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x804e);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 4)
    {
        upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x101f);
        upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x101f);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 8)
    {
        upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x201f);
        upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x201f);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 16)
    {
        upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x401f);
        upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x401f);
        temp_sum.val += upper_sum.val;
    }

    if(WF_SIZE > 32)
    {
        upper_sum.b32[0] = __llvm_amdgcn_readlane(temp_sum.b32[0], 32);
        upper_sum.b32[1] = __llvm_amdgcn_readlane(temp_sum.b32[1], 32);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_UTILS_HPP_
