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

#ifndef ROCALUTION_HIP_HIP_KERNELS_VECTOR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_VECTOR_HPP_

#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleadd(IndexType n,
                                ValueType alpha,
                                const ValueType* __restrict__ x,
                                ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = alpha * out[ind] + x[ind];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleaddscale(IndexType n,
                                     ValueType alpha,
                                     ValueType beta,
                                     const ValueType* __restrict__ x,
                                     ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = alpha * out[ind] + beta * x[ind];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleaddscale_offset(IndexType n,
                                            IndexType src_offset,
                                            IndexType dst_offset,
                                            ValueType alpha,
                                            ValueType beta,
                                            const ValueType* __restrict__ x,
                                            ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind + dst_offset] = alpha * out[ind + dst_offset] + beta * x[ind + src_offset];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleadd2(IndexType n,
                                 ValueType alpha,
                                 ValueType beta,
                                 ValueType gamma,
                                 const ValueType* __restrict__ x,
                                 const ValueType* __restrict__ y,
                                 ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = alpha * out[ind] + beta * x[ind] + gamma * y[ind];
}

template <typename ValueType, typename IndexType>
__global__ void
kernel_pointwisemult(IndexType n, const ValueType* __restrict__ x, ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = out[ind] * x[ind];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_pointwisemult2(IndexType n,
                                      const ValueType* __restrict__ x,
                                      const ValueType* __restrict__ y,
                                      ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = y[ind] * x[ind];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_copy_offset_from(IndexType n,
                                        IndexType src_offset,
                                        IndexType dst_offset,
                                        const ValueType* __restrict__ in,
                                        ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind + dst_offset] = in[ind + src_offset];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_permute(IndexType n,
                               const IndexType* __restrict__ permute,
                               const ValueType* __restrict__ in,
                               ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[permute[ind]] = in[ind];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_permute_backward(IndexType n,
                                        const IndexType* __restrict__ permute,
                                        const ValueType* __restrict__ in,
                                        ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = in[permute[ind]];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_get_index_values(IndexType size,
                                        const IndexType* __restrict__ index,
                                        const ValueType* __restrict__ in,
                                        ValueType* __restrict__ out)
{
    IndexType i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i >= size)
    {
        return;
    }

    out[i] = in[index[i]];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_set_index_values(IndexType size,
                                        const IndexType* __restrict__ index,
                                        const ValueType* __restrict__ in,
                                        ValueType* __restrict__ out)
{
    IndexType i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i >= size)
    {
        return;
    }

    out[index[i]] = in[i];
}

template <typename ValueType, typename IndexType>
__global__ void kernel_power(IndexType n, double power, ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = hip_pow(out[ind], power);
}

template <typename ValueType, typename IndexType>
__global__ void
kernel_copy_from_float(IndexType n, const float* __restrict__ in, ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = static_cast<ValueType>(in[ind]);
}

template <typename ValueType, typename IndexType>
__global__ void
kernel_copy_from_double(IndexType n, const double* __restrict__ in, ValueType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[ind] = static_cast<ValueType>(in[ind]);
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_VECTOR_HPP_
