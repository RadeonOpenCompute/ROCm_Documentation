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

#ifndef ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Compute non-zero entries per row
__global__ void kernel_hyb_coo_nnz(int m,
                                   int ell_width,
                                   const int* __restrict__ csr_row_ptr,
                                   int* __restrict__ coo_row_nnz)
{
    int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= m)
    {
        return;
    }

    int row_nnz      = csr_row_ptr[gid + 1] - csr_row_ptr[gid] - ell_width;
    coo_row_nnz[gid] = row_nnz > 0 ? row_nnz : 0;
}

// CSR to HYB format conversion kernel
template <typename ValueType>
__global__ void kernel_hyb_csr2hyb(int m,
                                   const ValueType* __restrict__ csr_val,
                                   const int* __restrict__ csr_row_ptr,
                                   const int* __restrict__ csr_col_ind,
                                   int ell_width,
                                   int* __restrict__ ell_col_ind,
                                   ValueType* __restrict__ ell_val,
                                   int* __restrict__ coo_row_ind,
                                   int* __restrict__ coo_col_ind,
                                   ValueType* __restrict__ coo_val,
                                   int* __restrict__ workspace)
{
    int ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    int p = 0;

    int row_begin = csr_row_ptr[ai];
    int row_end   = csr_row_ptr[ai + 1];
    int coo_idx   = coo_row_ind ? workspace[ai] : 0;

    // Fill HYB matrix
    for(int aj = row_begin; aj < row_end; ++aj)
    {
        if(p < ell_width)
        {
            // Fill ELL part
            int idx          = ELL_IND(ai, p++, m, ell_width);
            ell_col_ind[idx] = csr_col_ind[aj];
            ell_val[idx]     = csr_val[aj];
        }
        else
        {
            // Fill COO part
            coo_row_ind[coo_idx] = ai;
            coo_col_ind[coo_idx] = csr_col_ind[aj];
            coo_val[coo_idx]     = csr_val[aj];
            ++coo_idx;
        }
    }

    // Pad remaining ELL structure
    for(int aj = row_end - row_begin; aj < ell_width; ++aj)
    {
        int idx          = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx] = -1;
        ell_val[idx]     = static_cast<ValueType>(0);
    }
}

template <typename IndexType>
__global__ void kernel_dia_diag_idx(IndexType nrow,
                                    IndexType* __restrict__ row_offset,
                                    IndexType* __restrict__ col,
                                    IndexType* __restrict__ diag_idx)
{
    IndexType row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(row >= nrow)
    {
        return;
    }

    for(IndexType j = row_offset[row]; j < row_offset[row + 1]; ++j)
    {
        IndexType idx = col[j] - row + nrow;
        diag_idx[idx] = 1;
    }
}

template <typename IndexType>
__global__ void kernel_dia_fill_offset(IndexType nrow,
                                       IndexType ncol,
                                       IndexType* __restrict__ diag_idx,
                                       const IndexType* __restrict__ offset_map,
                                       IndexType* __restrict__ offset)
{
    IndexType i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(i >= nrow + ncol)
    {
        return;
    }

    if(diag_idx[i] == 1)
    {
        offset[offset_map[i]] = i - nrow;
        diag_idx[i]           = offset_map[i];
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_dia_convert(IndexType nrow,
                                   IndexType ndiag,
                                   const IndexType* __restrict__ row_offset,
                                   const IndexType* __restrict__ col,
                                   const ValueType* __restrict__ val,
                                   const IndexType* __restrict__ diag_idx,
                                   ValueType* __restrict__ dia_val)
{
    IndexType row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(row >= nrow)
    {
        return;
    }

    for(IndexType j = row_offset[row]; j < row_offset[row + 1]; ++j)
    {
        IndexType idx = col[j] - row + nrow;

        dia_val[DIA_IND(row, diag_idx[idx], nrow, ndiag)] = val[j];
    }
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_
