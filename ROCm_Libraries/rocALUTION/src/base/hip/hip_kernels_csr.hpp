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

#ifndef ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_scale_diagonal(IndexType nrow,
                                          const IndexType* __restrict__ row_offset,
                                          const IndexType* __restrict__ col,
                                          ValueType alpha,
                                          ValueType* __restrict__ val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(ai == col[aj])
        {
            val[aj] = alpha * val[aj];
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_scale_offdiagonal(IndexType nrow,
                                             const IndexType* __restrict__ row_offset,
                                             const IndexType* __restrict__ col,
                                             ValueType alpha,
                                             ValueType* __restrict__ val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(ai != col[aj])
        {
            val[aj] = alpha * val[aj];
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_diagonal(IndexType nrow,
                                        const IndexType* __restrict__ row_offset,
                                        const IndexType* __restrict__ col,
                                        ValueType alpha,
                                        ValueType* __restrict__ val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(ai == col[aj])
        {
            val[aj] = val[aj] + alpha;
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_offdiagonal(IndexType nrow,
                                           const IndexType* __restrict__ row_offset,
                                           const IndexType* __restrict__ col,
                                           ValueType alpha,
                                           ValueType* __restrict__ val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(ai != col[aj])
        {
            val[aj] = val[aj] + alpha;
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_diag(IndexType nrow,
                                        const IndexType* __restrict__ row_offset,
                                        const IndexType* __restrict__ col,
                                        const ValueType* __restrict__ val,
                                        ValueType* __restrict__ vec)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(ai == col[aj])
        {
            vec[ai] = val[aj];
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_inv_diag(IndexType nrow,
                                            const IndexType* __restrict__ row_offset,
                                            const IndexType* __restrict__ col,
                                            const ValueType* __restrict__ val,
                                            ValueType* __restrict__ vec)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(ai == col[aj])
        {
            make_ValueType(vec[ai], 1);
            vec[ai] = vec[ai] / val[aj];
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_submatrix_row_nnz(const IndexType* __restrict__ row_offset,
                                                     const IndexType* __restrict__ col,
                                                     const ValueType* __restrict__ val,
                                                     IndexType smrow_offset,
                                                     IndexType smcol_offset,
                                                     IndexType smrow_size,
                                                     IndexType smcol_size,
                                                     IndexType* __restrict__ row_nnz)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= smrow_size)
    {
        return;
    }

    IndexType nnz = 0;
    IndexType ind = ai + smrow_offset;

    for(IndexType aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
    {
        IndexType c = col[aj];

        if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
        {
            ++nnz;
        }
    }

    row_nnz[ai] = nnz;
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_submatrix_copy(const IndexType* __restrict__ row_offset,
                                                  const IndexType* __restrict__ col,
                                                  const ValueType* __restrict__ val,
                                                  IndexType smrow_offset,
                                                  IndexType smcol_offset,
                                                  IndexType smrow_size,
                                                  IndexType smcol_size,
                                                  const IndexType* __restrict__ sm_row_offset,
                                                  IndexType* __restrict__ sm_col,
                                                  ValueType* __restrict__ sm_val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= smrow_size)
    {
        return;
    }

    IndexType row_nnz = sm_row_offset[ai];
    IndexType ind     = ai + smrow_offset;

    for(IndexType aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
    {
        IndexType c = col[aj];

        if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
        {
            sm_col[row_nnz] = c - smcol_offset;
            sm_val[row_nnz] = val[aj];
            ++row_nnz;
        }
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_diagmatmult_r(IndexType nrow,
                                         const IndexType* __restrict__ row_offset,
                                         const IndexType* __restrict__ col,
                                         const ValueType* __restrict__ diag,
                                         ValueType* __restrict__ val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        val[aj] = val[aj] * diag[col[aj]];
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_diagmatmult_l(IndexType nrow,
                                         const IndexType* __restrict__ row_offset,
                                         const ValueType* __restrict__ diag,
                                         ValueType* __restrict__ val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        val[aj] = val[aj] * diag[ai];
    }
}

// Calculates the number of non-zero elements per row
template <typename IndexType>
__global__ void kernel_calc_row_nnz(IndexType nrow,
                                    const IndexType* __restrict__ row_offset,
                                    IndexType* __restrict__ row_nnz)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    row_nnz[ai] = row_offset[ai + 1] - row_offset[ai];
}

// Performs a permutation on the vector of non-zero elements per row
//
// Inputs:   nrow:         number of rows in matrix
//           row_nnz_src:  original number of non-zero elements per row
//           perm_vec:     permutation vector
// Outputs:  row_nnz_dst   permuted number of non-zero elements per row
template <typename IndexType>
__global__ void kernel_permute_row_nnz(IndexType nrow,
                                       const IndexType* __restrict__ row_offset,
                                       const IndexType* __restrict__ perm_vec,
                                       IndexType* __restrict__ row_nnz_dst)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    if(ai == 0)
    {
        row_nnz_dst[0] = 0;
    }

    row_nnz_dst[perm_vec[ai] + 1] = row_offset[ai + 1] - row_offset[ai];
}

// Permutes rows
//
// Inputs:   nrow:             number of rows in matrix
//           row_offset:       original row pointer
//           perm_row_offset:  permuted row pointer
//           col:              original column indices of elements
//           data:             original data vector
//           perm_vec:         permutation vector
//           row_nnz:          number of non-zero elements per row
// Outputs:  perm_col:         permuted column indices of elements
//           perm_data:        permuted data vector
template <typename ValueType, typename IndexType, unsigned int WF_SIZE>
__global__ void kernel_permute_rows(IndexType nrow,
                                    const IndexType* __restrict__ row_offset,
                                    const IndexType* __restrict__ perm_row_offset,
                                    const IndexType* __restrict__ col,
                                    const ValueType* __restrict__ data,
                                    const IndexType* __restrict__ perm_vec,
                                    IndexType* __restrict__ perm_col,
                                    ValueType* __restrict__ perm_data)
{
    IndexType tid = hipThreadIdx_x;
    IndexType gid = hipBlockIdx_x * hipBlockDim_x + tid;
    IndexType lid = tid & (WF_SIZE - 1);
    IndexType row = gid / WF_SIZE;

    if(row >= nrow)
    {
        return;
    }

    IndexType perm_index = perm_row_offset[perm_vec[row]];
    IndexType prev_index = row_offset[row];
    IndexType num_elems  = row_offset[row + 1] - prev_index;

    for(IndexType i = lid; i < num_elems; i += WF_SIZE)
    {
        perm_data[perm_index + i] = data[prev_index + i];
        perm_col[perm_index + i]  = col[prev_index + i];
    }
}

// Permutes columns
//
// Inputs:   nrow:             number of rows in matrix
//           row_offset:       row pointer
//           perm_vec:         permutation vector
//           perm_col:         row-permuted column indices of elements
//           perm_data:        row-permuted data
// Outputs:  col:              fully permuted column indices of elements
//           data:             fully permuted data
template <unsigned int size, typename ValueType, typename IndexType>
__global__ void kernel_permute_cols(IndexType nrow,
                                    const IndexType* __restrict__ row_offset,
                                    const IndexType* __restrict__ perm_vec,
                                    const IndexType* __restrict__ perm_col,
                                    const ValueType* __restrict__ perm_data,
                                    IndexType* __restrict__ col,
                                    ValueType* __restrict__ data)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType elem_index = row_offset[ai];
    IndexType num_elems  = row_offset[ai + 1] - elem_index;

    IndexType ccol[size];
    ValueType cval[size];

    col += elem_index;
    data += elem_index;
    perm_col += elem_index;
    perm_data += elem_index;

    for(IndexType i = 0; i < num_elems; ++i)
    {
        ccol[i] = col[i];
        cval[i] = data[i];
    }

    for(IndexType i = 0; i < num_elems; ++i)
    {
        IndexType comp = perm_vec[perm_col[i]];

        IndexType j;
        for(j = i - 1; j >= 0; --j)
        {
            IndexType c = ccol[j];
            if(c > comp)
            {
                cval[j + 1] = cval[j];
                ccol[j + 1] = c;
            }
            else
            {
                break;
            }
        }

        cval[j + 1] = perm_data[i];
        ccol[j + 1] = comp;
    }

    for(IndexType i = 0; i < num_elems; ++i)
    {
        col[i]  = ccol[i];
        data[i] = cval[i];
    }
}

// Permutes columns
//
// Inputs:   nrow:             number of rows in matrix
//           row_offset:       row pointer
//           perm_vec:         permutation vector
//           perm_col:         row-permuted column indices of elements
//           perm_data:        row-permuted data
// Outputs:  col:              fully permuted column indices of elements
//           data:             fully permuted data
template <typename ValueType, typename IndexType>
__global__ void kernel_permute_cols_fallback(IndexType nrow,
                                             const IndexType* __restrict__ row_offset,
                                             const IndexType* __restrict__ perm_vec,
                                             const IndexType* __restrict__ perm_col,
                                             const ValueType* __restrict__ perm_data,
                                             IndexType* __restrict__ col,
                                             ValueType* __restrict__ data)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType elem_index = row_offset[ai];
    IndexType num_elems  = row_offset[ai + 1] - elem_index;

    col += elem_index;
    data += elem_index;
    perm_col += elem_index;
    perm_data += elem_index;

    for(IndexType i = 0; i < num_elems; ++i)
    {
        IndexType comp = perm_vec[perm_col[i]];

        IndexType j;
        for(j = i - 1; j >= 0; --j)
        {
            IndexType c = col[j];
            if(c > comp)
            {
                data[j + 1] = data[j];
                col[j + 1]  = c;
            }
            else
            {
                break;
            }
        }

        data[j + 1] = perm_data[i];
        col[j + 1]  = comp;
    }
}

// TODO
// kind of ugly and inefficient ... but works
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_csr_same_struct(IndexType nrow,
                                               const IndexType* __restrict__ out_row_offset,
                                               const IndexType* __restrict__ out_col,
                                               const IndexType* __restrict__ in_row_offset,
                                               const IndexType* __restrict__ in_col,
                                               const ValueType* __restrict__ in_val,
                                               ValueType alpha,
                                               ValueType beta,
                                               ValueType* __restrict__ out_val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType first_col = in_row_offset[ai];

    for(IndexType ajj = out_row_offset[ai]; ajj < out_row_offset[ai + 1]; ++ajj)
    {
        for(IndexType aj = first_col; aj < in_row_offset[ai + 1]; ++aj)
        {
            if(in_col[aj] == out_col[ajj])
            {
                out_val[ajj] = alpha * out_val[ajj] + beta * in_val[aj];
                ++first_col;
                break;
            }
        }
    }
}

// Computes the lower triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_lower_nnz_per_row(IndexType nrow,
                                             const IndexType* __restrict__ src_row_offset,
                                             const IndexType* __restrict__ src_col,
                                             IndexType* __restrict__ nnz_per_row)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    nnz_per_row[ai] = 0;

    for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
    {
        if(src_col[aj] <= ai)
        {
            ++nnz_per_row[ai];
        }
    }
}

// Computes the upper triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_upper_nnz_per_row(IndexType nrow,
                                             const IndexType* __restrict__ src_row_offset,
                                             const IndexType* __restrict__ src_col,
                                             IndexType* __restrict__ nnz_per_row)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    nnz_per_row[ai] = 0;

    for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
    {
        if(src_col[aj] >= ai)
        {
            ++nnz_per_row[ai];
        }
    }
}

// Computes the stricktly lower triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_slower_nnz_per_row(IndexType nrow,
                                              const IndexType* __restrict__ src_row_offset,
                                              const IndexType* __restrict__ src_col,
                                              IndexType* __restrict__ nnz_per_row)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    nnz_per_row[ai] = 0;

    for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
    {
        if(src_col[aj] < ai)
        {
            ++nnz_per_row[ai];
        }
    }
}

// Computes the stricktly upper triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_supper_nnz_per_row(IndexType nrow,
                                              const IndexType* __restrict__ src_row_offset,
                                              const IndexType* __restrict__ src_col,
                                              IndexType* __restrict__ nnz_per_row)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    nnz_per_row[ai] = 0;

    for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
    {
        if(src_col[aj] > ai)
        {
            ++nnz_per_row[ai];
        }
    }
}

// Extracts lower triangular part for given nnz per row array (partial sums nnz)
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_l_triangular(IndexType nrow,
                                                const IndexType* __restrict__ src_row_offset,
                                                const IndexType* __restrict__ src_col,
                                                const ValueType* __restrict__ src_val,
                                                IndexType* __restrict__ nnz_per_row,
                                                IndexType* __restrict__ dst_col,
                                                ValueType* __restrict__ dst_val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType dst_index = nnz_per_row[ai];
    IndexType src_index = src_row_offset[ai];

    for(IndexType aj = 0; aj < nnz_per_row[ai + 1] - nnz_per_row[ai]; ++aj)
    {
        dst_col[dst_index] = src_col[src_index];
        dst_val[dst_index] = src_val[src_index];

        ++dst_index;
        ++src_index;
    }
}

// Extracts upper triangular part for given nnz per row array (partial sums nnz)
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_u_triangular(IndexType nrow,
                                                const IndexType* __restrict__ src_row_offset,
                                                const IndexType* __restrict__ src_col,
                                                const ValueType* __restrict__ src_val,
                                                IndexType* __restrict__ nnz_per_row,
                                                IndexType* __restrict__ dst_col,
                                                ValueType* __restrict__ dst_val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType num_elements = nnz_per_row[ai + 1] - nnz_per_row[ai];
    IndexType src_index    = src_row_offset[ai + 1] - num_elements;
    IndexType dst_index    = nnz_per_row[ai];

    for(IndexType aj = 0; aj < num_elements; ++aj)
    {
        dst_col[dst_index] = src_col[src_index];
        dst_val[dst_index] = src_val[src_index];

        ++dst_index;
        ++src_index;
    }
}

// Compress
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_compress_count_nrow(const IndexType* __restrict__ row_offset,
                                               const IndexType* __restrict__ col,
                                               const ValueType* __restrict__ val,
                                               IndexType nrow,
                                               double drop_off,
                                               IndexType* __restrict__ row_offset_new)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
        {
            ++row_offset_new[ai];
        }
    }
}

// Compress
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_compress_copy(const IndexType* __restrict__ row_offset,
                                         const IndexType* __restrict__ col,
                                         const ValueType* __restrict__ val,
                                         IndexType nrow,
                                         double drop_off,
                                         const IndexType* __restrict__ row_offset_new,
                                         IndexType* __restrict__ col_new,
                                         ValueType* __restrict__ val_new)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType ajj = row_offset_new[ai];

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
        {
            col_new[ajj] = col[aj];
            val_new[ajj] = val[aj];
            ++ajj;
        }
    }
}

// Extract column vector
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_column_vector(const IndexType* __restrict__ row_offset,
                                                 const IndexType* __restrict__ col,
                                                 const ValueType* __restrict__ val,
                                                 IndexType nrow,
                                                 IndexType idx,
                                                 ValueType* __restrict__ vec)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    make_ValueType(vec[ai], 0);

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(idx == col[aj])
        {
            vec[ai] = val[aj];
        }
    }
}

// Replace column vector - compute new offset
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_replace_column_vector_offset(const IndexType* __restrict__ row_offset,
                                                        const IndexType* __restrict__ col,
                                                        IndexType nrow,
                                                        IndexType idx,
                                                        const ValueType* __restrict__ vec,
                                                        IndexType* __restrict__ offset)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    bool add      = true;
    IndexType val = row_offset[ai + 1] - row_offset[ai];

    for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
    {
        if(col[aj] == idx)
        {
            add = false;
            break;
        }
    }

    if(add == true && hip_abs(vec[ai]) != 0.0)
    {
        ++val;
    }

    if(add == false && hip_abs(vec[ai]) == 0.0)
    {
        --val;
    }

    offset[ai + 1] = val;
}

// Replace column vector - compute new offset
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_replace_column_vector(const IndexType* __restrict__ row_offset,
                                                 const IndexType* __restrict__ col,
                                                 const ValueType* __restrict__ val,
                                                 IndexType nrow,
                                                 IndexType idx,
                                                 const ValueType* __restrict__ vec,
                                                 const IndexType* __restrict__ offset,
                                                 IndexType* __restrict__ new_col,
                                                 ValueType* __restrict__ new_val)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    IndexType aj = row_offset[ai];
    IndexType k  = offset[ai];

    for(; aj < row_offset[ai + 1]; ++aj)
    {
        if(col[aj] < idx)
        {
            new_col[k] = col[aj];
            new_val[k] = val[aj];
            ++k;
        }
        else
        {
            break;
        }
    }

    if(hip_abs(vec[ai]) != 0.0)
    {
        new_col[k] = idx;
        new_val[k] = vec[ai];
        ++k;
        ++aj;
    }

    for(; aj < row_offset[ai + 1]; ++aj)
    {
        if(col[aj] > idx)
        {
            new_col[k] = col[aj];
            new_val[k] = val[aj];
            ++k;
        }
    }
}

// Extract row vector
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_row_vector(const IndexType* __restrict__ row_offset,
                                              const IndexType* __restrict__ col,
                                              const ValueType* __restrict__ val,
                                              IndexType row_nnz,
                                              IndexType idx,
                                              ValueType* __restrict__ vec)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= row_nnz)
    {
        return;
    }

    IndexType aj = row_offset[idx] + ai;
    vec[col[aj]] = val[aj];
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
