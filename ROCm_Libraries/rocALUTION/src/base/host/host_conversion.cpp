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
#include "host_conversion.hpp"
#include "../matrix_formats.hpp"
#include "../matrix_formats_ind.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"

#include <stdlib.h>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution {

template <typename ValueType, typename IndexType>
bool csr_to_dense(int omp_threads,
                  IndexType nnz,
                  IndexType nrow,
                  IndexType ncol,
                  const MatrixCSR<ValueType, IndexType>& src,
                  MatrixDENSE<ValueType>* dst)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    allocate_host(nrow * ncol, &dst->val);
    set_to_zero_host(nrow * ncol, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
        {
            dst->val[DENSE_IND(i, src.col[j], nrow, ncol)] = src.val[j];
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool dense_to_csr(int omp_threads,
                  IndexType nrow,
                  IndexType ncol,
                  const MatrixDENSE<ValueType>& src,
                  MatrixCSR<ValueType, IndexType>* dst,
                  IndexType* nnz)
{
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    allocate_host(nrow + 1, &dst->row_offset);
    set_to_zero_host(nrow + 1, dst->row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = 0; j < ncol; ++j)
        {
            if(src.val[DENSE_IND(i, j, nrow, ncol)] != static_cast<ValueType>(0))
            {
                dst->row_offset[i] += 1;
            }
        }
    }

    *nnz = 0;
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType tmp      = dst->row_offset[i];
        dst->row_offset[i] = *nnz;
        *nnz += tmp;
    }

    dst->row_offset[nrow] = *nnz;

    allocate_host(*nnz, &dst->col);
    allocate_host(*nnz, &dst->val);

    set_to_zero_host(*nnz, dst->col);
    set_to_zero_host(*nnz, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType ind = dst->row_offset[i];

        for(IndexType j = 0; j < ncol; ++j)
        {
            if(src.val[DENSE_IND(i, j, nrow, ncol)] != static_cast<ValueType>(0))
            {
                dst->val[ind] = src.val[DENSE_IND(i, j, nrow, ncol)];
                dst->col[ind] = j;
                ++ind;
            }
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_mcsr(int omp_threads,
                 IndexType nnz,
                 IndexType nrow,
                 IndexType ncol,
                 const MatrixCSR<ValueType, IndexType>& src,
                 MatrixMCSR<ValueType, IndexType>* dst)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    // No support for non-squared matrices
    if(nrow != ncol)
    {
        return false;
    }

    omp_set_num_threads(omp_threads);

    // Pre-analysing step to check zero diagonal entries
    IndexType diag_entries = 0;

    for(int i = 0; i < nrow; ++i)
    {
        for(int j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
        {
            if(i == src.col[j])
            {
                ++diag_entries;
            }
        }
    }

    IndexType zero_diag_entries = nrow - diag_entries;

    // MCSR does not support zero diagonal entries (yet)
    if(zero_diag_entries > 0)
    {
        return false;
    }

    allocate_host(nrow + 1, &dst->row_offset);
    allocate_host(nnz, &dst->col);
    allocate_host(nnz, &dst->val);

    set_to_zero_host(nrow + 1, dst->row_offset);
    set_to_zero_host(nnz, dst->col);
    set_to_zero_host(nnz, dst->val);

    for(IndexType ai = 0; ai < nrow + 1; ++ai)
    {
        dst->row_offset[ai] = nrow + src.row_offset[ai] - ai;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType ai = 0; ai < nrow; ++ai)
    {
        IndexType correction = ai;
        for(IndexType aj = src.row_offset[ai]; aj < src.row_offset[ai + 1]; ++aj)
        {
            if(ai != src.col[aj])
            {
                IndexType ind = nrow + aj - correction;

                // non-diag
                dst->col[ind] = src.col[aj];
                dst->val[ind] = src.val[aj];
            }
            else
            {
                // diag
                dst->val[ai] = src.val[aj];
                ++correction;
            }
        }
    }

    if(dst->row_offset[nrow] != src.row_offset[nrow])
    {
        return false;
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool mcsr_to_csr(int omp_threads,
                 IndexType nnz,
                 IndexType nrow,
                 IndexType ncol,
                 const MatrixMCSR<ValueType, IndexType>& src,
                 MatrixCSR<ValueType, IndexType>* dst)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    // Only support square matrices
    if(nrow != ncol)
    {
        return false;
    }

    omp_set_num_threads(omp_threads);

    allocate_host(nrow + 1, &dst->row_offset);
    allocate_host(nnz, &dst->col);
    allocate_host(nnz, &dst->val);

    set_to_zero_host(nrow + 1, dst->row_offset);
    set_to_zero_host(nnz, dst->col);
    set_to_zero_host(nnz, dst->val);

    for(IndexType ai = 0; ai < nrow + 1; ++ai)
    {
        dst->row_offset[ai] = src.row_offset[ai] - nrow + ai;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType ai = 0; ai < nrow; ++ai)
    {
        for(IndexType aj = src.row_offset[ai]; aj < src.row_offset[ai + 1]; ++aj)
        {
            IndexType ind = aj - nrow + ai;

            // non-diag
            dst->col[ind] = src.col[aj];
            dst->val[ind] = src.val[aj];
        }

        IndexType diag_ind = src.row_offset[ai + 1] - nrow + ai;

        // diag
        dst->val[diag_ind] = src.val[ai];
        dst->col[diag_ind] = ai;
    }

    if(dst->row_offset[nrow] != src.row_offset[nrow])
    {
        return false;
    }

// Sorting the col (per row)
// Bubble sort algorithm

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = dst->row_offset[i]; j < dst->row_offset[i + 1]; ++j)
        {
            for(IndexType jj = dst->row_offset[i]; jj < dst->row_offset[i + 1] - 1; ++jj)
            {
                if(dst->col[jj] > dst->col[jj + 1])
                {
                    // swap elements
                    IndexType ind = dst->col[jj];
                    ValueType val = dst->val[jj];

                    dst->col[jj] = dst->col[jj + 1];
                    dst->val[jj] = dst->val[jj + 1];

                    dst->col[jj + 1] = ind;
                    dst->val[jj + 1] = val;
                }
            }
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_coo(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixCSR<ValueType, IndexType>& src,
                MatrixCOO<ValueType, IndexType>* dst)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    allocate_host(nnz, &dst->row);
    allocate_host(nnz, &dst->col);
    allocate_host(nnz, &dst->val);

    set_to_zero_host(nnz, dst->row);
    set_to_zero_host(nnz, dst->col);
    set_to_zero_host(nnz, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
        {
            dst->row[j] = i;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nnz; ++i)
    {
        dst->col[i] = src.col[i];
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nnz; ++i)
    {
        dst->val[i] = src.val[i];
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_ell(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixCSR<ValueType, IndexType>& src,
                MatrixELL<ValueType, IndexType>* dst,
                IndexType* nnz_ell)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    dst->max_row = 0;
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType max_row = src.row_offset[i + 1] - src.row_offset[i];

        if(max_row > dst->max_row)
        {
            dst->max_row = max_row;
        }
    }

    *nnz_ell = dst->max_row * nrow;

    // Limit ELL size to 5 times CSR nnz
    if(dst->max_row > 5 * (nnz / nrow))
    {
        return false;
    }

    allocate_host(*nnz_ell, &dst->val);
    allocate_host(*nnz_ell, &dst->col);

    set_to_zero_host(*nnz_ell, dst->val);
    set_to_zero_host(*nnz_ell, dst->col);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType n = 0;

        for(IndexType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
        {
            IndexType ind = ELL_IND(i, n, nrow, dst->max_row);

            dst->val[ind] = src.val[j];
            dst->col[ind] = src.col[j];
            ++n;
        }

        for(IndexType j = src.row_offset[i + 1] - src.row_offset[i]; j < dst->max_row; ++j)
        {
            IndexType ind = ELL_IND(i, n, nrow, dst->max_row);

            dst->val[ind] = static_cast<ValueType>(0);
            dst->col[ind] = static_cast<IndexType>(-1);
            ++n;
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool ell_to_csr(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixELL<ValueType, IndexType>& src,
                MatrixCSR<ValueType, IndexType>* dst,
                IndexType* nnz_csr)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    allocate_host(nrow + 1, &dst->row_offset);
    set_to_zero_host(nrow + 1, dst->row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType ai = 0; ai < nrow; ++ai)
    {
        for(IndexType n = 0; n < src.max_row; ++n)
        {
            IndexType aj = ELL_IND(ai, n, nrow, src.max_row);

            if((src.col[aj] >= 0) && (src.col[aj] < ncol))
            {
                ++dst->row_offset[ai];
            }
        }
    }

    *nnz_csr = 0;
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType tmp      = dst->row_offset[i];
        dst->row_offset[i] = *nnz_csr;
        *nnz_csr += tmp;
    }

    dst->row_offset[nrow] = *nnz_csr;

    allocate_host(*nnz_csr, &dst->col);
    allocate_host(*nnz_csr, &dst->val);

    set_to_zero_host(*nnz_csr, dst->col);
    set_to_zero_host(*nnz_csr, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType ai = 0; ai < nrow; ++ai)
    {
        IndexType ind = dst->row_offset[ai];

        for(IndexType n = 0; n < src.max_row; ++n)
        {
            IndexType aj = ELL_IND(ai, n, nrow, src.max_row);

            if((src.col[aj] >= 0) && (src.col[aj] < ncol))
            {
                dst->col[ind] = src.col[aj];
                dst->val[ind] = src.val[aj];
                ++ind;
            }
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool hyb_to_csr(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                IndexType nnz_ell,
                IndexType nnz_coo,
                const MatrixHYB<ValueType, IndexType>& src,
                MatrixCSR<ValueType, IndexType>* dst,
                IndexType* nnz_csr)
{
    assert(nnz > 0);
    assert(nnz == nnz_ell + nnz_coo);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    allocate_host(nrow + 1, &dst->row_offset);
    set_to_zero_host(nrow + 1, dst->row_offset);

    IndexType start;
    start = 0;

    // TODO
    //#ifdef _OPENMP
    // #pragma omp parallel for private(start)
    //#endif
    for(IndexType ai = 0; ai < nrow; ++ai)
    {
        // ELL
        for(IndexType n = 0; n < src.ELL.max_row; ++n)
        {
            IndexType aj = ELL_IND(ai, n, nrow, src.ELL.max_row);

            if((src.ELL.col[aj] >= 0) && (src.ELL.col[aj] < ncol))
            {
                dst->row_offset[ai] += 1;
            }
        }

        // COO
        for(IndexType i = start; i < nnz_coo; ++i)
        {
            if(src.COO.row[i] == ai)
            {
                dst->row_offset[ai] += 1;
                ++start;
            }

            if(src.COO.row[i] > ai)
            {
                break;
            }
        }
    }

    *nnz_csr = 0;
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType tmp      = dst->row_offset[i];
        dst->row_offset[i] = *nnz_csr;
        *nnz_csr += tmp;
    }

    dst->row_offset[nrow] = *nnz_csr;

    allocate_host(*nnz_csr, &dst->col);
    allocate_host(*nnz_csr, &dst->val);

    set_to_zero_host(*nnz_csr, dst->col);
    set_to_zero_host(*nnz_csr, dst->val);

    start = 0;

    // TODO
    //#ifdef _OPENMP
    //#pragma omp parallel for private(start)
    //#endif
    for(IndexType ai = 0; ai < nrow; ++ai)
    {
        IndexType ind = dst->row_offset[ai];

        // ELL
        for(IndexType n = 0; n < src.ELL.max_row; ++n)
        {
            IndexType aj = ELL_IND(ai, n, nrow, src.ELL.max_row);

            if((src.ELL.col[aj] >= 0) && (src.ELL.col[aj] < ncol))
            {
                dst->col[ind] = src.ELL.col[aj];
                dst->val[ind] = src.ELL.val[aj];
                ++ind;
            }
        }

        // COO
        for(IndexType i = start; i < nnz_coo; ++i)
        {
            if(src.COO.row[i] == ai)
            {
                dst->col[ind] = src.COO.col[i];
                dst->val[ind] = src.COO.val[i];
                ++ind;
                ++start;
            }

            if(src.COO.row[i] > ai)
            {
                break;
            }
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool coo_to_csr(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixCOO<ValueType, IndexType>& src,
                MatrixCSR<ValueType, IndexType>* dst)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    allocate_host(nrow + 1, &dst->row_offset);
    allocate_host(nnz, &dst->col);
    allocate_host(nnz, &dst->val);

    // COO has to be sorted by rows
    for(IndexType i = 1; i < nnz; ++i)
    {
        assert(src.row[i] >= src.row[i - 1]);
    }

    // Initialize row offset with zeros
    set_to_zero_host(nrow + 1, dst->row_offset);

    // Compute nnz entries per row of CSR
    for(IndexType i = 0; i < nnz; ++i)
    {
        ++dst->row_offset[src.row[i] + 1];
    }

    // Do exclusive scan to obtain row ptrs
    for(IndexType i = 0; i < nrow; ++i)
    {
        dst->row_offset[i + 1] += dst->row_offset[i];
    }

    assert(dst->row_offset[nrow] == nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nnz; ++i)
    {
        dst->col[i] = src.col[i];
        dst->val[i] = src.val[i];
    }

// Sorting the col (per row)
// Bubble sort algorithm
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = dst->row_offset[i]; j < dst->row_offset[i + 1]; ++j)
        {
            for(IndexType jj = dst->row_offset[i]; jj < dst->row_offset[i + 1] - 1; ++jj)
            {
                if(dst->col[jj] > dst->col[jj + 1])
                {
                    // swap elements
                    IndexType ind = dst->col[jj];
                    ValueType val = dst->val[jj];

                    dst->col[jj] = dst->col[jj + 1];
                    dst->val[jj] = dst->val[jj + 1];

                    dst->col[jj + 1] = ind;
                    dst->val[jj + 1] = val;
                }
            }
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_dia(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixCSR<ValueType, IndexType>& src,
                MatrixDIA<ValueType, IndexType>* dst,
                IndexType* nnz_dia)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    // Determine number of populated diagonals
    dst->num_diag = 0;

    std::vector<IndexType> diag_idx(nrow + ncol, 0);

    // Loop over rows and increment ndiag counter if diag offset has not been visited yet
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
        {
            // Diagonal offset the current entry belongs to
            IndexType offset = src.col[j] - i + nrow;

            if(!diag_idx[offset])
            {
                diag_idx[offset] = 1;
                ++dst->num_diag;
            }
        }
    }

    IndexType size = nrow > ncol ? nrow : ncol;
    *nnz_dia       = size * dst->num_diag;

    // Conversion fails if DIA nnz exceeds 5 times CSR nnz
    if(dst->num_diag > 5 * (nnz / size))
    {
        return false;
    }

    // Allocate DIA matrix
    allocate_host(dst->num_diag, &dst->offset);
    allocate_host(*nnz_dia, &dst->val);

    set_to_zero_host(*nnz_dia, dst->val);

    for(IndexType i = 0, d = 0; i < nrow + ncol; ++i)
    {
        // Fill DIA offset, if i-th diagonal is populated
        if(diag_idx[i])
        {
            // Store offset index for reverse index access
            diag_idx[i] = d;
            // Store diagonals offset, where the diagonal is offset 0
            // Left from diagonal offsets are decreasing
            // Right from diagonal offsets are increasing
            dst->offset[d++] = i - nrow;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        for(IndexType j = src.row_offset[i]; j < src.row_offset[i + 1]; ++j)
        {
            // Diagonal offset the current entry belongs to
            IndexType offset = src.col[j] - i + nrow;
            dst->val[DIA_IND(i, diag_idx[offset], nrow, dst->num_diag)] = src.val[j];
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool dia_to_csr(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixDIA<ValueType, IndexType>& src,
                MatrixCSR<ValueType, IndexType>* dst,
                IndexType* nnz_csr)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    // Allocate CSR row pointer array
    allocate_host(nrow + 1, &dst->row_offset);

    // Extract CSR row offsets
    dst->row_offset[0] = 0;
    for(IndexType i = 0; i < nrow; ++i)
    {
        // Initialize row offset of i-th row
        dst->row_offset[i + 1] = dst->row_offset[i];

        for(IndexType n = 0; n < src.num_diag; ++n)
        {
            IndexType j = i + src.offset[n];

            if(j >= 0 && j < ncol)
            {
                // Exclude padded zeros
                if(src.val[DIA_IND(i, n, nrow, src.num_diag)] != static_cast<ValueType>(0))
                {
                    ++dst->row_offset[i + 1];
                }
            }
        }
    }

    // CSR nnz
    *nnz_csr = dst->row_offset[nrow];

    // Allocate CSR matrix
    allocate_host(*nnz_csr, &dst->col);
    allocate_host(*nnz_csr, &dst->val);

// Fill CSR col and val arrays
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType idx = dst->row_offset[i];

        for(IndexType n = 0; n < src.num_diag; ++n)
        {
            IndexType j = i + src.offset[n];

            if(j >= 0 && j < ncol)
            {
                ValueType val = src.val[DIA_IND(i, n, nrow, src.num_diag)];

                if(val != static_cast<ValueType>(0))
                {
                    dst->col[idx] = j;
                    dst->val[idx] = val;
                    ++idx;
                }
            }
        }
    }

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_hyb(int omp_threads,
                IndexType nnz,
                IndexType nrow,
                IndexType ncol,
                const MatrixCSR<ValueType, IndexType>& src,
                MatrixHYB<ValueType, IndexType>* dst,
                IndexType* nnz_hyb,
                IndexType* nnz_ell,
                IndexType* nnz_coo)
{
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    omp_set_num_threads(omp_threads);

    // Determine ELL width by average nnz per row
    if(dst->ELL.max_row == 0)
    {
        dst->ELL.max_row = (nnz - 1) / nrow + 1;
    }

    // ELL nnz is ELL width times nrow
    *nnz_ell = dst->ELL.max_row * nrow;
    *nnz_coo = 0;

    // Array to hold COO part nnz per row
    IndexType* coo_row_ptr = NULL;
    allocate_host(nrow + 1, &coo_row_ptr);

    // If there is no ELL part, its easy...
    if(*nnz_ell == 0)
    {
        *nnz_coo = nnz;
        // copy rowoffset to coorownnz
    }
    else
    {
// Compute COO nnz per row and COO nnz
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexType i = 0; i < nrow; ++i)
        {
            IndexType row_nnz  = src.row_offset[i + 1] - src.row_offset[i] - dst->ELL.max_row;
            coo_row_ptr[i + 1] = (row_nnz > 0) ? row_nnz : 0;
        }

        // Exclusive scan
        coo_row_ptr[0] = 0;
        for(IndexType i = 0; i < nrow; ++i)
        {
            coo_row_ptr[i + 1] += coo_row_ptr[i];
        }

        *nnz_coo = coo_row_ptr[nrow];
    }

    *nnz_hyb = *nnz_coo + *nnz_ell;

    if(*nnz_hyb <= 0)
    {
        return false;
    }

    // ELL
    if(*nnz_ell > 0)
    {
        allocate_host(*nnz_ell, &dst->ELL.val);
        allocate_host(*nnz_ell, &dst->ELL.col);
    }

    // COO
    if(*nnz_coo > 0)
    {
        allocate_host(*nnz_coo, &dst->COO.row);
        allocate_host(*nnz_coo, &dst->COO.col);
        allocate_host(*nnz_coo, &dst->COO.val);
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IndexType i = 0; i < nrow; ++i)
    {
        IndexType p         = 0;
        IndexType row_begin = src.row_offset[i];
        IndexType row_end   = src.row_offset[i + 1];
        IndexType coo_idx   = dst->COO.row ? coo_row_ptr[i] : 0;

        // Fill HYB matrix
        for(IndexType j = row_begin; j < row_end; ++j)
        {
            if(p < dst->ELL.max_row)
            {
                // Fill ELL part
                IndexType idx     = ELL_IND(i, p++, nrow, dst->ELL.max_row);
                dst->ELL.col[idx] = src.col[j];
                dst->ELL.val[idx] = src.val[j];
            }
            else
            {
                dst->COO.row[coo_idx] = i;
                dst->COO.col[coo_idx] = src.col[j];
                dst->COO.val[coo_idx] = src.val[j];
                ++coo_idx;
            }
        }

        // Pad remaining ELL structure
        for(IndexType j = row_end - row_begin; j < dst->ELL.max_row; ++j)
        {
            IndexType idx     = ELL_IND(i, p++, nrow, dst->ELL.max_row);
            dst->ELL.col[idx] = -1;
            dst->ELL.val[idx] = static_cast<ValueType>(0);
        }
    }

    free_host(&coo_row_ptr);

    return true;
}

template bool csr_to_coo(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<double, int>& src,
                         MatrixCOO<double, int>* dst);

template bool csr_to_coo(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<float, int>& src,
                         MatrixCOO<float, int>* dst);

#ifdef SUPPORT_COMPLEX
template bool csr_to_coo(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<double>, int>& src,
                         MatrixCOO<std::complex<double>, int>* dst);

template bool csr_to_coo(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<float>, int>& src,
                         MatrixCOO<std::complex<float>, int>* dst);
#endif

template bool csr_to_coo(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<int, int>& src,
                         MatrixCOO<int, int>* dst);

template bool csr_to_mcsr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixCSR<double, int>& src,
                          MatrixMCSR<double, int>* dst);

template bool csr_to_mcsr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixCSR<float, int>& src,
                          MatrixMCSR<float, int>* dst);

#ifdef SUPPORT_COMPLEX
template bool csr_to_mcsr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixCSR<std::complex<double>, int>& src,
                          MatrixMCSR<std::complex<double>, int>* dst);

template bool csr_to_mcsr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixCSR<std::complex<float>, int>& src,
                          MatrixMCSR<std::complex<float>, int>* dst);
#endif

template bool csr_to_mcsr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixCSR<int, int>& src,
                          MatrixMCSR<int, int>* dst);

template bool mcsr_to_csr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixMCSR<double, int>& src,
                          MatrixCSR<double, int>* dst);

template bool mcsr_to_csr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixMCSR<float, int>& src,
                          MatrixCSR<float, int>* dst);

#ifdef SUPPORT_COMPLEX
template bool mcsr_to_csr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixMCSR<std::complex<double>, int>& src,
                          MatrixCSR<std::complex<double>, int>* dst);

template bool mcsr_to_csr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixMCSR<std::complex<float>, int>& src,
                          MatrixCSR<std::complex<float>, int>* dst);
#endif

template bool mcsr_to_csr(int omp_threads,
                          int nnz,
                          int nrow,
                          int ncol,
                          const MatrixMCSR<int, int>& src,
                          MatrixCSR<int, int>* dst);

template bool csr_to_dia(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<double, int>& src,
                         MatrixDIA<double, int>* dst,
                         int* nnz_dia);

template bool csr_to_dia(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<float, int>& src,
                         MatrixDIA<float, int>* dst,
                         int* nnz_dia);

#ifdef SUPPORT_COMPLEX
template bool csr_to_dia(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<double>, int>& src,
                         MatrixDIA<std::complex<double>, int>* dst,
                         int* nnz_dia);

template bool csr_to_dia(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<float>, int>& src,
                         MatrixDIA<std::complex<float>, int>* dst,
                         int* nnz_dia);
#endif

template bool csr_to_dia(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<int, int>& src,
                         MatrixDIA<int, int>* dst,
                         int* nnz_dia);

template bool csr_to_hyb(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<double, int>& src,
                         MatrixHYB<double, int>* dst,
                         int* nnz_hyb,
                         int* nnz_ell,
                         int* nnz_coo);

template bool csr_to_hyb(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<float, int>& src,
                         MatrixHYB<float, int>* dst,
                         int* nnz_hyb,
                         int* nnz_ell,
                         int* nnz_coo);

#ifdef SUPPORT_COMPLEX
template bool csr_to_hyb(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<double>, int>& src,
                         MatrixHYB<std::complex<double>, int>* dst,
                         int* nnz_hyb,
                         int* nnz_ell,
                         int* nnz_coo);

template bool csr_to_hyb(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<float>, int>& src,
                         MatrixHYB<std::complex<float>, int>* dst,
                         int* nnz_hyb,
                         int* nnz_ell,
                         int* nnz_coo);
#endif

template bool csr_to_hyb(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<int, int>& src,
                         MatrixHYB<int, int>* dst,
                         int* nnz_hyb,
                         int* nnz_ell,
                         int* nnz_coo);

template bool csr_to_ell(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<double, int>& src,
                         MatrixELL<double, int>* dst,
                         int* nnz_ell);

template bool csr_to_ell(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<float, int>& src,
                         MatrixELL<float, int>* dst,
                         int* nnz_ell);

#ifdef SUPPORT_COMPLEX
template bool csr_to_ell(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<double>, int>& src,
                         MatrixELL<std::complex<double>, int>* dst,
                         int* nnz_ell);

template bool csr_to_ell(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<std::complex<float>, int>& src,
                         MatrixELL<std::complex<float>, int>* dst,
                         int* nnz_ell);
#endif

template bool csr_to_ell(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCSR<int, int>& src,
                         MatrixELL<int, int>* dst,
                         int* nnz_ell);

template bool csr_to_dense(int omp_threads,
                           int nnz,
                           int nrow,
                           int ncol,
                           const MatrixCSR<double, int>& src,
                           MatrixDENSE<double>* dst);

template bool csr_to_dense(int omp_threads,
                           int nnz,
                           int nrow,
                           int ncol,
                           const MatrixCSR<float, int>& src,
                           MatrixDENSE<float>* dst);

#ifdef SUPPORT_COMPLEX
template bool csr_to_dense(int omp_threads,
                           int nnz,
                           int nrow,
                           int ncol,
                           const MatrixCSR<std::complex<double>, int>& src,
                           MatrixDENSE<std::complex<double>>* dst);

template bool csr_to_dense(int omp_threads,
                           int nnz,
                           int nrow,
                           int ncol,
                           const MatrixCSR<std::complex<float>, int>& src,
                           MatrixDENSE<std::complex<float>>* dst);
#endif

template bool csr_to_dense(int omp_threads,
                           int nnz,
                           int nrow,
                           int ncol,
                           const MatrixCSR<int, int>& src,
                           MatrixDENSE<int>* dst);

template bool dense_to_csr(int omp_threads,
                           int nrow,
                           int ncol,
                           const MatrixDENSE<double>& src,
                           MatrixCSR<double, int>* dst,
                           int* nnz);

template bool dense_to_csr(int omp_threads,
                           int nrow,
                           int ncol,
                           const MatrixDENSE<float>& src,
                           MatrixCSR<float, int>* dst,
                           int* nnz);

#ifdef SUPPORT_COMPLEX
template bool dense_to_csr(int omp_threads,
                           int nrow,
                           int ncol,
                           const MatrixDENSE<std::complex<double>>& src,
                           MatrixCSR<std::complex<double>, int>* dst,
                           int* nnz);

template bool dense_to_csr(int omp_threads,
                           int nrow,
                           int ncol,
                           const MatrixDENSE<std::complex<float>>& src,
                           MatrixCSR<std::complex<float>, int>* dst,
                           int* nnz);
#endif

template bool dense_to_csr(int omp_threads,
                           int nrow,
                           int ncol,
                           const MatrixDENSE<int>& src,
                           MatrixCSR<int, int>* dst,
                           int* nnz);

template bool dia_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixDIA<double, int>& src,
                         MatrixCSR<double, int>* dst,
                         int* nnz_csr);

template bool dia_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixDIA<float, int>& src,
                         MatrixCSR<float, int>* dst,
                         int* nnz_csr);

#ifdef SUPPORT_COMPLEX
template bool dia_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixDIA<std::complex<double>, int>& src,
                         MatrixCSR<std::complex<double>, int>* dst,
                         int* nnz_csr);

template bool dia_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixDIA<std::complex<float>, int>& src,
                         MatrixCSR<std::complex<float>, int>* dst,
                         int* nnz_csr);
#endif

template bool dia_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixDIA<int, int>& src,
                         MatrixCSR<int, int>* dst,
                         int* nnz_csr);

template bool ell_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixELL<double, int>& src,
                         MatrixCSR<double, int>* dst,
                         int* nnz_csr);

template bool ell_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixELL<float, int>& src,
                         MatrixCSR<float, int>* dst,
                         int* nnz_csr);

#ifdef SUPPORT_COMPLEX
template bool ell_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixELL<std::complex<double>, int>& src,
                         MatrixCSR<std::complex<double>, int>* dst,
                         int* nnz_csr);

template bool ell_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixELL<std::complex<float>, int>& src,
                         MatrixCSR<std::complex<float>, int>* dst,
                         int* nnz_csr);
#endif

template bool ell_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixELL<int, int>& src,
                         MatrixCSR<int, int>* dst,
                         int* nnz_csr);

template bool coo_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCOO<double, int>& src,
                         MatrixCSR<double, int>* dst);

template bool coo_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCOO<float, int>& src,
                         MatrixCSR<float, int>* dst);

#ifdef SUPPORT_COMPLEX
template bool coo_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCOO<std::complex<double>, int>& src,
                         MatrixCSR<std::complex<double>, int>* dst);

template bool coo_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCOO<std::complex<float>, int>& src,
                         MatrixCSR<std::complex<float>, int>* dst);
#endif

template bool coo_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         const MatrixCOO<int, int>& src,
                         MatrixCSR<int, int>* dst);

template bool hyb_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         int nnz_ell,
                         int nnz_coo,
                         const MatrixHYB<double, int>& src,
                         MatrixCSR<double, int>* dst,
                         int* nnz_csr);

template bool hyb_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         int nnz_ell,
                         int nnz_coo,
                         const MatrixHYB<float, int>& src,
                         MatrixCSR<float, int>* dst,
                         int* nnz_csr);

#ifdef SUPPORT_COMPLEX
template bool hyb_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         int nnz_ell,
                         int nnz_coo,
                         const MatrixHYB<std::complex<double>, int>& src,
                         MatrixCSR<std::complex<double>, int>* dst,
                         int* nnz_csr);

template bool hyb_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         int nnz_ell,
                         int nnz_coo,
                         const MatrixHYB<std::complex<float>, int>& src,
                         MatrixCSR<std::complex<float>, int>* dst,
                         int* nnz_csr);
#endif

template bool hyb_to_csr(int omp_threads,
                         int nnz,
                         int nrow,
                         int ncol,
                         int nnz_ell,
                         int nnz_coo,
                         const MatrixHYB<int, int>& src,
                         MatrixCSR<int, int>* dst,
                         int* nnz_csr);

} // namespace rocalution
