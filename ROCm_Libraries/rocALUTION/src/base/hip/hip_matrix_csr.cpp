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
#include "hip_matrix_csr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_matrix_bcsr.hpp"
#include "hip_matrix_dense.hpp"
#include "hip_vector.hpp"
#include "hip_conversion.hpp"
#include "../host/host_matrix_csr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_csr.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"
#include "hip_sparse.hpp"
#include "../matrix_formats_ind.hpp"

#include <vector>
#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixCSR<ValueType>::HIPAcceleratorMatrixCSR()
{
    // no default constructors
    LOG_INFO("no default constructor");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorMatrixCSR<ValueType>::HIPAcceleratorMatrixCSR(
    const Rocalution_Backend_Descriptor local_backend)
{
    log_debug(this,
              "HIPAcceleratorMatrixCSR::HIPAcceleratorMatrixCSR()",
              "constructor with local_backend");

    this->mat_.row_offset = NULL;
    this->mat_.col        = NULL;
    this->mat_.val        = NULL;
    this->set_backend(local_backend);

    this->L_mat_descr_ = 0;
    this->U_mat_descr_ = 0;

    this->mat_descr_ = 0;
    this->mat_info_  = 0;

    this->mat_buffer_size_ = 0;
    this->mat_buffer_      = NULL;

    this->tmp_vec_ = NULL;

    CHECK_HIP_ERROR(__FILE__, __LINE__);

    rocsparse_status status;

    status = rocsparse_create_mat_descr(&this->mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_index_base(this->mat_descr_, rocsparse_index_base_zero);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_type(this->mat_descr_, rocsparse_matrix_type_general);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_create_mat_info(&this->mat_info_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorMatrixCSR<ValueType>::~HIPAcceleratorMatrixCSR()
{
    log_debug(this, "HIPAcceleratorMatrixCSR::~HIPAcceleratorMatrixCSR()", "destructor");

    this->Clear();

    rocsparse_status status;

    status = rocsparse_destroy_mat_descr(this->mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_destroy_mat_info(this->mat_info_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::Info(void) const
{
    LOG_INFO("HIPAcceleratorMatrixCSR<ValueType>");
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::AllocateCSR(int nnz, int nrow, int ncol)
{
    assert(nnz >= 0);
    assert(ncol >= 0);
    assert(nrow >= 0);

    if(this->nnz_ > 0)
    {
        this->Clear();
    }

    if(nnz > 0)
    {
        allocate_hip(nrow + 1, &this->mat_.row_offset);
        allocate_hip(nnz, &this->mat_.col);
        allocate_hip(nnz, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nrow + 1, mat_.row_offset);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.col);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::SetDataPtrCSR(
    int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol)
{
    assert(*row_offset != NULL);
    assert(*col != NULL);
    assert(*val != NULL);
    assert(nnz > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    this->Clear();

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

    hipDeviceSynchronize();

    this->mat_.row_offset = *row_offset;
    this->mat_.col        = *col;
    this->mat_.val        = *val;

    this->ApplyAnalysis();
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LeaveDataPtrCSR(int** row_offset,
                                                         int** col,
                                                         ValueType** val)
{
    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);
    assert(this->nnz_ > 0);

    hipDeviceSynchronize();

    // see free_host function for details
    *row_offset = this->mat_.row_offset;
    *col        = this->mat_.col;
    *val        = this->mat_.val;

    this->mat_.row_offset = NULL;
    this->mat_.col        = NULL;
    this->mat_.val        = NULL;

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::Clear(void)
{
    if(this->nnz_ > 0)
    {
        free_hip(&this->mat_.row_offset);
        free_hip(&this->mat_.col);
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;

        this->LAnalyseClear();
        this->UAnalyseClear();
        this->LUAnalyseClear();
        this->LLAnalyseClear();
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Zeros()
{
    if(this->nnz_ > 0)
    {
        set_to_zero_hip(this->local_backend_.HIP_block_size, this->nnz_, mat_.val);
    }

    return true;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
{
    const HostMatrixCSR<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // CPU to HIP copy
    if((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->nrow_ == cast_mat->nrow_);
        assert(this->ncol_ == cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(this->mat_.row_offset,
                      cast_mat->mat_.row_offset,
                      (this->nrow_ + 1) * sizeof(int),
                      hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->mat_.col,
                      cast_mat->mat_.col,
                      this->nnz_ * sizeof(int),
                      hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->mat_.val,
                      cast_mat->mat_.val,
                      this->nnz_ * sizeof(ValueType),
                      hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP matrix type");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    this->ApplyAnalysis();
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
{
    const HostMatrixCSR<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // CPU to HIP copy
    if((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->nrow_ == cast_mat->nrow_);
        assert(this->ncol_ == cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpyAsync(this->mat_.row_offset,
                           cast_mat->mat_.row_offset,
                           (this->nrow_ + 1) * sizeof(int),
                           hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpyAsync(this->mat_.col,
                           cast_mat->mat_.col,
                           this->nnz_ * sizeof(int),
                           hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpyAsync(this->mat_.val,
                           cast_mat->mat_.val,
                           this->nnz_ * sizeof(ValueType),
                           hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP matrix type");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    this->ApplyAnalysis();
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
{
    HostMatrixCSR<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to CPU copy
    if((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(dst)) != NULL)
    {
        cast_mat->set_backend(this->local_backend_);

        if(cast_mat->nnz_ == 0)
        {
            cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->ncol_ == cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(cast_mat->mat_.row_offset,
                      this->mat_.row_offset,
                      (this->nrow_ + 1) * sizeof(int),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(cast_mat->mat_.col,
                      this->mat_.col,
                      this->nnz_ * sizeof(int),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(cast_mat->mat_.val,
                      this->mat_.val,
                      this->nnz_ * sizeof(ValueType),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP matrix type");
        this->Info();
        dst->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
{
    HostMatrixCSR<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to CPU copy
    if((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(dst)) != NULL)
    {
        cast_mat->set_backend(this->local_backend_);

        if(cast_mat->nnz_ == 0)
        {
            cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->ncol_ == cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpyAsync(cast_mat->mat_.row_offset,
                           this->mat_.row_offset,
                           (this->nrow_ + 1) * sizeof(int),
                           hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpyAsync(cast_mat->mat_.col,
                           this->mat_.col,
                           this->nnz_ * sizeof(int),
                           hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpyAsync(cast_mat->mat_.val,
                           this->mat_.val,
                           this->nnz_ * sizeof(ValueType),
                           hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP matrix type");
        this->Info();
        dst->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
{
    const HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
    const HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(this->mat_.row_offset,
                      hip_cast_mat->mat_.row_offset,
                      (this->nrow_ + 1) * sizeof(int),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->mat_.col,
                      hip_cast_mat->mat_.col,
                      this->nnz_ * sizeof(int),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->mat_.val,
                      hip_cast_mat->mat_.val,
                      this->nnz_ * sizeof(ValueType),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        // CPU to HIP
        if((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*>(&src)) != NULL)
        {
            this->CopyFromHost(*host_cast_mat);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    this->ApplyAnalysis();
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
{
    const HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
    const HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(this->mat_.row_offset,
                      hip_cast_mat->mat_.row_offset,
                      (this->nrow_ + 1) * sizeof(int),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->mat_.col,
                      hip_cast_mat->mat_.col,
                      this->nnz_ * sizeof(int),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->mat_.val,
                      hip_cast_mat->mat_.val,
                      this->nnz_ * sizeof(ValueType),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        // CPU to HIP
        if((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*>(&src)) != NULL)
        {
            this->CopyFromHostAsync(*host_cast_mat);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    this->ApplyAnalysis();
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
{
    HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
    HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(dst)) != NULL)
    {
        hip_cast_mat->set_backend(this->local_backend_);

        if(hip_cast_mat->nnz_ == 0)
        {
            hip_cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(hip_cast_mat->mat_.row_offset,
                      this->mat_.row_offset,
                      (this->nrow_ + 1) * sizeof(int),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(hip_cast_mat->mat_.col,
                      this->mat_.col,
                      this->nnz_ * sizeof(int),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(hip_cast_mat->mat_.val,
                      this->mat_.val,
                      this->nnz_ * sizeof(ValueType),
                      hipMemcpyDeviceToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        // HIP to CPU
        if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)) != NULL)
        {
            this->CopyToHost(host_cast_mat);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
{
    HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
    HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(dst)) != NULL)
    {
        hip_cast_mat->set_backend(this->local_backend_);

        if(hip_cast_mat->nnz_ == 0)
        {
            hip_cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(hip_cast_mat->mat_.row_offset,
                      this->mat_.row_offset,
                      (this->nrow_ + 1) * sizeof(int),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(hip_cast_mat->mat_.col,
                      this->mat_.col,
                      this->nnz_ * sizeof(int),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(hip_cast_mat->mat_.val,
                      this->mat_.val,
                      this->nnz_ * sizeof(ValueType),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        // HIP to CPU
        if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)) != NULL)
        {
            this->CopyToHostAsync(host_cast_mat);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromCSR(const int* row_offsets,
                                                     const int* col,
                                                     const ValueType* val)
{
    // assert CSR format
    assert(this->GetMatFormat() == CSR);

    if(this->nnz_ > 0)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        hipMemcpy(this->mat_.row_offset,
                  row_offsets,
                  (this->nrow_ + 1) * sizeof(int),
                  hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(this->mat_.col, col, this->nnz_ * sizeof(int), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(this->mat_.val, val, this->nnz_ * sizeof(ValueType), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    this->ApplyAnalysis();
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToCSR(int* row_offsets, int* col, ValueType* val) const
{
    // assert CSR format
    assert(this->GetMatFormat() == CSR);

    if(this->nnz_ > 0)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        hipMemcpy(row_offsets,
                  this->mat_.row_offset,
                  (this->nrow_ + 1) * sizeof(int),
                  hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(col, this->mat_.col, this->nnz_ * sizeof(int), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(val, this->mat_.val, this->nnz_ * sizeof(ValueType), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
    this->Clear();

    // empty matrix is empty matrix
    if(mat.GetNnz() == 0)
    {
        return true;
    }

    const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;
    if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
    {
        this->CopyFrom(*cast_mat_csr);
        return true;
    }

    // Convert from COO to CSR
    const HIPAcceleratorMatrixCOO<ValueType>* cast_mat_coo;
    if((cast_mat_coo = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&mat)) != NULL)
    {
        this->Clear();

        if(coo_to_csr_hip(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                          cast_mat_coo->nnz_,
                          cast_mat_coo->nrow_,
                          cast_mat_coo->ncol_,
                          cast_mat_coo->mat_,
                          &this->mat_) == true)
        {
            this->nrow_ = cast_mat_coo->nrow_;
            this->ncol_ = cast_mat_coo->ncol_;
            this->nnz_  = cast_mat_coo->nnz_;

            this->ApplyAnalysis();

            return true;
        }
    }

    const HIPAcceleratorMatrixELL<ValueType>* cast_mat_ell;
    if((cast_mat_ell = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*>(&mat)) != NULL)
    {
        this->Clear();
        int nnz;

        if(ell_to_csr_hip(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                          cast_mat_ell->nnz_,
                          cast_mat_ell->nrow_,
                          cast_mat_ell->ncol_,
                          cast_mat_ell->mat_,
                          cast_mat_ell->mat_descr_,
                          &this->mat_,
                          this->mat_descr_,
                          &nnz) == true)
        {
            this->nrow_ = cast_mat_ell->nrow_;
            this->ncol_ = cast_mat_ell->ncol_;
            this->nnz_  = nnz;

            this->ApplyAnalysis();

            return true;
        }
    }

    /*
    const HIPAcceleratorMatrixDENSE<ValueType> *cast_mat_dense;
    if ((cast_mat_dense = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*> (&mat)) != NULL)
    {
      this->Clear();
      int nnz = 0;

      FATAL_ERROR(__FILE__, __LINE__);

      this->nrow_ = cast_mat_dense->nrow_;
      this->ncol_ = cast_mat_dense->ncol_;
      this->nnz_  = nnz;

      return true;

    }
    */

    /*
    const HIPAcceleratorMatrixDIA<ValueType>   *cast_mat_dia;
    if ((cast_mat_dia = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*> (&mat)) != NULL) {
      this->Clear();
      int nnz = 0;

      FATAL_ERROR(__FILE__, __LINE__);

      this->nrow_ = cast_mat_dia->nrow_;
      this->ncol_ = cast_mat_dia->ncol_;
      this->nnz_  = nnz ;

      return true;

    }
    */

    /*
    const HIPAcceleratorMatrixMCSR<ValueType>  *cast_mat_mcsr;
    if ((cast_mat_mcsr = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&mat)) != NULL) {
      this->Clear();

      FATAL_ERROR(__FILE__, __LINE__);

      this->nrow_ = cast_mat_mcsr->nrow_;
      this->ncol_ = cast_mat_mcsr->ncol_;
      this->nnz_  = cast_mat_mcsr->nnz_;

      return true;

    }
    */

    /*
    const HIPAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
    if ((cast_mat_hyb = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {
      this->Clear();

      FATAL_ERROR(__FILE__, __LINE__);
      int nnz = 0;

      this->nrow_ = cast_mat_hyb->nrow_;
      this->ncol_ = cast_mat_hyb->ncol_;
      this->nnz_  = nnz;

      return true;

    }
    */

    return false;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHostCSR(
    const int* row_offset, const int* col, const ValueType* val, int nnz, int nrow, int ncol)
{
    assert(nnz >= 0);
    assert(ncol >= 0);
    assert(nrow >= 0);
    assert(row_offset != NULL);
    assert(col != NULL);
    assert(val != NULL);

    // Allocate matrix
    if(this->nnz_ > 0)
    {
        this->Clear();
    }

    if(nnz > 0)
    {
        allocate_hip(nrow + 1, &this->mat_.row_offset);
        allocate_hip(nnz, &this->mat_.col);
        allocate_hip(nnz, &this->mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        hipMemcpy(this->mat_.row_offset,
                  row_offset,
                  (this->nrow_ + 1) * sizeof(int),
                  hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(this->mat_.col, col, this->nnz_ * sizeof(int), hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(this->mat_.val, val, this->nnz_ * sizeof(ValueType), hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    this->ApplyAnalysis();
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Permute(const BaseVector<int>& permutation)
{
    assert(permutation.GetSize() == this->nrow_);
    assert(permutation.GetSize() == this->ncol_);

    if(this->nnz_ > 0)
    {
        int* d_nnzPerm    = NULL;
        int* d_offset     = NULL;
        ValueType* d_data = NULL;

        allocate_hip<int>((this->nrow_ + 1), &d_nnzPerm);
        allocate_hip<ValueType>(this->nnz_, &d_data);
        allocate_hip<int>(this->nnz_, &d_offset);

        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
        assert(cast_perm != NULL);

        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_permute_row_nnz<int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->nrow_,
                           this->mat_.row_offset,
                           cast_perm->vec_,
                           d_nnzPerm);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // hipcub buffer
        size_t size  = 0;
        void* buffer = NULL;

        // Determine maximum
        int* d_max = NULL;
        allocate_hip(1, &d_max);
        hipcub::DeviceReduce::Max(buffer, size, d_nnzPerm, d_max, nrow);
        hipMalloc(&buffer, size);
        hipcub::DeviceReduce::Max(buffer, size, d_nnzPerm, d_max, nrow);
        hipFree(buffer);
        buffer = NULL;

        int maxnnzrow;
        hipMemcpy(&maxnnzrow, d_max, sizeof(int), hipMemcpyDeviceToHost);
        free_hip(&d_max);

        // Inclusive sum
        hipcub::DeviceScan::InclusiveSum(buffer, size, d_nnzPerm + 1, d_nnzPerm + 1, nrow);
        hipMalloc(&buffer, size);
        hipcub::DeviceScan::InclusiveSum(buffer, size, d_nnzPerm + 1, d_nnzPerm + 1, nrow);
        hipFree(buffer);
        buffer = NULL;

        BlockSize = dim3(this->local_backend_.HIP_block_size);
        GridSize  = dim3(
            (this->local_backend_.HIP_warp * nrow - 1) / this->local_backend_.HIP_block_size + 1);

        if(this->local_backend_.HIP_warp == 32)
        {
            hipLaunchKernelGGL((kernel_permute_rows<ValueType, int, 32>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               d_nnzPerm,
                               this->mat_.col,
                               this->mat_.val,
                               cast_perm->vec_,
                               d_offset,
                               d_data);
        }
        else if(this->local_backend_.HIP_warp == 64)
        {
            hipLaunchKernelGGL((kernel_permute_rows<ValueType, int, 64>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               d_nnzPerm,
                               this->mat_.col,
                               this->mat_.val,
                               cast_perm->vec_,
                               d_offset,
                               d_data);
        }
        else
        {
            LOG_INFO("Unsupported HIP warp size of " << this->local_backend_.HIP_warp);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&this->mat_.row_offset);

        this->mat_.row_offset = d_nnzPerm;

        if(maxnnzrow > 64)
        {
            hipLaunchKernelGGL((kernel_permute_cols_fallback<ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               cast_perm->vec_,
                               d_offset,
                               d_data,
                               this->mat_.col,
                               this->mat_.val);
        }
        else if(maxnnzrow > 32)
        {
            hipLaunchKernelGGL((kernel_permute_cols<64, ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               cast_perm->vec_,
                               d_offset,
                               d_data,
                               this->mat_.col,
                               this->mat_.val);
        }
        else if(maxnnzrow > 16)
        {
            hipLaunchKernelGGL((kernel_permute_cols<32, ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               cast_perm->vec_,
                               d_offset,
                               d_data,
                               this->mat_.col,
                               this->mat_.val);
        }
        else if(maxnnzrow > 8)
        {
            hipLaunchKernelGGL((kernel_permute_cols<16, ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               cast_perm->vec_,
                               d_offset,
                               d_data,
                               this->mat_.col,
                               this->mat_.val);
        }
        else if(maxnnzrow > 4)
        {
            hipLaunchKernelGGL((kernel_permute_cols<8, ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               cast_perm->vec_,
                               d_offset,
                               d_data,
                               this->mat_.col,
                               this->mat_.val);
        }
        else
        {
            hipLaunchKernelGGL((kernel_permute_cols<4, ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->nrow_,
                               this->mat_.row_offset,
                               cast_perm->vec_,
                               d_offset,
                               d_data,
                               this->mat_.col,
                               this->mat_.val);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip<int>(&d_offset);
        free_hip<ValueType>(&d_data);
    }

    this->ApplyAnalysis();

    return true;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::ApplyAnalysis(void)
{
    if(this->nnz_ > 0)
    {
        rocsparse_status status;

        status = rocsparseTcsrmv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->ncol_,
                                          this->nnz_,
                                          this->mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
                                               BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HIPAcceleratorVector<ValueType>* cast_in =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
        HIPAcceleratorVector<ValueType>* cast_out =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        ValueType alpha = 1.0;
        ValueType beta  = 0.0;

        rocsparse_status status;
        status = rocsparseTcsrmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->ncol_,
                                 this->nnz_,
                                 &alpha,
                                 this->mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row_offset,
                                 this->mat_.col,
                                 this->mat_info_,
                                 cast_in->vec_,
                                 &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                                  ValueType scalar,
                                                  BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HIPAcceleratorVector<ValueType>* cast_in =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
        HIPAcceleratorVector<ValueType>* cast_out =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        ValueType beta = 1.0;

        rocsparse_status status;
        status = rocsparseTcsrmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->ncol_,
                                 this->nnz_,
                                 &scalar,
                                 this->mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row_offset,
                                 this->mat_.col,
                                 this->mat_info_,
                                 cast_in->vec_,
                                 &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ILU0Factorize(void)
{
    if(this->nnz_ > 0)
    {
        rocsparse_status status;

        // Create buffer, if not already available
        size_t buffer_size = 0;
        status =
            rocsparseTcsrilu0_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          this->nrow_,
                                          this->nnz_,
                                          this->mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          &buffer_size);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            hipMalloc(&this->mat_buffer_, buffer_size);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        status =
            rocsparseTcsrilu0_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->nrow_,
                                       this->nnz_,
                                       this->mat_descr_,
                                       this->mat_.val,
                                       this->mat_.row_offset,
                                       this->mat_.col,
                                       this->mat_info_,
                                       rocsparse_analysis_policy_reuse,
                                       rocsparse_solve_policy_auto,
                                       this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparseTcsrilu0(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                   this->nrow_,
                                   this->nnz_,
                                   this->mat_descr_,
                                   this->mat_.val,
                                   this->mat_.row_offset,
                                   this->mat_.col,
                                   this->mat_info_,
                                   rocsparse_solve_policy_auto,
                                   this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_csrilu0_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ICFactorize(BaseVector<ValueType>* inv_diag)
{
    return false;

    // TODO
    if(this->nnz_ > 0)
    {
        /*
            cusparseStatus_t status;

            cusparseSolveAnalysisInfo_t infoA = 0;

            status = cusparseCreateSolveAnalysisInfo(&infoA);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

            status = cusparseSetMatType(this->mat_descr_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

            status = cusparseSetMatFillMode(this->mat_descr_, CUSPARSE_FILL_MODE_LOWER);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

            status = cusparseSetMatDiagType(this->mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

            status =
           cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             this->nrow_, this->nnz_,
                                             this->mat_descr_,
                                             this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                             infoA);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

            status = cusparseDcsric0(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     this->nrow_,
                                     this->mat_descr_,
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                     infoA);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
        */
    }

    return true;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LUAnalyse(void)
{
    assert(this->ncol_ == this->nrow_);
    assert(this->tmp_vec_ == NULL);

    this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);

    assert(this->tmp_vec_ != NULL);

    rocsparse_status status;

    // L part
    status = rocsparse_create_mat_descr(&this->L_mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_unit);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    // U part
    status = rocsparse_create_mat_descr(&this->U_mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_type(this->U_mat_descr_, rocsparse_matrix_type_general);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_index_base(this->U_mat_descr_, rocsparse_index_base_zero);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_fill_mode(this->U_mat_descr_, rocsparse_fill_mode_upper);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_non_unit);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    // Create buffer, if not already available
    size_t buffer_size = 0;
    status = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->nnz_,
                                         this->L_mat_descr_,
                                         this->mat_.val,
                                         this->mat_.row_offset,
                                         this->mat_.col,
                                         this->mat_info_,
                                         &buffer_size);

    // Buffer is shared with ILU0 and other solve functions
    if(this->mat_buffer_ == NULL)
    {
        this->mat_buffer_size_ = buffer_size;
        hipMalloc(&this->mat_buffer_, buffer_size);
    }

    assert(this->mat_buffer_size_ >= buffer_size);
    assert(this->mat_buffer_ != NULL);

    // L part analysis
    status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                      rocsparse_operation_none,
                                      this->nrow_,
                                      this->nnz_,
                                      this->L_mat_descr_,
                                      this->mat_.val,
                                      this->mat_.row_offset,
                                      this->mat_.col,
                                      this->mat_info_,
                                      rocsparse_analysis_policy_reuse,
                                      rocsparse_solve_policy_auto,
                                      this->mat_buffer_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    // U part analysis
    status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                      rocsparse_operation_none,
                                      this->nrow_,
                                      this->nnz_,
                                      this->U_mat_descr_,
                                      this->mat_.val,
                                      this->mat_.row_offset,
                                      this->mat_.col,
                                      this->mat_info_,
                                      rocsparse_analysis_policy_reuse,
                                      rocsparse_solve_policy_auto,
                                      this->mat_buffer_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    // Allocate temporary vector
    tmp_vec_->Allocate(this->nrow_);
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LUAnalyseClear(void)
{
    rocsparse_status status;

    // Clear analysis info
    if(this->L_mat_descr_ != NULL)
    {
        status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->L_mat_descr_,
                                       this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    if(this->U_mat_descr_ != NULL)
    {
        status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->U_mat_descr_,
                                       this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    // Clear matrix descriptor
    if(this->L_mat_descr_ != NULL)
    {
        status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    if(this->U_mat_descr_ != NULL)
    {
        status = rocsparse_destroy_mat_descr(this->U_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    this->L_mat_descr_ = 0;
    this->U_mat_descr_ = 0;

    // Clear buffer
    if(this->mat_buffer_ != NULL)
    {
        hipFree(this->mat_buffer_);
        this->mat_buffer_ = NULL;
    }

    this->mat_buffer_size_ = 0;

    // Clear temporary vector
    if(this->tmp_vec_ != NULL)
    {
        delete this->tmp_vec_;
        this->tmp_vec_ = NULL;
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                                 BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(this->L_mat_descr_ != 0);
        assert(this->U_mat_descr_ != 0);
        assert(this->mat_info_ != 0);

        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);
        assert(this->ncol_ == this->nrow_);

        assert(this->tmp_vec_ != NULL);

        const HIPAcceleratorVector<ValueType>* cast_in =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
        HIPAcceleratorVector<ValueType>* cast_out =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        rocsparse_status status;

        ValueType alpha = static_cast<ValueType>(1);

        // Solve L
        status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->nnz_,
                                 &alpha,
                                 this->L_mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row_offset,
                                 this->mat_.col,
                                 this->mat_info_,
                                 cast_in->vec_,
                                 tmp_vec_->vec_,
                                 rocsparse_solve_policy_auto,
                                 this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Solve U
        status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->nnz_,
                                 &alpha,
                                 this->U_mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row_offset,
                                 this->mat_.col,
                                 this->mat_info_,
                                 tmp_vec_->vec_,
                                 cast_out->vec_,
                                 rocsparse_solve_policy_auto,
                                 this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LLAnalyse(void)
{
    /*
        cusparseStatus_t status;

        // L part
        status = cusparseCreateMatDescr(&this->L_mat_descr_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatType(this->L_mat_descr_,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatIndexBase(this->L_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatFillMode(this->L_mat_descr_, CUSPARSE_FILL_MODE_LOWER);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatDiagType(this->L_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseCreateSolveAnalysisInfo(&this->L_mat_info_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        // U part
        status = cusparseCreateMatDescr(&this->U_mat_descr_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatType(this->U_mat_descr_,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatIndexBase(this->U_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatFillMode(this->U_mat_descr_, CUSPARSE_FILL_MODE_LOWER);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseSetMatDiagType(this->U_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseCreateSolveAnalysisInfo(&this->U_mat_info_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        // Analysis
        status = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         this->nrow_, this->nnz_,
                                         this->L_mat_descr_,
                                         this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                         this->L_mat_info_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

        status = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         CUSPARSE_OPERATION_TRANSPOSE,
                                         this->nrow_, this->nnz_,
                                         this->U_mat_descr_,
                                         this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                         this->U_mat_info_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
    */
    assert(this->ncol_ == this->nrow_);
    assert(this->tmp_vec_ == NULL);
    this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);
    assert(this->tmp_vec_ != NULL);

    tmp_vec_->Allocate(this->nrow_);
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LLAnalyseClear(void)
{
    /*
      cusparseStatus_t status;

      if (this->L_mat_info_ != 0) {
        status = cusparseDestroySolveAnalysisInfo(this->L_mat_info_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
      }

      if (this->L_mat_descr_ != 0) {
        status = cusparseDestroyMatDescr(this->L_mat_descr_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
      }

      if (this->U_mat_info_ != 0) {
        status = cusparseDestroySolveAnalysisInfo(this->U_mat_info_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
      }

      if (this->U_mat_descr_ != 0) {
        status = cusparseDestroyMatDescr(this->U_mat_descr_);
        CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
      }

      this->L_mat_descr_ = 0;
      this->U_mat_descr_ = 0;
      this->L_mat_info_ = 0;
      this->U_mat_info_ = 0;
    */
    if(this->tmp_vec_ != NULL)
    {
        delete this->tmp_vec_;
        this->tmp_vec_ = NULL;
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                                 BaseVector<ValueType>* out) const
{
    return false;

    // TODO
    if(this->nnz_ > 0)
    {
        /*
            assert(this->L_mat_descr_ != 0);
            assert(this->U_mat_descr_ != 0);
            assert(this->L_mat_info_  != 0);
            assert(this->U_mat_info_  != 0);

            assert(in.  GetSize()  >= 0);
            assert(out->GetSize()  >= 0);
            assert(in.  GetSize()  == this->ncol_);
            assert(out->GetSize()  == this->nrow_);
            assert(this->ncol_ == this->nrow_);

            const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const
           HIPAcceleratorVector<ValueType>*> (&in);
            HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<
           HIPAcceleratorVector<ValueType>*> (out);

            assert(cast_in != NULL);
            assert(cast_out!= NULL);

            cusparseStatus_t status;

            ValueType one = ValueType(1.0);

            // Solve L
            status = cusparseScsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          this->nrow_,
                                          &one,
                                          this->L_mat_descr_,
                                          this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                          this->L_mat_info_,
                                          cast_in->vec_,
                                          this->tmp_vec_->vec_);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

            // Solve U
            status = cusparseScsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          CUSPARSE_OPERATION_TRANSPOSE,
                                          this->nrow_,
                                          &one,
                                          this->U_mat_descr_,
                                          this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                          this->U_mat_info_,
                                          this->tmp_vec_->vec_,
                                          cast_out->vec_);
            CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
        */
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                                 const BaseVector<ValueType>& inv_diag,
                                                 BaseVector<ValueType>* out) const
{
    return LLSolve(in, out);
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LAnalyse(bool diag_unit)
{
    rocsparse_status status;

    // L part
    status = rocsparse_create_mat_descr(&this->L_mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    if(diag_unit == true)
    {
        status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
    else
    {
        status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_non_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    // Create buffer, if not already available
    size_t buffer_size = 0;
    status = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->nnz_,
                                         this->L_mat_descr_,
                                         this->mat_.val,
                                         this->mat_.row_offset,
                                         this->mat_.col,
                                         this->mat_info_,
                                         &buffer_size);

    // Buffer is shared with ILU0 and other solve functions
    if(this->mat_buffer_ == NULL)
    {
        this->mat_buffer_size_ = buffer_size;
        hipMalloc(&this->mat_buffer_, buffer_size);
    }

    assert(this->mat_buffer_size_ >= buffer_size);
    assert(this->mat_buffer_ != NULL);

    status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                      rocsparse_operation_none,
                                      this->nrow_,
                                      this->nnz_,
                                      this->L_mat_descr_,
                                      this->mat_.val,
                                      this->mat_.row_offset,
                                      this->mat_.col,
                                      this->mat_info_,
                                      rocsparse_analysis_policy_reuse,
                                      rocsparse_solve_policy_auto,
                                      this->mat_buffer_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::UAnalyse(bool diag_unit)
{
    rocsparse_status status;

    // U part
    status = rocsparse_create_mat_descr(&this->U_mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_type(this->U_mat_descr_, rocsparse_matrix_type_general);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_index_base(this->U_mat_descr_, rocsparse_index_base_zero);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_fill_mode(this->U_mat_descr_, rocsparse_fill_mode_upper);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    if(diag_unit == true)
    {
        status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
    else
    {
        status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_non_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    // Create buffer, if not already available
    size_t buffer_size = 0;
    status = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->nnz_,
                                         this->U_mat_descr_,
                                         this->mat_.val,
                                         this->mat_.row_offset,
                                         this->mat_.col,
                                         this->mat_info_,
                                         &buffer_size);

    // Buffer is shared with ILU0 and other solve functions
    if(this->mat_buffer_ == NULL)
    {
        this->mat_buffer_size_ = buffer_size;
        hipMalloc(&this->mat_buffer_, buffer_size);
    }

    assert(this->mat_buffer_size_ >= buffer_size);
    assert(this->mat_buffer_ != NULL);

    status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                      rocsparse_operation_none,
                                      this->nrow_,
                                      this->nnz_,
                                      this->U_mat_descr_,
                                      this->mat_.val,
                                      this->mat_.row_offset,
                                      this->mat_.col,
                                      this->mat_info_,
                                      rocsparse_analysis_policy_reuse,
                                      rocsparse_solve_policy_auto,
                                      this->mat_buffer_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LAnalyseClear(void)
{
    rocsparse_status status;

    // Clear analysis info
    if(this->L_mat_descr_ != NULL)
    {
        status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->L_mat_descr_,
                                       this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    // Clear buffer
    if(this->mat_buffer_ != NULL)
    {
        hipFree(this->mat_buffer_);
        this->mat_buffer_ = NULL;
    }

    this->mat_buffer_size_ = 0;

    // Clear matrix descriptor
    if(this->L_mat_descr_ != NULL)
    {
        status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    this->L_mat_descr_ = 0;
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::UAnalyseClear(void)
{
    rocsparse_status status;

    // Clear analysis info
    if(this->U_mat_descr_ != NULL)
    {
        status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->U_mat_descr_,
                                       this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    // Clear buffer
    if(this->mat_buffer_ != NULL)
    {
        hipFree(this->mat_buffer_);
        this->mat_buffer_ = NULL;
    }

    this->mat_buffer_size_ = 0;

    // Clear matrix descriptor
    if(this->U_mat_descr_ != NULL)
    {
        status = rocsparse_destroy_mat_descr(this->U_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    this->U_mat_descr_ = 0;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LSolve(const BaseVector<ValueType>& in,
                                                BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(this->L_mat_descr_ != 0);
        assert(this->mat_info_ != 0);

        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);
        assert(this->ncol_ == this->nrow_);
        assert(this->mat_buffer_size_ > 0);
        assert(this->mat_buffer_ != NULL);

        const HIPAcceleratorVector<ValueType>* cast_in =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
        HIPAcceleratorVector<ValueType>* cast_out =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        rocsparse_status status;

        ValueType alpha = static_cast<ValueType>(1);

        // Solve L
        status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->nnz_,
                                 &alpha,
                                 this->L_mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row_offset,
                                 this->mat_.col,
                                 this->mat_info_,
                                 cast_in->vec_,
                                 cast_out->vec_,
                                 rocsparse_solve_policy_auto,
                                 this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::USolve(const BaseVector<ValueType>& in,
                                                BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(this->U_mat_descr_ != 0);
        assert(this->mat_info_ != 0);

        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);
        assert(this->ncol_ == this->nrow_);
        assert(this->mat_buffer_size_ > 0);
        assert(this->mat_buffer_ != NULL);

        const HIPAcceleratorVector<ValueType>* cast_in =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
        HIPAcceleratorVector<ValueType>* cast_out =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        rocsparse_status status;

        ValueType alpha = static_cast<ValueType>(1);

        // Solve U
        status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->nnz_,
                                 &alpha,
                                 this->U_mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row_offset,
                                 this->mat_.col,
                                 this->mat_info_,
                                 cast_in->vec_,
                                 cast_out->vec_,
                                 rocsparse_solve_policy_auto,
                                 this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType>* vec_diag) const
{
    if(this->nnz_ > 0)
    {
        assert(vec_diag != NULL);
        assert(vec_diag->GetSize() == this->nrow_);

        HIPAcceleratorVector<ValueType>* cast_vec_diag =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec_diag);

        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_extract_diag<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           cast_vec_diag->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractInverseDiagonal(
    BaseVector<ValueType>* vec_inv_diag) const
{
    if(this->nnz_ > 0)
    {
        assert(vec_inv_diag != NULL);
        assert(vec_inv_diag->GetSize() == this->nrow_);

        HIPAcceleratorVector<ValueType>* cast_vec_inv_diag =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec_inv_diag);

        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_extract_inv_diag<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           cast_vec_inv_diag->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractSubMatrix(
    int row_offset, int col_offset, int row_size, int col_size, BaseMatrix<ValueType>* mat) const
{
    assert(mat != NULL);

    assert(row_offset >= 0);
    assert(col_offset >= 0);

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    HIPAcceleratorMatrixCSR<ValueType>* cast_mat =
        dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(mat);
    assert(cast_mat != NULL);

    int mat_nnz = 0;

    int* row_nnz = NULL;
    allocate_hip(row_size + 1, &row_nnz);

    // compute the nnz per row in the new matrix

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(row_size / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_extract_submatrix_row_nnz<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       this->mat_.row_offset,
                       this->mat_.col,
                       this->mat_.val,
                       row_offset,
                       col_offset,
                       row_size,
                       col_size,
                       row_nnz);

    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // compute the new nnz by reduction
    std::vector<int> tmp(row_size + 1);
    hipMemcpy(&tmp[1], row_nnz, sizeof(int) * row_size, hipMemcpyDeviceToHost);

    tmp[0] = 0;
    for(int i = 0; i < row_size; ++i)
    {
        tmp[i + 1] += tmp[i];
    }

    mat_nnz = tmp[row_size];

    hipMemcpy(row_nnz, tmp.data(), sizeof(int) * (row_size + 1), hipMemcpyHostToDevice);

    /*
      // TODO replace when PR575 is fixed in HIP
      size_t size = 0;
      void* buffer = NULL;

      hipcub::DeviceScan::ExclusiveSum(buffer, size, row_nnz, row_nnz, row_size + 1);
      hipMalloc(&buffer, size);
      hipcub::DeviceScan::ExclusiveSum(buffer, size, row_nnz, row_nnz, row_size + 1);
      hipFree(buffer);

      hipMemcpy(&mat_nnz, &row_nnz[row_size], sizeof(int), hipMemcpyDeviceToHost);
    */

    // not empty submatrix
    if(mat_nnz > 0)
    {
        cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

        free_hip<int>(&cast_mat->mat_.row_offset);
        cast_mat->mat_.row_offset = row_nnz;
        // copying the sub matrix

        hipLaunchKernelGGL((kernel_csr_extract_submatrix_copy<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           row_offset,
                           col_offset,
                           row_size,
                           col_size,
                           cast_mat->mat_.row_offset,
                           cast_mat->mat_.col,
                           cast_mat->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
    else
    {
        free_hip(&row_nnz);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType>* L) const
{
    assert(L != NULL);

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    HIPAcceleratorMatrixCSR<ValueType>* cast_L =
        dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(L);

    assert(cast_L != NULL);

    cast_L->Clear();

    // compute nnz per row
    int nrow = this->nrow_;

    allocate_hip<int>(nrow + 1, &cast_L->mat_.row_offset);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_slower_nnz_per_row<int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       cast_L->mat_.row_offset + 1);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // partial sum row_nnz to obtain row_offset vector
    // TODO currently performing partial sum on host
    int* h_buffer = NULL;
    allocate_host(nrow + 1, &h_buffer);
    hipMemcpy(h_buffer + 1, cast_L->mat_.row_offset + 1, nrow * sizeof(int), hipMemcpyDeviceToHost);

    h_buffer[0] = 0;
    for(int i = 1; i < nrow + 1; ++i)
        h_buffer[i] += h_buffer[i - 1];

    int nnz_L = h_buffer[nrow];

    hipMemcpy(cast_L->mat_.row_offset, h_buffer, (nrow + 1) * sizeof(int), hipMemcpyHostToDevice);

    free_host(&h_buffer);
    // end TODO

    // allocate lower triangular part structure
    allocate_hip<int>(nnz_L, &cast_L->mat_.col);
    allocate_hip<ValueType>(nnz_L, &cast_L->mat_.val);

    // fill lower triangular part
    hipLaunchKernelGGL((kernel_csr_extract_l_triangular<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       this->mat_.val,
                       cast_L->mat_.row_offset,
                       cast_L->mat_.col,
                       cast_L->mat_.val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cast_L->nrow_ = this->nrow_;
    cast_L->ncol_ = this->ncol_;
    cast_L->nnz_  = nnz_L;

    cast_L->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType>* L) const
{
    assert(L != NULL);

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    HIPAcceleratorMatrixCSR<ValueType>* cast_L =
        dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(L);

    assert(cast_L != NULL);

    cast_L->Clear();

    // compute nnz per row
    int nrow = this->nrow_;

    allocate_hip<int>(nrow + 1, &cast_L->mat_.row_offset);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_lower_nnz_per_row<int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       cast_L->mat_.row_offset + 1);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // partial sum row_nnz to obtain row_offset vector
    // TODO currently performing partial sum on host
    int* h_buffer = NULL;
    allocate_host(nrow + 1, &h_buffer);
    hipMemcpy(h_buffer + 1, cast_L->mat_.row_offset + 1, nrow * sizeof(int), hipMemcpyDeviceToHost);

    h_buffer[0] = 0;
    for(int i = 1; i < nrow + 1; ++i)
        h_buffer[i] += h_buffer[i - 1];

    int nnz_L = h_buffer[nrow];

    hipMemcpy(cast_L->mat_.row_offset, h_buffer, (nrow + 1) * sizeof(int), hipMemcpyHostToDevice);

    free_host(&h_buffer);
    // end TODO

    // allocate lower triangular part structure
    allocate_hip<int>(nnz_L, &cast_L->mat_.col);
    allocate_hip<ValueType>(nnz_L, &cast_L->mat_.val);

    // fill lower triangular part
    hipLaunchKernelGGL((kernel_csr_extract_l_triangular<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       this->mat_.val,
                       cast_L->mat_.row_offset,
                       cast_L->mat_.col,
                       cast_L->mat_.val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cast_L->nrow_ = this->nrow_;
    cast_L->ncol_ = this->ncol_;
    cast_L->nnz_  = nnz_L;

    cast_L->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType>* U) const
{
    assert(U != NULL);

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    HIPAcceleratorMatrixCSR<ValueType>* cast_U =
        dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(U);

    assert(cast_U != NULL);

    cast_U->Clear();

    // compute nnz per row
    int nrow = this->nrow_;

    allocate_hip<int>(nrow + 1, &cast_U->mat_.row_offset);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_supper_nnz_per_row<int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       cast_U->mat_.row_offset + 1);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // partial sum row_nnz to obtain row_offset vector
    // TODO currently performing partial sum on host
    int* h_buffer = NULL;
    allocate_host(nrow + 1, &h_buffer);
    hipMemcpy(h_buffer + 1, cast_U->mat_.row_offset + 1, nrow * sizeof(int), hipMemcpyDeviceToHost);

    h_buffer[0] = 0;
    for(int i = 1; i < nrow + 1; ++i)
        h_buffer[i] += h_buffer[i - 1];

    int nnz_L = h_buffer[nrow];

    hipMemcpy(cast_U->mat_.row_offset, h_buffer, (nrow + 1) * sizeof(int), hipMemcpyHostToDevice);

    free_host(&h_buffer);
    // end TODO

    // allocate lower triangular part structure
    allocate_hip<int>(nnz_L, &cast_U->mat_.col);
    allocate_hip<ValueType>(nnz_L, &cast_U->mat_.val);

    // fill upper triangular part
    hipLaunchKernelGGL((kernel_csr_extract_u_triangular<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       this->mat_.val,
                       cast_U->mat_.row_offset,
                       cast_U->mat_.col,
                       cast_U->mat_.val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cast_U->nrow_ = this->nrow_;
    cast_U->ncol_ = this->ncol_;
    cast_U->nnz_  = nnz_L;

    cast_U->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType>* U) const
{
    assert(U != NULL);

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    HIPAcceleratorMatrixCSR<ValueType>* cast_U =
        dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(U);

    assert(cast_U != NULL);

    cast_U->Clear();

    // compute nnz per row
    int nrow = this->nrow_;

    allocate_hip<int>(nrow + 1, &cast_U->mat_.row_offset);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_upper_nnz_per_row<int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       cast_U->mat_.row_offset + 1);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // partial sum row_nnz to obtain row_offset vector
    // TODO currently performing partial sum on host
    int* h_buffer = NULL;
    allocate_host(nrow + 1, &h_buffer);
    hipMemcpy(h_buffer + 1, cast_U->mat_.row_offset + 1, nrow * sizeof(int), hipMemcpyDeviceToHost);

    h_buffer[0] = 0;
    for(int i = 1; i < nrow + 1; ++i)
        h_buffer[i] += h_buffer[i - 1];

    int nnz_L = h_buffer[nrow];

    hipMemcpy(cast_U->mat_.row_offset, h_buffer, (nrow + 1) * sizeof(int), hipMemcpyHostToDevice);

    free_host(&h_buffer);
    // end TODO

    // allocate lower triangular part structure
    allocate_hip<int>(nnz_L, &cast_U->mat_.col);
    allocate_hip<ValueType>(nnz_L, &cast_U->mat_.val);

    // fill lower triangular part
    hipLaunchKernelGGL((kernel_csr_extract_u_triangular<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       nrow,
                       this->mat_.row_offset,
                       this->mat_.col,
                       this->mat_.val,
                       cast_U->mat_.row_offset,
                       cast_U->mat_.col,
                       cast_U->mat_.val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cast_U->nrow_ = this->nrow_;
    cast_U->ncol_ = this->ncol_;
    cast_U->nnz_  = nnz_L;

    cast_U->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MaximalIndependentSet(int& size,
                                                               BaseVector<int>* permutation) const
{
    assert(permutation != NULL);

    HIPAcceleratorVector<int>* cast_perm = dynamic_cast<HIPAcceleratorVector<int>*>(permutation);

    assert(cast_perm != NULL);
    assert(this->nrow_ == this->ncol_);

    int* h_row_offset = NULL;
    int* h_col        = NULL;

    allocate_host(this->nrow_ + 1, &h_row_offset);
    allocate_host(this->nnz_, &h_col);

    hipMemcpy(h_row_offset,
              this->mat_.row_offset,
              (this->nrow_ + 1) * sizeof(int),
              hipMemcpyDeviceToHost);
    hipMemcpy(h_col, this->mat_.col, this->nnz_ * sizeof(int), hipMemcpyDeviceToHost);

    int* mis = NULL;
    allocate_host(this->nrow_, &mis);
    memset(mis, 0, sizeof(int) * this->nrow_);

    size = 0;

    for(int ai = 0; ai < this->nrow_; ++ai)
    {
        if(mis[ai] == 0)
        {
            // set the node
            mis[ai] = 1;
            ++size;

            // remove all nbh nodes (without diagonal)
            for(int aj = h_row_offset[ai]; aj < h_row_offset[ai + 1]; ++aj)
            {
                if(ai != h_col[aj])
                {
                    mis[h_col[aj]] = -1;
                }
            }
        }
    }

    int* h_perm = NULL;
    allocate_host(this->nrow_, &h_perm);

    int pos = 0;
    for(int ai = 0; ai < this->nrow_; ++ai)
    {
        if(mis[ai] == 1)
        {
            h_perm[ai] = pos;
            ++pos;
        }
        else
        {
            h_perm[ai] = size + ai - pos;
        }
    }

    // Check the permutation
    //
    //  for (int ai=0; ai<this->nrow_; ++ai) {
    //    assert( h_perm[ai] >= 0 );
    //    assert( h_perm[ai] < this->nrow_ );
    //  }

    cast_perm->Allocate(this->nrow_);
    hipMemcpy(cast_perm->vec_, h_perm, permutation->GetSize() * sizeof(int), hipMemcpyHostToDevice);

    free_host(&h_row_offset);
    free_host(&h_col);

    free_host(&h_perm);
    free_host(&mis);

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MultiColoring(int& num_colors,
                                                       int** size_colors,
                                                       BaseVector<int>* permutation) const
{
    assert(permutation != NULL);

    HIPAcceleratorVector<int>* cast_perm = dynamic_cast<HIPAcceleratorVector<int>*>(permutation);

    assert(cast_perm != NULL);

    // node colors (init value = 0 i.e. no color)
    int* color        = NULL;
    int* h_row_offset = NULL;
    int* h_col        = NULL;
    int size          = this->nrow_;

    allocate_host(size, &color);
    allocate_host(this->nrow_ + 1, &h_row_offset);
    allocate_host(this->nnz_, &h_col);

    hipMemcpy(h_row_offset,
              this->mat_.row_offset,
              (this->nrow_ + 1) * sizeof(int),
              hipMemcpyDeviceToHost);
    hipMemcpy(h_col, this->mat_.col, this->nnz_ * sizeof(int), hipMemcpyDeviceToHost);

    memset(color, 0, size * sizeof(int));
    num_colors = 0;
    std::vector<bool> row_col;

    for(int ai = 0; ai < this->nrow_; ++ai)
    {
        color[ai] = 1;
        row_col.clear();
        row_col.assign(num_colors + 2, false);

        for(int aj = h_row_offset[ai]; aj < h_row_offset[ai + 1]; ++aj)
        {
            if(ai != h_col[aj])
            {
                row_col[color[h_col[aj]]] = true;
            }
        }

        for(int aj = h_row_offset[ai]; aj < h_row_offset[ai + 1]; ++aj)
        {
            if(row_col[color[ai]] == true)
            {
                ++color[ai];
            }
        }

        if(color[ai] > num_colors)
        {
            num_colors = color[ai];
        }
    }

    free_host(&h_row_offset);
    free_host(&h_col);

    allocate_host(num_colors, size_colors);
    set_to_zero_host(num_colors, *size_colors);

    int* offsets_color = NULL;
    allocate_host(num_colors, &offsets_color);
    memset(offsets_color, 0, sizeof(int) * num_colors);

    for(int i = 0; i < this->nrow_; ++i)
    {
        ++(*size_colors)[color[i] - 1];
    }

    int total = 0;
    for(int i = 1; i < num_colors; ++i)
    {
        total += (*size_colors)[i - 1];
        offsets_color[i] = total;
        //   LOG_INFO("offsets = " << total);
    }

    int* h_perm = NULL;
    allocate_host(this->nrow_, &h_perm);

    for(int i = 0; i < this->nrow_; ++i)
    {
        h_perm[i] = offsets_color[color[i] - 1];
        ++offsets_color[color[i] - 1];
    }

    cast_perm->Allocate(this->nrow_);
    hipMemcpy(cast_perm->vec_, h_perm, permutation->GetSize() * sizeof(int), hipMemcpyHostToDevice);

    free_host(&h_perm);
    free_host(&color);
    free_host(&offsets_color);

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Scale(ValueType alpha)
{
    if(this->nnz_ > 0)
    {
        rocblas_status status;
        status = rocblasTscal(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                              this->nnz_,
                              &alpha,
                              this->mat_.val,
                              1);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ScaleDiagonal(ValueType alpha)
{
    if(this->nnz_ > 0)
    {
        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_scale_diagonal<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           alpha,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ScaleOffDiagonal(ValueType alpha)
{
    if(this->nnz_ > 0)
    {
        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_scale_offdiagonal<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           alpha,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::AddScalarDiagonal(ValueType alpha)
{
    if(this->nnz_ > 0)
    {
        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_add_diagonal<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           alpha,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::AddScalarOffDiagonal(ValueType alpha)
{
    if(this->nnz_ > 0)
    {
        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_add_offdiagonal<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           alpha,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::AddScalar(ValueType alpha)
{
    if(this->nnz_ > 0)
    {
        int nnz = this->nnz_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nnz / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_buffer_addscalar<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nnz,
                           alpha,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType>& diag)
{
    assert(diag.GetSize() == this->ncol_);

    const HIPAcceleratorVector<ValueType>* cast_diag =
        dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&diag);

    assert(cast_diag != NULL);

    if(this->nnz_ > 0)
    {
        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_diagmatmult_r<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           this->mat_.col,
                           cast_diag->vec_,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType>& diag)
{
    assert(diag.GetSize() == this->ncol_);

    const HIPAcceleratorVector<ValueType>* cast_diag =
        dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&diag);

    assert(cast_diag != NULL);

    if(this->nnz_ > 0)
    {
        int nrow = this->nrow_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_diagmatmult_l<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nrow,
                           this->mat_.row_offset,
                           cast_diag->vec_,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                                    const BaseMatrix<ValueType>& B)
{
    return false;

    // TODO
    assert(A.GetN() == B.GetM());
    assert(A.GetM() > 0);
    assert(B.GetN() > 0);
    assert(B.GetM() > 0);

    const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_A =
        dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&A);
    const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_B =
        dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&B);

    assert(cast_mat_A != NULL);
    assert(cast_mat_B != NULL);

    this->Clear();

    int m    = cast_mat_A->nrow_;
    int n    = cast_mat_B->ncol_;
    int k    = cast_mat_B->nrow_;
    int nnzC = 0;

    allocate_hip(m + 1, &this->mat_.row_offset);
    CHECK_HIP_ERROR(__FILE__, __LINE__);
    /*
      cusparseStatus_t status;

      status = cusparseSetPointerMode(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                      CUSPARSE_POINTER_MODE_HOST);
      CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

      status = cusparseXcsrgemmNnz(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   m, n, k,
                                   cast_mat_A->mat_descr_, cast_mat_A->nnz_,
                                   cast_mat_A->mat_.row_offset, cast_mat_A->mat_.col,
                                   cast_mat_B->mat_descr_, cast_mat_B->nnz_,
                                   cast_mat_B->mat_.row_offset, cast_mat_B->mat_.col,
                                   this->mat_descr_, this->mat_.row_offset,
                                   &nnzC);
      CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
    */
    allocate_hip(nnzC, &this->mat_.col);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    allocate_hip(nnzC, &this->mat_.val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    this->nrow_ = m;
    this->ncol_ = n;
    this->nnz_  = nnzC;
    /*
      status = __cusparseXcsrgemm__(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    m, n, k,
                                    // A
                                    cast_mat_A->mat_descr_, cast_mat_A->nnz_,
                                    cast_mat_A->mat_.val,
                                    cast_mat_A->mat_.row_offset, cast_mat_A->mat_.col,
                                    // B
                                    cast_mat_B->mat_descr_, cast_mat_B->nnz_,
                                    cast_mat_B->mat_.val,
                                    cast_mat_B->mat_.row_offset, cast_mat_B->mat_.col,
                                    // C
                                    this->mat_descr_,
                                    this->mat_.val,
                                    this->mat_.row_offset, this->mat_.col);
      CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
    */

    this->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Gershgorin(ValueType& lambda_min,
                                                    ValueType& lambda_max) const
{
    return false;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType>& mat,
                                                   ValueType alpha,
                                                   ValueType beta,
                                                   bool structure)
{
    return false;

    if(this->nnz_ > 0)
    {
        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat =
            dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat);

        assert(cast_mat != NULL);

        assert(cast_mat->nrow_ == this->nrow_);
        assert(cast_mat->ncol_ == this->ncol_);
        assert(this->nnz_ > 0);
        assert(cast_mat->nnz_ > 0);

        if(structure == false)
        {
            int nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            hipLaunchKernelGGL((kernel_csr_add_csr_same_struct<ValueType, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               nrow,
                               this->mat_.row_offset,
                               this->mat_.col,
                               cast_mat->mat_.row_offset,
                               cast_mat->mat_.col,
                               cast_mat->mat_.val,
                               alpha,
                               beta,
                               this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
        else
        {
            int m              = this->nrow_;
            int n              = this->ncol_;
            int* csrRowPtrC    = NULL;
            int* csrColC       = NULL;
            ValueType* csrValC = NULL;
            int nnzC;
            /*
                  allocate_hip(m+1, &csrRowPtrC);

                  cusparseStatus_t status;

                  cusparseMatDescr_t desc_mat_C = 0;

                  status = cusparseCreateMatDescr(&desc_mat_C);
                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

                  status = cusparseSetMatIndexBase(desc_mat_C, CUSPARSE_INDEX_BASE_ZERO);
                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

                  status = cusparseSetMatType(desc_mat_C, CUSPARSE_MATRIX_TYPE_GENERAL);
                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

                  status =
               cusparseSetPointerMode(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                                  CUSPARSE_POINTER_MODE_HOST);
                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

                  status =
               cusparseXcsrgeamNnz(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                               m, n,
                                               this->mat_descr_, this->nnz_,
                                               this->mat_.row_offset, this->mat_.col,
                                               cast_mat->mat_descr_, cast_mat->nnz_,
                                               cast_mat->mat_.row_offset, cast_mat->mat_.col,
                                               desc_mat_C, csrRowPtrC,
                                               &nnzC);
                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

                  allocate_hip(nnzC, &csrColC);
                  allocate_hip(nnzC, &csrValC);

                  status =
               __cusparseXcsrgeam__(CUSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                                m, n,
                                                // A
                                                &alpha,
                                                this->mat_descr_, this->nnz_,
                                                this->mat_.val,
                                                this->mat_.row_offset, this->mat_.col,
                                                // B
                                                &beta,
                                                cast_mat->mat_descr_, cast_mat->nnz_,
                                                cast_mat->mat_.val,
                                                cast_mat->mat_.row_offset, cast_mat->mat_.col,
                                                // C
                                                desc_mat_C,
                                                csrValC,
                                                csrRowPtrC, csrColC);

                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);

                  status = cusparseDestroyMatDescr(desc_mat_C);
                  CHECK_CUSPARSE_ERROR(status, __FILE__, __LINE__);
            */
            this->Clear();

            this->mat_.row_offset = csrRowPtrC;
            this->mat_.col        = csrColC;
            this->mat_.val        = csrValC;

            this->nrow_ = m;
            this->ncol_ = n;
            this->nnz_  = nnzC;
        }
    }

    this->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Compress(double drop_off)
{
    if(this->nnz_ > 0)
    {
        HIPAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

        tmp.CopyFrom(*this);

        int nrow    = this->nrow_;
        int mat_nnz = 0;

        int* row_offset = NULL;
        allocate_hip(nrow + 1, &row_offset);

        int* mat_row_offset = NULL;
        allocate_hip(nrow + 1, &mat_row_offset);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nrow + 1, row_offset);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_compress_count_nrow<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           nrow,
                           drop_off,
                           row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // TODO replace
        std::vector<int> htmp(nrow + 1);
        hipMemcpy(&htmp[1], row_offset, sizeof(int) * nrow, hipMemcpyDeviceToHost);

        htmp[0] = 0;
        for(int i = 0; i < nrow; ++i)
        {
            htmp[i + 1] += htmp[i];
        }

        hipMemcpy(mat_row_offset, htmp.data(), sizeof(int) * (nrow + 1), hipMemcpyHostToDevice);

        /*
            // TODO replace when PR575 is fixed in HIP
            size_t size = 0;
            void* buffer = NULL;

            hipcub::DeviceScan::ExclusiveSum(buffer, size, row_nnz, row_nnz, row_size + 1);
            hipMalloc(&buffer, size);
            hipcub::DeviceScan::ExclusiveSum(buffer, size, row_nnz, row_nnz, row_size + 1);
            hipFree(buffer);
            buffer = NULL;
        */

        // get the new mat nnz
        hipMemcpy(&mat_nnz, &mat_row_offset[nrow], sizeof(int), hipMemcpyDeviceToHost);

        this->AllocateCSR(mat_nnz, nrow, this->ncol_);

        // TODO - just exchange memory pointers
        // copy row_offset
        hipMemcpy(this->mat_.row_offset,
                  mat_row_offset,
                  (nrow + 1) * sizeof(int),
                  hipMemcpyDeviceToDevice);

        // copy col and val

        hipLaunchKernelGGL((kernel_csr_compress_copy<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           tmp.mat_.row_offset,
                           tmp.mat_.col,
                           tmp.mat_.val,
                           tmp.nrow_,
                           drop_off,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&row_offset);
        free_hip(&mat_row_offset);
    }

    this->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Transpose(void)
{
    if(this->nnz_ > 0)
    {
        HIPAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

        tmp.CopyFrom(*this);

        this->Clear();
        this->AllocateCSR(tmp.nnz_, tmp.ncol_, tmp.nrow_);

        rocsparse_status status;

        size_t buffer_size = 0;
        status =
            rocsparse_csr2csc_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          tmp.nrow_,
                                          tmp.ncol_,
                                          tmp.nnz_,
                                          tmp.mat_.row_offset,
                                          tmp.mat_.col,
                                          rocsparse_action_numeric,
                                          &buffer_size);

        void* buffer = NULL;
        hipMalloc(&buffer, buffer_size);

        status = rocsparseTcsr2csc(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                   tmp.nrow_,
                                   tmp.ncol_,
                                   tmp.nnz_,
                                   tmp.mat_.val,
                                   tmp.mat_.row_offset,
                                   tmp.mat_.col,
                                   this->mat_.val,
                                   this->mat_.col,
                                   this->mat_.row_offset,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        hipFree(buffer);
    }

    this->ApplyAnalysis();

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ReplaceColumnVector(int idx,
                                                             const BaseVector<ValueType>& vec)
{
    assert(vec.GetSize() == this->nrow_);

    if(this->nnz_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_vec =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&vec);
        assert(cast_vec != NULL);

        int* row_offset = NULL;
        int* col        = NULL;
        ValueType* val  = NULL;

        int nrow = this->nrow_;
        int ncol = this->ncol_;

        allocate_hip(nrow + 1, &row_offset);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_replace_column_vector_offset<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->mat_.row_offset,
                           this->mat_.col,
                           nrow,
                           idx,
                           cast_vec->vec_,
                           row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // TODO on GPU when PR575 fixed
        int* host_offset = NULL;
        allocate_host(nrow + 1, &host_offset);

        hipMemcpy(host_offset, row_offset, sizeof(int) * (nrow + 1), hipMemcpyDeviceToHost);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        host_offset[0] = 0;
        for(int i = 0; i < nrow; ++i)
            host_offset[i + 1] += host_offset[i];

        int nnz = host_offset[nrow];

        hipMemcpy(row_offset, host_offset, sizeof(int) * (nrow + 1), hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(nnz, &col);
        allocate_hip(nnz, &val);

        hipLaunchKernelGGL((kernel_csr_replace_column_vector<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           nrow,
                           idx,
                           cast_vec->vec_,
                           row_offset,
                           col,
                           val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        this->Clear();
        this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractColumnVector(int idx,
                                                             BaseVector<ValueType>* vec) const
{
    assert(vec != NULL);
    assert(vec->GetSize() == this->nrow_);

    if(this->nnz_ > 0)
    {
        HIPAcceleratorVector<ValueType>* cast_vec =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec);
        assert(cast_vec != NULL);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(this->nrow_ / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_extract_column_vector<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           this->nrow_,
                           idx,
                           cast_vec->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractRowVector(int idx, BaseVector<ValueType>* vec) const
{
    assert(vec != NULL);
    assert(vec->GetSize() == this->ncol_);

    if(this->nnz_ > 0)
    {
        HIPAcceleratorVector<ValueType>* cast_vec =
            dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec);
        assert(cast_vec != NULL);

        cast_vec->Zeros();

        // Get nnz of row idx
        int nnz[2];

        hipMemcpy(nnz, this->mat_.row_offset + idx, 2 * sizeof(int), hipMemcpyDeviceToHost);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        int row_nnz = nnz[1] - nnz[0];

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(row_nnz / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_csr_extract_row_vector<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           this->mat_.row_offset,
                           this->mat_.col,
                           this->mat_.val,
                           row_nnz,
                           idx,
                           cast_vec->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template class HIPAcceleratorMatrixCSR<double>;
template class HIPAcceleratorMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixCSR<std::complex<double>>;
template class HIPAcceleratorMatrixCSR<std::complex<float>>;
#endif

} // namespace rocalution
