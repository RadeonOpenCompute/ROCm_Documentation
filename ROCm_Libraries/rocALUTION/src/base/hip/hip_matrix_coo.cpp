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
#include "hip_vector.hpp"
#include "hip_conversion.hpp"
#include "../host/host_matrix_coo.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_coo.hpp"
#include "hip_allocate_free.hpp"
#include "hip_sparse.hpp"
#include "../matrix_formats_ind.hpp"

#include <algorithm>
#include <rocsparse.h>
#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixCOO<ValueType>::HIPAcceleratorMatrixCOO()
{
    // no default constructors
    LOG_INFO("no default constructor");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorMatrixCOO<ValueType>::HIPAcceleratorMatrixCOO(
    const Rocalution_Backend_Descriptor local_backend)
{
    log_debug(this,
              "HIPAcceleratorMatrixCOO::HIPAcceleratorMatrixCOO()",
              "constructor with local_backend");

    this->mat_.row = NULL;
    this->mat_.col = NULL;
    this->mat_.val = NULL;
    this->set_backend(local_backend);

    this->mat_descr_ = 0;

    CHECK_HIP_ERROR(__FILE__, __LINE__);

    rocsparse_status status;

    status = rocsparse_create_mat_descr(&this->mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_index_base(this->mat_descr_, rocsparse_index_base_zero);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    status = rocsparse_set_mat_type(this->mat_descr_, rocsparse_matrix_type_general);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorMatrixCOO<ValueType>::~HIPAcceleratorMatrixCOO()
{
    log_debug(this, "HIPAcceleratorMatrixCOO::~HIPAcceleratorMatrixCOO()", "destructor");

    this->Clear();

    rocsparse_status status;

    status = rocsparse_destroy_mat_descr(this->mat_descr_);
    CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::Info(void) const
{
    LOG_INFO("HIPAcceleratorMatrixCOO<ValueType>");
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::AllocateCOO(int nnz, int nrow, int ncol)
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
        allocate_hip(nnz, &this->mat_.row);
        allocate_hip(nnz, &this->mat_.col);
        allocate_hip(nnz, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.row);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.col);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::SetDataPtrCOO(
    int** row, int** col, ValueType** val, int nnz, int nrow, int ncol)
{
    assert(*row != NULL);
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

    this->mat_.row = *row;
    this->mat_.col = *col;
    this->mat_.val = *val;
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::LeaveDataPtrCOO(int** row, int** col, ValueType** val)
{
    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);
    assert(this->nnz_ > 0);

    hipDeviceSynchronize();

    // see free_host function for details
    *row = this->mat_.row;
    *col = this->mat_.col;
    *val = this->mat_.val;

    this->mat_.row = NULL;
    this->mat_.col = NULL;
    this->mat_.val = NULL;

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::Clear()
{
    if(this->nnz_ > 0)
    {
        free_hip(&this->mat_.row);
        free_hip(&this->mat_.col);
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
{
    const HostMatrixCOO<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // CPU to HIP copy
    if((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCOO(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
        }

        if(this->nnz_ > 0)
        {
            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            hipMemcpy(this->mat_.row,
                      cast_mat->mat_.row,
                      (this->nnz_) * sizeof(int),
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
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
{
    HostMatrixCOO<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to CPU copy
    if((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*>(dst)) != NULL)
    {
        cast_mat->set_backend(this->local_backend_);

        if(cast_mat->nnz_ == 0)
        {
            cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->nrow_ == cast_mat->nrow_);
        assert(this->ncol_ == cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(cast_mat->mat_.row,
                      this->mat_.row,
                      this->nnz_ * sizeof(int),
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
{
    const HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
    const HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCOO(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(this->mat_.row,
                      hip_cast_mat->mat_.row,
                      (this->nnz_) * sizeof(int),
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
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
{
    HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
    HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCOO<ValueType>*>(dst)) != NULL)
    {
        hip_cast_mat->set_backend(this->local_backend_);

        if(hip_cast_mat->nnz_ == 0)
        {
            hip_cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(hip_cast_mat->mat_.row,
                      this->mat_.row,
                      (this->nnz_) * sizeof(int),
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
{
    const HostMatrixCOO<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // CPU to HIP copy
    if((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCOO(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
        }

        if(this->nnz_ > 0)
        {
            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            hipMemcpyAsync(this->mat_.row,
                           cast_mat->mat_.row,
                           (this->nnz_) * sizeof(int),
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
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
{
    HostMatrixCOO<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to CPU copy
    if((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*>(dst)) != NULL)
    {
        cast_mat->set_backend(this->local_backend_);

        if(cast_mat->nnz_ == 0)
        {
            cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        }

        if(this->nnz_ > 0)
        {
            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            hipMemcpyAsync(cast_mat->mat_.row,
                           this->mat_.row,
                           this->nnz_ * sizeof(int),
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
{
    const HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
    const HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateCOO(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(this->mat_.row,
                      hip_cast_mat->mat_.row,
                      (this->nnz_) * sizeof(int),
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
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
{
    HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
    HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCOO<ValueType>*>(dst)) != NULL)
    {
        hip_cast_mat->set_backend(this->local_backend_);

        if(hip_cast_mat->nnz_ == 0)
        {
            hip_cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        if(this->nnz_ > 0)
        {
            hipMemcpy(hip_cast_mat->mat_.row,
                      this->mat_.row,
                      (this->nnz_) * sizeof(int),
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromCOO(const int* row,
                                                     const int* col,
                                                     const ValueType* val)
{
    // assert CSR format
    assert(this->GetMatFormat() == COO);

    if(this->nnz_ > 0)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        hipMemcpy(this->mat_.row, row, this->nnz_ * sizeof(int), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(this->mat_.col, col, this->nnz_ * sizeof(int), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(this->mat_.val, val, this->nnz_ * sizeof(ValueType), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyToCOO(int* row, int* col, ValueType* val) const
{
    // assert CSR format
    assert(this->GetMatFormat() == COO);

    if(this->nnz_ > 0)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        hipMemcpy(row, this->mat_.row, this->nnz_ * sizeof(int), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(col, this->mat_.col, this->nnz_ * sizeof(int), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(val, this->mat_.val, this->nnz_ * sizeof(ValueType), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
    this->Clear();

    // empty matrix is empty matrix
    if(mat.GetNnz() == 0)
    {
        return true;
    }

    const HIPAcceleratorMatrixCOO<ValueType>* cast_mat_coo;

    if((cast_mat_coo = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&mat)) != NULL)
    {
        this->CopyFrom(*cast_mat_coo);
        return true;
    }

    const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;

    if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
    {
        this->Clear();

        if(csr_to_coo_hip(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                          cast_mat_csr->nnz_,
                          cast_mat_csr->nrow_,
                          cast_mat_csr->ncol_,
                          cast_mat_csr->mat_,
                          &this->mat_) == true)
        {
            this->nrow_ = cast_mat_csr->nrow_;
            this->ncol_ = cast_mat_csr->ncol_;
            this->nnz_  = cast_mat_csr->nnz_;

            return true;
        }
    }

    return false;
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::Apply(const BaseVector<ValueType>& in,
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
        status = rocsparseTcoomv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->ncol_,
                                 this->nnz_,
                                 &alpha,
                                 this->mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row,
                                 this->mat_.col,
                                 cast_in->vec_,
                                 &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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
        status = rocsparseTcoomv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->nrow_,
                                 this->ncol_,
                                 this->nnz_,
                                 &scalar,
                                 this->mat_descr_,
                                 this->mat_.val,
                                 this->mat_.row,
                                 this->mat_.col,
                                 cast_in->vec_,
                                 &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }
}

template <typename ValueType>
bool HIPAcceleratorMatrixCOO<ValueType>::Permute(const BaseVector<int>& permutation)
{
    // symmetric permutation only
    assert(permutation.GetSize() == this->nrow_);
    assert(permutation.GetSize() == this->ncol_);

    if(this->nnz_ > 0)
    {
        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
        assert(cast_perm != NULL);

        HIPAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
        src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        src.CopyFrom(*this);

        int nnz = this->nnz_;
        int s   = nnz;
        int k =
            (nnz / this->local_backend_.HIP_block_size) / this->local_backend_.HIP_max_threads + 1;
        if(k > 1)
        {
            s = nnz / k;
        }

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(s / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_coo_permute<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           nnz,
                           src.mat_.row,
                           src.mat_.col,
                           cast_perm->vec_,
                           this->mat_.row,
                           this->mat_.col);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
{
    // symmetric permutation only
    assert(permutation.GetSize() == this->nrow_);
    assert(permutation.GetSize() == this->ncol_);

    if(this->nnz_ > 0)
    {
        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

        assert(cast_perm != NULL);

        int* pb = NULL;
        allocate_hip(this->nrow_, &pb);

        int n = this->nrow_;
        dim3 BlockSize1(this->local_backend_.HIP_block_size);
        dim3 GridSize1(n / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL(
            (kernel_reverse_index<int>), GridSize1, BlockSize1, 0, 0, n, cast_perm->vec_, pb);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        HIPAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
        src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        src.CopyFrom(*this);

        int nnz = this->nnz_;
        int s   = nnz;
        int k =
            (nnz / this->local_backend_.HIP_block_size) / this->local_backend_.HIP_max_threads + 1;
        if(k > 1)
        {
            s = nnz / k;
        }

        dim3 BlockSize2(this->local_backend_.HIP_block_size);
        dim3 GridSize2(s / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_coo_permute<ValueType, int>),
                           GridSize2,
                           BlockSize2,
                           0,
                           0,
                           nnz,
                           src.mat_.row,
                           src.mat_.col,
                           pb,
                           this->mat_.row,
                           this->mat_.col);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&pb);
    }

    return true;
}

template class HIPAcceleratorMatrixCOO<double>;
template class HIPAcceleratorMatrixCOO<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixCOO<std::complex<double>>;
template class HIPAcceleratorMatrixCOO<std::complex<float>>;
#endif

} // namespace rocalution
