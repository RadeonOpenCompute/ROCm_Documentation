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
#include "hip_matrix_bcsr.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_bcsr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_bcsr.hpp"
#include "hip_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixBCSR<ValueType>::HIPAcceleratorMatrixBCSR()
{
    // no default constructors
    LOG_INFO("no default constructor");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorMatrixBCSR<ValueType>::HIPAcceleratorMatrixBCSR(
    const Rocalution_Backend_Descriptor local_backend)
{
    log_debug(this,
              "HIPAcceleratorMatrixBCSR::HIPAcceleratorMatrixBCSR()",
              "constructor with local_backend");

    this->set_backend(local_backend);

    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // this is not working anyway...
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorMatrixBCSR<ValueType>::~HIPAcceleratorMatrixBCSR()
{
    log_debug(this, "HIPAcceleratorMatrixBCSR::~HIPAcceleratorMatrixBCSR()", "destructor");

    this->Clear();
}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::Info(void) const
{
    LOG_INFO("HIPAcceleratorMatrixBCSR<ValueType>");
}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::AllocateBCSR(int nnz, int nrow, int ncol)
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
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::Clear()
{
    if(this->nnz_ > 0)
    {
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
{
    const HostMatrixBCSR<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // CPU to HIP copy
    if((cast_mat = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateBCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->nrow_ == cast_mat->nrow_);
        assert(this->ncol_ == cast_mat->ncol_);

        cast_mat->GetNnz();

        FATAL_ERROR(__FILE__, __LINE__);
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
void HIPAcceleratorMatrixBCSR<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
{
    HostMatrixBCSR<ValueType>* cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to CPU copy
    if((cast_mat = dynamic_cast<HostMatrixBCSR<ValueType>*>(dst)) != NULL)
    {
        cast_mat->set_backend(this->local_backend_);

        if(cast_mat->nnz_ == 0)
        {
            cast_mat->AllocateBCSR(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == cast_mat->nnz_);
        assert(this->nrow_ == cast_mat->nrow_);
        assert(this->ncol_ == cast_mat->ncol_);

        FATAL_ERROR(__FILE__, __LINE__);
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
void HIPAcceleratorMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
{
    const HIPAcceleratorMatrixBCSR<ValueType>* hip_cast_mat;
    const HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == src.GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*>(&src)) != NULL)
    {
        if(this->nnz_ == 0)
        {
            this->AllocateBCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        FATAL_ERROR(__FILE__, __LINE__);
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
void HIPAcceleratorMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
{
    HIPAcceleratorMatrixBCSR<ValueType>* hip_cast_mat;
    HostMatrix<ValueType>* host_cast_mat;

    // copy only in the same format
    assert(this->GetMatFormat() == dst->GetMatFormat());

    // HIP to HIP copy
    if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixBCSR<ValueType>*>(dst)) != NULL)
    {
        hip_cast_mat->set_backend(this->local_backend_);

        if(hip_cast_mat->nnz_ == 0)
        {
            hip_cast_mat->AllocateBCSR(this->nnz_, this->nrow_, this->ncol_);
        }

        assert(this->nnz_ == hip_cast_mat->nnz_);
        assert(this->nrow_ == hip_cast_mat->nrow_);
        assert(this->ncol_ == hip_cast_mat->ncol_);

        FATAL_ERROR(__FILE__, __LINE__);
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
bool HIPAcceleratorMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
    this->Clear();

    // empty matrix is empty matrix
    if(mat.GetNnz() == 0)
    {
        return true;
    }

    const HIPAcceleratorMatrixBCSR<ValueType>* cast_mat_bcsr;
    if((cast_mat_bcsr = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*>(&mat)) != NULL)
    {
        this->CopyFrom(*cast_mat_bcsr);
        return true;
    }

    /*
    const HIPAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
    if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

      this->Clear();

      FATAL_ERROR(__FILE__, __LINE__);

      this->nrow_ = cast_mat_csr->nrow_;
      this->ncol_ = cast_mat_csr->ncol_;
      this->nnz_  = cast_mat_csr->nnz_;

      return true;

    }
    */

    return false;
}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
                                                BaseVector<ValueType>* out) const
{
    /*
      assert(in.  GetSize() >= 0);
      assert(out->GetSize() >= 0);
      assert(in.  GetSize() == this->ncol_);
      assert(out->GetSize() == this->nrow_);

      const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const
      HIPAcceleratorVector<ValueType>*> (&in) ;
      HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<
      HIPAcceleratorVector<ValueType>*> (out) ;

      assert(cast_in != NULL);
      assert(cast_out!= NULL);
    */
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                                   ValueType scalar,
                                                   BaseVector<ValueType>* out) const
{
    FATAL_ERROR(__FILE__, __LINE__);
}

template class HIPAcceleratorMatrixBCSR<double>;
template class HIPAcceleratorMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixBCSR<std::complex<double>>;
template class HIPAcceleratorMatrixBCSR<std::complex<float>>;
#endif

} // namespace rocalution
