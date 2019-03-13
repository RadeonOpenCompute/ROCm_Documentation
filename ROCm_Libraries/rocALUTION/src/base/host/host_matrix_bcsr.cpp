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
#include "host_matrix_bcsr.hpp"
#include "host_matrix_csr.hpp"
#include "host_conversion.hpp"
#include "host_vector.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution {

template <typename ValueType>
HostMatrixBCSR<ValueType>::HostMatrixBCSR()
{
    // no default constructors
    LOG_INFO("no default constructor");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HostMatrixBCSR<ValueType>::HostMatrixBCSR(const Rocalution_Backend_Descriptor local_backend)
{
    log_debug(this, "HostMatrixBCSR::HostMatrixBCSR()", "constructor with local_backend");

    this->set_backend(local_backend);

    // not implemented yet
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HostMatrixBCSR<ValueType>::~HostMatrixBCSR()
{
    log_debug(this, "HostMatrixBCSR::~HostMatrixBCSR()", "destructor");

    this->Clear();
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::Info(void) const
{
    // TODO
    LOG_INFO("HostMatrixBCSR<ValueType>");
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::Clear()
{
    if(this->nnz_ > 0)
    {
        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::AllocateBCSR(int nnz, int nrow, int ncol)
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
        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
{
    // copy only in the same format
    assert(this->GetMatFormat() == mat.GetMatFormat());

    if(const HostMatrixBCSR<ValueType>* cast_mat =
           dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
    {
        this->AllocateBCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);

        assert((this->nnz_ == cast_mat->nnz_) && (this->nrow_ == cast_mat->nrow_) &&
               (this->ncol_ == cast_mat->ncol_));

        if(this->nnz_ > 0)
        {
            _set_omp_backend_threads(this->local_backend_, this->nrow_);

            // TODO
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        // Host matrix knows only host matrices
        // -> dispatching
        mat.CopyTo(this);
    }
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
{
    mat->CopyFrom(*this);
}

template <typename ValueType>
bool HostMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
    this->Clear();

    // empty matrix is empty matrix
    if(mat.GetNnz() == 0)
    {
        return true;
    }

    if(const HostMatrixBCSR<ValueType>* cast_mat =
           dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
    {
        this->CopyFrom(*cast_mat);
        return true;
    }

    if(const HostMatrixCSR<ValueType>* cast_mat =
           dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
    {
        this->Clear();
        int nnz = 0;

        // TODO
        //      csr_to_bcsr(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_,
        //                 cast_mat->mat_, &this->mat_, &nnz);
        FATAL_ERROR(__FILE__, __LINE__);

        this->nrow_ = cast_mat->nrow_;
        this->ncol_ = cast_mat->ncol_;
        this->nnz_  = nnz;

        return true;
    }

    return false;
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
                                      BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        //    const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*>
        //    (&in);
        //    HostVector<ValueType> *cast_out      = dynamic_cast<      HostVector<ValueType>*>
        //    (out);

        //    assert(cast_in != NULL);
        //    assert(cast_out!= NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

        // TODO
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                         ValueType scalar,
                                         BaseVector<ValueType>* out) const
{
    if(this->nnz_ > 0)
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        //    const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*>
        //    (&in) ;
        //    HostVector<ValueType> *cast_out      = dynamic_cast<      HostVector<ValueType>*>
        //    (out) ;

        //    assert(cast_in != NULL);
        //    assert(cast_out!= NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

        // TODO
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template class HostMatrixBCSR<double>;
template class HostMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrixBCSR<std::complex<double>>;
template class HostMatrixBCSR<std::complex<float>>;
#endif

} // namespace rocalution
