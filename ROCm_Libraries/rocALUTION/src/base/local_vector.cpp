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

#include "../utils/def.hpp"
#include "backend_manager.hpp"
#include "local_vector.hpp"
#include "local_matrix.hpp"
#include "base_vector.hpp"
#include "host/host_vector.hpp"
#include "backend_manager.hpp"
#include "../utils/log.hpp"
#include "../utils/allocate_free.hpp"

#include <stdlib.h>
#include <sstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rocalution {

template <typename ValueType>
LocalVector<ValueType>::LocalVector()
{
    log_debug(this, "LocalVector::LocalVector()");

    this->object_name_ = "";

    // Create empty vector on the host
    this->vector_host_  = new HostVector<ValueType>(this->local_backend_);
    this->vector_accel_ = NULL;
    this->vector_       = this->vector_host_;
}

template <typename ValueType>
LocalVector<ValueType>::~LocalVector()
{
    log_debug(this, "LocalVector::~LocalVector()");

    this->Clear();
    delete this->vector_;
}

template <typename ValueType>
IndexType2 LocalVector<ValueType>::GetSize(void) const
{
    return (IndexType2) this->vector_->GetSize();
}

template <typename ValueType>
LocalVector<ValueType>& LocalVector<ValueType>::GetInterior()
{
    return *this;
}

template <typename ValueType>
const LocalVector<ValueType>& LocalVector<ValueType>::GetInterior() const
{
    return *this;
}

template <typename ValueType>
void LocalVector<ValueType>::Allocate(std::string name, IndexType2 size)
{
    log_debug(this, "LocalVector::Allocate()", name, size);

    assert(size <= std::numeric_limits<int>::max());
    assert(size >= 0);

    this->object_name_ = name;

    if(size > 0)
    {
        Rocalution_Backend_Descriptor backend = this->local_backend_;

        // init host vector
        if(this->vector_ == this->vector_host_)
        {
            delete this->vector_host_;

            this->vector_host_ = new HostVector<ValueType>(backend);
            assert(this->vector_host_ != NULL);
            this->vector_host_->Allocate(IndexTypeToInt(size));
            this->vector_ = this->vector_host_;
        }
        else
        {
            // init accel vector
            assert(this->vector_ == this->vector_accel_);

            delete this->vector_accel_;

            this->vector_accel_ = _rocalution_init_base_backend_vector<ValueType>(backend);
            assert(this->vector_accel_ != NULL);
            this->vector_accel_->Allocate(IndexTypeToInt(size));
            this->vector_ = this->vector_accel_;
        }
    }
}

template <typename ValueType>
bool LocalVector<ValueType>::Check(void) const
{
    log_debug(this, "LocalVector::Check()");

    bool check = false;

    if(this->is_accel_() == true)
    {
        LocalVector<ValueType> vec;
        vec.CopyFrom(*this);

        check = vec.Check();

        LOG_VERBOSE_INFO(2, "*** warning: LocalVector::Check() is performed on the host");
    }
    else
    {
        check = this->vector_->Check();
    }

    return check;
}

template <typename ValueType>
void LocalVector<ValueType>::SetDataPtr(ValueType** ptr, std::string name, int size)
{
    log_debug(this, "LocalVector::SetDataPtr()", ptr, name, size);

    assert(ptr != NULL);
    assert(*ptr != NULL);
    assert(size > 0);

    this->Clear();

    this->object_name_ = name;

    this->vector_->SetDataPtr(ptr, size);

    *ptr = NULL;
}

template <typename ValueType>
void LocalVector<ValueType>::LeaveDataPtr(ValueType** ptr)
{
    log_debug(this, "LocalVector::LeaveDataPtr()", ptr);

    assert(*ptr == NULL);
    assert(this->GetSize() > 0);

    this->vector_->LeaveDataPtr(ptr);
}

template <typename ValueType>
void LocalVector<ValueType>::Clear(void)
{
    log_debug(this, "LocalVector::Clear()");

    this->vector_->Clear();
}

template <typename ValueType>
void LocalVector<ValueType>::Zeros(void)
{
    log_debug(this, "LocalVector::Zeros()");

    if(this->GetSize() > 0)
    {
        this->vector_->Zeros();
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Ones(void)
{
    log_debug(this, "LocalVector::Ones()");

    if(this->GetSize() > 0)
    {
        this->vector_->Ones();
    }
}

template <typename ValueType>
void LocalVector<ValueType>::SetValues(ValueType val)
{
    log_debug(this, "LocalVector::SetValues()", val);

    if(this->GetSize() > 0)
    {
        this->vector_->SetValues(val);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::SetRandomUniform(unsigned long long seed, ValueType a, ValueType b)
{
    log_debug(this, "LocalVector::SetRandomUniform()", seed, a, b);

    assert(a <= b);

    if(this->GetSize() > 0)
    {
        // host only
        bool on_host = this->is_host_();
        if(on_host == false)
        {
            this->MoveToHost();
        }

        assert(this->vector_ == this->vector_host_);
        this->vector_host_->SetRandomUniform(seed, a, b);

        if(on_host == false)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalVector::SetRandomUniform() is performed on the host");

            this->MoveToAccelerator();
        }
    }
}

template <typename ValueType>
void LocalVector<ValueType>::SetRandomNormal(unsigned long long seed, ValueType mean, ValueType var)
{
    log_debug(this, "LocalVector::SetRandomNormal()", seed, mean, var);

    if(this->GetSize() > 0)
    {
        // host only
        bool on_host = this->is_host_();
        if(on_host == false)
        {
            this->MoveToHost();
        }

        assert(this->vector_ == this->vector_host_);
        this->vector_host_->SetRandomNormal(seed, mean, var);

        if(on_host == false)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalVector::SetRandomNormal() is performed on the host");

            this->MoveToAccelerator();
        }
    }
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFrom(const LocalVector<ValueType>& src)
{
    log_debug(this, "LocalVector::CopyFrom()", (const void*&)src);

    assert(this != &src);

    this->vector_->CopyFrom(*src.vector_);
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFromAsync(const LocalVector<ValueType>& src)
{
    log_debug(this, "LocalVector::CopyFromAsync()", (const void*&)src);

    assert(this->asyncf_ == false);
    assert(this != &src);

    this->vector_->CopyFromAsync(*src.vector_);

    this->asyncf_ = true;
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFromFloat(const LocalVector<float>& src)
{
    log_debug(this, "LocalVector::CopyFromFloat()", (const void*&)src);

    this->vector_->CopyFromFloat(*src.vector_);
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFromDouble(const LocalVector<double>& src)
{
    log_debug(this, "LocalVector::CopyFromDouble()", (const void*&)src);

    this->vector_->CopyFromDouble(*src.vector_);
}

template <typename ValueType>
bool LocalVector<ValueType>::is_host_(void) const
{
    return (this->vector_ == this->vector_host_);
}

template <typename ValueType>
bool LocalVector<ValueType>::is_accel_(void) const
{
    return (this->vector_ == this->vector_accel_);
}

template <typename ValueType>
void LocalVector<ValueType>::CloneFrom(const LocalVector<ValueType>& src)
{
    log_debug(this, "LocalVector::CloneFrom()", (const void*&)src);

    assert(this != &src);

    this->CloneBackend(src);
    this->CopyFrom(src);
}

template <typename ValueType>
void LocalVector<ValueType>::MoveToAccelerator(void)
{
    log_debug(this, "LocalVector::MoveToAccelerator()");

    if(_rocalution_available_accelerator() == false)
    {
        LOG_VERBOSE_INFO(
            4,
            "*** info: LocalVector::MoveToAccelerator() no accelerator available - doing nothing");
    }

    if((_rocalution_available_accelerator() == true) && (this->vector_ == this->vector_host_))
    {
        this->vector_accel_ = _rocalution_init_base_backend_vector<ValueType>(this->local_backend_);

        // Copy to accel
        this->vector_accel_->CopyFrom(*this->vector_host_);

        this->vector_ = this->vector_accel_;
        delete this->vector_host_;
        this->vector_host_ = NULL;

        LOG_VERBOSE_INFO(4,
                         "*** info: LocalVector::MoveToAccelerator() host to accelerator transfer");
    }
}

template <typename ValueType>
void LocalVector<ValueType>::MoveToHost(void)
{
    log_debug(this, "LocalVector::MoveToHost()");

    if(_rocalution_available_accelerator() == false)
    {
        LOG_VERBOSE_INFO(
            4, "*** info: LocalVector::MoveToHost() no accelerator available - doing nothing");
    }

    if((_rocalution_available_accelerator() == true) && (this->vector_ == this->vector_accel_))
    {
        this->vector_host_ = new HostVector<ValueType>(this->local_backend_);

        // Copy to host
        this->vector_host_->CopyFrom(*this->vector_accel_);

        this->vector_ = this->vector_host_;
        delete this->vector_accel_;
        this->vector_accel_ = NULL;

        LOG_VERBOSE_INFO(4, "*** info: LocalVector::MoveToHost() accelerator to host transfer");
    }
}

template <typename ValueType>
void LocalVector<ValueType>::MoveToAcceleratorAsync(void)
{
    log_debug(this, "LocalVector::MoveToAcceleratorAsync()");

    assert(this->asyncf_ == false);

    if(_rocalution_available_accelerator() == false)
    {
        LOG_VERBOSE_INFO(4,
                         "*** info: LocalVector::MoveToAcceleratorAsync() no accelerator "
                         "available - doing nothing");
    }

    if((_rocalution_available_accelerator() == true) && (this->vector_ == this->vector_host_))
    {
        this->vector_accel_ = _rocalution_init_base_backend_vector<ValueType>(this->local_backend_);

        // Copy to accel
        this->vector_accel_->CopyFromAsync(*this->vector_host_);

        this->asyncf_ = true;

        LOG_VERBOSE_INFO(
            4, "*** info: LocalVector::MoveToAcceleratorAsync() host to accelerator transfer");
    }
}

template <typename ValueType>
void LocalVector<ValueType>::MoveToHostAsync(void)
{
    log_debug(this, "LocalVector::MoveToHostAsync()");

    assert(this->asyncf_ == false);

    if(_rocalution_available_accelerator() == false)
    {
        LOG_VERBOSE_INFO(4,
                         "*** info: LocalVector::MoveToAcceleratorAsync() no accelerator "
                         "available - doing nothing");
    }

    if((_rocalution_available_accelerator() == true) && (this->vector_ == this->vector_accel_))
    {
        this->vector_host_ = new HostVector<ValueType>(this->local_backend_);

        // Copy to host
        this->vector_host_->CopyFromAsync(*this->vector_accel_);

        this->asyncf_ = true;

        LOG_VERBOSE_INFO(
            4, "*** info: LocalVector::MoveToHostAsync() accelerator to host transfer (started)");
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Sync(void)
{
    log_debug(this, "LocalVector::Sync()");

    // check for active async transfer
    if(this->asyncf_ == true)
    {
        // The Move*Async function is active
        if((this->vector_accel_ != NULL) && (this->vector_host_ != NULL))
        {
            // MoveToHostAsync();
            if((_rocalution_available_accelerator() == true) &&
               (this->vector_ == this->vector_accel_))
            {
                _rocalution_sync();

                this->vector_ = this->vector_host_;
                delete this->vector_accel_;
                this->vector_accel_ = NULL;

                LOG_VERBOSE_INFO(4,
                                 "*** info: LocalVector::MoveToHostAsync() accelerator to host "
                                 "transfer (synced)");
            }

            // MoveToAcceleratorAsync();
            if((_rocalution_available_accelerator() == true) &&
               (this->vector_ == this->vector_host_))
            {
                _rocalution_sync();

                this->vector_ = this->vector_accel_;
                delete this->vector_host_;
                this->vector_host_ = NULL;
                LOG_VERBOSE_INFO(4,
                                 "*** info: LocalVector::MoveToAcceleratorAsync() host to "
                                 "accelerator transfer (synced)");
            }
        }
        else
        {
            // The Copy*Async function is active
            _rocalution_sync();
            LOG_VERBOSE_INFO(4, "*** info: LocalVector::Copy*Async() transfer (synced)");
        }
    }

    this->asyncf_ = false;
}

template <typename ValueType>
ValueType& LocalVector<ValueType>::operator[](int i)
{
    log_debug(this, "LocalVector::operator[]()", i);

    assert(this->vector_host_ != NULL);
    assert((i >= 0) && (i < vector_host_->size_));

    return vector_host_->vec_[i];
}

template <typename ValueType>
const ValueType& LocalVector<ValueType>::operator[](int i) const
{
    log_debug(this, "LocalVector::operator[]()", i);

    assert(this->vector_host_ != NULL);
    assert((i >= 0) && (i < vector_host_->size_));

    return vector_host_->vec_[i];
}

template <typename ValueType>
void LocalVector<ValueType>::Info(void) const
{
    std::string current_backend_name;

    if(this->vector_ == this->vector_host_)
    {
        current_backend_name = _rocalution_host_name[0];
    }
    else
    {
        assert(this->vector_ == this->vector_accel_);
        current_backend_name = _rocalution_backend_name[this->local_backend_.backend];
    }

    LOG_INFO("LocalVector"
             << " name="
             << this->object_name_
             << ";"
             << " size="
             << this->GetSize()
             << ";"
             << " prec="
             << 8 * sizeof(ValueType)
             << "bit;"
             << " host backend={"
             << _rocalution_host_name[0]
             << "};"
             << " accelerator backend={"
             << _rocalution_backend_name[this->local_backend_.backend]
             << "};"
             << " current="
             << current_backend_name);
}

template <typename ValueType>
void LocalVector<ValueType>::ReadFileASCII(const std::string name)
{
    log_debug(this, "LocalVector::ReadFileASCII()", name);

    this->Clear();

    // host only
    bool on_host = this->is_host_();
    if(on_host == false)
    {
        this->MoveToHost();
    }

    assert(this->vector_ == this->vector_host_);
    this->vector_host_->ReadFileASCII(name);

    this->object_name_ = name;

    if(on_host == false)
    {
        this->MoveToAccelerator();
    }
}

template <typename ValueType>
void LocalVector<ValueType>::WriteFileASCII(const std::string name) const
{
    log_debug(this, "LocalVector::WriteFileASCII()", name);

    if(this->is_host_() == true)
    {
        assert(this->vector_ == this->vector_host_);
        this->vector_host_->WriteFileASCII(name);
    }
    else
    {
        LocalVector<ValueType> vec_host;
        vec_host.CopyFrom(*this);

        assert(vec_host.vector_ == vec_host.vector_host_);
        vec_host.vector_host_->WriteFileASCII(name);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::ReadFileBinary(const std::string name)
{
    log_debug(this, "LocalVector::ReadFileBinary()", name);

    // host only
    bool on_host = this->is_host_();
    if(on_host == false)
    {
        this->MoveToHost();
    }

    assert(this->vector_ == this->vector_host_);
    this->vector_host_->ReadFileBinary(name);

    this->object_name_ = name;

    if(on_host == false)
    {
        this->MoveToAccelerator();
    }
}

template <typename ValueType>
void LocalVector<ValueType>::WriteFileBinary(const std::string name) const
{
    log_debug(this, "LocalVector::WriteFileBinary()", name);

    if(this->is_host_() == true)
    {
        assert(this->vector_ == this->vector_host_);
        this->vector_host_->WriteFileBinary(name);
    }
    else
    {
        LocalVector<ValueType> vec_host;
        vec_host.CopyFrom(*this);

        assert(vec_host.vector_ == vec_host.vector_host_);
        vec_host.vector_host_->WriteFileBinary(name);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::AddScale(const LocalVector<ValueType>& x, ValueType alpha)
{
    log_debug(this, "LocalVector::AddScale()", (const void*&)x, alpha);

    assert(this->GetSize() == x.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->AddScale(*x.vector_, alpha);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x)
{
    log_debug(this, "LocalVector::ScaleAdd()", alpha, (const void*&)x);

    assert(this->GetSize() == x.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->ScaleAdd(alpha, *x.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::ScaleAddScale(ValueType alpha,
                                           const LocalVector<ValueType>& x,
                                           ValueType beta)
{
    log_debug(this, "LocalVector::ScaleAddScale()", alpha, (const void*&)x, beta);

    assert(this->GetSize() == x.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->ScaleAddScale(alpha, *x.vector_, beta);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::ScaleAddScale(ValueType alpha,
                                           const LocalVector<ValueType>& x,
                                           ValueType beta,
                                           int src_offset,
                                           int dst_offset,
                                           int size)
{
    log_debug(this,
              "LocalVector::ScaleAddScale()",
              alpha,
              (const void*&)x,
              beta,
              src_offset,
              dst_offset,
              size);

    assert((IndexType2)src_offset < x.GetSize());
    assert((IndexType2)dst_offset < this->GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->ScaleAddScale(alpha, *x.vector_, beta, src_offset, dst_offset, size);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::ScaleAdd2(ValueType alpha,
                                       const LocalVector<ValueType>& x,
                                       ValueType beta,
                                       const LocalVector<ValueType>& y,
                                       ValueType gamma)
{
    log_debug(
        this, "LocalVector::ScaleAdd2()", alpha, (const void*&)x, beta, (const void*&)y, gamma);

    assert(this->GetSize() == x.GetSize());
    assert(this->GetSize() == y.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_) &&
            (y.vector_ == y.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_) &&
            (y.vector_ == y.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->ScaleAdd2(alpha, *x.vector_, beta, *y.vector_, gamma);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Scale(ValueType alpha)
{
    log_debug(this, "LocalVector::Scale()", alpha);

    if(this->GetSize() > 0)
    {
        this->vector_->Scale(alpha);
    }
}

template <typename ValueType>
ValueType LocalVector<ValueType>::Dot(const LocalVector<ValueType>& x) const
{
    log_debug(this, "LocalVector::Dot()", (const void*&)x);

    assert(this->GetSize() == x.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        return this->vector_->Dot(*x.vector_);
    }
    else
    {
        return static_cast<ValueType>(0);
    }
}

template <typename ValueType>
ValueType LocalVector<ValueType>::DotNonConj(const LocalVector<ValueType>& x) const
{
    log_debug(this, "LocalVector::DotNonConj()", (const void*&)x);

    assert(this->GetSize() == x.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        return this->vector_->DotNonConj(*x.vector_);
    }
    else
    {
        return static_cast<ValueType>(0);
    }
}

template <typename ValueType>
ValueType LocalVector<ValueType>::Norm(void) const
{
    log_debug(this, "LocalVector::Norm()");

    if(this->GetSize() > 0)
    {
        return this->vector_->Norm();
    }
    else
    {
        return static_cast<ValueType>(0);
    }
}

template <typename ValueType>
ValueType LocalVector<ValueType>::Reduce(void) const
{
    log_debug(this, "LocalVector::Reduce()");

    if(this->GetSize() > 0)
    {
        return this->vector_->Reduce();
    }
    else
    {
        return static_cast<ValueType>(0);
    }
}

template <typename ValueType>
ValueType LocalVector<ValueType>::Asum(void) const
{
    log_debug(this, "LocalVector::Asum()");

    if(this->GetSize() > 0)
    {
        return this->vector_->Asum();
    }
    else
    {
        return static_cast<ValueType>(0);
    }
}

template <typename ValueType>
int LocalVector<ValueType>::Amax(ValueType& value) const
{
    log_debug(this, "LocalVector::Amax()", value);

    if(this->GetSize() > 0)
    {
        return this->vector_->Amax(value);
    }
    else
    {
        value = static_cast<ValueType>(0);
        return -1;
    }
}

template <typename ValueType>
void LocalVector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x)
{
    log_debug(this, "LocalVector::PointWiseMult()", (const void*&)x);

    assert(this->GetSize() == x.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->PointWiseMult(*x.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x,
                                           const LocalVector<ValueType>& y)
{
    log_debug(this, "LocalVector::PointWiseMult()", (const void*&)x, (const void*&)y);

    assert(this->GetSize() == x.GetSize());
    assert(this->GetSize() == y.GetSize());
    assert(((this->vector_ == this->vector_host_) && (x.vector_ == x.vector_host_) &&
            (y.vector_ == y.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (x.vector_ == x.vector_accel_) &&
            (y.vector_ == y.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->PointWiseMult(*x.vector_, *y.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFrom(const LocalVector<ValueType>& src,
                                      int src_offset,
                                      int dst_offset,
                                      int size)
{
    log_debug(this, "LocalVector::CopyFrom()", (const void*&)src, src_offset, dst_offset, size);

    assert(&src != this);
    assert((IndexType2)src_offset < src.GetSize());
    assert((IndexType2)dst_offset < this->GetSize());

    assert(((this->vector_ == this->vector_host_) && (src.vector_ == src.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (src.vector_ == src.vector_accel_)));

    this->vector_->CopyFrom(*src.vector_, src_offset, dst_offset, size);
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFromData(const ValueType* data)
{
    log_debug(this, "LocalVector::CopyFromData()", data);

    assert(data != NULL);

    if(this->GetSize() > 0)
    {
        this->vector_->CopyFromData(data);
    }

    this->object_name_ = "Imported from vector";
}

template <typename ValueType>
void LocalVector<ValueType>::CopyToData(ValueType* data) const
{
    log_debug(this, "LocalVector::CopyToData()", data);

    assert(data != NULL);

    if(this->GetSize() > 0)
    {
        this->vector_->CopyToData(data);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Permute(const LocalVector<int>& permutation)
{
    log_debug(this, "LocalVector::Permute()", (const void*&)permutation);

    assert(permutation.GetSize() == this->GetSize());
    assert(((this->vector_ == this->vector_host_) &&
            (permutation.vector_ == permutation.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) &&
            (permutation.vector_ == permutation.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->Permute(*permutation.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::PermuteBackward(const LocalVector<int>& permutation)
{
    log_debug(this, "LocalVector::PermuteBackward()", (const void*&)permutation);

    assert(permutation.GetSize() == this->GetSize());
    assert(((this->vector_ == this->vector_host_) &&
            (permutation.vector_ == permutation.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) &&
            (permutation.vector_ == permutation.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->PermuteBackward(*permutation.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFromPermute(const LocalVector<ValueType>& src,
                                             const LocalVector<int>& permutation)
{
    log_debug(this, "LocalVector::CopyFromPermute()", (const void*&)src, (const void*&)permutation);

    assert(&src != this);
    assert(permutation.GetSize() == this->GetSize());
    assert(this->GetSize() == src.GetSize());
    assert(((this->vector_ == this->vector_host_) && (src.vector_ == src.vector_host_) &&
            (permutation.vector_ == permutation.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (src.vector_ == src.vector_accel_) &&
            (permutation.vector_ == permutation.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->CopyFromPermute(*src.vector_, *permutation.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::CopyFromPermuteBackward(const LocalVector<ValueType>& src,
                                                     const LocalVector<int>& permutation)
{
    log_debug(this,
              "LocalVector::CopyFromPermuteBackward()",
              (const void*&)src,
              (const void*&)permutation);

    assert(&src != this);
    assert(permutation.GetSize() == this->GetSize());
    assert(this->GetSize() == src.GetSize());
    assert(((this->vector_ == this->vector_host_) && (src.vector_ == src.vector_host_) &&
            (permutation.vector_ == permutation.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (src.vector_ == src.vector_accel_) &&
            (permutation.vector_ == permutation.vector_accel_)));

    if(this->GetSize() > 0)
    {
        this->vector_->CopyFromPermuteBackward(*src.vector_, *permutation.vector_);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Restriction(const LocalVector<ValueType>& vec_fine,
                                         const LocalVector<int>& map)
{
    log_debug(this, "LocalVector::Restriction()", (const void*&)vec_fine, (const void*&)map);

    assert(&vec_fine != this);
    assert(
        ((this->vector_ == this->vector_host_) && (vec_fine.vector_ == vec_fine.vector_host_)) ||
        ((this->vector_ == this->vector_accel_) && (vec_fine.vector_ == vec_fine.vector_accel_)));
    assert(((this->vector_ == this->vector_host_) && (map.vector_ == map.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (map.vector_ == map.vector_accel_)));

    if(this->GetSize() > 0)
    {
        bool err = this->vector_->Restriction(*vec_fine.vector_, *map.vector_);

        if((err == false) && (this->is_host_() == true))
        {
            LOG_INFO("Computation of LocalVector::Restriction() fail");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            this->MoveToHost();

            LocalVector<int> map_tmp;
            map_tmp.CopyFrom(map);

            LocalVector<ValueType> vec_fine_tmp;
            vec_fine_tmp.CopyFrom(vec_fine);

            if(this->vector_->Restriction(*vec_fine_tmp.vector_, *map_tmp.vector_) == false)
            {
                LOG_INFO("Computation of LocalVector::Restriction() fail");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            LOG_VERBOSE_INFO(2, "*** warning: LocalVector::Restriction() is performed on the host");

            this->MoveToAccelerator();
        }
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Prolongation(const LocalVector<ValueType>& vec_coarse,
                                          const LocalVector<int>& map)
{
    log_debug(this, "LocalVector::Prolongation()", (const void*&)vec_coarse, (const void*&)map);

    assert(&vec_coarse != this);
    assert(((this->vector_ == this->vector_host_) &&
            (vec_coarse.vector_ == vec_coarse.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) &&
            (vec_coarse.vector_ == vec_coarse.vector_accel_)));
    assert(((this->vector_ == this->vector_host_) && (map.vector_ == map.vector_host_)) ||
           ((this->vector_ == this->vector_accel_) && (map.vector_ == map.vector_accel_)));

    if(this->GetSize() > 0)
    {
        bool err = this->vector_->Prolongation(*vec_coarse.vector_, *map.vector_);

        if((err == false) && (this->is_host_() == true))
        {
            LOG_INFO("Computation of LocalVector::Prolongation() fail");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            this->MoveToHost();

            LocalVector<int> map_tmp;
            map_tmp.CopyFrom(map);

            LocalVector<ValueType> vec_coarse_tmp;
            vec_coarse_tmp.CopyFrom(vec_coarse);

            if(this->vector_->Prolongation(*vec_coarse_tmp.vector_, *map_tmp.vector_) == false)
            {
                LOG_INFO("Computation of LocalVector::Prolongation() fail");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            LOG_VERBOSE_INFO(2,
                             "*** warning: LocalVector::Prolongation() is performed on the host");

            this->MoveToAccelerator();
        }
    }
}

template <typename ValueType>
void LocalVector<ValueType>::SetIndexArray(int size, const int* index)
{
    log_debug(this, "LocalVector::SetIndexArray()", size, index);

    assert(size > 0);
    assert(index != NULL);

    this->vector_->SetIndexArray(size, index);
}

template <typename ValueType>
void LocalVector<ValueType>::GetIndexValues(ValueType* values) const
{
    log_debug(this, "LocalVector::GetIndexValues()", values);

    assert(values != NULL);

    this->vector_->GetIndexValues(values);
}

template <typename ValueType>
void LocalVector<ValueType>::SetIndexValues(const ValueType* values)
{
    log_debug(this, "LocalVector::SetIndexValues()", values);

    assert(values != NULL);

    this->vector_->SetIndexValues(values);
}

template <typename ValueType>
void LocalVector<ValueType>::GetContinuousValues(int start, int end, ValueType* values) const
{
    log_debug(this, "LocalVector::GetContinuousValues()", start, end, values);

    assert(values != NULL);
    assert(start >= 0);
    assert(end >= start);
    assert(end <= this->GetSize());

    this->vector_->GetContinuousValues(start, end, values);
}

template <typename ValueType>
void LocalVector<ValueType>::SetContinuousValues(int start, int end, const ValueType* values)
{
    log_debug(this, "LocalVector::SetContinuousValues()", start, end, values);

    assert(values != NULL);
    assert(start >= 0);
    assert(end >= start);
    assert(end <= this->GetSize());

    this->vector_->SetContinuousValues(start, end, values);
}

template <typename ValueType>
void LocalVector<ValueType>::ExtractCoarseMapping(
    int start, int end, const int* index, int nc, int* size, int* map) const
{
    log_debug(this, "LocalVector::ExtractCoarseMapping()", start, end, index, nc, size, map);

    assert(index != NULL);
    assert(size != NULL);
    assert(map != NULL);
    assert(start >= 0);
    assert(end >= start);

    bool on_host = this->is_host_();

    if(on_host == true)
    {
        this->vector_->ExtractCoarseMapping(start, end, index, nc, size, map);
    }
    else
    {
        LOG_VERBOSE_INFO(
            2, "*** warning: LocalVector::ExtractCoarseMapping() is performed on the host");

        LocalVector<ValueType> vec_host;
        vec_host.CloneFrom(*this);

        vec_host.MoveToHost();

        vec_host.ExtractCoarseMapping(start, end, index, nc, size, map);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::ExtractCoarseBoundary(
    int start, int end, const int* index, int nc, int* size, int* boundary) const
{
    log_debug(this, "LocalVector::ExtractCoarseBoundary()", start, end, index, nc, size, boundary);

    assert(index != NULL);
    assert(size != NULL);
    assert(boundary != NULL);
    assert(start >= 0);
    assert(end >= start);

    bool on_host = this->is_host_();

    if(on_host == true)
    {
        this->vector_->ExtractCoarseBoundary(start, end, index, nc, size, boundary);
    }
    else
    {
        LOG_VERBOSE_INFO(
            2, "*** warning: LocalVector::ExtractCoarseBoundary() is performed on the host");

        LocalVector<ValueType> vec_host;
        vec_host.CloneFrom(*this);

        vec_host.MoveToHost();

        vec_host.ExtractCoarseBoundary(start, end, index, nc, size, boundary);
    }
}

template <typename ValueType>
void LocalVector<ValueType>::Power(double power)
{
    log_debug(this, "LocalVector::Power()", power);

    if(this->GetSize() > 0)
    {
        this->vector_->Power(power);
    }
}

template class LocalVector<double>;
template class LocalVector<float>;
#ifdef SUPPORT_COMPLEX
template class LocalVector<std::complex<double>>;
template class LocalVector<std::complex<float>>;
#endif

template class LocalVector<int>;

} // namespace rocalution
