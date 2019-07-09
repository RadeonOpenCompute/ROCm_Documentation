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
#include "global_vector.hpp"
#include "local_vector.hpp"
#include "../utils/log.hpp"
#include "../utils/allocate_free.hpp"

#ifdef SUPPORT_MULTINODE
#include "../utils/log_mpi.hpp"
#include "../utils/communicator.hpp"
#endif

#include <math.h>
#include <sstream>
#include <limits>
#include <algorithm>
#include <complex>

namespace rocalution {

template <typename ValueType>
GlobalVector<ValueType>::GlobalVector()
{
    log_debug(this, "GlobalVector::GlobalVector()");

#ifndef SUPPORT_MULTINODE
    LOG_INFO("Multinode support disabled");
    FATAL_ERROR(__FILE__, __LINE__);
#endif

    this->object_name_ = "";

    this->recv_boundary_ = NULL;
    this->send_boundary_ = NULL;

#ifdef SUPPORT_MULTINODE
    this->recv_event_ = NULL;
    this->send_event_ = NULL;
#endif
}

template <typename ValueType>
GlobalVector<ValueType>::GlobalVector(const ParallelManager& pm)
{
    log_debug(this, "GlobalVector::GlobalVector()", (const void*&)pm);

    assert(pm.Status() == true);

    this->object_name_ = "";

    this->pm_ = &pm;

    this->recv_boundary_ = NULL;
    this->send_boundary_ = NULL;

#ifdef SUPPORT_MULTINODE
    this->recv_event_ = new MRequest[pm.nrecv_];
    this->send_event_ = new MRequest[pm.nsend_];
#endif
}

template <typename ValueType>
GlobalVector<ValueType>::~GlobalVector()
{
    log_debug(this, "GlobalVector::~GlobalVector()");

    this->Clear();

#ifdef SUPPORT_MULTINODE
    if(this->recv_event_ != NULL)
    {
        delete[] this->recv_event_;
        this->recv_event_ = NULL;
    }

    if(this->send_event_ != NULL)
    {
        delete[] this->send_event_;
        this->send_event_ = NULL;
    }
#endif
}

template <typename ValueType>
void GlobalVector<ValueType>::Clear(void)
{
    log_debug(this, "GlobalVector::Clear()");

    this->vector_interior_.Clear();
    this->vector_ghost_.Clear();

    if(this->recv_boundary_ != NULL)
    {
        free_host(&this->recv_boundary_);
    }

    if(this->send_boundary_ != NULL)
    {
        free_host(&this->send_boundary_);
    }
}

template <typename ValueType>
void GlobalVector<ValueType>::SetParallelManager(const ParallelManager& pm)
{
    log_debug(this, "GlobalVector::SetParallelManager()", (const void*&)pm);

    assert(pm.Status() == true);

    this->pm_ = &pm;
}

template <typename ValueType>
IndexType2 GlobalVector<ValueType>::GetSize(void) const
{
    return this->pm_->global_size_;
}

template <typename ValueType>
int GlobalVector<ValueType>::GetLocalSize(void) const
{
    return this->vector_interior_.GetLocalSize();
}

template <typename ValueType>
int GlobalVector<ValueType>::GetGhostSize(void) const
{
    return this->vector_ghost_.GetLocalSize();
}

template <typename ValueType>
const LocalVector<ValueType>& GlobalVector<ValueType>::GetInterior() const
{
    log_debug(this, "GlobalVector::GetInterior() const");

    return this->vector_interior_;
}

template <typename ValueType>
LocalVector<ValueType>& GlobalVector<ValueType>::GetInterior()
{
    log_debug(this, "GlobalVector::GetInterior()");

    return this->vector_interior_;
}

template <typename ValueType>
const LocalVector<ValueType>& GlobalVector<ValueType>::GetGhost() const
{
    log_debug(this, "GlobalVector::GetGhost()");

    return this->vector_ghost_;
}

template <typename ValueType>
void GlobalVector<ValueType>::Allocate(std::string name, IndexType2 size)
{
    log_debug(this, "GlobalVector::Allocate()", name, size);

    assert(this->pm_ != NULL);
    assert(this->pm_->global_size_ == size);
    assert(size <= std::numeric_limits<IndexType2>::max());

    std::string interior_name = "Interior of " + name;
    std::string ghost_name    = "Ghost of " + name;

#ifdef SUPPORT_MULTINODE
    if(this->recv_event_ == NULL)
    {
        this->recv_event_ = new MRequest[this->pm_->nrecv_];
    }

    if(this->send_event_ == NULL)
    {
        this->send_event_ = new MRequest[this->pm_->nsend_];
    }
#endif

    this->object_name_ = name;

    this->vector_interior_.Allocate(interior_name, this->pm_->GetLocalSize());
    this->vector_ghost_.Allocate(ghost_name, this->pm_->GetNumReceivers());

    this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

    // Allocate send and receive buffer
    allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
    allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);
}

template <typename ValueType>
void GlobalVector<ValueType>::Zeros(void)
{
    log_debug(this, "GlobalVector::Zeros()");

    this->vector_interior_.Zeros();
}

template <typename ValueType>
void GlobalVector<ValueType>::Ones(void)
{
    log_debug(this, "GlobalVector::Ones()");

    this->vector_interior_.Ones();
}

template <typename ValueType>
void GlobalVector<ValueType>::SetValues(ValueType val)
{
    log_debug(this, "GlobalVector::SetValues()", val);

    this->vector_interior_.SetValues(val);
}

template <typename ValueType>
void GlobalVector<ValueType>::SetDataPtr(ValueType** ptr, std::string name, IndexType2 size)
{
    log_debug(this, "GlobalVector::SetDataPtr()", ptr, name, size);

    assert(ptr != NULL);
    assert(*ptr != NULL);
    assert(this->pm_ != NULL);
    assert(this->pm_->global_size_ == size);
    assert(size <= std::numeric_limits<IndexType2>::max());

    this->Clear();

    std::string interior_name = "Interior of " + name;
    std::string ghost_name    = "Ghost of " + name;

#ifdef SUPPORT_MULTINODE
    if(this->recv_event_ == NULL)
    {
        this->recv_event_ = new MRequest[this->pm_->nrecv_];
    }

    if(this->send_event_ == NULL)
    {
        this->send_event_ = new MRequest[this->pm_->nsend_];
    }
#endif

    this->object_name_ = name;

    this->vector_interior_.SetDataPtr(ptr, interior_name, this->pm_->local_size_);
    this->vector_ghost_.Allocate(ghost_name, this->pm_->GetNumReceivers());

    this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

    // Allocate send and receive buffer
    allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
    allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);
}

template <typename ValueType>
void GlobalVector<ValueType>::LeaveDataPtr(ValueType** ptr)
{
    log_debug(this, "GlobalVector::LeaveDataPtr()", ptr);

    assert(*ptr == NULL);
    assert(this->vector_interior_.GetSize() > 0);

    this->vector_interior_.LeaveDataPtr(ptr);

    free_host(&this->recv_boundary_);
    free_host(&this->send_boundary_);

    this->vector_ghost_.Clear();
}

template <typename ValueType>
void GlobalVector<ValueType>::SetRandomUniform(unsigned long long seed, ValueType a, ValueType b)
{
    log_debug(this, "GlobalVector::SetRandomUniform()", seed, a, b);

    this->vector_interior_.SetRandomUniform(seed, a, b);
}

template <typename ValueType>
void GlobalVector<ValueType>::SetRandomNormal(unsigned long long seed,
                                              ValueType mean,
                                              ValueType var)
{
    log_debug(this, "GlobalVector::SetRandomNormal()", seed, mean, var);

    this->vector_interior_.SetRandomUniform(seed, mean, var);
}

template <typename ValueType>
void GlobalVector<ValueType>::CopyFrom(const GlobalVector<ValueType>& src)
{
    log_debug(this, "GlobalVector::CopyFrom()", (const void*&)src);

    assert(this != &src);
    assert(this->pm_ == src.pm_);
    assert(this->recv_boundary_ != NULL);
    assert(this->send_boundary_ != NULL);

    this->vector_interior_.CopyFrom(src.vector_interior_);
}

template <typename ValueType>
void GlobalVector<ValueType>::CloneFrom(const GlobalVector<ValueType>& src)
{
    log_debug(this, "GlobalVector::CloneFrom()", (const void*&)src);

    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GlobalVector<ValueType>::MoveToAccelerator(void)
{
    log_debug(this, "GlobalVector::MoveToAccelerator()");

    this->vector_interior_.MoveToAccelerator();
    this->vector_ghost_.MoveToAccelerator();
}

template <typename ValueType>
void GlobalVector<ValueType>::MoveToHost(void)
{
    log_debug(this, "GlobalVector::MoveToHost()");

    this->vector_interior_.MoveToHost();
    this->vector_ghost_.MoveToHost();
}

template <typename ValueType>
ValueType& GlobalVector<ValueType>::operator[](int i)
{
    log_debug(this, "GlobalVector::operator[]()", i);

    assert((i >= 0) && (i < this->pm_->local_size_));

    return this->vector_interior_[i];
}

template <typename ValueType>
const ValueType& GlobalVector<ValueType>::operator[](int i) const
{
    log_debug(this, "GlobalVector::operator[]() const", i);

    assert((i >= 0) && (i < this->pm_->local_size_));

    return this->vector_interior_[i];
}

template <typename ValueType>
void GlobalVector<ValueType>::Info(void) const
{
    std::string current_backend_name;

    if(this->is_host_() == true)
    {
        current_backend_name = _rocalution_host_name[0];
    }
    else
    {
        assert(this->is_accel_() == true);
        current_backend_name = _rocalution_backend_name[this->local_backend_.backend];
    }

    LOG_INFO("GlobalVector"
             << " name="
             << this->object_name_
             << ";"
             << " size="
             << this->GetSize()
             << ";"
             << " prec="
             << 8 * sizeof(ValueType)
             << "bit;"
             << " subdomains="
             << this->pm_->num_procs_
             << ";"
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
bool GlobalVector<ValueType>::Check(void) const
{
    log_debug(this, "GlobalVector::Check()");

    bool interior_check = this->vector_interior_.Check();
    bool ghost_check    = this->vector_ghost_.Check();

    if(interior_check == true && ghost_check == true)
    {
        return true;
    }

    return false;
}

template <typename ValueType>
void GlobalVector<ValueType>::ReadFileASCII(const std::string filename)
{
    log_debug(this, "GlobalVector::ReadFileASCII()", filename);

    assert(this->pm_->Status() == true);

    // Read header file
    std::ifstream headfile(filename.c_str(), std::ifstream::in);

    if(!headfile.is_open())
    {
        LOG_INFO("Cannot open GlobalVector file [read]: " << filename);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // Go to this ranks line in the headfile
    for(int i = 0; i < this->pm_->rank_; ++i)
    {
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string name;
    std::getline(headfile, name);

    headfile.close();

    // Extract directory containing the subfiles
    size_t found     = filename.find_last_of("\\/");
    std::string path = filename.substr(0, found + 1);

    name.erase(remove_if(name.begin(), name.end(), isspace), name.end());

    this->vector_interior_.ReadFileASCII(path + name);

#ifdef SUPPORT_MULTINODE
    if(this->recv_event_ == NULL)
    {
        this->recv_event_ = new MRequest[this->pm_->nrecv_];
    }

    if(this->send_event_ == NULL)
    {
        this->send_event_ = new MRequest[this->pm_->nsend_];
    }
#endif

    this->object_name_ = filename;

    this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

    // Allocate ghost vector
    this->vector_ghost_.Allocate("ghost", this->pm_->GetNumReceivers());

    // Allocate send and receive buffer
    allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
    allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);
}

template <typename ValueType>
void GlobalVector<ValueType>::WriteFileASCII(const std::string filename) const
{
    log_debug(this, "GlobalVector::WriteFileASCII()", filename);

    // Master rank writes the global headfile
    if(this->pm_->rank_ == 0)
    {
        std::ofstream headfile;

        headfile.open((char*)filename.c_str(), std::ofstream::out);
        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalVector file [write]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        for(int i = 0; i < this->pm_->num_procs_; ++i)
        {
            std::ostringstream rs;
            rs << i;

            std::string name = filename + ".rank." + rs.str();

            headfile << name << "\n";
        }
    }

    std::ostringstream rs;
    rs << this->pm_->rank_;

    std::string name = filename + ".rank." + rs.str();

    this->vector_interior_.WriteFileASCII(name);
}

template <typename ValueType>
void GlobalVector<ValueType>::ReadFileBinary(const std::string filename)
{
    log_debug(this, "GlobalVector::ReadFileBinary()", filename);

    assert(this->pm_->Status() == true);

    // Read header file
    std::ifstream headfile(filename.c_str(), std::ifstream::in);

    if(!headfile.is_open())
    {
        LOG_INFO("Cannot open GlobalVector file [read]: " << filename);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // Go to this ranks line in the headfile
    for(int i = 0; i < this->pm_->rank_; ++i)
    {
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string name;
    std::getline(headfile, name);

    headfile.close();

    // Extract directory containing the subfiles
    size_t found     = filename.find_last_of("\\/");
    std::string path = filename.substr(0, found + 1);

    name.erase(remove_if(name.begin(), name.end(), isspace), name.end());

    this->vector_interior_.ReadFileBinary(path + name);

#ifdef SUPPORT_MULTINODE
    if(this->recv_event_ == NULL)
    {
        this->recv_event_ = new MRequest[this->pm_->nrecv_];
    }

    if(this->send_event_ == NULL)
    {
        this->send_event_ = new MRequest[this->pm_->nsend_];
    }
#endif

    this->object_name_ = filename;

    this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

    // Allocate ghost vector
    this->vector_ghost_.Allocate("ghost", this->pm_->GetNumReceivers());

    // Allocate send and receive buffer
    allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
    allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);
}

template <typename ValueType>
void GlobalVector<ValueType>::WriteFileBinary(const std::string filename) const
{
    log_debug(this, "GlobalVector::WriteFileBinary()", filename);

    // Master rank writes the global headfile
    if(this->pm_->rank_ == 0)
    {
        std::ofstream headfile;

        headfile.open((char*)filename.c_str(), std::ofstream::out);
        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalVector file [write]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        for(int i = 0; i < this->pm_->num_procs_; ++i)
        {
            std::ostringstream rs;
            rs << i;

            std::string name = filename + ".rank." + rs.str();

            headfile << name << "\n";
        }
    }

    std::ostringstream rs;
    rs << this->pm_->rank_;

    std::string name = filename + ".rank." + rs.str();

    this->vector_interior_.WriteFileBinary(name);
}

template <typename ValueType>
void GlobalVector<ValueType>::AddScale(const GlobalVector<ValueType>& x, ValueType alpha)
{
    log_debug(this, "GlobalVector::Addscale()", (const void*&)x, alpha);

    this->vector_interior_.AddScale(x.vector_interior_, alpha);
}

template <typename ValueType>
void GlobalVector<ValueType>::ScaleAdd2(ValueType alpha,
                                        const GlobalVector<ValueType>& x,
                                        ValueType beta,
                                        const GlobalVector<ValueType>& y,
                                        ValueType gamma)
{
    log_debug(
        this, "GlobalVector::ScaleAdd2()", alpha, (const void*&)x, beta, (const void*&)y, gamma);

    this->vector_interior_.ScaleAdd2(alpha, x.vector_interior_, beta, y.vector_interior_, gamma);
}

template <typename ValueType>
void GlobalVector<ValueType>::ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x)
{
    log_debug(this, "GlobalVector::ScaleAdd()", alpha, (const void*&)x);

    this->vector_interior_.ScaleAdd(alpha, x.vector_interior_);
}

template <typename ValueType>
void GlobalVector<ValueType>::Scale(ValueType alpha)
{
    log_debug(this, "GlobalVector::Scale()", alpha);

    this->vector_interior_.Scale(alpha);
}

template <typename ValueType>
void GlobalVector<ValueType>::ScaleAddScale(ValueType alpha,
                                            const GlobalVector<ValueType>& x,
                                            ValueType beta)
{
    log_debug(this, "GlobalVector::ScaleAddScale()", alpha, (const void*&)x, beta);

    this->vector_interior_.ScaleAddScale(alpha, x.vector_interior_, beta);
}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Dot(const GlobalVector<ValueType>& x) const
{
    log_debug(this, "GlobalVector::Dot()", (const void*&)x);

    ValueType local = this->vector_interior_.Dot(x.vector_interior_);
    ValueType global;

#ifdef SUPPORT_MULTINODE
    communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
    global = local;
#endif

    return global;
}

template <typename ValueType>
ValueType GlobalVector<ValueType>::DotNonConj(const GlobalVector<ValueType>& x) const
{
    log_debug(this, "GlobalVector::DotNonConj()", (const void*&)x);

    ValueType local = this->vector_interior_.DotNonConj(x.vector_interior_);
    ValueType global;

#ifdef SUPPORT_MULTINODE
    communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
    global = local;
#endif

    return global;
}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Norm(void) const
{
    log_debug(this, "GlobalVector::Norm()");

    ValueType result = this->Dot(*this);
    return sqrt(result);
}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Reduce(void) const
{
    log_debug(this, "GlobalVector::Reduce()");

    ValueType local = this->vector_interior_.Reduce();
    ValueType global;

#ifdef SUPPORT_MULTINODE
    communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
    global = local;
#endif

    return global;
}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Asum(void) const
{
    log_debug(this, "GlobalVector::Asum()");

    ValueType local = this->vector_interior_.Asum();
    ValueType global;

#ifdef SUPPORT_MULTINODE
    communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
    global = local;
#endif

    return global;
}

template <typename ValueType>
int GlobalVector<ValueType>::Amax(ValueType& value) const
{
    log_debug(this, "GlobalVector::Amax()", value);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GlobalVector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x)
{
    log_debug(this, "GlobalVector::PointWiseMult()", (const void*&)x);

    this->vector_interior_.PointWiseMult(x.vector_interior_);
}

template <typename ValueType>
void GlobalVector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x,
                                            const GlobalVector<ValueType>& y)
{
    log_debug(this, "GlobalVector::PointWiseMult()", (const void*&)x, (const void*&)y);

    this->vector_interior_.PointWiseMult(x.vector_interior_, y.vector_interior_);
}

template <typename ValueType>
void GlobalVector<ValueType>::UpdateGhostValuesAsync_(const GlobalVector<ValueType>& in)
{
    log_debug(this, "GlobalVector::UpdateGhostValuesAsync_()", "#*# begin", (const void*&)in);

#ifdef SUPPORT_MULTINODE
    int tag = 0;

    // async recv boundary from neighbors
    for(int i = 0; i < this->pm_->nrecv_; ++i)
    {
        // nnz that we receive from process i
        int boundary_nnz = this->pm_->recv_offset_index_[i + 1] - this->pm_->recv_offset_index_[i];

        // if this has ghost values that belong to process i
        if(boundary_nnz > 0)
        {
            communication_async_recv(this->recv_boundary_ + this->pm_->recv_offset_index_[i],
                                     boundary_nnz,
                                     this->pm_->recvs_[i],
                                     tag,
                                     &this->recv_event_[i],
                                     this->pm_->comm_);
        }
    }

    // prepare send buffer
    in.vector_interior_.GetIndexValues(this->send_boundary_);

    // async send boundary to neighbors
    for(int i = 0; i < this->pm_->nsend_; ++i)
    {
        // nnz that we send to process i
        int boundary_nnz = this->pm_->send_offset_index_[i + 1] - this->pm_->send_offset_index_[i];

        // if process i has ghost values that belong to this
        if(boundary_nnz > 0)
        {
            communication_async_send(this->send_boundary_ + this->pm_->send_offset_index_[i],
                                     boundary_nnz,
                                     this->pm_->sends_[i],
                                     tag,
                                     &this->send_event_[i],
                                     this->pm_->comm_);
        }
    }
#endif

    log_debug(this, "GlobalVector::UpdateGhostValuesAsync_()", "#*# end");
}

template <typename ValueType>
void GlobalVector<ValueType>::UpdateGhostValuesSync_(void)
{
    log_debug(this, "GlobalVector::UpdateGhostValuesSync_()", "#*# begin");

#ifdef SUPPORT_MULTINODE
    // Sync before updating ghost values
    communication_syncall(this->pm_->nrecv_, this->recv_event_);
    communication_syncall(this->pm_->nsend_, this->send_event_);

    this->vector_ghost_.SetContinuousValues(0, this->pm_->GetNumReceivers(), this->recv_boundary_);
#endif

    log_debug(this, "GlobalVector::UpdateGhostValuesSync_()", "#*# end");
}

template <typename ValueType>
void GlobalVector<ValueType>::Power(double power)
{
    log_debug(this, "GlobalVector::Power()", power);

    this->vector_interior_.Power(power);
}

template <typename ValueType>
void GlobalVector<ValueType>::Restriction(const GlobalVector<ValueType>& vec_fine,
                                          const LocalVector<int>& map)
{
}

template <typename ValueType>
void GlobalVector<ValueType>::Prolongation(const GlobalVector<ValueType>& vec_coarse,
                                           const LocalVector<int>& map)
{
}

template <typename ValueType>
bool GlobalVector<ValueType>::is_host_(void) const
{
    assert(this->vector_interior_.is_host_() == this->vector_ghost_.is_host_());
    return this->vector_interior_.is_host_();
}

template <typename ValueType>
bool GlobalVector<ValueType>::is_accel_(void) const
{
    assert(this->vector_interior_.is_accel_() == this->vector_ghost_.is_accel_());
    return this->vector_interior_.is_accel_();
}

template class GlobalVector<double>;
template class GlobalVector<float>;
#ifdef SUPPORT_COMPLEX
template class GlobalVector<std::complex<double>>;
template class GlobalVector<std::complex<float>>;
#endif

} // namespace rocalution
