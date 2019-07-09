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
#include "hip_vector.hpp"
#include "../base_vector.hpp"
#include "../host/host_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/math_functions.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"

#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorVector<ValueType>::HIPAcceleratorVector()
{
    // no default constructors
    LOG_INFO("no default constructor");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorVector<ValueType>::HIPAcceleratorVector(
    const Rocalution_Backend_Descriptor local_backend)
{
    log_debug(
        this, "HIPAcceleratorVector::HIPAcceleratorVector()", "constructor with local_backend");

    this->vec_ = NULL;
    this->set_backend(local_backend);

    this->index_array_  = NULL;
    this->index_buffer_ = NULL;

    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HIPAcceleratorVector<ValueType>::~HIPAcceleratorVector()
{
    log_debug(this, "HIPAcceleratorVector::~HIPAcceleratorVector()", "destructor");

    this->Clear();
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Info(void) const
{
    LOG_INFO("HIPAcceleratorVector<ValueType>");
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Allocate(int n)
{
    assert(n >= 0);

    if(this->size_ > 0)
    {
        this->Clear();
    }

    if(n > 0)
    {
        allocate_hip(n, &this->vec_);
        set_to_zero_hip(this->local_backend_.HIP_block_size, n, this->vec_);

        this->size_ = n;
    }

    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetDataPtr(ValueType** ptr, int size)
{
    assert(*ptr != NULL);
    assert(size > 0);

    hipDeviceSynchronize();

    this->vec_  = *ptr;
    this->size_ = size;
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::LeaveDataPtr(ValueType** ptr)
{
    assert(this->size_ > 0);

    hipDeviceSynchronize();
    *ptr       = this->vec_;
    this->vec_ = NULL;

    this->size_ = 0;
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Clear(void)
{
    if(this->size_ > 0)
    {
        free_hip(&this->vec_);
        this->size_ = 0;
    }

    if(this->index_size_ > 0)
    {
        free_hip(&this->index_buffer_);
        free_hip(&this->index_array_);
        this->index_size_ = 0;
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromHost(const HostVector<ValueType>& src)
{
    // CPU to HIP copy
    const HostVector<ValueType>* cast_vec;
    if((cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
    {
        if(this->size_ == 0)
        {
            // Allocate local structure
            this->Allocate(cast_vec->size_);

            // Check for boundary
            assert(this->index_size_ == 0);
            if(cast_vec->index_size_ > 0)
            {
                this->index_size_ = cast_vec->index_size_;
                allocate_hip<int>(this->index_size_, &this->index_array_);
                allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);
            }
        }

        assert(cast_vec->size_ == this->size_);
        assert(cast_vec->index_size_ == this->index_size_);

        if(this->size_ > 0)
        {
            hipMemcpy(
                this->vec_, cast_vec->vec_, this->size_ * sizeof(ValueType), hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(this->index_array_,
                      cast_vec->index_array_,
                      this->index_size_ * sizeof(int),
                      hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP vector type");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToHost(HostVector<ValueType>* dst) const
{
    // HIP to CPU copy
    HostVector<ValueType>* cast_vec;
    if((cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
    {
        if(cast_vec->size_ == 0)
        {
            // Allocate local vector
            cast_vec->Allocate(this->size_);

            // Check for boundary
            assert(cast_vec->index_size_ == 0);
            if(this->index_size_ > 0)
            {
                cast_vec->index_size_ = this->index_size_;
                allocate_host(this->index_size_, &cast_vec->index_array_);
            }
        }

        assert(cast_vec->size_ == this->size_);
        assert(cast_vec->index_size_ == this->index_size_);

        if(this->size_ > 0)
        {
            hipMemcpy(
                cast_vec->vec_, this->vec_, this->size_ * sizeof(ValueType), hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpy(cast_vec->index_array_,
                      this->index_array_,
                      this->index_size_ * sizeof(int),
                      hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP vector type");
        this->Info();
        dst->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromHostAsync(const HostVector<ValueType>& src)
{
    // CPU to HIP copy
    const HostVector<ValueType>* cast_vec;
    if((cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
    {
        if(this->size_ == 0)
        {
            // Allocate local vector
            this->Allocate(cast_vec->size_);

            // Check for boundary
            assert(this->index_size_ == 0);
            if(cast_vec->index_size_ > 0)
            {
                this->index_size_ = cast_vec->index_size_;
                allocate_hip<int>(this->index_size_, &this->index_array_);
                allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);
            }
        }

        assert(cast_vec->size_ == this->size_);
        assert(cast_vec->index_size_ == this->index_size_);

        if(this->size_ > 0)
        {
            hipMemcpyAsync(
                this->vec_, cast_vec->vec_, this->size_ * sizeof(ValueType), hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpyAsync(this->index_array_,
                           cast_vec->index_array_,
                           this->index_size_ * sizeof(int),
                           hipMemcpyHostToDevice);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP vector type");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToHostAsync(HostVector<ValueType>* dst) const
{
    // HIP to CPU copy
    HostVector<ValueType>* cast_vec;
    if((cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
    {
        if(cast_vec->size_ == 0)
        {
            // Allocate local vector
            cast_vec->Allocate(this->size_);

            // Check for boundary
            assert(cast_vec->index_size_ == 0);
            if(this->index_size_ > 0)
            {
                cast_vec->index_size_ = this->index_size_;
                allocate_host(this->index_size_, &cast_vec->index_array_);
            }
        }

        assert(cast_vec->size_ == this->size_);
        assert(cast_vec->index_size_ == this->index_size_);

        if(this->size_ > 0)
        {
            hipMemcpyAsync(
                cast_vec->vec_, this->vec_, this->size_ * sizeof(ValueType), hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            hipMemcpyAsync(cast_vec->index_array_,
                           this->index_array_,
                           this->index_size_ * sizeof(int),
                           hipMemcpyDeviceToHost);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP vector type");
        this->Info();
        dst->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src)
{
    const HIPAcceleratorVector<ValueType>* hip_cast_vec;
    const HostVector<ValueType>* host_cast_vec;

    // HIP to HIP copy
    if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src)) != NULL)
    {
        if(this->size_ == 0)
        {
            // Allocate local vector
            this->Allocate(hip_cast_vec->size_);

            // Check for boundary
            assert(this->index_size_ == 0);
            if(hip_cast_vec->index_size_ > 0)
            {
                this->index_size_ = hip_cast_vec->index_size_;
                allocate_hip<int>(this->index_size_, &this->index_array_);
                allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);
            }
        }

        assert(hip_cast_vec->size_ == this->size_);
        assert(hip_cast_vec->index_size_ == this->index_size_);

        if(this != hip_cast_vec)
        {
            if(this->size_ > 0)
            {
                hipMemcpy(this->vec_,
                          hip_cast_vec->vec_,
                          this->size_ * sizeof(ValueType),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipMemcpy(this->index_array_,
                          hip_cast_vec->index_array_,
                          this->index_size_ * sizeof(int),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
    }
    else
    {
        // HIP to CPU copy
        if((host_cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
        {
            this->CopyFromHost(*host_cast_vec);
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromAsync(const BaseVector<ValueType>& src)
{
    const HIPAcceleratorVector<ValueType>* hip_cast_vec;
    const HostVector<ValueType>* host_cast_vec;

    // HIP to HIP copy
    if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src)) != NULL)
    {
        if(this->size_ == 0)
        {
            // Allocate local vector
            this->Allocate(hip_cast_vec->size_);

            // Check for boundary
            assert(this->index_size_ == 0);
            if(hip_cast_vec->index_size_ > 0)
            {
                this->index_size_ = hip_cast_vec->index_size_;
                allocate_hip<int>(this->index_size_, &this->index_array_);
                allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);
            }
        }

        assert(hip_cast_vec->size_ == this->size_);
        assert(hip_cast_vec->index_size_ == this->index_size_);

        if(this != hip_cast_vec)
        {
            if(this->size_ > 0)
            {
                hipMemcpy(this->vec_,
                          hip_cast_vec->vec_,
                          this->size_ * sizeof(ValueType),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipMemcpy(this->index_array_,
                          hip_cast_vec->index_array_,
                          this->index_size_ * sizeof(int),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
    }
    else
    {
        // HIP to CPU copy
        if((host_cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
        {
            this->CopyFromHostAsync(*host_cast_vec);
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src,
                                               int src_offset,
                                               int dst_offset,
                                               int size)
{
    assert(this->size_ > 0);
    assert(size > 0);
    assert(dst_offset + size <= this->size_);

    const HIPAcceleratorVector<ValueType>* cast_src =
        dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src);

    assert(cast_src != NULL);
    assert(cast_src->size_ > 0);
    assert(src_offset + size <= cast_src->size_);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_copy_offset_from<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       size,
                       src_offset,
                       dst_offset,
                       cast_src->vec_,
                       this->vec_);
    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyTo(BaseVector<ValueType>* dst) const
{
    HIPAcceleratorVector<ValueType>* hip_cast_vec;
    HostVector<ValueType>* host_cast_vec;

    // HIP to HIP copy
    if((hip_cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*>(dst)) != NULL)
    {
        if(hip_cast_vec->size_ == 0)
        {
            // Allocate local vector
            hip_cast_vec->Allocate(this->size_);

            // Check for boundary
            assert(hip_cast_vec->index_size_ == 0);
            if(this->index_size_ > 0)
            {
                hip_cast_vec->index_size_ = this->index_size_;
                allocate_hip<int>(this->index_size_, &hip_cast_vec->index_array_);
                allocate_hip<ValueType>(this->index_size_, &hip_cast_vec->index_buffer_);
            }
        }

        assert(hip_cast_vec->size_ == this->size_);
        assert(hip_cast_vec->index_size_ == this->index_size_);

        if(this != hip_cast_vec)
        {
            if(this->size_ > 0)
            {
                hipMemcpy(hip_cast_vec->vec_,
                          this->vec_,
                          this->size_ * sizeof(ValueType),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipMemcpy(hip_cast_vec->index_array_,
                          this->index_array_,
                          this->index_size_ * sizeof(int),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
    }
    else
    {
        // HIP to CPU copy
        if((host_cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
        {
            this->CopyToHost(host_cast_vec);
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToAsync(BaseVector<ValueType>* dst) const
{
    HIPAcceleratorVector<ValueType>* hip_cast_vec;
    HostVector<ValueType>* host_cast_vec;

    // HIP to HIP copy
    if((hip_cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*>(dst)) != NULL)
    {
        if(hip_cast_vec->size_ == 0)
        {
            // Allocate local vector
            hip_cast_vec->Allocate(this->size_);

            // Check for boundary
            assert(hip_cast_vec->index_size_ == 0);
            if(this->index_size_ > 0)
            {
                hip_cast_vec->index_size_ = this->index_size_;
                allocate_hip<int>(this->index_size_, &hip_cast_vec->index_array_);
                allocate_hip<ValueType>(this->index_size_, &hip_cast_vec->index_buffer_);
            }
        }

        assert(hip_cast_vec->size_ == this->size_);
        assert(hip_cast_vec->index_size_ == this->index_size_);

        if(this != hip_cast_vec)
        {
            if(this->size_ > 0)
            {
                hipMemcpy(hip_cast_vec->vec_,
                          this->vec_,
                          this->size_ * sizeof(ValueType),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipMemcpy(hip_cast_vec->index_array_,
                          this->index_array_,
                          this->index_size_ * sizeof(int),
                          hipMemcpyDeviceToDevice);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
    }
    else
    {
        // HIP to CPU copy
        if((host_cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
        {
            this->CopyToHostAsync(host_cast_vec);
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromFloat(const BaseVector<float>& src)
{
    LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
void HIPAcceleratorVector<double>::CopyFromFloat(const BaseVector<float>& src)
{
    const HIPAcceleratorVector<float>* hip_cast_vec;

    // HIP to HIP copy
    if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<float>*>(&src)) != NULL)
    {
        if(this->size_ == 0)
        {
            this->Allocate(hip_cast_vec->size_);
        }

        assert(hip_cast_vec->size_ == this->size_);

        if(this->size_ > 0)
        {
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            hipLaunchKernelGGL((kernel_copy_from_float<double, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->size_,
                               hip_cast_vec->vec_,
                               this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP vector type");
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromDouble(const BaseVector<double>& src)
{
    LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
void HIPAcceleratorVector<float>::CopyFromDouble(const BaseVector<double>& src)
{
    const HIPAcceleratorVector<double>* hip_cast_vec;

    // HIP to HIP copy
    if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<double>*>(&src)) != NULL)
    {
        if(this->size_ == 0)
        {
            this->Allocate(hip_cast_vec->size_);
        }

        assert(hip_cast_vec->size_ == this->size_);

        if(this->size_ > 0)
        {
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            hipLaunchKernelGGL((kernel_copy_from_double<float, int>),
                               GridSize,
                               BlockSize,
                               0,
                               0,
                               this->size_,
                               hip_cast_vec->vec_,
                               this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }
    else
    {
        LOG_INFO("Error unsupported HIP vector type");
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromData(const ValueType* data)
{
    if(this->size_ > 0)
    {
        hipMemcpy(this->vec_, data, this->size_ * sizeof(ValueType), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToData(ValueType* data) const
{
    if(this->size_ > 0)
    {
        hipMemcpy(data, this->vec_, this->size_ * sizeof(ValueType), hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Zeros(void)
{
    if(this->size_ > 0)
    {
        set_to_zero_hip(this->local_backend_.HIP_block_size, this->size_, this->vec_);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Ones(void)
{
    if(this->size_ > 0)
    {
        set_to_one_hip(this->local_backend_.HIP_block_size, this->size_, this->vec_);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetValues(ValueType val)
{
    LOG_INFO("HIPAcceleratorVector::SetValues NYI");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::AddScale(const BaseVector<ValueType>& x, ValueType alpha)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        rocblas_status status;
        status = rocblasTaxpy(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                              this->size_,
                              &alpha,
                              cast_x->vec_,
                              1,
                              this->vec_,
                              1);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }
}

template <>
void HIPAcceleratorVector<int>::AddScale(const BaseVector<int>& x, int alpha)
{
    LOG_INFO("No int axpy function");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_scaleadd<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           alpha,
                           cast_x->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAddScale(ValueType alpha,
                                                    const BaseVector<ValueType>& x,
                                                    ValueType beta)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_scaleaddscale<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           alpha,
                           beta,
                           cast_x->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAddScale(ValueType alpha,
                                                    const BaseVector<ValueType>& x,
                                                    ValueType beta,
                                                    int src_offset,
                                                    int dst_offset,
                                                    int size)
{
    if(this->size_ > 0)
    {
        assert(this->size_ > 0);
        assert(size > 0);
        assert(dst_offset + size <= this->size_);

        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(cast_x->size_ > 0);
        assert(src_offset + size <= cast_x->size_);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_scaleaddscale_offset<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           src_offset,
                           dst_offset,
                           alpha,
                           beta,
                           cast_x->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAdd2(ValueType alpha,
                                                const BaseVector<ValueType>& x,
                                                ValueType beta,
                                                const BaseVector<ValueType>& y,
                                                ValueType gamma)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);
        const HIPAcceleratorVector<ValueType>* cast_y =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&y);

        assert(cast_x != NULL);
        assert(cast_y != NULL);
        assert(this->size_ == cast_x->size_);
        assert(this->size_ == cast_y->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_scaleadd2<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           alpha,
                           beta,
                           gamma,
                           cast_x->vec_,
                           cast_y->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Scale(ValueType alpha)
{
    if(this->size_ > 0)
    {
        rocblas_status status;
        status = rocblasTscal(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                              this->size_,
                              &alpha,
                              this->vec_,
                              1);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }
}

template <>
void HIPAcceleratorVector<int>::Scale(int alpha)
{
    LOG_INFO("No int CUBLAS scale function");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::Dot(const BaseVector<ValueType>& x) const
{
    const HIPAcceleratorVector<ValueType>* cast_x =
        dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

    assert(cast_x != NULL);
    assert(this->size_ == cast_x->size_);

    ValueType res = static_cast<ValueType>(0);

    if(this->size_ > 0)
    {
        rocblas_status status;
        status = rocblasTdot(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                             this->size_,
                             this->vec_,
                             1,
                             cast_x->vec_,
                             1,
                             &res);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }

    return res;
}

template <>
int HIPAcceleratorVector<int>::Dot(const BaseVector<int>& x) const
{
    LOG_INFO("No int dot function");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::DotNonConj(const BaseVector<ValueType>& x) const
{
    const HIPAcceleratorVector<ValueType>* cast_x =
        dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

    assert(cast_x != NULL);
    assert(this->size_ == cast_x->size_);

    ValueType res = static_cast<ValueType>(0);

    if(this->size_ > 0)
    {
        rocblas_status status;
        status = rocblasTdotc(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                              this->size_,
                              this->vec_,
                              1,
                              cast_x->vec_,
                              1,
                              &res);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }

    return res;
}

template <>
int HIPAcceleratorVector<int>::DotNonConj(const BaseVector<int>& x) const
{
    LOG_INFO("No int dotc function");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::Norm(void) const
{
    ValueType res = static_cast<ValueType>(0);

    if(this->size_ > 0)
    {
        rocblas_status status;
        status = rocblasTnrm2(
            ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle), this->size_, this->vec_, 1, &res);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }

    return res;
}

template <>
int HIPAcceleratorVector<int>::Norm(void) const
{
    LOG_INFO("What is int HIPAcceleratorVector<ValueType>::Norm(void) const?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::Reduce(void) const
{
    ValueType res = static_cast<ValueType>(0);

    if(this->size_ > 0)
    {
        void* buffer = NULL;
        size_t size  = 0;

        ValueType* dres = NULL;
        allocate_hip(1, &dres);

        hipcub::DeviceReduce::Sum(buffer, size, this->vec_, dres, this->size_);

        hipMalloc(&buffer, size);

        hipcub::DeviceReduce::Sum(buffer, size, this->vec_, dres, this->size_);

        hipFree(buffer);

        hipMemcpy(&res, dres, sizeof(ValueType), hipMemcpyDeviceToHost);

        free_hip(&dres);
    }

    return res;
}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::Asum(void) const
{
    ValueType res = static_cast<ValueType>(0);

    if(this->size_ > 0)
    {
        rocblas_status status;
        status = rocblasTasum(
            ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle), this->size_, this->vec_, 1, &res);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
    }

    return res;
}

template <>
int HIPAcceleratorVector<int>::Asum(void) const
{
    LOG_INFO("Asum<int> not implemented");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
int HIPAcceleratorVector<ValueType>::Amax(ValueType& value) const
{
    int index = 0;
    value     = static_cast<ValueType>(0.0);

    if(this->size_ > 0)
    {
        rocblas_status status;
        status = rocblasTamax(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                              this->size_,
                              this->vec_,
                              1,
                              &index);
        CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);

        hipMemcpy(&value, this->vec_ + index, sizeof(ValueType), hipMemcpyDeviceToHost);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    value = rocalution_abs(value);
    return index;
}

template <>
int HIPAcceleratorVector<int>::Amax(int& value) const
{
    LOG_INFO("Amax<int> not implemented");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType>& x)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_pointwisemult<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           cast_x->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType>& x,
                                                    const BaseVector<ValueType>& y)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_x =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);
        const HIPAcceleratorVector<ValueType>* cast_y =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&y);

        assert(cast_x != NULL);
        assert(cast_y != NULL);
        assert(this->size_ == cast_x->size_);
        assert(this->size_ == cast_y->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL((kernel_pointwisemult2<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           cast_x->vec_,
                           cast_y->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Permute(const BaseVector<int>& permutation)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

        assert(cast_perm != NULL);
        assert(this->size_ == cast_perm->size_);

        HIPAcceleratorVector<ValueType> vec_tmp(this->local_backend_);
        vec_tmp.Allocate(this->size_);
        vec_tmp.CopyFrom(*this);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        // this->vec_[ cast_perm->vec_[i] ] = vec_tmp.vec_[i];
        hipLaunchKernelGGL((kernel_permute<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           cast_perm->vec_,
                           vec_tmp.vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

        assert(cast_perm != NULL);
        assert(this->size_ == cast_perm->size_);

        HIPAcceleratorVector<ValueType> vec_tmp(this->local_backend_);
        vec_tmp.Allocate(this->size_);
        vec_tmp.CopyFrom(*this);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        //    this->vec_[i] = vec_tmp.vec_[ cast_perm->vec_[i] ];
        hipLaunchKernelGGL((kernel_permute_backward<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           cast_perm->vec_,
                           vec_tmp.vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromPermute(const BaseVector<ValueType>& src,
                                                      const BaseVector<int>& permutation)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_vec =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src);
        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
        assert(cast_perm != NULL);
        assert(cast_vec != NULL);

        assert(cast_vec->size_ == this->size_);
        assert(cast_perm->size_ == this->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        //    this->vec_[ cast_perm->vec_[i] ] = cast_vec->vec_[i];
        hipLaunchKernelGGL((kernel_permute<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           cast_perm->vec_,
                           cast_vec->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                                              const BaseVector<int>& permutation)
{
    if(this->size_ > 0)
    {
        const HIPAcceleratorVector<ValueType>* cast_vec =
            dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src);
        const HIPAcceleratorVector<int>* cast_perm =
            dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
        assert(cast_perm != NULL);
        assert(cast_vec != NULL);

        assert(cast_vec->size_ == this->size_);
        assert(cast_perm->size_ == this->size_);

        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        //    this->vec_[i] = cast_vec->vec_[ cast_perm->vec_[i] ];
        hipLaunchKernelGGL((kernel_permute_backward<ValueType, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           cast_perm->vec_,
                           cast_vec->vec_,
                           this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetIndexArray(int size, const int* index)
{
    assert(size > 0);
    assert(this->size_ >= size);

    this->index_size_ = size;

    allocate_hip<int>(this->index_size_, &this->index_array_);
    allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);

    hipMemcpy(this->index_array_, index, this->index_size_ * sizeof(int), hipMemcpyHostToDevice);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::GetIndexValues(ValueType* values) const
{
    assert(values != NULL);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(this->index_size_ / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_get_index_values<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       this->index_size_,
                       this->index_array_,
                       this->vec_,
                       this->index_buffer_);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(
        values, this->index_buffer_, this->index_size_ * sizeof(ValueType), hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetIndexValues(const ValueType* values)
{
    assert(values != NULL);

    hipMemcpy(
        this->index_buffer_, values, this->index_size_ * sizeof(ValueType), hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(this->index_size_ / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_set_index_values<ValueType, int>),
                       GridSize,
                       BlockSize,
                       0,
                       0,
                       this->index_size_,
                       this->index_array_,
                       this->index_buffer_,
                       this->vec_);
    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::GetContinuousValues(int start,
                                                          int end,
                                                          ValueType* values) const
{
    assert(start >= 0);
    assert(end >= start);
    assert(end <= this->size_);
    assert(values != NULL);

    hipMemcpy(values, this->vec_ + start, (end - start) * sizeof(ValueType), hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetContinuousValues(int start,
                                                          int end,
                                                          const ValueType* values)
{
    assert(start >= 0);
    assert(end >= start);
    assert(end <= this->size_);
    assert(values != NULL);

    hipMemcpy(this->vec_ + start, values, (end - start) * sizeof(ValueType), hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ExtractCoarseMapping(
    int start, int end, const int* index, int nc, int* size, int* map) const
{
    LOG_INFO("ExtractCoarseMapping() NYI for HIP");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ExtractCoarseBoundary(
    int start, int end, const int* index, int nc, int* size, int* boundary) const
{
    LOG_INFO("ExtractCoarseBoundary() NYI for HIP");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Power(double power)
{
    if(this->size_ > 0)
    {
        int size = this->size_;
        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        hipLaunchKernelGGL(
            (kernel_power<ValueType, int>), GridSize, BlockSize, 0, 0, size, power, this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <>
void HIPAcceleratorVector<int>::Power(double power)
{
    if(this->size_ > 0)
    {
        LOG_INFO("HIPAcceleratorVector::Power(), no pow() for int in HIP");
        FATAL_ERROR(__FILE__, __LINE__);
    }
}

template class HIPAcceleratorVector<double>;
template class HIPAcceleratorVector<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorVector<std::complex<double>>;
template class HIPAcceleratorVector<std::complex<float>>;
#endif
template class HIPAcceleratorVector<int>;

} // namespace rocalution
