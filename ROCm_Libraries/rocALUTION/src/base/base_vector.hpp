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

#ifndef ROCALUTION_BASE_VECTOR_HPP_
#define ROCALUTION_BASE_VECTOR_HPP_

#include "backend_manager.hpp"

namespace rocalution {

// Forward declarations
template <typename ValueType>
class HostVector;

template <typename ValueType>
class AcceleratorVector;

template <typename ValueType>
class HIPAcceleratorVector;

/// Base class for all host/accelerator vectors
template <typename ValueType>
class BaseVector
{
    public:
    BaseVector();
    virtual ~BaseVector();

    /// Shows info about the object
    virtual void Info(void) const = 0;

    /// Returns the size of the vector
    int GetSize(void) const;

    /// Copy the backend descriptor information
    void set_backend(const Rocalution_Backend_Descriptor local_backend);

    /// Check if everything is ok
    virtual bool Check(void) const;

    /// Allocate a local vector with name and size
    virtual void Allocate(int n) = 0;

    /// Initialize a vector with externally allocated data
    virtual void SetDataPtr(ValueType** ptr, int size) = 0;
    /// Get a pointer from the vector data and free the vector object
    virtual void LeaveDataPtr(ValueType** ptr) = 0;

    /// Clear (free) the vector
    virtual void Clear(void) = 0;
    /// Set the values of the vector to zero
    virtual void Zeros(void) = 0;
    /// Set the values of the vector to one
    virtual void Ones(void) = 0;
    /// Set the values of the vector to given argument
    virtual void SetValues(ValueType val) = 0;

    /// Perform inplace permutation (forward) of the vector
    virtual void Permute(const BaseVector<int>& permutation) = 0;
    /// Perform inplace permutation (backward) of the vector
    virtual void PermuteBackward(const BaseVector<int>& permutation) = 0;

    /// Copy values from another vector
    virtual void CopyFrom(const BaseVector<ValueType>& vec) = 0;
    /// Async copy values from another vector
    virtual void CopyFromAsync(const BaseVector<ValueType>& vec);
    /// Copy values from another (float) vector
    virtual void CopyFromFloat(const BaseVector<float>& vec);
    /// Copy values from another (double) vector
    virtual void CopyFromDouble(const BaseVector<double>& vec);
    /// Copy values to another vector
    virtual void CopyTo(BaseVector<ValueType>* vec) const = 0;
    /// Async copy values to another vector
    virtual void CopyToAsync(BaseVector<ValueType>* vec) const;
    /// Copy data (not entire vector) from another vector with specified
    /// src/dst offsets and size
    virtual void
    CopyFrom(const BaseVector<ValueType>& src, int src_offset, int dst_offset, int size) = 0;

    /// Copy a vector under specified permutation (forward permutation)
    virtual void CopyFromPermute(const BaseVector<ValueType>& src,
                                 const BaseVector<int>& permutation) = 0;
    /// Copy a vector under specified permutation (backward permutation)
    virtual void CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                         const BaseVector<int>& permutation) = 0;

    virtual void CopyFromData(const ValueType* data);
    virtual void CopyToData(ValueType* data) const;

    /// Restriction operator based on restriction mapping vector
    virtual bool Restriction(const BaseVector<ValueType>& vec_fine, const BaseVector<int>& map);

    /// Prolongation operator based on restriction(!) mapping vector
    virtual bool Prolongation(const BaseVector<ValueType>& vec_coarse, const BaseVector<int>& map);

    /// Perform vector update of type this = this + alpha*x
    virtual void AddScale(const BaseVector<ValueType>& x, ValueType alpha) = 0;
    /// Perform vector update of type this = alpha*this + x
    virtual void ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x) = 0;
    /// Perform vector update of type this = alpha*this + x*beta
    virtual void ScaleAddScale(ValueType alpha, const BaseVector<ValueType>& x, ValueType beta) = 0;

    virtual void ScaleAddScale(ValueType alpha,
                               const BaseVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size) = 0;

    /// Perform vector update of type this = alpha*this + x*beta + y*gamma
    virtual void ScaleAdd2(ValueType alpha,
                           const BaseVector<ValueType>& x,
                           ValueType beta,
                           const BaseVector<ValueType>& y,
                           ValueType gamma) = 0;
    /// Perform vector scaling this = alpha*this
    virtual void Scale(ValueType alpha) = 0;
    /// Compute dot (scalar) product, return this^T y
    virtual ValueType Dot(const BaseVector<ValueType>& x) const = 0;
    /// Compute non-conjugated dot (scalar) product, return this^T y
    virtual ValueType DotNonConj(const BaseVector<ValueType>& x) const = 0;
    /// Compute L2 norm of the vector, return =  srqt(this^T this)
    virtual ValueType Norm(void) const = 0;
    /// Reduce vector
    virtual ValueType Reduce(void) const = 0;
    /// Compute sum of absolute values of the vector (L1 norm), return =  sum(|this|)
    virtual ValueType Asum(void) const = 0;
    /// Compute the absolute max value of the vector, return =  max(|this|)
    virtual int Amax(ValueType& value) const = 0;
    /// Perform point-wise multiplication (element-wise) of type this = this * x
    virtual void PointWiseMult(const BaseVector<ValueType>& x) = 0;
    /// Perform point-wise multiplication (element-wise) of type this = x*y
    virtual void PointWiseMult(const BaseVector<ValueType>& x, const BaseVector<ValueType>& y) = 0;
    virtual void Power(double power) = 0;

    /// Sets index array
    virtual void SetIndexArray(int size, const int* index) = 0;
    /// Gets index values
    virtual void GetIndexValues(ValueType* values) const = 0;
    /// Sets index values
    virtual void SetIndexValues(const ValueType* values) = 0;
    /// Gets continuous index values
    virtual void GetContinuousValues(int start, int end, ValueType* values) const = 0;
    /// Sets continuous index values
    virtual void SetContinuousValues(int start, int end, const ValueType* values) = 0;
    /// Extract coarse boundary mapping
    virtual void ExtractCoarseMapping(
        int start, int end, const int* index, int nc, int* size, int* map) const = 0;
    /// Extract coarse boundary index
    virtual void ExtractCoarseBoundary(
        int start, int end, const int* index, int nc, int* size, int* boundary) const = 0;

    protected:
    /// The size of the vector
    int size_;
    /// The size of the boundary index
    int index_size_;

    /// Backend descriptor (local copy)
    Rocalution_Backend_Descriptor local_backend_;
};

template <typename ValueType>
class AcceleratorVector : public BaseVector<ValueType>
{
    public:
    AcceleratorVector();
    virtual ~AcceleratorVector();

    /// Copy (accelerator vector) from host vector
    virtual void CopyFromHost(const HostVector<ValueType>& src) = 0;
    /// Copy (host vector) from accelerator vector
    virtual void CopyToHost(HostVector<ValueType>* dst) const = 0;

    /// Async copy (accelerator vector) from host vector
    virtual void CopyFromHostAsync(const HostVector<ValueType>& src);
    /// Async copy (host vector) from accelerator vector
    virtual void CopyToHostAsync(HostVector<ValueType>* dst) const;
};

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
