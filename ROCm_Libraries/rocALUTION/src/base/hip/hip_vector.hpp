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

#ifndef ROCALUTION_HIP_VECTOR_HPP_
#define ROCALUTION_HIP_VECTOR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorVector : public AcceleratorVector<ValueType>
{
    public:
    HIPAcceleratorVector();
    HIPAcceleratorVector(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HIPAcceleratorVector();

    virtual void Info(void) const;

    virtual void Allocate(int n);
    virtual void SetDataPtr(ValueType** ptr, int size);
    virtual void LeaveDataPtr(ValueType** ptr);
    virtual void Clear(void);
    virtual void Zeros(void);
    virtual void Ones(void);
    virtual void SetValues(ValueType val);

    virtual void CopyFrom(const BaseVector<ValueType>& src);
    virtual void CopyFromAsync(const BaseVector<ValueType>& src);
    virtual void
    CopyFrom(const BaseVector<ValueType>& src, int src_offset, int dst_offset, int size);

    virtual void CopyTo(BaseVector<ValueType>* dst) const;
    virtual void CopyToAsync(BaseVector<ValueType>* dst) const;
    virtual void CopyFromFloat(const BaseVector<float>& src);
    virtual void CopyFromDouble(const BaseVector<double>& src);

    virtual void CopyFromHostAsync(const HostVector<ValueType>& src);
    virtual void CopyFromHost(const HostVector<ValueType>& src);
    virtual void CopyToHostAsync(HostVector<ValueType>* dst) const;
    virtual void CopyToHost(HostVector<ValueType>* dst) const;

    virtual void CopyFromData(const ValueType* data);
    virtual void CopyToData(ValueType* data) const;

    virtual void CopyFromPermute(const BaseVector<ValueType>& src,
                                 const BaseVector<int>& permutation);
    virtual void CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                         const BaseVector<int>& permutation);

    virtual void Permute(const BaseVector<int>& permutation);
    virtual void PermuteBackward(const BaseVector<int>& permutation);

    // this = this + alpha*x
    virtual void AddScale(const BaseVector<ValueType>& x, ValueType alpha);
    // this = alpha*this + x
    virtual void ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x);
    // this = alpha*this + x*beta
    virtual void ScaleAddScale(ValueType alpha, const BaseVector<ValueType>& x, ValueType beta);
    virtual void ScaleAddScale(ValueType alpha,
                               const BaseVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size);
    // this = alpha*this + x*beta + y*gamma
    virtual void ScaleAdd2(ValueType alpha,
                           const BaseVector<ValueType>& x,
                           ValueType beta,
                           const BaseVector<ValueType>& y,
                           ValueType gamma);
    // this = alpha*this
    virtual void Scale(ValueType alpha);

    // this^T x
    virtual ValueType Dot(const BaseVector<ValueType>& x) const;
    // this^T x
    virtual ValueType DotNonConj(const BaseVector<ValueType>& x) const;
    // srqt(this^T this)
    virtual ValueType Norm(void) const;
    // reduce
    virtual ValueType Reduce(void) const;
    // Compute sum of absolute values of this
    virtual ValueType Asum(void) const;
    // Compute absolute value of this
    virtual int Amax(ValueType& value) const;
    // point-wise multiplication
    virtual void PointWiseMult(const BaseVector<ValueType>& x);
    virtual void PointWiseMult(const BaseVector<ValueType>& x, const BaseVector<ValueType>& y);
    virtual void Power(double power);

    // set index array
    virtual void SetIndexArray(int size, const int* index);
    // get index values
    virtual void GetIndexValues(ValueType* values) const;
    // set index values
    virtual void SetIndexValues(const ValueType* values);
    // get continuous index values
    virtual void GetContinuousValues(int start, int end, ValueType* values) const;
    // set continuous index values
    virtual void SetContinuousValues(int start, int end, const ValueType* values);

    // get coarse boundary mapping
    virtual void
    ExtractCoarseMapping(int start, int end, const int* index, int nc, int* size, int* map) const;
    // get coarse boundary index
    virtual void ExtractCoarseBoundary(
        int start, int end, const int* index, int nc, int* size, int* boundary) const;

    private:
    ValueType* vec_;

    int* index_array_;
    ValueType* index_buffer_;

    friend class HIPAcceleratorVector<float>;
    friend class HIPAcceleratorVector<double>;
    friend class HIPAcceleratorVector<std::complex<float>>;
    friend class HIPAcceleratorVector<std::complex<double>>;
    friend class HIPAcceleratorVector<int>;

    friend class HostVector<ValueType>;
    friend class AcceleratorMatrix<ValueType>;

    friend class HIPAcceleratorMatrixCSR<ValueType>;
    friend class HIPAcceleratorMatrixMCSR<ValueType>;
    friend class HIPAcceleratorMatrixBCSR<ValueType>;
    friend class HIPAcceleratorMatrixCOO<ValueType>;
    friend class HIPAcceleratorMatrixDIA<ValueType>;
    friend class HIPAcceleratorMatrixELL<ValueType>;
    friend class HIPAcceleratorMatrixDENSE<ValueType>;
    friend class HIPAcceleratorMatrixHYB<ValueType>;

    friend class HIPAcceleratorMatrixCOO<double>;
    friend class HIPAcceleratorMatrixCOO<float>;
    friend class HIPAcceleratorMatrixCOO<std::complex<double>>;
    friend class HIPAcceleratorMatrixCOO<std::complex<float>>;

    friend class HIPAcceleratorMatrixCSR<double>;
    friend class HIPAcceleratorMatrixCSR<float>;
    friend class HIPAcceleratorMatrixCSR<std::complex<double>>;
    friend class HIPAcceleratorMatrixCSR<std::complex<float>>;
};

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
