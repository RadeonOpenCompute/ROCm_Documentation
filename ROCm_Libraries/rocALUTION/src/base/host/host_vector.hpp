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

#ifndef ROCALUTION_HOST_VECTOR_HPP_
#define ROCALUTION_HOST_VECTOR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../base_stencil.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
class LocalVector;

template <typename ValueType>
class HostVector : public BaseVector<ValueType>
{
    public:
    HostVector();
    HostVector(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HostVector();

    virtual void Info(void) const;

    virtual bool Check(void) const;
    virtual void Allocate(int n);
    virtual void SetDataPtr(ValueType** ptr, int size);
    virtual void LeaveDataPtr(ValueType** ptr);
    virtual void Clear(void);
    virtual void Zeros(void);
    virtual void Ones(void);
    virtual void SetValues(ValueType val);
    virtual void SetRandomUniform(unsigned long long seed, ValueType a, ValueType b);
    virtual void SetRandomNormal(unsigned long long seed, ValueType mean, ValueType var);

    virtual void CopyFrom(const BaseVector<ValueType>& vec);
    virtual void CopyFromFloat(const BaseVector<float>& vec);
    virtual void CopyFromDouble(const BaseVector<double>& vec);
    virtual void CopyTo(BaseVector<ValueType>* vec) const;
    virtual void
    CopyFrom(const BaseVector<ValueType>& src, int src_offset, int dst_offset, int size);
    virtual void CopyFromPermute(const BaseVector<ValueType>& src,
                                 const BaseVector<int>& permutation);
    virtual void CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                         const BaseVector<int>& permutation);

    virtual void CopyFromData(const ValueType* data);
    virtual void CopyToData(ValueType* data) const;

    virtual void Permute(const BaseVector<int>& permutation);
    virtual void PermuteBackward(const BaseVector<int>& permutation);

    virtual bool Restriction(const BaseVector<ValueType>& vec_fine, const BaseVector<int>& map);
    virtual bool Prolongation(const BaseVector<ValueType>& vec_coarse, const BaseVector<int>& map);

    /// Read vector from ASCII file
    void ReadFileASCII(const std::string filename);
    /// Write vector to ASCII file
    void WriteFileASCII(const std::string filename) const;
    /// Read vector from binary file
    void ReadFileBinary(const std::string filename);
    /// Write vector to binary file
    void WriteFileBinary(const std::string filename) const;

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
    // reduce vector
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

    // for [] operator in LocalVector
    friend class LocalVector<ValueType>;

    friend class HostVector<double>;
    friend class HostVector<float>;
    friend class HostVector<std::complex<double>>;
    friend class HostVector<std::complex<float>>;
    friend class HostVector<int>;

    friend class HostMatrix<ValueType>;
    friend class HostMatrixCSR<ValueType>;
    friend class HostMatrixCOO<ValueType>;
    friend class HostMatrixDIA<ValueType>;
    friend class HostMatrixELL<ValueType>;
    friend class HostMatrixHYB<ValueType>;
    friend class HostMatrixDENSE<ValueType>;
    friend class HostMatrixMCSR<ValueType>;
    friend class HostMatrixBCSR<ValueType>;

    friend class HostMatrixCOO<float>;
    friend class HostMatrixCOO<double>;
    friend class HostMatrixCOO<std::complex<float>>;
    friend class HostMatrixCOO<std::complex<double>>;

    friend class HostMatrixCSR<double>;
    friend class HostMatrixCSR<float>;
    friend class HostMatrixCSR<std::complex<float>>;
    friend class HostMatrixCSR<std::complex<double>>;

    friend class HostMatrixDENSE<double>;
    friend class HostMatrixDENSE<float>;

    friend class HIPAcceleratorVector<ValueType>;

    friend class HostStencil<ValueType>;
    friend class HostStencilLaplace2D<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_HOST_VECTOR_HPP_
