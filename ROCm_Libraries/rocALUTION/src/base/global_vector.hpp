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

#ifndef ROCALUTION_GLOBAL_VECTOR_HPP_
#define ROCALUTION_GLOBAL_VECTOR_HPP_

#include "../utils/types.hpp"
#include "vector.hpp"
#include "parallel_manager.hpp"

namespace rocalution {

template <typename ValueType>
class LocalVector;
template <typename ValueType>
class LocalMatrix;
template <typename ValueType>
class GlobalMatrix;
struct MRequest;

/** \ingroup op_vec_module
  * \class GlobalVector
  * \brief GlobalVector class
  * \details
  * A GlobalVector is called global, because it can stay on a single or on multiple nodes
  * in a network. For this type of communication, MPI is used.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
template <typename ValueType>
class GlobalVector : public Vector<ValueType>
{
    public:
    GlobalVector();
    /** \brief Initialize a global vector with a parallel manager */
    GlobalVector(const ParallelManager& pm);
    virtual ~GlobalVector();

    virtual void MoveToAccelerator(void);
    virtual void MoveToHost(void);

    virtual void Info(void) const;
    virtual bool Check(void) const;

    virtual IndexType2 GetSize(void) const;
    virtual int GetLocalSize(void) const;
    virtual int GetGhostSize(void) const;

    /** \private */
    const LocalVector<ValueType>& GetInterior() const;
    /** \private */
    LocalVector<ValueType>& GetInterior();
    /** \private */
    const LocalVector<ValueType>& GetGhost() const;

    /** \brief Allocate a global vector with name and size */
    virtual void Allocate(std::string name, IndexType2 size);
    virtual void Clear(void);

    /** \brief Set the parallel manager of a global vector */
    void SetParallelManager(const ParallelManager& pm);

    virtual void Zeros(void);
    virtual void Ones(void);
    virtual void SetValues(ValueType val);
    virtual void SetRandomUniform(unsigned long long seed,
                                  ValueType a = static_cast<ValueType>(-1),
                                  ValueType b = static_cast<ValueType>(1));
    virtual void SetRandomNormal(unsigned long long seed,
                                 ValueType mean = static_cast<ValueType>(0),
                                 ValueType var  = static_cast<ValueType>(1));
    void CloneFrom(const GlobalVector<ValueType>& src);

    /** \brief Access operator (only for host data) */
    ValueType& operator[](int i);
    /** \brief Access operator (only for host data) */
    const ValueType& operator[](int i) const;

    /** \brief Initialize the local part of a global vector with externally allocated
      * data
      */
    void SetDataPtr(ValueType** ptr, std::string name, IndexType2 size);
    /** \brief Get a pointer to the data from the local part of a global vector and free
      * the global vector object
      */
    void LeaveDataPtr(ValueType** ptr);

    virtual void CopyFrom(const GlobalVector<ValueType>& src);
    virtual void ReadFileASCII(const std::string filename);
    virtual void WriteFileASCII(const std::string filename) const;
    virtual void ReadFileBinary(const std::string filename);
    virtual void WriteFileBinary(const std::string filename) const;

    virtual void AddScale(const GlobalVector<ValueType>& x, ValueType alpha);
    virtual void ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x);
    virtual void ScaleAdd2(ValueType alpha,
                           const GlobalVector<ValueType>& x,
                           ValueType beta,
                           const GlobalVector<ValueType>& y,
                           ValueType gamma);
    virtual void ScaleAddScale(ValueType alpha, const GlobalVector<ValueType>& x, ValueType beta);
    virtual void Scale(ValueType alpha);
    virtual ValueType Dot(const GlobalVector<ValueType>& x) const;
    virtual ValueType DotNonConj(const GlobalVector<ValueType>& x) const;
    virtual ValueType Norm(void) const;
    virtual ValueType Reduce(void) const;
    virtual ValueType Asum(void) const;
    virtual int Amax(ValueType& value) const;
    virtual void PointWiseMult(const GlobalVector<ValueType>& x);
    virtual void PointWiseMult(const GlobalVector<ValueType>& x, const GlobalVector<ValueType>& y);

    virtual void Power(double power);

    /** \brief Restriction operator based on restriction mapping vector */
    void Restriction(const GlobalVector<ValueType>& vec_fine, const LocalVector<int>& map);

    /** \brief Prolongation operator based on restriction mapping vector */
    void Prolongation(const GlobalVector<ValueType>& vec_coarse, const LocalVector<int>& map);

    protected:
    virtual bool is_host_(void) const;
    virtual bool is_accel_(void) const;

    /** \brief Update ghost values asynchronously */
    void UpdateGhostValuesAsync_(const GlobalVector<ValueType>& in);
    /** \brief Update ghost values synchronously */
    void UpdateGhostValuesSync_(void);

    private:
    MRequest* recv_event_;
    MRequest* send_event_;

    ValueType* recv_boundary_;
    ValueType* send_boundary_;

    LocalVector<ValueType> vector_interior_;
    LocalVector<ValueType> vector_ghost_;

    friend class LocalMatrix<ValueType>;
    friend class GlobalMatrix<ValueType>;

    friend class BaseRocalution<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_GLOBAL_VECTOR_HPP_
