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

#ifndef ROCALUTION_VECTOR_HPP_
#define ROCALUTION_VECTOR_HPP_

#include "../utils/types.hpp"
#include "base_rocalution.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

namespace rocalution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class LocalVector;

/** \ingroup op_vec_module
  * \class Vector
  * \brief Vector class
  * \details
  * The Vector class defines the generic interface for local and global vectors.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
template <typename ValueType>
class Vector : public BaseRocalution<ValueType>
{
    public:
    Vector();
    virtual ~Vector();

    /** \brief Return the size of the vector */
    virtual IndexType2 GetSize(void) const = 0;
    /** \brief Return the size of the local vector */
    virtual int GetLocalSize(void) const;
    /** \brief Return the size of the ghost vector */
    virtual int GetGhostSize(void) const;

    /** \brief Perform a sanity check of the vector
      * \details
      * Checks, if the vector contains valid data, i.e. if the values are not infinity
      * and not NaN (not a number).
      *
      * \retval true if the vector is ok (empty vector is also ok).
      * \retval false if there is something wrong with the values.
      */
    virtual bool Check(void) const = 0;

    virtual void Clear(void) = 0;

    /** \brief Set all values of the vector to 0 */
    virtual void Zeros(void) = 0;

    /** \brief Set all values of the vector to 1 */
    virtual void Ones(void) = 0;

    /** \brief Set all values of the vector to given argument */
    virtual void SetValues(ValueType val) = 0;

    /** \brief Fill the vector with random values from interval [a,b] */
    virtual void SetRandomUniform(unsigned long long seed,
                                  ValueType a = static_cast<ValueType>(-1),
                                  ValueType b = static_cast<ValueType>(1)) = 0;

    /** \brief Fill the vector with random values from normal distribution */
    virtual void SetRandomNormal(unsigned long long seed,
                                 ValueType mean = static_cast<ValueType>(0),
                                 ValueType var = static_cast<ValueType>(1)) = 0;

    /** \brief Read vector from ASCII file
      * \details
      * Read a vector from ASCII file.
      *
      * @param[in]
      * filename    name of the file containing the ASCII data.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *   vec.ReadFileASCII("my_vector.dat");
      * \endcode
      */
    virtual void ReadFileASCII(const std::string filename) = 0;

    /** \brief Write vector to ASCII file
      * \details
      * Write a vector to ASCII file.
      *
      * @param[in]
      * filename    name of the file to write the ASCII data to.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate and fill vec
      *   // ...
      *
      *   vec.WriteFileASCII("my_vector.dat");
      * \endcode
      */
    virtual void WriteFileASCII(const std::string filename) const = 0;

    /** \brief Read vector from binary file
      * \details
      * Read a vector from binary file. For details on the format, see WriteFileBinary().
      *
      * @param[in]
      * filename    name of the file containing the data.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *   vec.ReadFileBinary("my_vector.bin");
      * \endcode
      */
    virtual void ReadFileBinary(const std::string filename) = 0;

    /** \brief Write vector to binary file
      * \details
      * Write a vector to binary file.
      *
      * The binary format contains a header, the rocALUTION version and the vector data
      * as follows
      * \code{.cpp}
      *   // Header
      *   out << "#rocALUTION binary vector file" << std::endl;
      *
      *   // rocALUTION version
      *   out.write((char*)&version, sizeof(int));
      *
      *   // Vector data
      *   out.write((char*)&size, sizeof(int));
      *   out.write((char*)vec_val, size * sizeof(double));
      * \endcode
      *
      * \note
      * Vector values array is always stored in double precision (e.g. double or
      * std::complex<double>).
      *
      * @param[in]
      * filename    name of the file to write the data to.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate and fill vec
      *   // ...
      *
      *   vec.WriteFileBinary("my_vector.bin");
      * \endcode
      */
    virtual void WriteFileBinary(const std::string filename) const = 0;

    /** \brief Copy vector from another vector
      * \details
      * \p CopyFrom copies values from another vector.
      *
      * \note
      * This function allows cross platform copying. One of the objects could be
      * allocated on the accelerator backend.
      *
      * @param[in]
      * src Vector, where values should be copied from.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec1, vec2;
      *
      *   // Allocate and initialize vec1 and vec2
      *   // ...
      *
      *   // Move vec1 to accelerator
      *   // vec1.MoveToAccelerator();
      *
      *   // Now, vec1 is on the accelerator (if available)
      *   // and vec2 is on the host
      *
      *   // Copy vec1 to vec2 (or vice versa) will move data between host and
      *   // accelerator backend
      *   vec1.CopyFrom(vec2);
      * \endcode
      */
    /**@{*/
    virtual void CopyFrom(const LocalVector<ValueType>& src);
    virtual void CopyFrom(const GlobalVector<ValueType>& src);
    /**@}*/

    /** \brief Async copy from another local vector */
    virtual void CopyFromAsync(const LocalVector<ValueType>& src);

    /** \brief Copy values from another local float vector */
    virtual void CopyFromFloat(const LocalVector<float>& src);

    /** \brief Copy values from another local double vector */
    virtual void CopyFromDouble(const LocalVector<double>& src);

    /** \brief Copy vector from another vector with offsets and size
      * \details
      * \p CopyFrom copies values with specific source and destination offsets and sizes
      * from another vector.
      *
      * \note
      * This function allows cross platform copying. One of the objects could be
      * allocated on the accelerator backend.
      *
      * @param[in]
      * src         Vector, where values should be copied from.
      * @param[in]
      * src_offset  source offset.
      * @param[in]
      * dst_offset  destination offset.
      * @param[in]
      * size        number of entries to be copied.
      */
    virtual void
    CopyFrom(const LocalVector<ValueType>& src, int src_offset, int dst_offset, int size);

    /** \brief Clone the vector
      * \details
      * \p CloneFrom clones the entire vector, with data and backend descriptor from another Vector.
      *
      * @param[in]
      * src Vector to clone from.
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate and initialize vec (host or accelerator)
      *   // ...
      *
      *   LocalVector<ValueType> tmp;
      *
      *   // By cloning vec, tmp will have identical values and will be on the same
      *   // backend as vec
      *   tmp.CloneFrom(vec);
      * \endcode
      */
    /**@{*/
    virtual void CloneFrom(const LocalVector<ValueType>& src);
    virtual void CloneFrom(const GlobalVector<ValueType>& src);
    /**@}*/

    /** \brief Perform vector update of type this = this + alpha * x */
    virtual void AddScale(const LocalVector<ValueType>& x, ValueType alpha);
    /** \brief Perform vector update of type this = this + alpha * x */
    virtual void AddScale(const GlobalVector<ValueType>& x, ValueType alpha);

    /** \brief Perform vector update of type this = alpha * this + x */
    virtual void ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x);
    /** \brief Perform vector update of type this = alpha * this + x */
    virtual void ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x);

    /** \brief Perform vector update of type this = alpha * this + x * beta */
    virtual void ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, ValueType beta);
    /** \brief Perform vector update of type this = alpha * this + x * beta */
    virtual void ScaleAddScale(ValueType alpha, const GlobalVector<ValueType>& x, ValueType beta);

    /** \brief Perform vector update of type this = alpha * this + x * beta with offsets */
    virtual void ScaleAddScale(ValueType alpha,
                               const LocalVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size);
    /** \brief Perform vector update of type this = alpha * this + x * beta with offsets */
    virtual void ScaleAddScale(ValueType alpha,
                               const GlobalVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size);

    /** \brief Perform vector update of type this = alpha * this + x * beta + y * gamma */
    virtual void ScaleAdd2(ValueType alpha,
                           const LocalVector<ValueType>& x,
                           ValueType beta,
                           const LocalVector<ValueType>& y,
                           ValueType gamma);
    /** \brief Perform vector update of type this = alpha * this + x * beta + y * gamma */
    virtual void ScaleAdd2(ValueType alpha,
                           const GlobalVector<ValueType>& x,
                           ValueType beta,
                           const GlobalVector<ValueType>& y,
                           ValueType gamma);

    /** \brief Perform vector scaling this = alpha * this */
    virtual void Scale(ValueType alpha) = 0;

    /** \brief Compute dot (scalar) product, return this^T y */
    virtual ValueType Dot(const LocalVector<ValueType>& x) const;
    /** \brief Compute dot (scalar) product, return this^T y */
    virtual ValueType Dot(const GlobalVector<ValueType>& x) const;

    /** \brief Compute non-conjugate dot (scalar) product, return this^T y */
    virtual ValueType DotNonConj(const LocalVector<ValueType>& x) const;
    /** \brief Compute non-conjugate dot (scalar) product, return this^T y */
    virtual ValueType DotNonConj(const GlobalVector<ValueType>& x) const;

    /** \brief Compute \f$L_2\f$ norm of the vector, return = srqt(this^T this) */
    virtual ValueType Norm(void) const = 0;

    /** \brief Reduce the vector */
    virtual ValueType Reduce(void) const = 0;

    /** \brief Compute the sum of absolute values of the vector, return = sum(|this|) */
    virtual ValueType Asum(void) const = 0;

    /** \brief Compute the absolute max of the vector, return = index(max(|this|)) */
    virtual int Amax(ValueType& value) const = 0;

    /** \brief Perform point-wise multiplication (element-wise) of this = this * x */
    virtual void PointWiseMult(const LocalVector<ValueType>& x);
    /** \brief Perform point-wise multiplication (element-wise) of this = this * x */
    virtual void PointWiseMult(const GlobalVector<ValueType>& x);

    /** \brief Perform point-wise multiplication (element-wise) of this = x * y */
    virtual void PointWiseMult(const LocalVector<ValueType>& x, const LocalVector<ValueType>& y);
    /** \brief Perform point-wise multiplication (element-wise) of this = x * y */
    virtual void PointWiseMult(const GlobalVector<ValueType>& x, const GlobalVector<ValueType>& y);

    /** \brief Perform power operation to a vector */
    virtual void Power(double power) = 0;
};

} // namespace rocalution

#endif // ROCALUTION_VECTOR_HPP_
