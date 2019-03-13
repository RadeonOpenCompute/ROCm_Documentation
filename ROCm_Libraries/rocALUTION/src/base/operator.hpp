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

#ifndef ROCALUTION_OPERATOR_HPP_
#define ROCALUTION_OPERATOR_HPP_

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
  * \class Operator
  * \brief Operator class
  * \details
  * The Operator class defines the generic interface for applying an operator (e.g.
  * matrix or stencil) from/to global and local vectors.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
template <typename ValueType>
class Operator : public BaseRocalution<ValueType>
{
    public:
    Operator();
    virtual ~Operator();

    /** \brief Return the number of rows in the matrix/stencil */
    virtual IndexType2 GetM(void) const = 0;
    /** \brief Return the number of columns in the matrix/stencil */
    virtual IndexType2 GetN(void) const = 0;
    /** \brief Return the number of non-zeros in the matrix/stencil */
    virtual IndexType2 GetNnz(void) const = 0;

    /** \brief Return the number of rows in the local matrix/stencil */
    virtual int GetLocalM(void) const;
    /** \brief Return the number of columns in the local matrix/stencil */
    virtual int GetLocalN(void) const;
    /** \brief Return the number of non-zeros in the local matrix/stencil */
    virtual int GetLocalNnz(void) const;

    /** \brief Return the number of rows in the ghost matrix/stencil */
    virtual int GetGhostM(void) const;
    /** \brief Return the number of columns in the ghost matrix/stencil */
    virtual int GetGhostN(void) const;
    /** \brief Return the number of non-zeros in the ghost matrix/stencil */
    virtual int GetGhostNnz(void) const;

    /** \brief Apply the operator, out = Operator(in), where in and out are local
      * vectors
      */
    virtual void Apply(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

    /** \brief Apply and add the operator, out += scalar * Operator(in), where in and out
      * are local vectors
      */
    virtual void
    ApplyAdd(const LocalVector<ValueType>& in, ValueType scalar, LocalVector<ValueType>* out) const;

    /** \brief Apply the operator, out = Operator(in), where in and out are global
      * vectors
      */
    virtual void Apply(const GlobalVector<ValueType>& in, GlobalVector<ValueType>* out) const;

    /** \brief Apply and add the operator, out += scalar * Operator(in), where in and out
      * are global vectors
      */
    virtual void ApplyAdd(const GlobalVector<ValueType>& in,
                          ValueType scalar,
                          GlobalVector<ValueType>* out) const;
};

} // namespace rocalution

#endif // ROCALUTION_OPERTOR_HPP_
