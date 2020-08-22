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

#ifndef ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_
#define ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "preconditioner_multicolored.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace rocalution {

/** \ingroup precond_module
  * \class MultiColoredSGS
  * \brief Multi-Colored Symmetric Gauss-Seidel / SSOR Preconditioner
  * \details
  * The Multi-Colored Symmetric Gauss-Seidel / SSOR preconditioner is based on the
  * splitting of the original matrix. Higher parallelism in solving the forward and
  * backward substitution is obtained by performing a multi-colored decomposition.
  * Details on the Symmetric Gauss-Seidel / SSOR algorithm can be found in the SGS
  * preconditioner.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredSGS : public MultiColored<OperatorType, VectorType, ValueType>
{
    public:
    MultiColoredSGS();
    virtual ~MultiColoredSGS();

    virtual void Print(void) const;

    virtual void ReBuildNumeric(void);

    /** \brief Set the relaxation parameter for the SOR/SSOR scheme */
    void SetRelaxation(ValueType omega);

    protected:
    virtual void PostAnalyse_(void);

    virtual void SolveL_(void);
    virtual void SolveD_(void);
    virtual void SolveR_(void);
    virtual void Solve_(const VectorType& rhs, VectorType* x);

    /** \brief Relaxation parameter */
    ValueType omega_;
};

/** \ingroup precond_module
  * \class MultiColoredGS
  * \brief Multi-Colored Gauss-Seidel / SOR Preconditioner
  * \details
  * The Multi-Colored Symmetric Gauss-Seidel / SOR preconditioner is based on the
  * splitting of the original matrix. Higher parallelism in solving the forward
  * substitution is obtained by performing a multi-colored decomposition. Details on the
  * Gauss-Seidel / SOR algorithm can be found in the GS preconditioner.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredGS : public MultiColoredSGS<OperatorType, VectorType, ValueType>
{
    public:
    MultiColoredGS();
    virtual ~MultiColoredGS();

    virtual void Print(void) const;

    protected:
    virtual void PostAnalyse_(void);

    virtual void SolveL_(void);
    virtual void SolveD_(void);
    virtual void SolveR_(void);
    virtual void Solve_(const VectorType& rhs, VectorType* x);
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_
