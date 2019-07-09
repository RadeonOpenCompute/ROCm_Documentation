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

#ifndef ROCALUTION_KRYLOV_IDR_HPP_
#define ROCALUTION_KRYLOV_IDR_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class IDR
  * \brief Induced Dimension Reduction Method
  * \details
  * The Induced Dimension Reduction method is a Krylov subspace method for solving
  * sparse (non) symmetric linear systems \f$Ax=b\f$. IDR(s) generates residuals in a
  * sequence of nested subspaces.
  * \cite IDR1
  * \cite IDR2
  *
  * The dimension of the shadow space can be set by SetShadowSpace(). The default size
  * of the shadow space is 4.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class IDR : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    IDR();
    virtual ~IDR();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    /** \brief Set the size of the Shadow Space */
    void SetShadowSpace(int s);
    /** \brief Set random seed for ONB creation (seed must be greater than 0) */
    void SetRandomSeed(unsigned long long seed);

    protected:
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    int s_;
    unsigned long long seed_;

    ValueType kappa_;

    ValueType* c_;
    ValueType* f_;
    ValueType* M_;

    VectorType r_;
    VectorType v_;
    VectorType t_;

    VectorType** G_;
    VectorType** U_;
    VectorType** P_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_IDR_HPP_
