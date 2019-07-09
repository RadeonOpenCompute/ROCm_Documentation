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

#ifndef ROCALUTION_SMOOTHED_AMG_HPP_
#define ROCALUTION_SMOOTHED_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class SAAMG
  * \brief Smoothed Aggregation Algebraic MultiGrid Method
  * \details
  * The Smoothed Aggregation Algebraic MultiGrid method is based on smoothed
  * aggregation based interpolation scheme.
  * \cite vanek
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class SAAMG : public BaseAMG<OperatorType, VectorType, ValueType>
{
    public:
    SAAMG();
    virtual ~SAAMG();

    virtual void Print(void) const;
    virtual void BuildSmoothers(void);

    /** \brief Set coupling strength */
    void SetCouplingStrength(ValueType eps);
    /** \brief Set the relaxation parameter */
    void SetInterpRelax(ValueType relax);

    virtual void ReBuildNumeric(void);

    protected:
    virtual void Aggregate_(const OperatorType& op,
                            Operator<ValueType>* pro,
                            Operator<ValueType>* res,
                            OperatorType* coarse);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    private:
    /** \brief Coupling strength */
    ValueType eps_;

    /** \brief Relaxation parameter */
    ValueType relax_;
};

} // namespace rocalution

#endif // ROCALUTION_SMOOTHED_AMG_HPP_
