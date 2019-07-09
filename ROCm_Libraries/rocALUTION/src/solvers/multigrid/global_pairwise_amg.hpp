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

#ifndef ROCALUTION_MULTIGRID_GLOBAL_PAIRWISE_AMG_HPP_
#define ROCALUTION_MULTIGRID_GLOBAL_PAIRWISE_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"
#include "pairwise_amg.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class GlobalPairwiseAMG
  * \brief Pairwise Aggregation Algebraic MultiGrid Method (multi-node)
  * \details
  * The Pairwise Aggregation Algebraic MultiGrid method is based on a pairwise
  * aggregation matching scheme. It delivers very efficient building phase which is
  * suitable for Poisson-like equation. Most of the time it requires K-cycle for the
  * solving phase to provide low number of iterations. This version has multi-node
  * support.
  * \cite pairwiseamg
  *
  * \tparam OperatorType - can be GlobalMatrix
  * \tparam VectorType - can be GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class GlobalPairwiseAMG : public BaseAMG<OperatorType, VectorType, ValueType>
{
    public:
    GlobalPairwiseAMG();
    virtual ~GlobalPairwiseAMG();

    virtual void Print(void) const;
    virtual void BuildHierarchy(void);
    virtual void ClearLocal(void);

    /** \brief Set beta for pairwise aggregation */
    virtual void SetBeta(ValueType beta);
    /** \brief Set re-ordering for aggregation */
    virtual void SetOrdering(const _aggregation_ordering ordering);
    /** \brief Set target coarsening factor */
    virtual void SetCoarseningFactor(double factor);

    virtual void ReBuildNumeric(void);

    protected:
    /** \brief Constructs the prolongation, restriction and coarse operator */
    void Aggregate_(const OperatorType& op,
                    Operator<ValueType>* pro,
                    Operator<ValueType>* res,
                    OperatorType* coarse,
                    ParallelManager* pm,
                    LocalVector<int>* trans);

    /** \private */
    virtual void Aggregate_(const OperatorType& op,
                            Operator<ValueType>* pro,
                            Operator<ValueType>* res,
                            OperatorType* coarse);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    private:
    // Beta factor
    ValueType beta_;

    // Target factor for coarsening ratio
    double coarsening_factor_;
    // Ordering for the aggregation scheme
    int aggregation_ordering_;

    // Parallel Manager for coarser levels
    ParallelManager** pm_level_;

    // Transfer mapping
    LocalVector<int>** trans_level_;

    // Dimension of the coarse operators
    std::vector<int> dim_level_;
    std::vector<int> Gsize_level_;
    std::vector<int> rGsize_level_;
    std::vector<int*> rG_level_;
};

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_GLOBAL_PAIRWISE_AMG_HPP_
