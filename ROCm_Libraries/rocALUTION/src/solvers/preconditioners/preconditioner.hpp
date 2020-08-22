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

#ifndef ROCALUTION_PRECONDITIONER_HPP_
#define ROCALUTION_PRECONDITIONER_HPP_

#include "../solver.hpp"

namespace rocalution {

/** \ingroup precond_module
  * \class Preconditioner
  * \brief Base class for all preconditioners
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class Preconditioner : public Solver<OperatorType, VectorType, ValueType>
{
    public:
    Preconditioner();
    virtual ~Preconditioner();

    virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

    protected:
    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;
};

/** \ingroup precond_module
  * \class Jacobi
  * \brief Jacobi Method
  * \details
  * The Jacobi method is for solving a diagonally dominant system of linear equations
  * \f$Ax=b\f$. It solves for each diagonal element iteratively until convergence, such
  * that
  * \f[
  *   x_{i}^{(k+1)} = (1 - \omega)x_{i}^{(k)} + \frac{\omega}{a_{ii}}
  *   \left(
  *     b_{i} - \sum\limits_{j=1}^{i-1}{a_{ij}x_{j}^{(k)}} -
  *     \sum\limits_{j=i}^{n}{a_{ij}x_{j}^{(k)}}
  *   \right)
  * \f]
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class Jacobi : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    Jacobi();
    virtual ~Jacobi();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);
    virtual void Build(void);
    virtual void Clear(void);

    virtual void ResetOperator(const OperatorType& op);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    VectorType inv_diag_entries_;
};

/** \ingroup precond_module
  * \class GS
  * \brief Gauss-Seidel / Successive Over-Relaxation Method
  * \details
  * The Gauss-Seidel / SOR method is for solving system of linear equations \f$Ax=b\f$.
  * It approximates the solution iteratively with
  * \f[
  *    x_{i}^{(k+1)} = (1 - \omega) x_{i}^{(k)} + \frac{\omega}{a_{ii}}
  *    \left(
  *      b_{i} - \sum\limits_{j=1}^{i-1}{a_{ij}x_{j}^{(k+1)}} -
  *      \sum\limits_{j=i}^{n}{a_{ij}x_{j}^{(k)}}
  *    \right),
  * \f]
  * with \f$\omega \in (0,2)\f$.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class GS : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    GS();
    virtual ~GS();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);
    virtual void Build(void);
    virtual void Clear(void);

    virtual void ResetOperator(const OperatorType& op);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType GS_;
};

/** \ingroup precond_module
  * \class SGS
  * \brief Symmetric Gauss-Seidel / Symmetric Successive Over-Relaxation Method
  * \details
  * The Symmetric Gauss-Seidel / SSOR method is for solving system of linear equations
  * \f$Ax=b\f$. It approximates the solution iteratively.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class SGS : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    SGS();
    virtual ~SGS();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);
    virtual void Build(void);
    virtual void Clear(void);

    virtual void ResetOperator(const OperatorType& op);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType SGS_;

    VectorType diag_entries_;
    VectorType v_;
};

/** \ingroup precond_module
  * \class ILU
  * \brief Incomplete LU Factorization based on levels
  * \details
  * The Incomplete LU Factorization based on levels computes a sparse lower and sparse
  * upper triangular matrix such that \f$A = LU - R\f$.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class ILU : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    ILU();
    virtual ~ILU();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);

    /** \brief Initialize ILU(p) factorization
      * \details
      * Initialize ILU(p) factorization based on power.
      * \cite SAAD
      * - level = true build the structure based on levels
      * - level = false build the structure only based on the power(p+1)
      */
    virtual void Set(int p, bool level = true);
    virtual void Build(void);
    virtual void Clear(void);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType ILU_;
    int p_;
    bool level_;
};

/** \ingroup precond_module
  * \class ILUT
  * \brief Incomplete LU Factorization based on threshold
  * \details
  * The Incomplete LU Factorization based on threshold computes a sparse lower and sparse
  * upper triangular matrix such that \f$A = LU - R\f$. Fill-in values are dropped
  * depending on a threshold and number of maximal fill-ins per row.
  * \cite SAAD
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class ILUT : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    ILUT();
    virtual ~ILUT();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);

    /** \brief Set drop-off threshold */
    virtual void Set(double t);

    /** \brief Set drop-off threshold and maximum fill-ins per row */
    virtual void Set(double t, int maxrow);

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType ILUT_;
    double t_;
    int max_row_;
};

/** \ingroup precond_module
  * \class IC
  * \brief Incomplete Cholesky Factorization without fill-ins
  * \details
  * The Incomplete Cholesky Factorization computes a sparse lower triangular matrix
  * such that \f$A=LL^{T} - R\f$. Additional fill-ins are dropped and the sparsity
  * pattern of the original matrix is preserved.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class IC : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    IC();
    virtual ~IC();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);
    virtual void Build(void);
    virtual void Clear(void);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType IC_;
    VectorType inv_diag_entries_;
};

/** \ingroup precond_module
  * \class VariablePreconditioner
  * \brief Variable Preconditioner
  * \details
  * The Variable Preconditioner can hold a selection of preconditioners. Thus, any type
  * of preconditioners can be combined. As example, the variable preconditioner can
  * combine Jacobi, GS and ILU – then, the first iteration of the iterative solver will
  * apply Jacobi, the second iteration will apply GS and the third iteration will apply
  * ILU. After that, the solver will start again with Jacobi, GS, ILU.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class VariablePreconditioner : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    VariablePreconditioner();
    virtual ~VariablePreconditioner();

    virtual void Print(void) const;
    virtual void Solve(const VectorType& rhs, VectorType* x);
    virtual void Build(void);
    virtual void Clear(void);

    /** \brief Set the preconditioner sequence */
    virtual void SetPreconditioner(int n, Solver<OperatorType, VectorType, ValueType>** precond);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    int num_precond_;
    int counter_;
    Solver<OperatorType, VectorType, ValueType>** precond_;
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_HPP_
