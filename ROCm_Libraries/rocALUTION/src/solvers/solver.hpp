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

#ifndef ROCALUTION_SOLVER_HPP_
#define ROCALUTION_SOLVER_HPP_

#include "iter_ctrl.hpp"
#include "../base/base_rocalution.hpp"
#include "../base/local_vector.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class Solver
  * \brief Base class for all solvers and preconditioners
  * \details
  * Most of the solvers can be performed on linear operators LocalMatrix, LocalStencil
  * and GlobalMatrix - i.e. the solvers can be performed locally (on a shared memory
  * system) or in a distributed manner (on a cluster) via MPI. The only exception is the
  * AMG (Algebraic Multigrid) solver which has two versions (one for LocalMatrix and one
  * for GlobalMatrix class). The only pure local solvers (which do not support global/MPI
  * operations) are the mixed-precision defect-correction solver and all direct solvers.
  *
  * All solvers need three template parameters - Operators, Vectors and Scalar type.
  *
  * The Solver class is purely virtual and provides an interface for
  * - SetOperator() to set the operator \f$A\f$, i.e. the user can pass the matrix here.
  * - Build() to build the solver (including preconditioners, sub-solvers, etc.). The
  *   user need to specify the operator first before calling Build().
  * - Solve() to solve the system \f$Ax = b\f$. The user need to pass a right-hand-side
  *   \f$b\f$ and a vector \f$x\f$, where the solution will be obtained.
  * - Print() to show solver information.
  * - ReBuildNumeric() to only re-build the solver numerically (if possible).
  * - MoveToHost() and MoveToAccelerator() to offload the solver (including
  *   preconditioners and sub-solvers) to the host/accelerator.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class Solver : public RocalutionObj
{
    public:
    Solver();
    virtual ~Solver();

    /** \brief Set the Operator of the solver */
    void SetOperator(const OperatorType& op);

    /** \brief Reset the operator; see ReBuildNumeric() */
    virtual void ResetOperator(const OperatorType& op);

    /** \brief Print information about the solver */
    virtual void Print(void) const = 0;

    /** \brief Solve Operator x = rhs */
    virtual void Solve(const VectorType& rhs, VectorType* x) = 0;

    /** \brief Solve Operator x = rhs, setting initial x = 0 */
    virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

    /** \brief Clear (free all local data) the solver */
    virtual void Clear(void);

    /** \brief Build the solver (data allocation, structure and numerical computation) */
    virtual void Build(void);

    /** \brief Build the solver and move it to the accelerator asynchronously */
    virtual void BuildMoveToAcceleratorAsync(void);

    /** \brief Synchronize the solver */
    virtual void Sync(void);

    /** \brief Rebuild the solver only with numerical computation (no allocation or data
      * structure computation)
      */
    virtual void ReBuildNumeric(void);

    /** \brief Move all data (i.e. move the solver) to the host */
    virtual void MoveToHost(void);
    /** \brief Move all data (i.e. move the solver) to the accelerator */
    virtual void MoveToAccelerator(void);

    /** \brief Provide verbose output of the solver
      * \details
      * - verb = 0 -> no output
      * - verb = 1 -> print info about the solver (start, end);
      * - verb = 2 -> print (iter, residual) via iteration control;
      */
    virtual void Verbose(int verb = 1);

    protected:
    /** \brief Pointer to the operator */
    const OperatorType* op_;

    /** \brief Pointer to the defined preconditioner */
    Solver<OperatorType, VectorType, ValueType>* precond_;

    /** \brief Flag == true after building the solver (e.g. Build()) */
    bool build_;

    /** \brief Permutation vector (used if the solver performs permutation/re-ordering
      * techniques)
      */
    LocalVector<int> permutation_;

    /** \brief Verbose flag */
    int verb_;

    /** \brief Print starting message of the solver */
    virtual void PrintStart_(void) const = 0;
    /** \brief Print ending message of the solver */
    virtual void PrintEnd_(void) const = 0;

    /** \brief Move all local data to the host */
    virtual void MoveToHostLocalData_(void) = 0;
    /** \brief Move all local data to the accelerator */
    virtual void MoveToAcceleratorLocalData_(void) = 0;
};

/** \ingroup solver_module
  * \class IterativeLinearSolver
  * \brief Base class for all linear iterative solvers
  * \details
  * The iterative solvers are controlled by an iteration control object, which monitors
  * the convergence properties of the solver, i.e. maximum number of iteration, relative
  * tolerance, absolute tolerance and divergence tolerance. The iteration control can
  * also record the residual history and store it in an ASCII file.
  * - Init(), InitMinIter(), InitMaxIter() and InitTol() initialize the solver and set the
  * stopping criteria.
  * - RecordResidualHistory() and RecordHistory() start the recording of the residual and
  * write it into a file.
  * - Verbose() sets the level of verbose output of the solver (0 - no output, 2 - detailed
  * output, including residual and iteration information).
  * - SetPreconditioner() sets the preconditioning.
  *
  * All iterative solvers are controlled based on
  * - Absolute stopping criteria, when \f$|r_{k}|_{L_{p}} \lt \epsilon_{abs}\f$
  * - Relative stopping criteria, when \f$|r_{k}|_{L_{p}} / |r_{1}|_{L_{p}} \leq
  *   \epsilon_{rel}\f$
  * - Divergence stopping criteria, when \f$|r_{k}|_{L_{p}} / |r_{1}|_{L_{p}} \geq
  *   \epsilon_{div}\f$
  * - Maximum number of iteration \f$N\f$, when \f$k = N\f$
  *
  * where \f$k\f$ is the current iteration, \f$r_{k}\f$ the residual for the current
  * iteration \f$k\f$ (i.e. \f$r_{k} = b - Ax_{k}\f$) and \f$r_{1}\f$ the starting
  * residual (i.e. \f$r_{1} = b - Ax_{init}\f$). In addition, the minimum number of
  * iterations \f$M\f$ can be specified. In this case, the solver will not stop to
  * iterate, before \f$k \geq M\f$.
  *
  * The \f$L_{p}\f$ norm is used for the computation, where \f$p\f$ could be 1, 2 and
  * \f$\infty\f$. The norm computation can be set with SetResidualNorm() with 1 for
  * \f$L_{1}\f$, 2 for \f$L_{2}\f$ and 3 for \f$L_{\infty}\f$. For the computation with
  * \f$L_{\infty}\f$, the index of the maximum value can be obtained with
  * GetAmaxResidualIndex(). If this function is called and \f$L_{\infty}\f$ was not
  * selected, this function will return -1.
  *
  * The reached criteria can be obtained with GetSolverStatus(), returning
  * - 0, if no criteria has been reached yet
  * - 1, if absolute tolerance has been reached
  * - 2, if relative tolerance has been reached
  * - 3, if divergence tolerance has been reached
  * - 4, if maximum number of iteration has been reached
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class IterativeLinearSolver : public Solver<OperatorType, VectorType, ValueType>
{
    public:
    IterativeLinearSolver();
    virtual ~IterativeLinearSolver();

    /** \brief Initialize the solver with absolute/relative/divergence tolerance and
      * maximum number of iterations
      */
    void Init(double abs_tol, double rel_tol, double div_tol, int max_iter);

    /** \brief Initialize the solver with absolute/relative/divergence tolerance and
      * minimum/maximum number of iterations
      */
    void Init(double abs_tol, double rel_tol, double div_tol, int min_iter, int max_iter);

    /** \brief Set the minimum number of iterations */
    void InitMinIter(int min_iter);

    /** \brief Set the maximum number of iterations */
    void InitMaxIter(int max_iter);

    /** \brief Set the absolute/relative/divergence tolerance */
    void InitTol(double abs, double rel, double div);

    /** \brief Set the residual norm to \f$L_1\f$, \f$L_2\f$ or \f$L_\infty\f$ norm
      * \details
      * - resnorm = 1 -> \f$L_1\f$ norm
      * - resnorm = 2 -> \f$L_2\f$ norm
      * - resnorm = 3 -> \f$L_\infty\f$ norm
      */
    void SetResidualNorm(int resnorm);

    /** \brief Record the residual history */
    void RecordResidualHistory(void);

    /** \brief Write the history to file */
    void RecordHistory(const std::string filename) const;

    /** \brief Set the solver verbosity output */
    virtual void Verbose(int verb = 1);

    /** \brief Solve Operator x = rhs */
    virtual void Solve(const VectorType& rhs, VectorType* x);

    /** \brief Set a preconditioner of the linear solver */
    virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType>& precond);

    /** \brief Return the iteration count */
    virtual int GetIterationCount(void);

    /** \brief Return the current residual */
    virtual double GetCurrentResidual(void);

    /** \brief Return the current status */
    virtual int GetSolverStatus(void);

    /** \brief Return absolute maximum index of residual vector when using
      * \f$L_\infty\f$ norm
      */
    virtual int GetAmaxResidualIndex(void);

    protected:
    // Iteration control (monitor)
    IterationControl iter_ctrl_; /**< \private */

    /** \brief Non-preconditioner solution procedure */
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x) = 0;

    /** \brief Preconditioned solution procedure */
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x) = 0;

    /** \brief Residual norm */
    int res_norm_;

    /** \brief Absolute maximum index of residual vector when using \f$L_\infty\f$ */
    int index_;

    /** \brief Computes the vector norm */
    ValueType Norm_(const VectorType& vec);
};

/** \ingroup solver_module
  * \class FixedPoint
  * \brief Fixed-Point Iteration Scheme
  * \details
  * The Fixed-Point iteration scheme is based on additive splitting of the matrix
  * \f$A = M + N\f$. The scheme reads
  * \f[
  *   x_{k+1} = M^{-1} (b - N x_{k}).
  * \f]
  * It can also be reformulated as a weighted defect correction scheme
  * \f[
  *   x_{k+1} = x_{k} - \omega M^{-1} (Ax_{k} - b).
  * \f]
  * The inversion of \f$M\f$ can be performed by preconditioners (Jacobi, Gauss-Seidel,
  * ILU, etc.) or by any type of solvers.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class FixedPoint : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    FixedPoint();
    virtual ~FixedPoint();

    virtual void Print(void) const;

    virtual void ReBuildNumeric(void);

    /** \brief Set relaxation parameter \f$\omega\f$ */
    void SetRelaxation(ValueType omega);

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    /** \brief Relaxation parameter */
    ValueType omega_;
    VectorType x_old_; /**< \private */
    VectorType x_res_; /**< \private */

    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);
};

/** \ingroup solver_module
  * \class DirectLinearSolver
  * \brief Base class for all direct linear solvers
  * \details
  * The library provides three direct methods - LU, QR and Inversion (based on QR
  * decomposition). The user can pass a sparse matrix, internally it will be converted to
  * dense and then the selected method will be applied. These methods are not very
  * optimal and due to the fact that the matrix is converted to a dense format, these
  * methods should be used only for very small matrices.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class DirectLinearSolver : public Solver<OperatorType, VectorType, ValueType>
{
    public:
    DirectLinearSolver();
    virtual ~DirectLinearSolver();

    virtual void Verbose(int verb = 1);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
    /** \brief Solve Operator x = rhs */
    virtual void Solve_(const VectorType& rhs, VectorType* x) = 0;
};

} // namespace rocalution

#endif // ROCALUTION_SOLVER_HPP_
