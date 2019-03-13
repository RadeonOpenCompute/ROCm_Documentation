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

#include "../../utils/def.hpp"
#include "preconditioner.hpp"
#include "../solver.hpp"
#include "../../base/global_matrix.hpp"
#include "../../base/local_matrix.hpp"

#include "../../base/global_vector.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <math.h>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
Preconditioner<OperatorType, VectorType, ValueType>::Preconditioner()
{
    log_debug(this, "Preconditioner::Preconditioner()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
Preconditioner<OperatorType, VectorType, ValueType>::~Preconditioner()
{
    log_debug(this, "Preconditioner::~Preconditioner()", "destructor");
}

template <class OperatorType, class VectorType, typename ValueType>
void Preconditioner<OperatorType, VectorType, ValueType>::PrintStart_(void) const
{
    // do nothing
}

template <class OperatorType, class VectorType, typename ValueType>
void Preconditioner<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
{
    // do nothing
}

template <class OperatorType, class VectorType, typename ValueType>
void Preconditioner<OperatorType, VectorType, ValueType>::SolveZeroSol(const VectorType& rhs,
                                                                       VectorType* x)
{
    log_debug(this, "Preconditioner::SolveZeroSol()", (const void*&)rhs, x);

    this->Solve(rhs, x);
}

template <class OperatorType, class VectorType, typename ValueType>
Jacobi<OperatorType, VectorType, ValueType>::Jacobi()
{
    log_debug(this, "Jacobi::Jacobi()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
Jacobi<OperatorType, VectorType, ValueType>::~Jacobi()
{
    log_debug(this, "Jacobi::~Jacobi()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("Jacobi preconditioner");
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "Jacobi::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    this->inv_diag_entries_.CloneBackend(*this->op_);
    this->op_->ExtractInverseDiagonal(&this->inv_diag_entries_);

    log_debug(this, "Jacobi::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::ResetOperator(const OperatorType& op)
{
    log_debug(this, "Jacobi::ResetOperator()", this->build_, (const void*&)op);

    assert(this->op_ != NULL);

    this->inv_diag_entries_.Clear();
    this->inv_diag_entries_.CloneBackend(*this->op_);
    this->op_->ExtractInverseDiagonal(&this->inv_diag_entries_);
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "Jacobi::Clear()", this->build_);

    this->inv_diag_entries_.Clear();
    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "Jacobi::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);

    if(x != &rhs)
    {
        x->PointWiseMult(this->inv_diag_entries_, rhs);
    }
    else
    {
        x->PointWiseMult(this->inv_diag_entries_);
    }

    log_debug(this, "Jacobi::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "Jacobi::MoveToHostLocalData_()", this->build_);

    this->inv_diag_entries_.MoveToHost();
}

template <class OperatorType, class VectorType, typename ValueType>
void Jacobi<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "Jacobi::MoveToAcceleratorLocalData_()", this->build_);

    this->inv_diag_entries_.MoveToAccelerator();
}

template <class OperatorType, class VectorType, typename ValueType>
GS<OperatorType, VectorType, ValueType>::GS()
{
    log_debug(this, "GS::GS()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
GS<OperatorType, VectorType, ValueType>::~GS()
{
    log_debug(this, "GS::~GS()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("Gauss-Seidel (GS) preconditioner");
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "GS::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    this->GS_.CloneFrom(*this->op_);
    this->GS_.LAnalyse(false);

    log_debug(this, "GS::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::ResetOperator(const OperatorType& op)
{
    log_debug(this, "GS::ResetOperator()", this->build_, (const void*&)op);

    assert(this->op_ != NULL);

    this->GS_.Clear();
    this->GS_.CloneFrom(*this->op_);
    this->GS_.LAnalyse(false);
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "GS::Clear()", this->build_);

    this->GS_.Clear();
    this->GS_.LAnalyseClear();

    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "GS::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);

    this->GS_.LSolve(rhs, x);

    log_debug(this, "GS::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "GS::MoveToHostLocalData_()", this->build_);

    this->GS_.MoveToHost();
    this->GS_.LAnalyse(false);
}

template <class OperatorType, class VectorType, typename ValueType>
void GS<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "GS::MoveToAcceleratorLocalData_()", this->build_);

    this->GS_.MoveToAccelerator();
    this->GS_.LAnalyse(false);
}

template <class OperatorType, class VectorType, typename ValueType>
SGS<OperatorType, VectorType, ValueType>::SGS()
{
    log_debug(this, "SGS::SGS()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
SGS<OperatorType, VectorType, ValueType>::~SGS()
{
    log_debug(this, "SGS::~SGS()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("Symmetric Gauss-Seidel (SGS) preconditioner");
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "SGS::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    this->SGS_.CloneFrom(*this->op_);
    this->SGS_.LAnalyse(false);
    this->SGS_.UAnalyse(false);

    this->diag_entries_.CloneBackend(*this->op_);
    this->SGS_.ExtractInverseDiagonal(&this->diag_entries_);

    this->v_.CloneBackend(*this->op_);
    this->v_.Allocate("v", this->op_->GetM());

    log_debug(this, "SGS::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::ResetOperator(const OperatorType& op)
{
    log_debug(this, "SGS::ResetOperator()", this->build_, (const void*&)op);

    assert(this->op_ != NULL);

    this->SGS_.Clear();
    this->SGS_.CloneFrom(*this->op_);

    this->diag_entries_.Clear();
    this->diag_entries_.CloneBackend(*this->op_);
    this->SGS_.ExtractDiagonal(&this->diag_entries_);

    this->SGS_.LAnalyse(false);
    this->SGS_.UAnalyse(false);

    this->v_.Clear();
    this->v_.CloneBackend(*this->op_);
    this->v_.Allocate("v", this->op_->GetM());
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "SGS::Clear()", this->build_);

    this->SGS_.Clear();
    this->SGS_.LAnalyseClear();
    this->SGS_.UAnalyseClear();

    this->diag_entries_.Clear();
    this->v_.Clear();

    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "SGS::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);

    this->SGS_.LSolve(rhs, &this->v_);
    this->v_.PointWiseMult(this->diag_entries_);
    this->SGS_.USolve(this->v_, x);

    log_debug(this, "SGS::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "SGS::MoveToHostLocalData_()", this->build_);

    this->SGS_.MoveToHost();
    this->SGS_.LAnalyse(false);
    this->SGS_.UAnalyse(false);

    this->diag_entries_.MoveToHost();
    this->v_.MoveToHost();
}

template <class OperatorType, class VectorType, typename ValueType>
void SGS<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "SGS::MoveToAcceleratorLocalData_()", this->build_);

    this->SGS_.MoveToAccelerator();
    this->SGS_.LAnalyse(false);
    this->SGS_.UAnalyse(false);

    this->diag_entries_.MoveToAccelerator();
    this->v_.MoveToAccelerator();
}

template <class OperatorType, class VectorType, typename ValueType>
ILU<OperatorType, VectorType, ValueType>::ILU()
{
    log_debug(this, "ILU::ILU()", "default constructor");

    this->p_     = 0;
    this->level_ = true;
}

template <class OperatorType, class VectorType, typename ValueType>
ILU<OperatorType, VectorType, ValueType>::~ILU()
{
    log_debug(this, "ILU::ILU()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("ILU(" << this->p_ << ") preconditioner");

    if(this->build_ == true)
    {
        LOG_INFO("ILU nnz = " << this->ILU_.GetNnz());
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::Set(int p, bool level)
{
    log_debug(this, "ILU::Set()", p, level);

    assert(p >= 0);
    assert(this->build_ == false);

    this->p_     = p;
    this->level_ = level;
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "ILU::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    this->ILU_.CloneFrom(*this->op_);

    this->ILU_.ILUpFactorize(this->p_, this->level_);

    this->ILU_.LUAnalyse();

    log_debug(this, "ILU::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "ILU::Clear()", this->build_);

    this->ILU_.Clear();
    this->ILU_.LUAnalyseClear();
    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "ILU::MoveToHostLocalData_()", this->build_);

    this->ILU_.MoveToHost();
    this->ILU_.LUAnalyse();
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "ILU::MoveToAcceleratorLocalData_()", this->build_);

    this->ILU_.MoveToAccelerator();
    this->ILU_.LUAnalyse();
}

template <class OperatorType, class VectorType, typename ValueType>
void ILU<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "ILU::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);
    assert(x != &rhs);

    this->ILU_.LUSolve(rhs, x);

    log_debug(this, "ILU::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
ILUT<OperatorType, VectorType, ValueType>::ILUT()
{
    log_debug(this, "ILUT::ILUT()", "default constructor");

    this->t_       = 0.05;
    this->max_row_ = 100;
}

template <class OperatorType, class VectorType, typename ValueType>
ILUT<OperatorType, VectorType, ValueType>::~ILUT()
{
    log_debug(this, "ILUT::~ILUT()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("ILUT(" << this->t_ << "," << this->max_row_ << ") preconditioner");

    if(this->build_ == true)
    {
        LOG_INFO("ILUT nnz = " << this->ILUT_.GetNnz());
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::Set(double t)
{
    log_debug(this, "ILUT::Set()", t);

    assert(t >= 0);
    assert(this->build_ == false);

    this->t_ = t;
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::Set(double t, int maxrow)
{
    log_debug(this, "ILUT::Set()", t, maxrow);

    assert(t >= 0);
    assert(this->build_ == false);

    this->t_       = t;
    this->max_row_ = maxrow;
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "ILUT::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    this->ILUT_.CloneFrom(*this->op_);
    this->ILUT_.ILUTFactorize(this->t_, this->max_row_);
    this->ILUT_.LUAnalyse();

    log_debug(this, "ILUT::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "ILUT::Clear()", this->build_);

    this->ILUT_.Clear();
    this->ILUT_.LUAnalyseClear();
    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "ILUT::MoveToHostLocalData_()", this->build_);

    this->ILUT_.MoveToHost();
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "ILUT::MoveToAcceleratorLocalData_()", this->build_);

    this->ILUT_.MoveToAccelerator();
}

template <class OperatorType, class VectorType, typename ValueType>
void ILUT<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "ILUT::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);
    assert(x != &rhs);

    this->ILUT_.LUSolve(rhs, x);

    log_debug(this, "ILUT::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
IC<OperatorType, VectorType, ValueType>::IC()
{
    log_debug(this, "IC::IC()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
IC<OperatorType, VectorType, ValueType>::~IC()
{
    log_debug(this, "IC::IC()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void IC<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("IC preconditioner");

    if(this->build_ == true)
    {
        LOG_INFO("IC nnz = " << this->IC_.GetNnz());
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void IC<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "IC::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    this->IC_.CloneBackend(*this->op_);
    this->inv_diag_entries_.CloneBackend(*this->op_);

    this->op_->ExtractL(&this->IC_, true);
    this->IC_.ICFactorize(&this->inv_diag_entries_);
    this->IC_.LLAnalyse();

    log_debug(this, "IC::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void IC<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "IC::Clear()", this->build_);

    this->inv_diag_entries_.Clear();
    this->IC_.Clear();
    this->IC_.LLAnalyseClear();
    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void IC<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "IC::MoveToHostLocalData_()", this->build_);

    // this->inv_diag_entries_ is NOT needed on accelerator!
    this->IC_.MoveToHost();
}

template <class OperatorType, class VectorType, typename ValueType>
void IC<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "IC::MoveToAcceleratorLocalData_()", this->build_);

    // this->inv_diag_entries_ is NOT needed on accelerator!
    this->IC_.MoveToAccelerator();
}

template <class OperatorType, class VectorType, typename ValueType>
void IC<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "IC::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);
    assert(x != &rhs);

    this->IC_.LLSolve(rhs, this->inv_diag_entries_, x);

    log_debug(this, "IC::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
VariablePreconditioner<OperatorType, VectorType, ValueType>::VariablePreconditioner()
{
    log_debug(this, "VariablePreconditioner::VariablePreconditioner()", "default constructor");

    this->num_precond_ = 0;
    this->precond_     = NULL;
    this->counter_     = 0;
}

template <class OperatorType, class VectorType, typename ValueType>
VariablePreconditioner<OperatorType, VectorType, ValueType>::~VariablePreconditioner()
{
    log_debug(this, "VariablePreconditioner::~VariablePreconditioner()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::Print(void) const
{
    if(this->build_ == true)
    {
        LOG_INFO("VariablePreconditioner with " << this->num_precond_ << " preconditioners:");
        for(int i = 0; i < this->num_precond_; ++i)
        {
            this->precond_[i]->Print();
        }
    }
    else
    {
        LOG_INFO("VariablePreconditioner preconditioner");
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::SetPreconditioner(
    int n, Solver<OperatorType, VectorType, ValueType>** precond)
{
    assert(this->precond_ == NULL);
    assert(n > 0);

    this->precond_ = new Solver<OperatorType, VectorType, ValueType>*[n];

    for(int i = 0; i < n; ++i)
    {
        assert(precond[i] != NULL);
        this->precond_[i] = precond[i];
    }
    this->num_precond_ = n;
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "VariablePreconditioner::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    assert(this->op_ != NULL);

    assert(this->precond_ != NULL);
    assert(this->num_precond_ > 0);

    for(int i = 0; i < this->num_precond_; ++i)
    {
        assert(this->precond_[i] != NULL);
        this->precond_[i]->SetOperator(*this->op_);
        this->precond_[i]->Build();
    }

    log_debug(this, "VariablePreconditioner::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "VariablePreconditioner::Clear()", this->build_);

    if(this->precond_ != NULL)
    {
        for(int i = 0; i < this->num_precond_; ++i)
        {
            this->precond_[i]->Clear();
        }

        delete[] this->precond_;
        this->precond_ = NULL;
    }

    this->num_precond_ = 0;
    this->counter_     = 0;

    this->build_ = false;
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs,
                                                                        VectorType* x)
{
    log_debug(this, "VariablePreconditioner::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->build_ == true);
    assert(x != NULL);

    this->precond_[this->counter_]->Solve(rhs, x);
    ;
    ++this->counter_;

    if(this->counter_ >= this->num_precond_)
    {
        this->counter_ = 0;
    }

    log_debug(this, "VariablePreconditioner::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "VariablePreconditioner::MoveToHostLocalData_()", this->build_);

    if(this->build_ == true)
    {
        assert(this->precond_ != NULL);
        assert(this->num_precond_ > 0);

        for(int i = 0; i < this->num_precond_; ++i)
        {
            this->precond_[i]->MoveToHost();
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void VariablePreconditioner<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "VariablePreconditioner::MoveToAcceleratorLocalData_()", this->build_);

    if(this->build_ == true)
    {
        assert(this->precond_ != NULL);
        assert(this->num_precond_ > 0);

        for(int i = 0; i < this->num_precond_; ++i)
        {
            this->precond_[i]->MoveToAccelerator();
        }
    }
}

template class Preconditioner<LocalMatrix<double>, LocalVector<double>, double>;
template class Preconditioner<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class Preconditioner<LocalMatrix<std::complex<double>>,
                              LocalVector<std::complex<double>>,
                              std::complex<double>>;
template class Preconditioner<LocalMatrix<std::complex<float>>,
                              LocalVector<std::complex<float>>,
                              std::complex<float>>;
#endif

template class Preconditioner<GlobalMatrix<double>, GlobalVector<double>, double>;
template class Preconditioner<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class Preconditioner<GlobalMatrix<std::complex<double>>,
                              GlobalVector<std::complex<double>>,
                              std::complex<double>>;
template class Preconditioner<GlobalMatrix<std::complex<float>>,
                              GlobalVector<std::complex<float>>,
                              std::complex<float>>;
#endif

template class Jacobi<LocalMatrix<double>, LocalVector<double>, double>;
template class Jacobi<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class Jacobi<LocalMatrix<std::complex<double>>,
                      LocalVector<std::complex<double>>,
                      std::complex<double>>;
template class Jacobi<LocalMatrix<std::complex<float>>,
                      LocalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

template class Jacobi<GlobalMatrix<double>, GlobalVector<double>, double>;
template class Jacobi<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class Jacobi<GlobalMatrix<std::complex<double>>,
                      GlobalVector<std::complex<double>>,
                      std::complex<double>>;
template class Jacobi<GlobalMatrix<std::complex<float>>,
                      GlobalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

template class GS<LocalMatrix<double>, LocalVector<double>, double>;
template class GS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class GS<LocalMatrix<std::complex<double>>,
                  LocalVector<std::complex<double>>,
                  std::complex<double>>;
template class GS<LocalMatrix<std::complex<float>>,
                  LocalVector<std::complex<float>>,
                  std::complex<float>>;
#endif

template class SGS<LocalMatrix<double>, LocalVector<double>, double>;
template class SGS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class SGS<LocalMatrix<std::complex<double>>,
                   LocalVector<std::complex<double>>,
                   std::complex<double>>;
template class SGS<LocalMatrix<std::complex<float>>,
                   LocalVector<std::complex<float>>,
                   std::complex<float>>;
#endif

template class ILU<LocalMatrix<double>, LocalVector<double>, double>;
template class ILU<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class ILU<LocalMatrix<std::complex<double>>,
                   LocalVector<std::complex<double>>,
                   std::complex<double>>;
template class ILU<LocalMatrix<std::complex<float>>,
                   LocalVector<std::complex<float>>,
                   std::complex<float>>;
#endif

template class ILUT<LocalMatrix<double>, LocalVector<double>, double>;
template class ILUT<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class ILUT<LocalMatrix<std::complex<double>>,
                    LocalVector<std::complex<double>>,
                    std::complex<double>>;
template class ILUT<LocalMatrix<std::complex<float>>,
                    LocalVector<std::complex<float>>,
                    std::complex<float>>;
#endif

template class IC<LocalMatrix<double>, LocalVector<double>, double>;
template class IC<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class IC<LocalMatrix<std::complex<double>>,
                  LocalVector<std::complex<double>>,
                  std::complex<double>>;
template class IC<LocalMatrix<std::complex<float>>,
                  LocalVector<std::complex<float>>,
                  std::complex<float>>;
#endif

template class VariablePreconditioner<LocalMatrix<double>, LocalVector<double>, double>;
template class VariablePreconditioner<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class VariablePreconditioner<LocalMatrix<std::complex<double>>,
                                      LocalVector<std::complex<double>>,
                                      std::complex<double>>;
template class VariablePreconditioner<LocalMatrix<std::complex<float>>,
                                      LocalVector<std::complex<float>>,
                                      std::complex<float>>;
#endif

} // namespace rocalution
