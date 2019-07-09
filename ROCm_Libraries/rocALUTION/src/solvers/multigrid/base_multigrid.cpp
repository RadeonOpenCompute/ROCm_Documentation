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
#include "base_multigrid.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include <math.h>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
BaseMultiGrid<OperatorType, VectorType, ValueType>::BaseMultiGrid()
{
    log_debug(this, "BaseMultiGrid::BaseMultiGrid()", "default constructor");

    this->levels_        = -1;
    this->current_level_ = 0;

    this->iter_pre_smooth_  = 1;
    this->iter_post_smooth_ = 2;

    this->scaling_ = true;

    this->op_level_ = NULL;

    this->restrict_op_level_ = NULL;
    this->prolong_op_level_  = NULL;

    this->d_level_ = NULL;
    this->r_level_ = NULL;
    this->t_level_ = NULL;
    this->s_level_ = NULL;
    this->p_level_ = NULL;
    this->q_level_ = NULL;
    this->k_level_ = NULL;
    this->l_level_ = NULL;

    this->solver_coarse_  = NULL;
    this->smoother_level_ = NULL;

    this->cycle_      = Vcycle;
    this->host_level_ = 0;

    this->kcycle_full_ = true;
}

template <class OperatorType, class VectorType, typename ValueType>
BaseMultiGrid<OperatorType, VectorType, ValueType>::~BaseMultiGrid()
{
    log_debug(this, "BaseMultiGrid::~BaseMultiGrid()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::InitLevels(int levels)
{
    log_debug(this, "BaseMultiGrid::InitLevels()", levels);

    assert(this->build_ == false);
    assert(levels > 0);

    this->levels_ = levels;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetPreconditioner(
    Solver<OperatorType, VectorType, ValueType>& precond)
{
    LOG_INFO("BaseMultiGrid::SetPreconditioner() Perhaps you want to set the smoothers on all "
             "levels? use SetSmootherLevel() instead of SetPreconditioner!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSmoother(
    IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother)
{
    log_debug(this, "BaseMultiGrid::SetSmoother()", smoother);

    //  assert(this->build_ == false); not possible due to AMG
    assert(smoother != NULL);

    this->smoother_level_ = smoother;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSmootherPreIter(int iter)
{
    log_debug(this, "BaseMultiGrid::SetSmootherPreIter()", iter);

    this->iter_pre_smooth_ = iter;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSmootherPostIter(int iter)
{
    log_debug(this, "BaseMultiGrid::SetSmootherPostIter()", iter);

    this->iter_post_smooth_ = iter;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSolver(
    Solver<OperatorType, VectorType, ValueType>& solver)
{
    log_debug(this, "BaseMultiGrid::SetSolver()", (const void*&)solver);

    //  assert(this->build_ == false); not possible due to AMG

    this->solver_coarse_ = &solver;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetScaling(bool scaling)
{
    log_debug(this, "BaseMultiGrid::SetScaling()", scaling);

    this->scaling_ = scaling;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetHostLevels(int levels)
{
    log_debug(this, "BaseMultiGrid::SetHostLevels()", levels);

    assert(this->build_ == true);
    assert(levels > 0);
    assert(levels < this->levels_);

    this->host_level_ = levels;
    this->MoveHostLevels_();
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetCycle(unsigned int cycle)
{
    log_debug(this, "BaseMultiGrid::SetCycle()", cycle);

    this->cycle_ = cycle;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetKcycleFull(bool kcycle_full)
{
    log_debug(this, "BaseMultiGrid::SetKcycleFull()", kcycle_full);

    this->kcycle_full_ = kcycle_full;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("MultiGrid solver");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::PrintStart_(void) const
{
    assert(this->levels_ > 0);

    LOG_INFO("MultiGrid solver starts");
    LOG_INFO("MultiGrid Number of levels " << this->levels_);
    LOG_INFO("MultiGrid with smoother:");
    this->smoother_level_[0]->Print();
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
{
    LOG_INFO("MultiGrid ends");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "BaseMultiGrid::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);
    this->build_ = true;

    for(int i = 0; i < this->levels_ - 1; ++i)
    {
        assert(this->op_level_[i] != NULL);
        assert(this->smoother_level_[i] != NULL);
        assert(this->restrict_op_level_[i] != NULL);
        assert(this->prolong_op_level_[i] != NULL);
    }
    assert(this->op_ != NULL);
    assert(this->solver_coarse_ != NULL);
    assert(this->levels_ > 0);

    log_debug(this, "BaseMultiGrid::Build()", "#*# setup finest level 0");

    // Setup finest level 0
    this->smoother_level_[0]->SetOperator(*this->op_);
    this->smoother_level_[0]->Build();

    log_debug(this, "BaseMultiGrid::Build()", "#*# setup coarser levels");

    // Setup coarser levels
    for(int i = 1; i < this->levels_ - 1; ++i)
    {
        this->smoother_level_[i]->SetOperator(*this->op_level_[i - 1]);
        this->smoother_level_[i]->Build();
    }

    log_debug(this, "BaseMultiGrid::Build()", "#*# setup coarse grid solver");
    // Setup coarse grid solver
    this->solver_coarse_->SetOperator(*op_level_[this->levels_ - 2]);
    this->solver_coarse_->Build();

    log_debug(this, "BaseMultiGrid::Build()", "#*# setup all tmp vectors");

    // Setup all temporary vectors for the cycles - needed on all levels
    this->d_level_ = new VectorType*[this->levels_];
    this->r_level_ = new VectorType*[this->levels_];
    this->t_level_ = new VectorType*[this->levels_];
    this->s_level_ = new VectorType*[this->levels_];

    // Extra structure for K-cycle
    if(this->cycle_ == Kcycle)
    {
        this->p_level_ = new VectorType*[this->levels_ - 2];
        this->q_level_ = new VectorType*[this->levels_ - 2];
        this->k_level_ = new VectorType*[this->levels_ - 2];
        this->l_level_ = new VectorType*[this->levels_ - 2];

        for(int i = 0; i < this->levels_ - 2; ++i)
        {
            this->p_level_[i] = new VectorType;
            this->p_level_[i]->CloneBackend(*this->op_level_[i]);
            this->p_level_[i]->Allocate("p", this->op_level_[i]->GetM());

            this->q_level_[i] = new VectorType;
            this->q_level_[i]->CloneBackend(*this->op_level_[i]);
            this->q_level_[i]->Allocate("q", this->op_level_[i]->GetM());

            this->k_level_[i] = new VectorType;
            this->k_level_[i]->CloneBackend(*this->op_level_[i]);
            this->k_level_[i]->Allocate("k", this->op_level_[i]->GetM());

            this->l_level_[i] = new VectorType;
            this->l_level_[i]->CloneBackend(*this->op_level_[i]);
            this->l_level_[i]->Allocate("l", this->op_level_[i]->GetM());
        }
    }

    for(int i = 1; i < this->levels_; ++i)
    {
        // On finest level, we need to get the size from this->op_ instead
        this->d_level_[i] = new VectorType;
        this->d_level_[i]->CloneBackend(*this->op_level_[i - 1]);
        this->d_level_[i]->Allocate("defect correction", this->op_level_[i - 1]->GetM());

        this->r_level_[i] = new VectorType;
        this->r_level_[i]->CloneBackend(*this->op_level_[i - 1]);
        this->r_level_[i]->Allocate("residual", this->op_level_[i - 1]->GetM());

        this->t_level_[i] = new VectorType;
        this->t_level_[i]->CloneBackend(*this->op_level_[i - 1]);
        this->t_level_[i]->Allocate("temporary", this->op_level_[i - 1]->GetM());

        this->s_level_[i] = new VectorType;
        this->s_level_[i]->CloneBackend(*this->op_level_[i - 1]);
        this->s_level_[i]->Allocate("temporary", this->op_level_[i - 1]->GetM());
    }

    this->r_level_[0] = new VectorType;
    this->r_level_[0]->CloneBackend(*this->op_);
    this->r_level_[0]->Allocate("residual", this->op_->GetM());

    this->t_level_[0] = new VectorType;
    this->t_level_[0]->CloneBackend(*this->op_);
    this->t_level_[0]->Allocate("temporary", this->op_->GetM());

    this->s_level_[0] = new VectorType;
    this->s_level_[0]->CloneBackend(*this->op_);
    this->s_level_[0]->Allocate("temporary", this->op_->GetM());

    log_debug(this, "BaseMultiGrid::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "BaseMultiGrid::Clear()", this->build_);

    if(this->build_ == true)
    {
        for(int i = 0; i < this->levels_; ++i)
        {
            // Clear temporary VectorTypes
            if(i > 0)
            {
                delete this->d_level_[i];
            }
            delete this->r_level_[i];
            delete this->t_level_[i];
            delete this->s_level_[i];
        }

        delete[] this->d_level_;
        delete[] this->r_level_;
        delete[] this->t_level_;
        delete[] this->s_level_;

        // Extra structure for K-cycle
        if(this->cycle_ == Kcycle)
        {
            for(int i = 0; i < this->levels_ - 2; ++i)
            {
                delete this->p_level_[i];
                delete this->q_level_[i];
                delete this->k_level_[i];
                delete this->l_level_[i];
            }

            delete[] this->p_level_;
            delete[] this->q_level_;
            delete[] this->k_level_;
            delete[] this->l_level_;
        }

        // Clear smoothers - we built it
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->smoother_level_[i]->Clear();
        }

        // Clear coarse grid solver - we built it
        this->solver_coarse_->Clear();

        this->levels_ = -1;

        this->iter_ctrl_.Clear();

        this->build_ = false;
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    log_debug(this, "BaseMultiGrid::MoveToHostLocalData_()", this->build_);

    if(this->build_ == true)
    {
        this->r_level_[this->levels_ - 1]->MoveToHost();
        this->d_level_[this->levels_ - 1]->MoveToHost();
        this->t_level_[this->levels_ - 1]->MoveToHost();
        this->s_level_[this->levels_ - 1]->MoveToHost();
        this->solver_coarse_->MoveToHost();

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i]->MoveToHost();
            this->smoother_level_[i]->MoveToHost();
            this->r_level_[i]->MoveToHost();
            if(i > 0)
            {
                this->d_level_[i]->MoveToHost();
            }
            this->t_level_[i]->MoveToHost();
            this->s_level_[i]->MoveToHost();

            this->restrict_op_level_[i]->MoveToHost();
            this->prolong_op_level_[i]->MoveToHost();
        }

        // Extra structure for K-cycle
        if(this->cycle_ == Kcycle)
        {
            for(int i = 0; i < this->levels_ - 2; ++i)
            {
                this->p_level_[i]->MoveToHost();
                this->q_level_[i]->MoveToHost();
                this->k_level_[i]->MoveToHost();
                this->l_level_[i]->MoveToHost();
            }
        }

        if(this->precond_ != NULL)
        {
            this->precond_->MoveToHost();
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    log_debug(this, "BaseMultiGrid::MoveToAcceleratorLocalData_()", this->build_);

    if(this->build_ == true)
    {
        // If coarsest level on accelerator
        if(this->host_level_ == 0)
        {
            this->solver_coarse_->MoveToAccelerator();
        }

        // Move operators
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            if(i < this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i]->MoveToAccelerator();
                this->restrict_op_level_[i]->MoveToAccelerator();
                this->prolong_op_level_[i]->MoveToAccelerator();
            }
        }

        // Move smoothers
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            if(i < this->levels_ - this->host_level_)
            {
                this->smoother_level_[i]->MoveToAccelerator();
            }
        }

        // Move temporary vectors
        for(int i = 0; i < this->levels_; ++i)
        {
            if(i < this->levels_ - this->host_level_)
            {
                this->r_level_[i]->MoveToAccelerator();
                if(i > 0)
                {
                    this->d_level_[i]->MoveToAccelerator();
                }
                this->t_level_[i]->MoveToAccelerator();
                this->s_level_[i]->MoveToAccelerator();
            }
        }

        // Extra structure for K-cycle
        if(this->cycle_ == Kcycle)
        {
            for(int i = 0; i < this->levels_ - 2; ++i)
            {
                if(i < this->levels_ - this->host_level_ - 1)
                {
                    this->p_level_[i]->MoveToAccelerator();
                    this->q_level_[i]->MoveToAccelerator();
                    this->k_level_[i]->MoveToAccelerator();
                    this->l_level_[i]->MoveToAccelerator();
                }
            }
        }

        if(this->precond_ != NULL)
        {
            this->precond_->MoveToAccelerator();
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::MoveHostLevels_(void)
{
    log_debug(this, "BaseMultiGrid::MoveHostLevels_()", this->build_);

    // If coarsest level on accelerator
    if(this->host_level_ != 0)
    {
        this->solver_coarse_->MoveToHost();
    }

    // Move operators
    for(int i = 0; i < this->levels_ - 1; ++i)
    {
        if(i >= this->levels_ - this->host_level_ - 1)
        {
            this->op_level_[i]->MoveToHost();
            this->restrict_op_level_[i]->MoveToHost();
            this->prolong_op_level_[i]->MoveToHost();
        }
    }

    // Move smoothers
    for(int i = 0; i < this->levels_ - 1; ++i)
    {
        if(i >= this->levels_ - this->host_level_)
        {
            this->smoother_level_[i]->MoveToHost();
        }
    }

    // Move temporary vectors
    for(int i = 0; i < this->levels_; ++i)
    {
        if(i >= this->levels_ - this->host_level_)
        {
            this->r_level_[i]->MoveToHost();
            if(i > 0)
            {
                this->d_level_[i]->MoveToHost();
            }
            this->t_level_[i]->MoveToHost();
            this->s_level_[i]->MoveToHost();
        }
    }

    // Extra structure for K-cycle
    if(this->cycle_ == Kcycle)
    {
        for(int i = 0; i < this->levels_ - 2; ++i)
        {
            if(i >= this->levels_ - this->host_level_ - 1)
            {
                this->p_level_[i]->MoveToHost();
                this->q_level_[i]->MoveToHost();
                this->k_level_[i]->MoveToHost();
                this->l_level_[i]->MoveToHost();
            }
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
{
    log_debug(this, "BaseMultiGrid::Solve()", " #*# begin", (const void*&)rhs, x);

    assert(this->levels_ > 1);
    assert(x != NULL);
    assert(x != &rhs);
    assert(this->op_ != NULL);
    assert(this->build_ == true);
    assert(this->precond_ == NULL);
    assert(this->solver_coarse_ != NULL);

    for(int i = 0; i < this->levels_; ++i)
    {
        if(i > 0)
        {
            assert(this->d_level_[i] != NULL);
        }
        assert(this->r_level_[i] != NULL);
        assert(this->t_level_[i] != NULL);
        assert(this->s_level_[i] != NULL);
    }

    if(this->cycle_ == Kcycle)
    {
        for(int i = 0; i < this->levels_ - 2; ++i)
        {
            assert(this->k_level_[i] != NULL);
            assert(this->l_level_[i] != NULL);
            assert(this->p_level_[i] != NULL);
            assert(this->q_level_[i] != NULL);
        }
    }

    for(int i = 0; i < this->levels_ - 1; ++i)
    {
        if(i > 0)
        {
            assert(this->op_level_[i] != NULL);
        }
        assert(this->smoother_level_[i] != NULL);

        assert(this->restrict_op_level_[i] != NULL);
        assert(this->prolong_op_level_[i] != NULL);
    }

    if(this->verb_ > 0)
    {
        this->PrintStart_();
        this->iter_ctrl_.PrintInit();
    }

    // initial residual = b - Ax
    this->op_->Apply(*x, this->r_level_[0]);
    this->r_level_[0]->ScaleAdd(static_cast<ValueType>(-1), rhs);

    this->res_norm_ = rocalution_abs(this->Norm_(*this->r_level_[0]));

    if(this->iter_ctrl_.InitResidual(this->res_norm_) == false)
    {
        log_debug(this, "BaseMultiGrid::Solve()", " #*# end");

        return;
    }

    this->Vcycle_(rhs, x);

    while(!this->iter_ctrl_.CheckResidual(this->res_norm_, this->index_))
    {
        this->Vcycle_(rhs, x);
    }

    if(this->verb_ > 0)
    {
        this->iter_ctrl_.PrintStatus();
        this->PrintEnd_();
    }

    log_debug(this, "BaseMultiGrid::Solve()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Restrict_(const VectorType& fine,
                                                                   VectorType* coarse,
                                                                   int level)
{
    log_debug(this, "BaseMultiGrid::Restrict_()", (const void*&)fine, coarse, level);

    this->restrict_op_level_[level]->Apply(fine.GetInterior(), &(coarse->GetInterior()));
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Prolong_(const VectorType& coarse,
                                                                  VectorType* fine,
                                                                  int level)
{
    log_debug(this, "BaseMultiGrid::Prolong_()", (const void*&)coarse, fine, level);

    this->prolong_op_level_[level]->Apply(coarse.GetInterior(), &(fine->GetInterior()));
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Vcycle_(const VectorType& rhs,
                                                                 VectorType* x)
{
    log_debug(this, "BaseMultiGrid::Vcycle_()", " #*# begin", (const void*&)rhs, x);

    // Perform cycle
    if(this->current_level_ < this->levels_ - 1)
    {
        ValueType factor;
        ValueType divisor;

        // Pre-smoothing on finest level
        this->smoother_level_[this->current_level_]->InitMaxIter(this->iter_pre_smooth_);
        this->smoother_level_[this->current_level_]->Solve(rhs, x);

        if(this->scaling_ == true)
        {
            if(this->current_level_ > 0 && this->current_level_ < this->levels_ - 2 &&
               this->iter_pre_smooth_ > 0)
            {
                this->r_level_[this->current_level_]->PointWiseMult(rhs, *x);
                factor = this->r_level_[this->current_level_]->Reduce();
                this->op_level_[this->current_level_ - 1]->Apply(
                    *x, this->r_level_[this->current_level_]);
                this->r_level_[this->current_level_]->PointWiseMult(*x);

                divisor = this->r_level_[this->current_level_]->Reduce();

                if(divisor == static_cast<ValueType>(0))
                {
                    factor = static_cast<ValueType>(1);
                }
                else
                {
                    factor /= divisor;
                }

                x->Scale(factor);
            }
        }

        // Update residual
        if(this->current_level_ == 0)
        {
            this->op_->Apply(*x, this->s_level_[this->current_level_]);
        }
        else
        {
            this->op_level_[this->current_level_ - 1]->Apply(*x,
                                                             this->s_level_[this->current_level_]);
        }

        this->s_level_[this->current_level_]->ScaleAdd(static_cast<ValueType>(-1), rhs);

        if(this->current_level_ == this->levels_ - this->host_level_ - 1)
        {
            this->s_level_[this->current_level_]->MoveToHost();
        }

        // Restrict residual vector on finest
        // level
        this->Restrict_(*this->s_level_[this->current_level_],
                        this->t_level_[this->current_level_ + 1],
                        this->current_level_);

        if(this->current_level_ == this->levels_ - this->host_level_ - 1)
        {
            if(this->current_level_ == 0)
            {
                this->s_level_[this->current_level_]->CloneBackend(*this->op_);
            }
            else
            {
                this->s_level_[this->current_level_]->CloneBackend(
                    *this->op_level_[this->current_level_ - 1]);
            }
        }

        ++this->current_level_;

        // Set new solution for recursion to
        // zero
        this->d_level_[this->current_level_]->Zeros();

        // Recursive call dependent on the
        // cycle
        switch(this->cycle_)
        {
        // V-cycle
        case 0:
            this->Vcycle_(*this->t_level_[this->current_level_], d_level_[this->current_level_]);
            break;

        // W-cycle
        case 1:
            this->Wcycle_(*this->t_level_[this->current_level_], d_level_[this->current_level_]);
            break;

        // K-cycle
        case 2:
            this->Kcycle_(*this->t_level_[this->current_level_], d_level_[this->current_level_]);
            break;

        // F-cycle
        case 3:
            this->Fcycle_(*this->t_level_[this->current_level_], d_level_[this->current_level_]);
            break;

        default: FATAL_ERROR(__FILE__, __LINE__); break;
        }

        if(this->current_level_ == this->levels_ - this->host_level_)
        {
            this->r_level_[this->current_level_ - 1]->MoveToHost();
        }

        // Prolong solution vector on finest
        // level
        this->Prolong_(*this->d_level_[this->current_level_],
                       this->r_level_[this->current_level_ - 1],
                       this->current_level_ - 1);

        if(this->current_level_ == this->levels_ - this->host_level_)
        {
            if(this->current_level_ == 1)
            {
                this->r_level_[this->current_level_ - 1]->CloneBackend(*this->op_);
            }
            else
            {
                this->r_level_[this->current_level_ - 1]->CloneBackend(
                    *this->op_level_[this->current_level_ - 2]);
            }
        }

        --this->current_level_;

        // Scaling
        if(this->scaling_ == true && this->current_level_ < this->levels_ - 2)
        {
            if(this->current_level_ == 0)
            {
                this->s_level_[this->current_level_]->PointWiseMult(
                    *this->r_level_[this->current_level_]);
            }
            else
            {
                this->s_level_[this->current_level_]->PointWiseMult(
                    *this->r_level_[this->current_level_], *this->t_level_[this->current_level_]);
            }

            factor = this->s_level_[this->current_level_]->Reduce();

            if(this->current_level_ == 0)
            {
                this->op_->Apply(*this->r_level_[this->current_level_],
                                 this->s_level_[this->current_level_]);
            }
            else
            {
                this->op_level_[this->current_level_ - 1]->Apply(
                    *this->r_level_[this->current_level_], this->s_level_[this->current_level_]);
            }

            this->s_level_[this->current_level_]->PointWiseMult(
                *this->r_level_[this->current_level_]);

            // Check for division by zero
            divisor = this->s_level_[this->current_level_]->Reduce();
            if(divisor == static_cast<ValueType>(0))
            {
                factor = static_cast<ValueType>(1);
            }
            else
            {
                factor /= divisor;
            }

            // Defect correction
            x->AddScale(*this->r_level_[this->current_level_], factor);
        }
        else
            // Defect correction
            x->AddScale(*this->r_level_[this->current_level_], static_cast<ValueType>(1));

        // Post-smoothing on finest level
        this->smoother_level_[this->current_level_]->InitMaxIter(this->iter_post_smooth_);
        this->smoother_level_[this->current_level_]->Solve(rhs, x);

        if(this->current_level_ == 0)
        {
            // Update residual
            this->op_->Apply(*x, this->r_level_[this->current_level_]);
            this->r_level_[this->current_level_]->ScaleAdd(static_cast<ValueType>(-1), rhs);

            this->res_norm_ = rocalution_abs(this->Norm_(*this->r_level_[this->current_level_]));
        }
    }
    else
        // Coarse grid solver
        this->solver_coarse_->SolveZeroSol(rhs, x);

    log_debug(this, "BaseMultiGrid::Vcycle_()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Wcycle_(const VectorType& rhs,
                                                                 VectorType* x)
{
    // gamma = 2 hardcoded
    for(int i = 0; i < 2; ++i)
    {
        this->Vcycle_(rhs, x);
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Fcycle_(const VectorType& rhs,
                                                                 VectorType* x)
{
    LOG_INFO("BaseMultiGrid:Fcycle_() not implemented yet");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::Kcycle_(const VectorType& rhs,
                                                                 VectorType* x)
{
    if(this->current_level_ != 1 && this->kcycle_full_ == false)
    {
        this->Vcycle_(rhs, x);
    }
    else if(this->current_level_ < this->levels_ - 1)
    {
        VectorType* r = this->k_level_[this->current_level_ - 1];
        VectorType* s = this->l_level_[this->current_level_ - 1];
        VectorType* p = this->p_level_[this->current_level_ - 1];
        VectorType* q = this->q_level_[this->current_level_ - 1];

        // Start 2 CG iterations

        ValueType rho = static_cast<ValueType>(0);
        ValueType rho_old;
        ValueType alpha;
        ValueType beta;

        // r = rhs
        r->CopyFrom(rhs);

        // Cycle
        s->Zeros();
        this->Vcycle_(*r, s);

        // rho = (r,s)
        rho = r->Dot(*s);

        // s = Ap
        this->op_level_[this->current_level_ - 1]->Apply(*s, q);

        // alpha = rho / (s,q)
        alpha = rho / s->Dot(*q);

        // x = x + alpha*s
        x->AddScale(*s, alpha);

        // r = r - alpha*q
        r->AddScale(*q, -alpha);

        // 2nd CG iteration

        // rho_old = rho
        rho_old = rho;

        // Cycle
        p->Zeros();
        this->Vcycle_(*r, p);

        // rho = (r,p)
        rho = r->Dot(*p);

        // beta = rho / rho_old
        beta = rho / rho_old;

        // s = beta*s + p
        s->ScaleAdd(beta, *p);

        // q = As
        this->op_level_[this->current_level_ - 1]->Apply(*s, q);

        // alpha = rho / (s,q)
        alpha = rho / s->Dot(*q);

        // x = x + alpha*s
        x->AddScale(*s, alpha);
    }
    else
    {
        this->solver_coarse_->SolveZeroSol(rhs, x);
    }
}

// do nothing
template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                          VectorType* x)
{
    LOG_INFO("BaseMultiGrid:SolveNonPrecond_() this function is disabled - something is very wrong "
             "if you are calling it ...");
    FATAL_ERROR(__FILE__, __LINE__);
}

// do nothing
template <class OperatorType, class VectorType, typename ValueType>
void BaseMultiGrid<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                       VectorType* x)
{
    LOG_INFO("BaseMultiGrid:SolvePrecond_() this function is disabled - something is very wrong if "
             "you are calling it ...");
    FATAL_ERROR(__FILE__, __LINE__);
}

template class BaseMultiGrid<LocalMatrix<double>, LocalVector<double>, double>;
template class BaseMultiGrid<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BaseMultiGrid<LocalMatrix<std::complex<double>>,
                             LocalVector<std::complex<double>>,
                             std::complex<double>>;
template class BaseMultiGrid<LocalMatrix<std::complex<float>>,
                             LocalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

template class BaseMultiGrid<GlobalMatrix<double>, GlobalVector<double>, double>;
template class BaseMultiGrid<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BaseMultiGrid<GlobalMatrix<std::complex<double>>,
                             GlobalVector<std::complex<double>>,
                             std::complex<double>>;
template class BaseMultiGrid<GlobalMatrix<std::complex<float>>,
                             GlobalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

} // namespace rocalution
