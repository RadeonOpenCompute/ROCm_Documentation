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
#include "../../utils/types.hpp"
#include "base_amg.hpp"

#include "../iter_ctrl.hpp"
#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"
#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"

#include "../krylov/cg.hpp"
#include "../preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"

#include <list>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
BaseAMG<OperatorType, VectorType, ValueType>::BaseAMG()
{
    log_debug(this, "BaseAMG::BaseAMG()", "default constructor");

    this->coarse_size_ = 300;

    // manual smoothers and coarse solver
    this->set_sm_ = false;
    this->set_s_  = false;

    // default smoother format
    this->sm_format_ = CSR;
    // default operator format
    this->op_format_ = CSR;

    // since hierarchy has not been built yet
    this->hierarchy_ = false;

    // initialize temp default smoother pointer
    this->sm_default_ = NULL;
}

template <class OperatorType, class VectorType, typename ValueType>
BaseAMG<OperatorType, VectorType, ValueType>::~BaseAMG()
{
    log_debug(this, "BaseAMG::BaseAMG()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetCoarsestLevel(int coarse_size)
{
    log_debug(this, "BaseAMG::SetCoarsestLevel()", coarse_size);

    assert(this->build_ == false);
    assert(this->hierarchy_ == false);

    this->coarse_size_ = coarse_size;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetManualSmoothers(bool sm_manual)
{
    log_debug(this, "BaseAMG::SetManualSmoothers()", sm_manual);

    assert(this->build_ == false);

    this->set_sm_ = sm_manual;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetManualSolver(bool s_manual)
{
    log_debug(this, "BaseAMG::SetManualSolver()", s_manual);

    assert(this->build_ == false);

    this->set_s_ = s_manual;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetDefaultSmootherFormat(unsigned int op_format)
{
    log_debug(this, "BaseAMG::SetDefaultSmootherFormat()", op_format);

    assert(this->build_ == false);

    this->sm_format_ = op_format;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetOperatorFormat(unsigned int op_format)
{
    log_debug(this, "BaseAMG::SetOperatorFormat()", op_format);

    this->op_format_ = op_format;
}

template <class OperatorType, class VectorType, typename ValueType>
int BaseAMG<OperatorType, VectorType, ValueType>::GetNumLevels(void)
{
    assert(this->hierarchy_ != false);

    return this->levels_;
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::Build(void)
{
    log_debug(this, "BaseAMG::Build()", this->build_, " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);

    this->BuildHierarchy();

    this->build_ = true;

    log_debug(this, "BaseAMG::Build()", "#*# allocate data");

    this->d_level_ = new VectorType*[this->levels_];
    this->r_level_ = new VectorType*[this->levels_];
    this->t_level_ = new VectorType*[this->levels_];
    this->s_level_ = new VectorType*[this->levels_];

    // Extra structure for K-cycle
    if(this->cycle_ == 2)
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

    for(int i = 0; i < this->levels_; ++i)
    {
        this->r_level_[i] = new VectorType;
        this->t_level_[i] = new VectorType;
        this->s_level_[i] = new VectorType;
        if(i > 0)
        {
            this->d_level_[i] = new VectorType;
        }

        if(i > 0)
        {
            this->d_level_[i]->CloneBackend(*this->op_level_[i - 1]);
            this->r_level_[i]->CloneBackend(*this->op_level_[i - 1]);
            this->t_level_[i]->CloneBackend(*this->op_level_[i - 1]);
            this->s_level_[i]->CloneBackend(*this->op_level_[i - 1]);
        }
        else
        {
            this->r_level_[i]->CloneBackend(*this->op_);
            this->t_level_[i]->CloneBackend(*this->op_);
            this->s_level_[i]->CloneBackend(*this->op_);
        }
    }

    // Allocate temporary vectors for cycles
    for(int level = 0; level < this->levels_; ++level)
    {
        if(level > 0)
        {
            this->d_level_[level]->Allocate("defect correction",
                                            this->op_level_[level - 1]->GetM());
            this->r_level_[level]->Allocate("residual", this->op_level_[level - 1]->GetM());
            this->t_level_[level]->Allocate("temporary", this->op_level_[level - 1]->GetM());
            this->s_level_[level]->Allocate("temporary", this->op_level_[level - 1]->GetM());
        }
        else
        {
            this->r_level_[level]->Allocate("residual", this->op_->GetM());
            this->t_level_[level]->Allocate("temporary", this->op_->GetM());
            this->s_level_[level]->Allocate("temporary", this->op_->GetM());
        }
    }

    log_debug(this, "BaseAMG::Build()", "#*# setup smoothers");

    // Setup and build smoothers
    if(this->set_sm_ == false)
    {
        this->BuildSmoothers();
    }

    for(int i = 0; i < this->levels_ - 1; ++i)
    {
        if(i > 0)
        {
            this->smoother_level_[i]->SetOperator(*this->op_level_[i - 1]);
        }
        else
        {
            this->smoother_level_[i]->SetOperator(*this->op_);
        }

        this->smoother_level_[i]->Build();
    }

    log_debug(this, "BaseAMG::Build()", "#*# setup coarse solver");

    // Setup and build coarse grid solver
    if(this->set_s_ == false)
    {
        // Coarse Grid Solver
        CG<OperatorType, VectorType, ValueType>* cgs = new CG<OperatorType, VectorType, ValueType>;

        // set absolute tolerance to 0 to avoid issues with very small numbers
        cgs->Init(0.0, 1e-6, 1e+8, 1000000);
        // be quite
        cgs->Verbose(0);

        this->solver_coarse_ = cgs;
    }

    this->solver_coarse_->SetOperator(*this->op_level_[this->levels_ - 2]);
    this->solver_coarse_->Build();

    log_debug(this, "BaseAMG::Build()", "#*# convert operators");

    // Convert operator to op_format
    if(this->op_format_ != CSR)
    {
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i]->ConvertTo(this->op_format_);
        }
    }

    log_debug(this, "BaseAMG::Build()", this->build_, " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::BuildHierarchy(void)
{
    log_debug(this, "BaseAMG::BuildHierarchy()", " #*# begin");

    if(this->hierarchy_ == false)
    {
        assert(this->build_ == false);
        this->hierarchy_ = true;

        // AMG will use operators for inter grid transfers
        assert(this->op_ != NULL);
        assert(this->coarse_size_ > 0);

        if(this->op_->GetM() <= (IndexType2) this->coarse_size_)
        {
            LOG_INFO("Problem size too small for AMG, use Krylov solver instead");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Lists for the building procedure
        std::list<OperatorType*> op_list_;
        std::list<OperatorType*> restrict_list_;
        std::list<OperatorType*> prolong_list_;

        this->levels_ = 1;

        // Build finest hierarchy
        op_list_.push_back(new OperatorType);
        restrict_list_.push_back(new OperatorType);
        prolong_list_.push_back(new OperatorType);

        op_list_.back()->CloneBackend(*this->op_);
        restrict_list_.back()->CloneBackend(*this->op_);
        prolong_list_.back()->CloneBackend(*this->op_);

        // Create prolongation and restriction operators
        this->Aggregate_(*this->op_, prolong_list_.back(), restrict_list_.back(), op_list_.back());

        ++this->levels_;

        while(op_list_.back()->GetM() > (IndexType2) this->coarse_size_)
        {
            // Add new list elements
            restrict_list_.push_back(new OperatorType);
            prolong_list_.push_back(new OperatorType);
            OperatorType* prev_op_ = op_list_.back();
            op_list_.push_back(new OperatorType);

            op_list_.back()->CloneBackend(*this->op_);
            restrict_list_.back()->CloneBackend(*this->op_);
            prolong_list_.back()->CloneBackend(*this->op_);

            this->Aggregate_(
                *prev_op_, prolong_list_.back(), restrict_list_.back(), op_list_.back());

            ++this->levels_;

            if(this->levels_ > 19)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: BaseAMG::Build() Current number of levels: " << this->levels_);
            }
        }

        // Allocate data structures
        this->op_level_          = new OperatorType*[this->levels_ - 1];
        this->restrict_op_level_ = new Operator<ValueType>*[this->levels_ - 1];
        this->prolong_op_level_  = new Operator<ValueType>*[this->levels_ - 1];

        typename std::list<OperatorType*>::iterator op_it  = op_list_.begin();
        typename std::list<OperatorType*>::iterator pro_it = prolong_list_.begin();
        typename std::list<OperatorType*>::iterator res_it = restrict_list_.begin();

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i] = *op_it;
            this->op_level_[i]->Sort();
            ++op_it;

            this->restrict_op_level_[i] = *res_it;
            ++res_it;

            this->prolong_op_level_[i] = *pro_it;
            ++pro_it;
        }
    }

    log_debug(this, "BaseAMG::BuildHierarchy()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::BuildSmoothers(void)
{
    log_debug(this, "BaseAMG::BuildSmoothers()", " #*# begin");

    // Smoother for each level
    this->smoother_level_ =
        new IterativeLinearSolver<OperatorType, VectorType, ValueType>*[this->levels_ - 1];
    this->sm_default_ = new Solver<OperatorType, VectorType, ValueType>*[this->levels_ - 1];

    for(int i = 0; i < this->levels_ - 1; ++i)
    {
        FixedPoint<OperatorType, VectorType, ValueType>* sm =
            new FixedPoint<OperatorType, VectorType, ValueType>;
        Jacobi<OperatorType, VectorType, ValueType>* jac =
            new Jacobi<OperatorType, VectorType, ValueType>;

        sm->SetRelaxation(static_cast<ValueType>(0.67));
        sm->SetPreconditioner(*jac);
        sm->Verbose(0);
        this->smoother_level_[i] = sm;
        this->sm_default_[i]     = jac;
    }

    log_debug(this, "BaseAMG::BuildSmoothers()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::Clear(void)
{
    log_debug(this, "BaseAMG::Clear()", this->build_);

    if(this->build_ == true)
    {
        this->ClearLocal();

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
        if(this->cycle_ == 2)
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

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            // Clear operator data structure
            delete this->op_level_[i];
            delete this->restrict_op_level_[i];
            delete this->prolong_op_level_[i];

            if(this->set_sm_ == false)
            {
                delete this->smoother_level_[i];
                delete this->sm_default_[i];
            }
            else
                this->smoother_level_[i]->Clear();
        }

        // Clear coarse grid solver - we built it
        if(this->set_s_ == false)
        {
            delete this->solver_coarse_;
        }
        else
        {
            this->solver_coarse_->Clear();
        }

        delete[] this->op_level_;
        delete[] this->restrict_op_level_;
        delete[] this->prolong_op_level_;

        if(this->set_sm_ == false)
        {
            delete[] this->smoother_level_;
            delete[] this->sm_default_;
        }

        this->levels_ = -1;

        this->iter_ctrl_.Clear();

        this->build_     = false;
        this->hierarchy_ = false;
    }
}

// do nothing
template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::ClearLocal(void)
{
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetRestrictOperator(OperatorType** op)
{
    LOG_INFO("BaseAMG::SetRestrictOperator() Perhaps you want to use the MultiGrid class to set "
             "external restriction operators");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetProlongOperator(OperatorType** op)
{
    LOG_INFO("BaseAMG::SetProlongOperator() Perhaps you want to use the MultiGrid class to set "
             "external prolongation operators");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <class OperatorType, class VectorType, typename ValueType>
void BaseAMG<OperatorType, VectorType, ValueType>::SetOperatorHierarchy(OperatorType** op)
{
    LOG_INFO("BaseAMG::SetOperatorHierarchy() Perhaps you want to use the MultiGrid class to set "
             "external operators");
    FATAL_ERROR(__FILE__, __LINE__);
}

template class BaseAMG<LocalMatrix<double>, LocalVector<double>, double>;
template class BaseAMG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BaseAMG<LocalMatrix<std::complex<double>>,
                       LocalVector<std::complex<double>>,
                       std::complex<double>>;
template class BaseAMG<LocalMatrix<std::complex<float>>,
                       LocalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

template class BaseAMG<GlobalMatrix<double>, GlobalVector<double>, double>;
template class BaseAMG<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BaseAMG<GlobalMatrix<std::complex<double>>,
                       GlobalVector<std::complex<double>>,
                       std::complex<double>>;
template class BaseAMG<GlobalMatrix<std::complex<float>>,
                       GlobalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

} // namespace rocalution
