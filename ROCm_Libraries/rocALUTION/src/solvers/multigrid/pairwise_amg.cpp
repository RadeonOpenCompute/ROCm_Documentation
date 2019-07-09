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
#include "pairwise_amg.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../../utils/allocate_free.hpp"

#include <list>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
PairwiseAMG<OperatorType, VectorType, ValueType>::PairwiseAMG()
{
    log_debug(this, "PairwiseAMG::PairwiseAMG()", "default constructor");

    this->beta_        = static_cast<ValueType>(0.25);
    this->coarse_size_ = 300;

    // target coarsening factor
    this->coarsening_factor_ = 4.0;

    // number of pre- and post-smoothing steps
    this->iter_pre_smooth_  = 2;
    this->iter_post_smooth_ = 2;

    // set K-cycle to default
    this->cycle_ = 2;

    // disable scaling
    this->scaling_ = false;

    // disable ordering by default
    this->aggregation_ordering_ = NoOrdering;
}

template <class OperatorType, class VectorType, typename ValueType>
PairwiseAMG<OperatorType, VectorType, ValueType>::~PairwiseAMG()
{
    log_debug(this, "PairwiseAMG::PairwiseAMG()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::Print(void) const
{
    LOG_INFO("AMG solver");
    LOG_INFO("AMG number of levels " << this->levels_);
    LOG_INFO("AMG using pairwise aggregation");
    LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
    LOG_INFO("AMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
    LOG_INFO("AMG with smoother:");
    this->smoother_level_[0]->Print();
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
{
    assert(this->levels_ > 0);

    LOG_INFO("AMG solver starts");
    LOG_INFO("AMG number of levels " << this->levels_);
    LOG_INFO("AMG using pairwise aggregation");
    LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
    LOG_INFO("AMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
    LOG_INFO("AMG with smoother:");
    this->smoother_level_[0]->Print();
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
{
    LOG_INFO("AMG ends");
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::SetBeta(ValueType beta)
{
    log_debug(this, "PairwiseAMG::SetBeta()", beta);

    assert(beta > static_cast<ValueType>(0));
    assert(beta < static_cast<ValueType>(1));

    this->beta_ = beta;
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::SetCoarseningFactor(double factor)
{
    log_debug(this, "PairwiseAMG::SetCoarseningFactor()", factor);

    assert(factor > 0.0);
    assert(factor < 20.0);

    this->coarsening_factor_ = factor;
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::SetOrdering(unsigned int ordering)
{
    log_debug(this, "PairwiseAMG::SetOrdering()", ordering);

    assert(ordering >= 0 && ordering <= 5);

    this->aggregation_ordering_ = ordering;
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::BuildHierarchy(void)
{
    log_debug(this, "PairwiseAMG::BuildHierarchy()", " #*# begin");

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
        std::list<LocalVector<int>*> trans_list_;

        this->levels_ = 1;

        // Build finest hierarchy
        op_list_.push_back(new OperatorType);
        restrict_list_.push_back(new OperatorType);
        prolong_list_.push_back(new OperatorType);
        trans_list_.push_back(new LocalVector<int>);

        op_list_.back()->CloneBackend(*this->op_);
        restrict_list_.back()->CloneBackend(*this->op_);
        prolong_list_.back()->CloneBackend(*this->op_);
        trans_list_.back()->CloneBackend(*this->op_);

        // Create prolongation and restriction operators
        this->Aggregate_(*this->op_,
                         prolong_list_.back(),
                         restrict_list_.back(),
                         op_list_.back(),
                         trans_list_.back());

        ++this->levels_;

        while(op_list_.back()->GetM() > (IndexType2) this->coarse_size_)
        {
            // Add new list elements
            restrict_list_.push_back(new OperatorType);
            prolong_list_.push_back(new OperatorType);
            OperatorType* prev_op_ = op_list_.back();
            op_list_.push_back(new OperatorType);
            trans_list_.push_back(new LocalVector<int>);

            op_list_.back()->CloneBackend(*this->op_);
            restrict_list_.back()->CloneBackend(*this->op_);
            prolong_list_.back()->CloneBackend(*this->op_);
            trans_list_.back()->CloneBackend(*this->op_);

            this->Aggregate_(*prev_op_,
                             prolong_list_.back(),
                             restrict_list_.back(),
                             op_list_.back(),
                             trans_list_.back());

            ++this->levels_;
        }

        // Allocate data structures
        this->op_level_          = new OperatorType*[this->levels_ - 1];
        this->restrict_op_level_ = new Operator<ValueType>*[this->levels_ - 1];
        this->prolong_op_level_  = new Operator<ValueType>*[this->levels_ - 1];
        this->trans_level_       = new LocalVector<int>*[this->levels_ - 1];

        typename std::list<OperatorType*>::iterator op_it        = op_list_.begin();
        typename std::list<OperatorType*>::iterator pro_it       = prolong_list_.begin();
        typename std::list<OperatorType*>::iterator res_it       = restrict_list_.begin();
        typename std::list<LocalVector<int>*>::iterator trans_it = trans_list_.begin();

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i] = *op_it;
            this->op_level_[i]->Sort();
            ++op_it;

            this->restrict_op_level_[i] = *res_it;
            ++res_it;

            this->prolong_op_level_[i] = *pro_it;
            ++pro_it;

            this->trans_level_[i] = *trans_it;
            ++trans_it;
        }
    }

    log_debug(this, "PairwiseAMG::BuildHierarchy()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::BuildSmoothers(void)
{
    log_debug(this, "PairwiseAMG::BuildSmoothers()", " #*# begin");

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

    log_debug(this, "PairwiseAMG::BuildSmoothers()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
{
    log_debug(this, "PairwiseAMG::ReBuildNumeric()", " #*# begin");

    assert(this->levels_ > 1);
    assert(this->build_ == true);
    assert(this->op_ != NULL);

    this->op_level_[0]->Clear();
    this->op_level_[0]->CloneBackend(*this->op_);
    this->op_level_[0]->ConvertToCSR();

    this->trans_level_[0]->CloneBackend(*this->op_);

    this->op_->CoarsenOperator(this->op_level_[0],
                               this->dim_level_[0],
                               this->dim_level_[0],
                               *this->trans_level_[0],
                               this->Gsize_level_[0],
                               this->rG_level_[0],
                               this->rGsize_level_[0]);

    for(int i = 1; i < this->levels_ - 1; ++i)
    {
        this->op_level_[i]->Clear();
        this->op_level_[i]->ConvertToCSR();

        this->trans_level_[i]->CloneBackend(*this->op_level_[i]);

        if(i == this->levels_ - this->host_level_ - 1)
        {
            this->op_level_[i - 1]->MoveToHost();
        }

        this->op_level_[i - 1]->CoarsenOperator(this->op_level_[i],
                                                this->dim_level_[i],
                                                this->dim_level_[i],
                                                *this->trans_level_[i],
                                                this->Gsize_level_[i],
                                                this->rG_level_[i],
                                                this->rGsize_level_[i]);

        if(i == this->levels_ - this->host_level_ - 1)
        {
            this->op_level_[i - 1]->CloneBackend(*this->restrict_op_level_[i - 1]);
        }
    }

    this->smoother_level_[0]->ResetOperator(*this->op_);
    this->smoother_level_[0]->ReBuildNumeric();
    this->smoother_level_[0]->Verbose(0);

    for(int i = 1; i < this->levels_ - 1; ++i)
    {
        this->smoother_level_[i]->ResetOperator(*this->op_level_[i - 1]);
        this->smoother_level_[i]->ReBuildNumeric();
        this->smoother_level_[i]->Verbose(0);
    }

    this->solver_coarse_->ResetOperator(*this->op_level_[this->levels_ - 2]);
    this->solver_coarse_->ReBuildNumeric();
    this->solver_coarse_->Verbose(0);

    // Convert operator to op_format
    if(this->op_format_ != CSR)
    {
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i]->ConvertTo(this->op_format_);
        }
    }

    log_debug(this, "PairwiseAMG::ReBuildNumeric()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::ClearLocal(void)
{
    log_debug(this, "PairwiseAMG::ClearLocal()", this->build_);

    if(this->build_ == true)
    {
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            delete this->trans_level_[i];

            free_host(&this->rG_level_[i]);
        }

        delete[] this->trans_level_;

        this->dim_level_.clear();
        this->Gsize_level_.clear();
        this->rGsize_level_.clear();
        this->rG_level_.clear();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::Aggregate_(const OperatorType& op,
                                                                  Operator<ValueType>* pro,
                                                                  Operator<ValueType>* res,
                                                                  OperatorType* coarse,
                                                                  LocalVector<int>* trans)
{
    log_debug(this, "PairwiseAMG::Aggregate_()", (const void*&)op, pro, res, coarse, trans);

    assert(pro != NULL);
    assert(res != NULL);
    assert(coarse != NULL);

    OperatorType* cast_res = dynamic_cast<OperatorType*>(res);
    OperatorType* cast_pro = dynamic_cast<OperatorType*>(pro);

    assert(cast_res != NULL);
    assert(cast_pro != NULL);

    int nc;
    int* rG = NULL;
    int Gsize;
    int rGsize;

    // Allocate transfer mapping for current level
    trans->Allocate("transfer map", op.GetLocalM());

    op.InitialPairwiseAggregation(
        this->beta_, nc, trans, Gsize, &rG, rGsize, this->aggregation_ordering_);
    op.CoarsenOperator(coarse, nc, nc, *trans, Gsize, rG, rGsize);

    int cycle = 0;

    while(static_cast<ValueType>(op.GetM()) / static_cast<ValueType>(coarse->GetM()) <
          this->coarsening_factor_)
    {
        coarse->FurtherPairwiseAggregation(
            this->beta_, nc, trans, Gsize, &rG, rGsize, this->aggregation_ordering_);
        op.CoarsenOperator(coarse, nc, nc, *trans, Gsize, rG, rGsize);

        ++cycle;

        if(cycle > 8)
        {
            LOG_VERBOSE_INFO(2,
                             "*** warning: PairwiseAMG::Build() Coarsening cannot obtain "
                             "satisfying coarsening factor");
        }
    }

    cast_res->CreateFromMap(*trans, op.GetLocalM(), nc, cast_pro);

    // Store data for possible coarse operator rebuild
    this->dim_level_.push_back(nc);
    this->Gsize_level_.push_back(Gsize);
    this->rGsize_level_.push_back(rGsize);
    this->rG_level_.push_back(rG);
}

// disabled function
template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::Aggregate_(const OperatorType& op,
                                                                  Operator<ValueType>* pro,
                                                                  Operator<ValueType>* res,
                                                                  OperatorType* coarse)
{
    FATAL_ERROR(__FILE__, __LINE__);
}

template class PairwiseAMG<LocalMatrix<double>, LocalVector<double>, double>;
template class PairwiseAMG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class PairwiseAMG<LocalMatrix<std::complex<double>>,
                           LocalVector<std::complex<double>>,
                           std::complex<double>>;
template class PairwiseAMG<LocalMatrix<std::complex<float>>,
                           LocalVector<std::complex<float>>,
                           std::complex<float>>;
#endif

} // namespace rocalution
