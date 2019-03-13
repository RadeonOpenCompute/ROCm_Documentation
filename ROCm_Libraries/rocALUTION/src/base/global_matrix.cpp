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

#include "../utils/def.hpp"
#include "global_matrix.hpp"
#include "global_vector.hpp"
#include "local_matrix.hpp"
#include "local_vector.hpp"
#include "matrix_formats.hpp"
#include "../utils/log.hpp"
#include "../utils/allocate_free.hpp"
#include "../utils/math_functions.hpp"

#ifdef SUPPORT_MULTINODE
#include "../utils/communicator.hpp"
#endif

#include <sstream>
#include <limits>
#include <algorithm>
#include <complex>

namespace rocalution {

template <typename ValueType>
GlobalMatrix<ValueType>::GlobalMatrix()
{
    log_debug(this, "GlobalMatrix::GlobalMatrix()");

#ifndef SUPPORT_MULTINODE
    LOG_INFO("Multinode support disabled");
    FATAL_ERROR(__FILE__, __LINE__);
#endif

    this->object_name_ = "";

    this->nnz_ = 0;
}

template <typename ValueType>
GlobalMatrix<ValueType>::GlobalMatrix(const ParallelManager& pm)
{
    log_debug(this, "GlobalMatrix::GlobalMatrix()", (const void*&)pm);

    assert(pm.Status() == true);

    this->object_name_ = "";

    this->pm_ = &pm;

    this->nnz_ = 0;
}

template <typename ValueType>
GlobalMatrix<ValueType>::~GlobalMatrix()
{
    log_debug(this, "GlobalMatrix::~GlobalMatrix()");

    this->Clear();
}

template <typename ValueType>
void GlobalMatrix<ValueType>::Clear(void)
{
    log_debug(this, "GlobalMatrix::Clear()");

    this->matrix_interior_.Clear();
    this->matrix_ghost_.Clear();

    this->nnz_ = 0;
}

template <typename ValueType>
IndexType2 GlobalMatrix<ValueType>::GetM(void) const
{
    return this->pm_->GetGlobalSize();
}

template <typename ValueType>
IndexType2 GlobalMatrix<ValueType>::GetN(void) const
{
    return this->pm_->GetGlobalSize();
}

template <typename ValueType>
IndexType2 GlobalMatrix<ValueType>::GetNnz(void) const
{
    return this->nnz_;
}

template <typename ValueType>
int GlobalMatrix<ValueType>::GetLocalM(void) const
{
    return this->matrix_interior_.GetLocalM();
}

template <typename ValueType>
int GlobalMatrix<ValueType>::GetLocalN(void) const
{
    return this->matrix_interior_.GetLocalN();
}

template <typename ValueType>
int GlobalMatrix<ValueType>::GetLocalNnz(void) const
{
    return this->matrix_interior_.GetLocalNnz();
}

template <typename ValueType>
int GlobalMatrix<ValueType>::GetGhostM(void) const
{
    return this->matrix_ghost_.GetLocalM();
}

template <typename ValueType>
int GlobalMatrix<ValueType>::GetGhostN(void) const
{
    return this->matrix_ghost_.GetLocalN();
}

template <typename ValueType>
int GlobalMatrix<ValueType>::GetGhostNnz(void) const
{
    return this->matrix_ghost_.GetLocalNnz();
}

template <typename ValueType>
const LocalMatrix<ValueType>& GlobalMatrix<ValueType>::GetInterior() const
{
    log_debug(this, "GlobalMatrix::GetInterior() const");

    return this->matrix_interior_;
}

template <typename ValueType>
const LocalMatrix<ValueType>& GlobalMatrix<ValueType>::GetGhost() const
{
    log_debug(this, "GlobalMatrix::GetGhost()");

    return this->matrix_ghost_;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetParallelManager(const ParallelManager& pm)
{
    log_debug(this, "GlobalMatrix::SetParallelManager()", (const void*&)pm);

    assert(pm.Status() == true);

    this->pm_ = &pm;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::AllocateCSR(const std::string name, int local_nnz, int ghost_nnz)
{
    log_debug(this, "GlobalMatrix::AllocateCSR()", name, local_nnz, ghost_nnz);

    assert(this->pm_ != NULL);
    assert(local_nnz > 0);
    assert(ghost_nnz >= 0);

    this->object_name_        = name;
    std::string interior_name = "Interior of " + name;
    std::string ghost_name    = "Ghost of " + name;

    this->matrix_interior_.AllocateCSR(
        interior_name, local_nnz, this->pm_->GetLocalSize(), this->pm_->GetLocalSize());
    this->matrix_ghost_.AllocateCSR(
        ghost_name, ghost_nnz, this->pm_->GetLocalSize(), this->pm_->GetNumReceivers());

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::AllocateCOO(const std::string name, int local_nnz, int ghost_nnz)
{
    log_debug(this, "GlobalMatrix::AllocateCOO()", name, local_nnz, ghost_nnz);

    assert(this->pm_ != NULL);
    assert(local_nnz > 0);
    assert(ghost_nnz >= 0);

    this->object_name_        = name;
    std::string interior_name = "Interior of " + name;
    std::string ghost_name    = "Ghost of " + name;

    this->matrix_interior_.AllocateCOO(
        interior_name, local_nnz, this->pm_->GetLocalSize(), this->pm_->GetLocalSize());
    this->matrix_ghost_.AllocateCOO(
        ghost_name, ghost_nnz, this->pm_->GetLocalSize(), this->pm_->GetNumReceivers());

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetDataPtrCSR(int** local_row_offset,
                                            int** local_col,
                                            ValueType** local_val,
                                            int** ghost_row_offset,
                                            int** ghost_col,
                                            ValueType** ghost_val,
                                            std::string name,
                                            int local_nnz,
                                            int ghost_nnz)
{
    log_debug(this,
              "GlobalMatrix::SetDataPtrCSR()",
              local_row_offset,
              local_col,
              local_val,
              ghost_row_offset,
              ghost_col,
              ghost_val,
              name,
              local_nnz,
              ghost_nnz);

    assert(local_row_offset != NULL);
    assert(local_col != NULL);
    assert(local_val != NULL);

    assert(ghost_row_offset != NULL);
    assert(ghost_col != NULL);
    assert(ghost_val != NULL);

    assert(*local_row_offset != NULL);
    assert(*local_col != NULL);
    assert(*local_val != NULL);
    assert(local_nnz > 0);

    assert(*ghost_row_offset != NULL);
    assert(*ghost_col != NULL);
    assert(*ghost_val != NULL);
    assert(ghost_nnz > 0);

    assert(this->pm_ != NULL);

    this->Clear();

    this->object_name_        = name;
    std::string interior_name = "Interior of " + name;
    std::string ghost_name    = "Ghost of " + name;

    this->matrix_interior_.SetDataPtrCSR(local_row_offset,
                                         local_col,
                                         local_val,
                                         interior_name,
                                         local_nnz,
                                         this->pm_->GetLocalSize(),
                                         this->pm_->GetLocalSize());
    this->matrix_ghost_.SetDataPtrCSR(ghost_row_offset,
                                      ghost_col,
                                      ghost_val,
                                      ghost_name,
                                      ghost_nnz,
                                      this->pm_->GetLocalSize(),
                                      this->pm_->GetNumReceivers());

    this->matrix_ghost_.ConvertTo(COO);

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetDataPtrCOO(int** local_row,
                                            int** local_col,
                                            ValueType** local_val,
                                            int** ghost_row,
                                            int** ghost_col,
                                            ValueType** ghost_val,
                                            std::string name,
                                            int local_nnz,
                                            int ghost_nnz)
{
    log_debug(this,
              "GlobalMatrix::SetDataPtrCOO()",
              local_row,
              local_col,
              local_val,
              ghost_row,
              ghost_col,
              ghost_val,
              name,
              local_nnz,
              ghost_nnz);

    assert(local_row != NULL);
    assert(local_col != NULL);
    assert(local_val != NULL);

    assert(ghost_row != NULL);
    assert(ghost_col != NULL);
    assert(ghost_val != NULL);

    assert(*local_row != NULL);
    assert(*local_col != NULL);
    assert(*local_val != NULL);
    assert(local_nnz > 0);

    assert(*ghost_row != NULL);
    assert(*ghost_col != NULL);
    assert(*ghost_val != NULL);
    assert(ghost_nnz > 0);

    assert(this->pm_ != NULL);

    this->Clear();

    this->object_name_        = name;
    std::string interior_name = "Interior of " + name;
    std::string ghost_name    = "Ghost of " + name;

    this->matrix_interior_.SetDataPtrCOO(local_row,
                                         local_col,
                                         local_val,
                                         interior_name,
                                         local_nnz,
                                         this->pm_->GetLocalSize(),
                                         this->pm_->GetLocalSize());
    this->matrix_ghost_.SetDataPtrCOO(ghost_row,
                                      ghost_col,
                                      ghost_val,
                                      ghost_name,
                                      ghost_nnz,
                                      this->pm_->GetLocalSize(),
                                      this->pm_->GetNumReceivers());

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetLocalDataPtrCSR(
    int** row_offset, int** col, ValueType** val, std::string name, int nnz)
{
    log_debug(this, "GlobalMatrix::SetLocalDataPtrCSR()", row_offset, col, val, name, nnz);

    assert(row_offset != NULL);
    assert(col != NULL);
    assert(val != NULL);

    assert(*row_offset != NULL);
    assert(*col != NULL);
    assert(*val != NULL);

    assert(nnz > 0);

    assert(this->pm_ != NULL);

    this->object_name_        = name;
    std::string interior_name = "Interior of " + name;

    this->matrix_interior_.SetDataPtrCSR(row_offset,
                                         col,
                                         val,
                                         interior_name,
                                         nnz,
                                         this->pm_->GetLocalSize(),
                                         this->pm_->GetLocalSize());

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetLocalDataPtrCOO(
    int** row, int** col, ValueType** val, std::string name, int nnz)
{
    log_debug(this, "GlobalMatrix::SetLocalDataPtrCOO()", row, col, val, name, nnz);

    assert(row != NULL);
    assert(col != NULL);
    assert(val != NULL);

    assert(*row != NULL);
    assert(*col != NULL);
    assert(*val != NULL);

    assert(nnz > 0);

    assert(this->pm_ != NULL);

    this->object_name_        = name;
    std::string interior_name = "Interior of " + name;

    this->matrix_interior_.SetDataPtrCOO(
        row, col, val, interior_name, nnz, this->pm_->GetLocalSize(), this->pm_->GetLocalSize());

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetGhostDataPtrCSR(
    int** row_offset, int** col, ValueType** val, std::string name, int nnz)
{
    log_debug(this, "GlobalMatrix::SetGhostDataPtrCSR()", row_offset, col, val, name, nnz);

    assert(row_offset != NULL);
    assert(col != NULL);
    assert(val != NULL);

    assert(*row_offset != NULL);
    assert(*col != NULL);
    assert(*val != NULL);

    assert(nnz > 0);

    assert(this->pm_ != NULL);

    std::string ghost_name = "Ghost of " + name;

    this->matrix_ghost_.SetDataPtrCSR(row_offset,
                                      col,
                                      val,
                                      ghost_name,
                                      nnz,
                                      this->pm_->GetLocalSize(),
                                      this->pm_->GetNumReceivers());

    this->matrix_ghost_.ConvertTo(COO);

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::SetGhostDataPtrCOO(
    int** row, int** col, ValueType** val, std::string name, int nnz)
{
    log_debug(this, "GlobalMatrix::SetGhostDataPtrCOO()", row, col, val, name, nnz);

    assert(row != NULL);
    assert(col != NULL);
    assert(val != NULL);

    assert(*row != NULL);
    assert(*col != NULL);
    assert(*val != NULL);

    assert(nnz > 0);

    assert(this->pm_ != NULL);

    std::string ghost_name = "Ghost of " + name;

    this->matrix_ghost_.SetDataPtrCOO(
        row, col, val, ghost_name, nnz, this->pm_->GetLocalSize(), this->pm_->GetNumReceivers());

    // Sort ghost matrix
    this->matrix_ghost_.Sort();

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::LeaveDataPtrCSR(int** local_row_offset,
                                              int** local_col,
                                              ValueType** local_val,
                                              int** ghost_row_offset,
                                              int** ghost_col,
                                              ValueType** ghost_val)
{
    log_debug(this,
              "GlobalMatrix::LeaveDataPtrCSR()",
              local_row_offset,
              local_col,
              local_val,
              ghost_row_offset,
              ghost_col,
              ghost_val);

    assert(*local_row_offset == NULL);
    assert(*local_col == NULL);
    assert(*local_val == NULL);

    assert(*ghost_row_offset == NULL);
    assert(*ghost_col == NULL);
    assert(*ghost_val == NULL);

    assert(this->GetLocalM() > 0);
    assert(this->GetLocalN() > 0);
    assert(this->GetLocalNnz() > 0);

    assert(this->GetGhostM() > 0);
    assert(this->GetGhostN() > 0);
    assert(this->GetGhostNnz() > 0);

    this->matrix_interior_.LeaveDataPtrCSR(local_row_offset, local_col, local_val);
    this->matrix_ghost_.LeaveDataPtrCSR(ghost_row_offset, ghost_col, ghost_val);

    this->nnz_ = 0;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::LeaveDataPtrCOO(int** local_row,
                                              int** local_col,
                                              ValueType** local_val,
                                              int** ghost_row,
                                              int** ghost_col,
                                              ValueType** ghost_val)
{
    log_debug(this,
              "GlobalMatrix::LeaveDataPtrCOO()",
              local_row,
              local_col,
              local_val,
              ghost_row,
              ghost_col,
              ghost_val);

    assert(*local_row == NULL);
    assert(*local_col == NULL);
    assert(*local_val == NULL);

    assert(*ghost_row == NULL);
    assert(*ghost_col == NULL);
    assert(*ghost_val == NULL);

    assert(this->GetLocalM() > 0);
    assert(this->GetLocalN() > 0);
    assert(this->GetLocalNnz() > 0);

    assert(this->GetGhostM() > 0);
    assert(this->GetGhostN() > 0);
    assert(this->GetGhostNnz() > 0);

    this->matrix_interior_.LeaveDataPtrCOO(local_row, local_col, local_val);
    this->matrix_ghost_.LeaveDataPtrCOO(ghost_row, ghost_col, ghost_val);

    this->nnz_ = 0;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::LeaveLocalDataPtrCSR(int** row_offset, int** col, ValueType** val)
{
    log_debug(this, "GlobalMatrix::LeaveLocalDataPtrCSR()", row_offset, col, val);

    assert(*row_offset == NULL);
    assert(*col == NULL);
    assert(*val == NULL);

    assert(this->GetLocalM() > 0);
    assert(this->GetLocalN() > 0);
    assert(this->GetLocalNnz() > 0);

    this->matrix_interior_.LeaveDataPtrCSR(row_offset, col, val);

    this->nnz_ = 0;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::LeaveLocalDataPtrCOO(int** row, int** col, ValueType** val)
{
    log_debug(this, "GlobalMatrix::LeaveLocalDataPtrCOO()", row, col, val);

    assert(*row == NULL);
    assert(*col == NULL);
    assert(*val == NULL);

    assert(this->GetLocalM() > 0);
    assert(this->GetLocalN() > 0);
    assert(this->GetLocalNnz() > 0);

    this->matrix_interior_.LeaveDataPtrCOO(row, col, val);

    this->nnz_ = 0;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::LeaveGhostDataPtrCSR(int** row_offset, int** col, ValueType** val)
{
    log_debug(this, "GlobalMatrix::LeaveGhostDataPtrCSR()", row_offset, col, val);

    assert(*row_offset == NULL);
    assert(*col == NULL);
    assert(*val == NULL);

    assert(this->GetGhostM() > 0);
    assert(this->GetGhostN() > 0);
    assert(this->GetGhostNnz() > 0);

    this->matrix_ghost_.LeaveDataPtrCSR(row_offset, col, val);

    this->nnz_ = 0;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::LeaveGhostDataPtrCOO(int** row, int** col, ValueType** val)
{
    log_debug(this, "GlobalMatrix::LeaveGhostDataPtrCOO()", row, col, val);

    assert(*row == NULL);
    assert(*col == NULL);
    assert(*val == NULL);

    assert(this->GetGhostM() > 0);
    assert(this->GetGhostN() > 0);
    assert(this->GetGhostNnz() > 0);

    this->matrix_ghost_.LeaveDataPtrCOO(row, col, val);

    this->nnz_ = 0;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::MoveToAccelerator(void)
{
    log_debug(this, "GlobalMatrix::MoveToAccelerator()");

    this->matrix_interior_.MoveToAccelerator();
    this->matrix_ghost_.MoveToAccelerator();
}

template <typename ValueType>
void GlobalMatrix<ValueType>::MoveToHost(void)
{
    log_debug(this, "GlobalMatrix::MoveToHost()");

    this->matrix_interior_.MoveToHost();
    this->matrix_ghost_.MoveToHost();
}

template <typename ValueType>
bool GlobalMatrix<ValueType>::is_host_(void) const
{
    assert(this->matrix_interior_.is_host_() == this->matrix_ghost_.is_host_());
    return this->matrix_interior_.is_host_();
}

template <typename ValueType>
bool GlobalMatrix<ValueType>::is_accel_(void) const
{
    assert(this->matrix_interior_.is_accel_() == this->matrix_ghost_.is_accel_());
    return this->matrix_interior_.is_accel_();
}

template <typename ValueType>
void GlobalMatrix<ValueType>::Info(void) const
{
    std::string current_backend_name;

    if(this->is_host_() == true)
    {
        current_backend_name = _rocalution_host_name[0];
    }
    else
    {
        assert(this->is_accel_() == true);
        current_backend_name = _rocalution_backend_name[this->local_backend_.backend];
    }

    LOG_INFO("GlobalMatrix"
             << " name="
             << this->object_name_
             << ";"
             << " rows="
             << this->GetM()
             << ";"
             << " cols="
             << this->GetN()
             << ";"
             << " nnz="
             << this->GetNnz()
             << ";"
             << " prec="
             << 8 * sizeof(ValueType)
             << "bit;"
             << " format="
             << _matrix_format_names[this->matrix_interior_.GetFormat()]
             << "/"
             << _matrix_format_names[this->matrix_ghost_.GetFormat()]
             << ";"
             << " subdomains="
             << this->pm_->num_procs_
             << ";"
             << " host backend={"
             << _rocalution_host_name[0]
             << "};"
             << " accelerator backend={"
             << _rocalution_backend_name[this->local_backend_.backend]
             << "};"
             << " current="
             << current_backend_name);
}

template <typename ValueType>
bool GlobalMatrix<ValueType>::Check(void) const
{
    log_debug(this, "GlobalMatrix::Check()");

    bool interior_check = this->matrix_interior_.Check();
    bool ghost_check    = this->matrix_ghost_.Check();

    if(interior_check == true && ghost_check == true)
    {
        return true;
    }

    return false;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::CloneFrom(const GlobalMatrix<ValueType>& src)
{
    log_debug(this, "GlobalMatrix::CloneFrom()");

    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::CopyFrom(const GlobalMatrix<ValueType>& src)
{
    log_debug(this, "GlobalMatrix::CopyFrom()", (const void*&)src);

    assert(this != &src);
    assert(src.GetLocalNnz() != 0);
    assert(src.GetGhostNnz() != 0);

    this->matrix_interior_.CopyFrom(src.GetInterior());
    this->matrix_ghost_.CopyFrom(src.GetGhost());

    this->object_name_ = "Copy from " + src.object_name_;
    this->pm_          = src.pm_;

    this->nnz_ = src.nnz_;
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToCSR(void)
{
    this->ConvertTo(CSR);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToMCSR(void)
{
    this->ConvertTo(MCSR);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToBCSR(void)
{
    this->ConvertTo(BCSR);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToCOO(void)
{
    this->ConvertTo(COO);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToELL(void)
{
    this->ConvertTo(ELL);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToDIA(void)
{
    this->ConvertTo(DIA);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToHYB(void)
{
    this->ConvertTo(HYB);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertToDENSE(void)
{
    this->ConvertTo(DENSE);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ConvertTo(unsigned int matrix_format)
{
    log_debug(this, "GlobalMatrix::ConverTo()", matrix_format);

    this->matrix_interior_.ConvertTo(matrix_format);

    // Ghost part remains COO
    this->matrix_ghost_.ConvertTo(COO);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::Apply(const GlobalVector<ValueType>& in,
                                    GlobalVector<ValueType>* out) const
{
    log_debug(this, "GlobalMatrix::Apply()", (const void*&)in, out);

    assert(out != NULL);
    assert(&in != out);

    assert(this->GetM() == out->GetSize());
    assert(this->GetN() == in.GetSize());
    assert(this->is_host_() == in.is_host_());
    assert(this->is_host_() == out->is_host_());

    out->UpdateGhostValuesAsync_(in);

    this->matrix_interior_.Apply(in.vector_interior_, &out->vector_interior_);

    out->UpdateGhostValuesSync_();

    this->matrix_ghost_.ApplyAdd(
        out->vector_ghost_, static_cast<ValueType>(1), &out->vector_interior_);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ApplyAdd(const GlobalVector<ValueType>& in,
                                       ValueType scalar,
                                       GlobalVector<ValueType>* out) const
{
    log_debug(this, "GlobalMatrix::ApplyAdd()", (const void*&)in, scalar, out);

    assert(out != NULL);
    assert(&in != out);

    assert(this->GetM() == out->GetSize());
    assert(this->GetN() == in.GetSize());
    assert(this->is_host_() == in.is_host_());
    assert(this->is_host_() == out->is_host_());

    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ReadFileMTX(const std::string filename)
{
    log_debug(this, "GlobalMatrix::ReadFileMTX()", filename);

    assert(this->pm_->Status() == true);

    // Read header file
    std::ifstream headfile(filename.c_str(), std::ifstream::in);

    if(!headfile.is_open())
    {
        LOG_INFO("Cannot open GlobalMatrix file [read]: " << filename);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // Go to this ranks line in the headfile
    for(int i = 0; i < this->pm_->rank_; ++i)
    {
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string interior_name;
    std::string ghost_name;

    std::getline(headfile, interior_name);
    std::getline(headfile, ghost_name);

    headfile.close();

    // Extract directory containing the subfiles
    size_t found     = filename.find_last_of("\\/");
    std::string path = filename.substr(0, found + 1);

    interior_name.erase(remove_if(interior_name.begin(), interior_name.end(), isspace),
                        interior_name.end());
    ghost_name.erase(remove_if(ghost_name.begin(), ghost_name.end(), isspace), ghost_name.end());

    this->matrix_interior_.ReadFileMTX(path + interior_name);
    this->matrix_ghost_.ReadFileMTX(path + ghost_name);

    // Convert ghost matrix to COO
    this->matrix_ghost_.ConvertToCOO();

    this->object_name_ = filename;

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::WriteFileMTX(const std::string filename) const
{
    log_debug(this, "GlobalMatrix::WriteFileMTX()", filename);

    // Master rank writes the global headfile
    if(this->pm_->rank_ == 0)
    {
        std::ofstream headfile;

        headfile.open((char*)filename.c_str(), std::ofstream::out);
        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalMatrix file [write]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        for(int i = 0; i < this->pm_->num_procs_; ++i)
        {
            std::ostringstream rs;
            rs << i;

            std::string interior_name = filename + ".interior.rank." + rs.str();
            std::string ghost_name    = filename + ".ghost.rank." + rs.str();

            headfile << interior_name << "\n";
            headfile << ghost_name << "\n";
        }
    }

    std::ostringstream rs;
    rs << this->pm_->rank_;

    std::string interior_name = filename + ".interior.rank." + rs.str();
    std::string ghost_name    = filename + ".ghost.rank." + rs.str();

    this->matrix_interior_.WriteFileMTX(interior_name);
    this->matrix_ghost_.WriteFileMTX(ghost_name);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ReadFileCSR(const std::string filename)
{
    log_debug(this, "GlobalMatrix::ReadFileCSR()", filename);

    assert(this->pm_->Status() == true);

    // Read header file
    std::ifstream headfile(filename.c_str(), std::ifstream::in);

    if(!headfile.is_open())
    {
        LOG_INFO("Cannot open GlobalMatrix file [read]: " << filename);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // Go to this ranks line in the headfile
    for(int i = 0; i < this->pm_->rank_; ++i)
    {
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string interior_name;
    std::string ghost_name;

    std::getline(headfile, interior_name);
    std::getline(headfile, ghost_name);

    headfile.close();

    // Extract directory containing the subfiles
    size_t found     = filename.find_last_of("\\/");
    std::string path = filename.substr(0, found + 1);

    interior_name.erase(remove_if(interior_name.begin(), interior_name.end(), isspace),
                        interior_name.end());
    ghost_name.erase(remove_if(ghost_name.begin(), ghost_name.end(), isspace), ghost_name.end());

    this->matrix_interior_.ReadFileCSR(path + interior_name);
    this->matrix_ghost_.ReadFileCSR(path + ghost_name);

    // Convert ghost matrix to COO
    this->matrix_ghost_.ConvertToCOO();

    this->object_name_ = filename;

#ifdef SUPPORT_MULTINODE
    IndexType2 nnz_local;
    IndexType2 nnz_ghost;

    communication_allreduce_single_sum(
        this->matrix_interior_.GetNnz(), &nnz_local, this->pm_->comm_);
    communication_allreduce_single_sum(this->matrix_ghost_.GetNnz(), &nnz_ghost, this->pm_->comm_);

    this->nnz_ = nnz_local + nnz_ghost;
#endif
}

template <typename ValueType>
void GlobalMatrix<ValueType>::WriteFileCSR(const std::string filename) const
{
    log_debug(this, "GlobalMatrix::WriteFileCSR()", filename);

    // Master rank writes the global headfile
    if(this->pm_->rank_ == 0)
    {
        std::ofstream headfile;

        headfile.open((char*)filename.c_str(), std::ofstream::out);
        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open GlobalMatrix file [write]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        for(int i = 0; i < this->pm_->num_procs_; ++i)
        {
            std::ostringstream rs;
            rs << i;

            std::string interior_name = filename + ".interior.rank." + rs.str();
            std::string ghost_name    = filename + ".ghost.rank." + rs.str();

            headfile << interior_name << "\n";
            headfile << ghost_name << "\n";
        }
    }

    std::ostringstream rs;
    rs << this->pm_->rank_;

    std::string interior_name = filename + ".interior.rank." + rs.str();
    std::string ghost_name    = filename + ".ghost.rank." + rs.str();

    this->matrix_interior_.WriteFileCSR(interior_name);
    this->matrix_ghost_.WriteFileCSR(ghost_name);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::ExtractInverseDiagonal(GlobalVector<ValueType>* vec_inv_diag) const
{
    log_debug(this, "GlobalMatrix::ExtractInverseDiagonal()", vec_inv_diag);

    assert(vec_inv_diag != NULL);
    assert(vec_inv_diag->GetSize() == this->GetM());

    this->matrix_interior_.ExtractInverseDiagonal(&vec_inv_diag->vector_interior_);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::Sort(void)
{
    log_debug(this, "GlobalMatrix::Sort()");

    this->matrix_interior_.Sort();
}

template <typename ValueType>
void GlobalMatrix<ValueType>::Scale(ValueType alpha)
{
    log_debug(this, "GlobalMatrix::Scale()", alpha);

    this->matrix_interior_.Scale(alpha);
    this->matrix_ghost_.Scale(alpha);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::InitialPairwiseAggregation(ValueType beta,
                                                         int& nc,
                                                         LocalVector<int>* G,
                                                         int& Gsize,
                                                         int** rG,
                                                         int& rGsize,
                                                         int ordering) const
{
    log_debug(this,
              "GlobalMatrix::InitialPairwiseAggregation()",
              beta,
              nc,
              G,
              Gsize,
              rG,
              rGsize,
              ordering);

    // TODO asserts

    LocalMatrix<ValueType> tmp;
    tmp.CloneFrom(this->matrix_ghost_);
    tmp.ConvertToCSR();

    this->matrix_interior_.InitialPairwiseAggregation(
        tmp, beta, nc, G, Gsize, rG, rGsize, ordering);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::FurtherPairwiseAggregation(ValueType beta,
                                                         int& nc,
                                                         LocalVector<int>* G,
                                                         int& Gsize,
                                                         int** rG,
                                                         int& rGsize,
                                                         int ordering) const
{
    log_debug(this,
              "GlobalMatrix::FurtherPairwiseAggregation()",
              beta,
              nc,
              G,
              Gsize,
              rG,
              rGsize,
              ordering);

    // TODO asserts

    LocalMatrix<ValueType> tmp;
    tmp.CloneFrom(this->matrix_ghost_);
    tmp.ConvertToCSR();

    this->matrix_interior_.FurtherPairwiseAggregation(
        tmp, beta, nc, G, Gsize, rG, rGsize, ordering);
}

template <typename ValueType>
void GlobalMatrix<ValueType>::CoarsenOperator(GlobalMatrix<ValueType>* Ac,
                                              ParallelManager* pm,
                                              int nrow,
                                              int ncol,
                                              const LocalVector<int>& G,
                                              int Gsize,
                                              const int* rG,
                                              int rGsize) const
{
    log_debug(this,
              "GlobalMatrix::CoarsenOperator()",
              Ac,
              pm,
              nrow,
              ncol,
              (const void*&)G,
              Gsize,
              rG,
              rGsize);

    assert(Ac != NULL);
    assert(pm != NULL);
    assert(rG != NULL);

// TODO asserts

#ifdef SUPPORT_MULTINODE
    // MPI Requests for sync
    std::vector<MRequest> req_mapping(this->pm_->nrecv_ + this->pm_->nsend_);
    std::vector<MRequest> req_offsets(this->pm_->nrecv_ + this->pm_->nsend_);

    // Determine connected pairs for the ghost layer of neighboring ranks
    int** send_ghost_map = new int*[this->pm_->nsend_];
    int** recv_ghost_map = new int*[this->pm_->nrecv_];

    int* send_map_size = NULL;
    int* recv_map_size = NULL;

    allocate_host(this->pm_->nsend_, &send_map_size);
    allocate_host(this->pm_->nrecv_, &recv_map_size);

    // Receive sizes
    for(int n = 0; n < this->pm_->nrecv_; ++n)
    {
        communication_async_recv(
            &recv_map_size[n], 1, this->pm_->recvs_[n], 0, &req_mapping[n], this->pm_->comm_);
    }

    // Loop over neighbor ranks
    for(int n = 0; n < this->pm_->nsend_; ++n)
    {
        send_ghost_map[n] =
            new int[this->pm_->send_offset_index_[n + 1] - this->pm_->send_offset_index_[n]];

        G.ExtractCoarseMapping(this->pm_->send_offset_index_[n],
                               this->pm_->send_offset_index_[n + 1],
                               this->pm_->boundary_index_,
                               nrow,
                               &send_map_size[n],
                               send_ghost_map[n]);

        // Send sizes
        communication_async_send(&send_map_size[n],
                                 1,
                                 this->pm_->sends_[n],
                                 0,
                                 &req_mapping[this->pm_->nrecv_ + n],
                                 this->pm_->comm_);
    }

    // Wait for mapping sizes communication to finish
    communication_syncall(this->pm_->nrecv_ + this->pm_->nsend_, &req_mapping[0]);

    // Receive mapping pairs
    for(int n = 0; n < this->pm_->nrecv_; ++n)
    {
        recv_ghost_map[n] = new int[recv_map_size[n]];
        communication_async_recv(recv_ghost_map[n],
                                 recv_map_size[n],
                                 this->pm_->recvs_[n],
                                 0,
                                 &req_mapping[n],
                                 this->pm_->comm_);
    }

    // Send mapping pairs
    for(int n = 0; n < this->pm_->nsend_; ++n)
    {
        communication_async_send(send_ghost_map[n],
                                 send_map_size[n],
                                 this->pm_->sends_[n],
                                 0,
                                 &req_mapping[this->pm_->nrecv_ + n],
                                 this->pm_->comm_);
    }

    // Get coarse boundary of current rank
    int* boundary_index    = NULL;
    int* send_offset_index = NULL;
    int* recv_offset_index = NULL;

    allocate_host(this->pm_->send_index_size_, &boundary_index);
    allocate_host(this->pm_->nsend_ + 1, &send_offset_index);
    allocate_host(this->pm_->nrecv_ + 1, &recv_offset_index);

    send_offset_index[0] = 0;

    int m = 0;
    for(int n = 0; n < this->pm_->nsend_; ++n)
    {
        G.ExtractCoarseBoundary(this->pm_->send_offset_index_[n],
                                this->pm_->send_offset_index_[n + 1],
                                this->pm_->boundary_index_,
                                nrow,
                                &m,
                                boundary_index);

        send_offset_index[n + 1] = m;
    }

    // Communicate boundary offsets
    for(int n = 0; n < this->pm_->nrecv_; ++n)
    {
        communication_async_recv(&recv_offset_index[n + 1],
                                 1,
                                 this->pm_->recvs_[n],
                                 0,
                                 &req_offsets[n],
                                 this->pm_->comm_);
    }

    for(int n = 0; n < this->pm_->nsend_; ++n)
    {
        communication_async_send(&send_offset_index[n + 1],
                                 1,
                                 this->pm_->sends_[n],
                                 0,
                                 &req_offsets[this->pm_->nrecv_ + n],
                                 this->pm_->comm_);
    }

    int boundary_size = m;

    // Coarsen interior part of the matrix on the host (no accelerator support)
    LocalMatrix<ValueType> tmp;
    LocalMatrix<ValueType> host_interior;

    if(this->is_accel_())
    {
        host_interior.ConvertTo(this->GetInterior().GetFormat());
        host_interior.CopyFrom(this->GetInterior());

        host_interior.CoarsenOperator(&tmp, nrow, nrow, G, Gsize, rG, rGsize);
    }
    else
    {
        this->matrix_interior_.CoarsenOperator(&tmp, nrow, nrow, G, Gsize, rG, rGsize);
    }

    int* Ac_interior_row_offset = NULL;
    int* Ac_interior_col        = NULL;
    ValueType* Ac_interior_val  = NULL;

    int nnzc = tmp.GetNnz();
    tmp.LeaveDataPtrCSR(&Ac_interior_row_offset, &Ac_interior_col, &Ac_interior_val);

    // Wait for boundary offset communication to finish
    communication_syncall(this->pm_->nrecv_ + this->pm_->nsend_, &req_offsets[0]);

    recv_offset_index[0] = 0;
    for(int n = 0; n < this->pm_->nrecv_; ++n)
    {
        recv_offset_index[n + 1] += recv_offset_index[n];
    }

    // Wait for mappings communication to finish
    communication_syncall(this->pm_->nrecv_ + this->pm_->nsend_, &req_mapping[0]);

    // Free send mapping buffers and sizes
    for(int n = 0; n < this->pm_->nsend_; ++n)
    {
        delete[] send_ghost_map[n];
    }

    delete[] send_ghost_map;
    free_host(&send_map_size);

    // Prepare ghost G sets
    int* ghost_G = NULL;

    allocate_host(this->pm_->recv_offset_index_[this->pm_->nrecv_], &ghost_G);

    int k = 0;
    for(int n = 0; n < this->pm_->nrecv_; ++n)
    {
        for(int i = 0; i < this->pm_->recv_offset_index_[n + 1] - this->pm_->recv_offset_index_[n];
            ++i)
        {
            ghost_G[k] =
                (i < recv_map_size[n]) ? (recv_offset_index[n] + recv_ghost_map[n][i]) : -1;
            ++k;
        }
    }

    // Free receive mapping buffers and sizes
    for(int n = 0; n < this->pm_->nrecv_; ++n)
    {
        delete[] recv_ghost_map[n];
    }

    delete[] recv_ghost_map;
    free_host(&recv_map_size);

    // Coarsen ghost part of the matrix on the host (no accelerator support)
    LocalVector<int> G_ghost;
    G_ghost.SetDataPtr(&ghost_G, "G ghost", this->pm_->recv_offset_index_[this->pm_->nrecv_]);

    LocalMatrix<ValueType> tmp_ghost;
    LocalMatrix<ValueType> host_ghost;

    if(this->is_accel_())
    {
        host_ghost.ConvertTo(this->GetGhost().GetFormat());
        host_ghost.CopyFrom(this->GetGhost());

        host_ghost.CoarsenOperator(
            &tmp_ghost, nrow, this->pm_->GetNumReceivers(), G_ghost, Gsize, rG, rGsize);
    }
    else
    {
        this->matrix_ghost_.CoarsenOperator(
            &tmp_ghost, nrow, this->pm_->GetNumReceivers(), G_ghost, Gsize, rG, rGsize);
    }

    G_ghost.Clear();

    int* Ac_ghost_row_offset = NULL;
    int* Ac_ghost_col        = NULL;
    ValueType* Ac_ghost_val  = NULL;

    int nnzg = tmp_ghost.GetNnz();
    tmp_ghost.LeaveDataPtrCSR(&Ac_ghost_row_offset, &Ac_ghost_col, &Ac_ghost_val);

    // Communicator
    pm->Clear();
    pm->SetMPICommunicator(this->pm_->comm_);

    // Get the global size
    int global_size;
    communication_allreduce_single_sum(nrow, &global_size, this->pm_->comm_);
    pm->SetGlobalSize(global_size);

    // Local size
    pm->SetLocalSize(nrow);

    // New boundary and boundary offsets
    pm->SetBoundaryIndex(boundary_size, boundary_index);
    free_host(&boundary_index);

    pm->SetReceivers(this->pm_->nrecv_, this->pm_->recvs_, recv_offset_index);
    free_host(&recv_offset_index);

    pm->SetSenders(this->pm_->nsend_, this->pm_->sends_, send_offset_index);
    free_host(&send_offset_index);

    // Allocate
    Ac->Clear();
    bool isaccel = Ac->is_accel_();
    Ac->MoveToHost();
    Ac->SetParallelManager(*pm);
    Ac->SetDataPtrCSR(&Ac_interior_row_offset,
                      &Ac_interior_col,
                      &Ac_interior_val,
                      &Ac_ghost_row_offset,
                      &Ac_ghost_col,
                      &Ac_ghost_val,
                      "",
                      nnzc,
                      nnzg);

    if(isaccel == true)
    {
        Ac->MoveToAccelerator();
    }
#endif
}

template class GlobalMatrix<double>;
template class GlobalMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class GlobalMatrix<std::complex<double>>;
template class GlobalMatrix<std::complex<float>>;
#endif

} // namespace rocalution
