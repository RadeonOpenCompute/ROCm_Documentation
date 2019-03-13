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
#include "base_matrix.hpp"
#include "base_vector.hpp"
#include "backend_manager.hpp"
#include "../utils/log.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
BaseMatrix<ValueType>::BaseMatrix()
{
    log_debug(this, "BaseMatrix::BaseMatrix()");

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
}

template <typename ValueType>
BaseMatrix<ValueType>::~BaseMatrix()
{
    log_debug(this, "BaseMatrix::~BaseMatrix()");
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetM(void) const
{
    return this->nrow_;
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetN(void) const
{
    return this->ncol_;
}

template <typename ValueType>
inline int BaseMatrix<ValueType>::GetNnz(void) const
{
    return this->nnz_;
}

template <typename ValueType>
void BaseMatrix<ValueType>::set_backend(const Rocalution_Backend_Descriptor local_backend)
{
    this->local_backend_ = local_backend;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Check(void) const
{
    LOG_INFO("BaseMatrix<ValueType>::Check()");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromCSR(const int* row_offsets,
                                        const int* col,
                                        const ValueType* val)
{
    LOG_INFO("CopyFromCSR(const int* row_offsets, const int* col, const ValueType* val)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This function is not available for this backend");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyToCSR(int* row_offsets, int* col, ValueType* val) const
{
    LOG_INFO("CopyToCSR(int *row_offsets, int *col, ValueType *val) const");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This function is not available for this backend");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromCOO(const int* row, const int* col, const ValueType* val)
{
    LOG_INFO("CopyFromCOO(const int* row, const int* col, const ValueType* val)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This function is not available for this backend");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyToCOO(int* row, int* col, ValueType* val) const
{
    LOG_INFO("CopyToCOO(const int* row, const int* col, const ValueType* val) const");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This function is not available for this backend");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromHostCSR(
    const int* row_offset, const int* col, const ValueType* val, int nnz, int nrow, int ncol)
{
    LOG_INFO("CopyFromHostCSR(const int* row_offsets, const int* col, const ValueType* val, int "
             "nnz, int nrow, int ncol)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This function is not available for this backend");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateCSR(int nnz, int nrow, int ncol)
{
    LOG_INFO("AllocateCSR(int nnz, int nrow, int ncol)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a CSR matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateCOO(int nnz, int nrow, int ncol)
{
    LOG_INFO("AllocateCOO(int nnz, int nrow, int ncol)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a COO matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateDIA(int nnz, int nrow, int ncol, int ndiag)
{
    LOG_INFO("AllocateDIA(int nnz, int nrow, int ncol, int ndiag)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a DIA matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateELL(int nnz, int nrow, int ncol, int max_row)
{
    LOG_INFO("AllocateELL(int nnz, int nrow, int ncol, int max_row)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a ELL matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateHYB(
    int ell_nnz, int coo_nnz, int ell_max_row, int nrow, int ncol)
{
    LOG_INFO("AllocateHYB(int ell_nnz, int coo_nnz, int ell_max_row, int nrow, int ncol)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a HYB matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateDENSE(int nrow, int ncol)
{
    LOG_INFO("AllocateDENSE(int nrow, int ncol)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a DENSE matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateMCSR(int nnz, int nrow, int ncol)
{
    LOG_INFO("AllocateMCSR(int nnz, int nrow, int ncol)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("This is NOT a MCSR matrix");
    FATAL_ERROR(__FILE__, __LINE__);
}

// The conversion CSR->COO (or X->CSR->COO)
template <typename ValueType>
bool BaseMatrix<ValueType>::ReadFileMTX(const std::string filename)
{
    return false;
}

// The conversion CSR->COO (or X->CSR->COO)
template <typename ValueType>
bool BaseMatrix<ValueType>::WriteFileMTX(const std::string filename) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ReadFileCSR(const std::string filename)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::WriteFileCSR(const std::string filename) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractDiagonal(BaseVector<ValueType>* vec_diag) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractSubMatrix(
    int row_offset, int col_offset, int row_size, int col_size, BaseMatrix<ValueType>* mat) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractL(BaseMatrix<ValueType>* L) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType>* L) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractU(BaseMatrix<ValueType>* U) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType>* U) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                    BaseVector<ValueType>* out) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                    BaseVector<ValueType>* out) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                    const BaseVector<ValueType>& inv_diag,
                                    BaseVector<ValueType>* out) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ILU0Factorize(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ILUTFactorize(double t, int maxrow)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ICFactorize(BaseVector<ValueType>* inv_diag)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Permute(const BaseVector<int>& permutation)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CMK(BaseVector<int>* permutation) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::RCMK(BaseVector<int>* permutation) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ConnectivityOrder(BaseVector<int>* permutation) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MultiColoring(int& num_colors,
                                          int** size_colors,
                                          BaseVector<int>* permutation) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MaximalIndependentSet(int& size, BaseVector<int>* permutation) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ZeroBlockPermutation(int& size, BaseVector<int>* permutation) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SymbolicPower(int p)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& src)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MatrixAdd(const BaseMatrix<ValueType>& mat,
                                      ValueType alpha,
                                      ValueType beta,
                                      bool structure)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Scale(ValueType alpha)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ScaleDiagonal(ValueType alpha)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ScaleOffDiagonal(ValueType alpha)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AddScalar(ValueType alpha)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AddScalarDiagonal(ValueType alpha)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AddScalarOffDiagonal(ValueType alpha)
{
    return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::LUAnalyse(void)
{
    LOG_INFO("BaseMatrix<ValueType>::LUAnalyse(void)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LUAnalyseClear(void)
{
    LOG_INFO("BaseMatrix<ValueType>::LUAnalyseClear(void)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LLAnalyse(void)
{
    LOG_INFO("BaseMatrix<ValueType>::LLAnalyse(void)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LLAnalyseClear(void)
{
    LOG_INFO("BaseMatrix<ValueType>::LLAnalyseClear(void)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LAnalyse(bool diag_unit)
{
    LOG_INFO("BaseMatrix<ValueType>::LAnalyse(bool diag_unit=false)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LAnalyseClear(void)
{
    LOG_INFO("BaseMatrix<ValueType>::LAnalyseClear(void)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LSolve(const BaseVector<ValueType>& in,
                                   BaseVector<ValueType>* out) const
{
    return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::UAnalyse(bool diag_unit)
{
    LOG_INFO("BaseMatrix<ValueType>::UAnalyse(bool diag_unit=false)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::UAnalyseClear(void)
{
    LOG_INFO("BaseMatrix<ValueType>::UAnalyseClear(void)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)!");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
bool BaseMatrix<ValueType>::USolve(const BaseVector<ValueType>& in,
                                   BaseVector<ValueType>* out) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::NumericMatMatMult(const BaseMatrix<ValueType>& A,
                                              const BaseMatrix<ValueType>& B)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& A,
                                               const BaseMatrix<ValueType>& B)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGConnect(ValueType eps, BaseVector<int>* connections) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGAggregate(const BaseVector<int>& connections,
                                         BaseVector<int>* aggregates) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGSmoothedAggregation(ValueType relax,
                                                   const BaseVector<int>& aggregates,
                                                   const BaseVector<int>& connections,
                                                   BaseMatrix<ValueType>* prolong,
                                                   BaseMatrix<ValueType>* restrict) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGAggregation(const BaseVector<int>& aggregates,
                                           BaseMatrix<ValueType>* prolong,
                                           BaseMatrix<ValueType>* restrict) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::RugeStueben(ValueType eps,
                                        BaseMatrix<ValueType>* prolong,
                                        BaseMatrix<ValueType>* restrict) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::InitialPairwiseAggregation(ValueType beta,
                                                       int& nc,
                                                       BaseVector<int>* G,
                                                       int& Gsize,
                                                       int** rG,
                                                       int& rGsize,
                                                       int ordering) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                       ValueType beta,
                                                       int& nc,
                                                       BaseVector<int>* G,
                                                       int& Gsize,
                                                       int** rG,
                                                       int& rGsize,
                                                       int ordering) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::FurtherPairwiseAggregation(ValueType beta,
                                                       int& nc,
                                                       BaseVector<int>* G,
                                                       int& Gsize,
                                                       int** rG,
                                                       int& rGsize,
                                                       int ordering) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                       ValueType beta,
                                                       int& nc,
                                                       BaseVector<int>* G,
                                                       int& Gsize,
                                                       int** rG,
                                                       int& rGsize,
                                                       int ordering) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                            int nrow,
                                            int ncol,
                                            const BaseVector<int>& G,
                                            int Gsize,
                                            const int* rG,
                                            int rGsize) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LUFactorize(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Householder(int idx, ValueType& beta, BaseVector<ValueType>* vec) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::QRDecompose(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::QRSolve(const BaseVector<ValueType>& in,
                                    BaseVector<ValueType>* out) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Invert(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::FSAI(int power, const BaseMatrix<ValueType>* pattern)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SPAI(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType>& diag)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType>& diag)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                       const BaseMatrix<ValueType>& B)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Zeros(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Compress(double drop_off)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Transpose(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Sort(void)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Key(long int& row_key, long int& col_key, long int& val_key) const
{
    return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrCOO(
    int** row, int** col, ValueType** val, int nnz, int nrow, int ncol)
{
    LOG_INFO("BaseMatrix<ValueType>::SetDataPtrCOO(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrCOO(int** row, int** col, ValueType** val)
{
    LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrCOO(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrCSR(
    int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol)
{
    LOG_INFO("BaseMatrix<ValueType>::SetDataPtrCSR(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val)
{
    LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrCSR(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrMCSR(
    int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol)
{
    LOG_INFO("BaseMatrix<ValueType>::SetDataPtrMCSR(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val)
{
    LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrMCSR(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrDENSE(ValueType** val, int nrow, int ncol)
{
    LOG_INFO("BaseMatrix<ValueType>::SetDataPtrDENSE(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrDENSE(ValueType** val)
{
    LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrDENSE(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrELL(
    int** col, ValueType** val, int nnz, int nrow, int ncol, int max_row)
{
    LOG_INFO("BaseMatrix<ValueType>::SetDataPtrELL(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrELL(int** col, ValueType** val, int& max_row)
{
    LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrELL(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrDIA(
    int** offset, ValueType** val, int nnz, int nrow, int ncol, int num_diag)
{
    LOG_INFO("BaseMatrix<ValueType>::SetDataPtrDIA(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag)
{
    LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrDIA(...)");
    LOG_INFO("Matrix format=" << _matrix_format_names[this->GetMatFormat()]);
    this->Info();
    LOG_INFO("The function is not implemented (yet)! Check the backend?");
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CreateFromMap(const BaseVector<int>& map, int n, int m)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CreateFromMap(const BaseVector<int>& map,
                                          int n,
                                          int m,
                                          BaseMatrix<ValueType>* pro)
{
    return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& mat)
{
    // default is no async
    LOG_VERBOSE_INFO(4, "*** info: BaseMatrix::CopyFromAsync() no async available)");

    this->CopyFrom(mat);
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyToAsync(BaseMatrix<ValueType>* mat) const
{
    // default is no async
    LOG_VERBOSE_INFO(4, "*** info: BaseMatrix::CopyToAsync() no async available)");

    this->CopyTo(mat);
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ReplaceRowVector(int idx, const BaseVector<ValueType>& vec)
{
    return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractRowVector(int idx, BaseVector<ValueType>* vec) const
{
    return false;
}

// TODO print also parameters info?

template <typename ValueType>
HostMatrix<ValueType>::HostMatrix()
{
}

template <typename ValueType>
HostMatrix<ValueType>::~HostMatrix()
{
}

template <typename ValueType>
AcceleratorMatrix<ValueType>::AcceleratorMatrix()
{
}

template <typename ValueType>
AcceleratorMatrix<ValueType>::~AcceleratorMatrix()
{
}

template <typename ValueType>
void AcceleratorMatrix<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
{
    // default is no async
    this->CopyFromHost(src);
}

template <typename ValueType>
void AcceleratorMatrix<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
{
    // default is no async
    this->CopyToHost(dst);
}

template <typename ValueType>
HIPAcceleratorMatrix<ValueType>::HIPAcceleratorMatrix()
{
}

template <typename ValueType>
HIPAcceleratorMatrix<ValueType>::~HIPAcceleratorMatrix()
{
}

template class BaseMatrix<double>;
template class BaseMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class BaseMatrix<std::complex<double>>;
template class BaseMatrix<std::complex<float>>;
#endif
template class BaseMatrix<int>;

template class HostMatrix<double>;
template class HostMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrix<std::complex<double>>;
template class HostMatrix<std::complex<float>>;
#endif
template class HostMatrix<int>;

template class AcceleratorMatrix<double>;
template class AcceleratorMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class AcceleratorMatrix<std::complex<double>>;
template class AcceleratorMatrix<std::complex<float>>;
#endif
template class AcceleratorMatrix<int>;

template class HIPAcceleratorMatrix<double>;
template class HIPAcceleratorMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrix<std::complex<double>>;
template class HIPAcceleratorMatrix<std::complex<float>>;
#endif
template class HIPAcceleratorMatrix<int>;

} // namespace rocalution
