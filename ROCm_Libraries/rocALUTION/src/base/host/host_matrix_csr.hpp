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

#ifndef ROCALUTION_HOST_MATRIX_CSR_HPP_
#define ROCALUTION_HOST_MATRIX_CSR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HostMatrixCSR : public HostMatrix<ValueType>
{
    public:
    HostMatrixCSR();
    HostMatrixCSR(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HostMatrixCSR();

    virtual void Info(void) const;
    virtual unsigned int GetMatFormat(void) const { return CSR; }

    virtual bool Check(void) const;
    virtual void AllocateCSR(int nnz, int nrow, int ncol);
    virtual void
    SetDataPtrCSR(int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol);
    virtual void LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val);

    virtual void Clear(void);
    virtual bool Zeros(void);

    virtual bool Scale(ValueType alpha);
    virtual bool ScaleDiagonal(ValueType alpha);
    virtual bool ScaleOffDiagonal(ValueType alpha);
    virtual bool AddScalar(ValueType alpha);
    virtual bool AddScalarDiagonal(ValueType alpha);
    virtual bool AddScalarOffDiagonal(ValueType alpha);

    virtual bool ExtractSubMatrix(int row_offset,
                                  int col_offset,
                                  int row_size,
                                  int col_size,
                                  BaseMatrix<ValueType>* mat) const;

    virtual bool ExtractDiagonal(BaseVector<ValueType>* vec_diag) const;
    virtual bool ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const;
    virtual bool ExtractU(BaseMatrix<ValueType>* U) const;
    virtual bool ExtractUDiagonal(BaseMatrix<ValueType>* U) const;
    virtual bool ExtractL(BaseMatrix<ValueType>* L) const;
    virtual bool ExtractLDiagonal(BaseMatrix<ValueType>* L) const;

    virtual bool
    MultiColoring(int& num_colors, int** size_colors, BaseVector<int>* permutation) const;

    virtual bool MaximalIndependentSet(int& size, BaseVector<int>* permutation) const;

    virtual bool ZeroBlockPermutation(int& size, BaseVector<int>* permutation) const;

    virtual bool SymbolicPower(int p);

    virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& src);
    virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
    virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
    virtual bool NumericMatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);

    virtual bool DiagonalMatrixMultR(const BaseVector<ValueType>& diag);
    virtual bool DiagonalMatrixMultL(const BaseVector<ValueType>& diag);

    virtual bool
    MatrixAdd(const BaseMatrix<ValueType>& mat, ValueType alpha, ValueType beta, bool structure);

    virtual bool Permute(const BaseVector<int>& permutation);

    virtual bool CMK(BaseVector<int>* permutation) const;
    virtual bool RCMK(BaseVector<int>* permutation) const;
    virtual bool ConnectivityOrder(BaseVector<int>* permutation) const;

    virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

    virtual void CopyFrom(const BaseMatrix<ValueType>& mat);

    virtual void CopyFromCSR(const int* row_offsets, const int* col, const ValueType* val);
    virtual void CopyToCSR(int* row_offsets, int* col, ValueType* val) const;

    virtual void CopyTo(BaseMatrix<ValueType>* mat) const;

    virtual void CopyFromHostCSR(
        const int* row_offset, const int* col, const ValueType* val, int nnz, int nrow, int ncol);

    virtual bool ReadFileCSR(const std::string);
    virtual bool WriteFileCSR(const std::string) const;

    virtual bool CreateFromMap(const BaseVector<int>& map, int n, int m);
    virtual bool
    CreateFromMap(const BaseVector<int>& map, int n, int m, BaseMatrix<ValueType>* pro);

    virtual bool ICFactorize(BaseVector<ValueType>* inv_diag);

    virtual bool ILU0Factorize(void);
    virtual bool ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat);
    virtual bool ILUTFactorize(double t, int maxrow);

    virtual void LUAnalyse(void);
    virtual void LUAnalyseClear(void);
    virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual void LLAnalyse(void);
    virtual void LLAnalyseClear(void);
    virtual bool LLSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual bool LLSolve(const BaseVector<ValueType>& in,
                         const BaseVector<ValueType>& inv_diag,
                         BaseVector<ValueType>* out) const;

    virtual void LAnalyse(bool diag_unit = false);
    virtual void LAnalyseClear(void);
    virtual bool LSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual void UAnalyse(bool diag_unit = false);
    virtual void UAnalyseClear(void);
    virtual bool USolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual bool Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual void
    ApplyAdd(const BaseVector<ValueType>& in, ValueType scalar, BaseVector<ValueType>* out) const;

    virtual bool Compress(double drop_off);
    virtual bool Transpose(void);
    virtual bool Sort(void);
    virtual bool Key(long int& row_key, long int& col_key, long int& val_key) const;

    virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);
    virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;

    virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);
    virtual bool ExtractRowVector(int idx, BaseVector<ValueType>* vec) const;

    virtual bool AMGConnect(ValueType eps, BaseVector<int>* connections) const;
    virtual bool AMGAggregate(const BaseVector<int>& connections,
                              BaseVector<int>* aggregates) const;
    virtual bool AMGSmoothedAggregation(ValueType relax,
                                        const BaseVector<int>& aggregates,
                                        const BaseVector<int>& connections,
                                        BaseMatrix<ValueType>* prolong,
                                        BaseMatrix<ValueType>* restrict) const;
    virtual bool AMGAggregation(const BaseVector<int>& aggregates,
                                BaseMatrix<ValueType>* prolong,
                                BaseMatrix<ValueType>* restrict) const;

    virtual bool RugeStueben(ValueType eps,
                             BaseMatrix<ValueType>* prolong,
                             BaseMatrix<ValueType>* restrict) const;

    virtual bool FSAI(int power, const BaseMatrix<ValueType>* pattern);
    virtual bool SPAI(void);

    virtual bool InitialPairwiseAggregation(ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    virtual bool InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                            ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    virtual bool FurtherPairwiseAggregation(ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    virtual bool FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                            ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    virtual bool CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                 int nrow,
                                 int ncol,
                                 const BaseVector<int>& G,
                                 int Gsize,
                                 const int* rG,
                                 int rGsize) const;

    private:
    MatrixCSR<ValueType, int> mat_;

    friend class BaseVector<ValueType>;
    friend class HostVector<ValueType>;
    friend class HostMatrixCOO<ValueType>;
    friend class HostMatrixDIA<ValueType>;
    friend class HostMatrixELL<ValueType>;
    friend class HostMatrixHYB<ValueType>;
    friend class HostMatrixDENSE<ValueType>;
    friend class HostMatrixMCSR<ValueType>;
    friend class HostMatrixBCSR<ValueType>;

    friend class HIPAcceleratorMatrixCSR<ValueType>;

    bool L_diag_unit_;
    bool U_diag_unit_;
};

} // namespace rocalution

#endif // ROCALUTION_HOST_MATRIX_CSR_HPP_
