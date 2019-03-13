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

#ifndef ROCALUTION_BASE_MATRIX_HPP_
#define ROCALUTION_BASE_MATRIX_HPP_

#include "matrix_formats.hpp"
#include "backend_manager.hpp"

namespace rocalution {

template <typename ValueType>
class BaseVector;
template <typename ValueType>
class HostVector;
template <typename ValueType>
class HIPAcceleratorVector;

template <typename ValueType>
class HostMatrixCSR;
template <typename ValueType>
class HostMatrixCOO;
template <typename ValueType>
class HostMatrixDIA;
template <typename ValueType>
class HostMatrixELL;
template <typename ValueType>
class HostMatrixHYB;
template <typename ValueType>
class HostMatrixDENSE;
template <typename ValueType>
class HostMatrixMCSR;
template <typename ValueType>
class HostMatrixBCSR;

template <typename ValueType>
class HIPAcceleratorMatrixCSR;
template <typename ValueType>
class HIPAcceleratorMatrixMCSR;
template <typename ValueType>
class HIPAcceleratorMatrixBCSR;
template <typename ValueType>
class HIPAcceleratorMatrixCOO;
template <typename ValueType>
class HIPAcceleratorMatrixDIA;
template <typename ValueType>
class HIPAcceleratorMatrixELL;
template <typename ValueType>
class HIPAcceleratorMatrixHYB;
template <typename ValueType>
class HIPAcceleratorMatrixDENSE;

/// Base class for all host/accelerator matrices
template <typename ValueType>
class BaseMatrix
{
    public:
    BaseMatrix();
    virtual ~BaseMatrix();

    /// Return the number of rows in the matrix
    int GetM(void) const;
    /// Return the number of columns in the matrix
    int GetN(void) const;
    /// Return the non-zeros of the matrix
    int GetNnz(void) const;
    /// Shows simple info about the object
    virtual void Info(void) const = 0;
    /// Return the matrix format id (see matrix_formats.hpp)
    virtual unsigned int GetMatFormat(void) const = 0;
    /// Copy the backend descriptor information
    virtual void set_backend(const Rocalution_Backend_Descriptor local_backend);

    virtual bool Check(void) const;

    /// Allocate CSR Matrix
    virtual void AllocateCSR(int nnz, int nrow, int ncol);
    /// Allocate MCSR Matrix
    virtual void AllocateMCSR(int nnz, int nrow, int ncol);
    /// Allocate COO Matrix
    virtual void AllocateCOO(int nnz, int nrow, int ncol);
    /// Allocate DIA Matrix
    virtual void AllocateDIA(int nnz, int nrow, int ncol, int ndiag);
    /// Allocate ELL Matrix
    virtual void AllocateELL(int nnz, int nrow, int ncol, int max_row);
    /// Allocate HYB Matrix
    virtual void AllocateHYB(int ell_nnz, int coo_nnz, int ell_max_row, int nrow, int ncol);
    /// Allocate DENSE Matrix
    virtual void AllocateDENSE(int nrow, int ncol);

    /// Initialize a COO matrix on the Host with externally allocated data
    virtual void SetDataPtrCOO(int** row, int** col, ValueType** val, int nnz, int nrow, int ncol);
    /// Leave a COO matrix to Host pointers
    virtual void LeaveDataPtrCOO(int** row, int** col, ValueType** val);

    /// Initialize a CSR matrix on the Host with externally allocated data
    virtual void
    SetDataPtrCSR(int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol);
    /// Leave a CSR matrix to Host pointers
    virtual void LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val);

    /// Initialize a MCSR matrix on the Host with externally allocated data
    virtual void
    SetDataPtrMCSR(int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol);
    /// Leave a MCSR matrix to Host pointers
    virtual void LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val);

    /// Initialize an ELL matrix on the Host with externally allocated data
    virtual void
    SetDataPtrELL(int** col, ValueType** val, int nnz, int nrow, int ncol, int max_row);
    /// Leave an ELL matrix to Host pointers
    virtual void LeaveDataPtrELL(int** col, ValueType** val, int& max_row);

    /// Initialize a DIA matrix on the Host with externally allocated data
    virtual void
    SetDataPtrDIA(int** offset, ValueType** val, int nnz, int nrow, int ncol, int num_diag);
    /// Leave a DIA matrix to Host pointers
    virtual void LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag);

    /// Initialize a DENSE matrix on the Host with externally allocated data
    virtual void SetDataPtrDENSE(ValueType** val, int nrow, int ncol);
    /// Leave a DENSE matrix to Host pointers
    virtual void LeaveDataPtrDENSE(ValueType** val);

    /// Clear (free) the matrix
    virtual void Clear(void) = 0;

    /// Set all the values to zero
    virtual bool Zeros(void);

    /// Scale all values
    virtual bool Scale(ValueType alpha);
    /// Scale the diagonal entries of the matrix with alpha
    virtual bool ScaleDiagonal(ValueType alpha);
    /// Scale the off-diagonal entries of the matrix with alpha
    virtual bool ScaleOffDiagonal(ValueType alpha);
    /// Add alpha to all values
    virtual bool AddScalar(ValueType alpha);
    /// Add alpha to the diagonal entries of the matrix
    virtual bool AddScalarDiagonal(ValueType alpha);
    /// Add alpha to the off-diagonal entries of the matrix
    virtual bool AddScalarOffDiagonal(ValueType alpha);

    /// Extrat a sub-matrix with row/col_offset and row/col_size
    virtual bool ExtractSubMatrix(int row_offset,
                                  int col_offset,
                                  int row_size,
                                  int col_size,
                                  BaseMatrix<ValueType>* mat) const;

    /// Extract the diagonal values of the matrix into a LocalVector
    virtual bool ExtractDiagonal(BaseVector<ValueType>* vec_diag) const;
    /// Extract the inverse (reciprocal) diagonal values of the matrix into a LocalVector
    virtual bool ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const;
    /// Extract the upper triangular matrix
    virtual bool ExtractU(BaseMatrix<ValueType>* U) const;
    /// Extract the upper triangular matrix including diagonal
    virtual bool ExtractUDiagonal(BaseMatrix<ValueType>* U) const;
    /// Extract the lower triangular matrix
    virtual bool ExtractL(BaseMatrix<ValueType>* L) const;
    /// Extract the lower triangular matrix including diagonal
    virtual bool ExtractLDiagonal(BaseMatrix<ValueType>* L) const;

    /// Perform (forward) permutation of the matrix
    virtual bool Permute(const BaseVector<int>& permutation);

    /// Perform (backward) permutation of the matrix
    virtual bool PermuteBackward(const BaseVector<int>& permutation);

    /// Create permutation vector for CMK reordering of the matrix
    virtual bool CMK(BaseVector<int>* permutation) const;
    /// Create permutation vector for reverse CMK reordering of the matrix
    virtual bool RCMK(BaseVector<int>* permutation) const;
    /// Create permutation vector for connectivity reordering of the matrix (increasing nnz per row)
    virtual bool ConnectivityOrder(BaseVector<int>* permutation) const;

    /// Perform multi-coloring decomposition of the matrix; Returns number of
    /// colors, the corresponding sizes (the array is allocated in the function)
    /// and the permutation
    virtual bool
    MultiColoring(int& num_colors, int** size_colors, BaseVector<int>* permutation) const;

    /// Perform maximal independent set decomposition of the matrix; Returns the
    /// size of the maximal independent set and the corresponding permutation
    virtual bool MaximalIndependentSet(int& size, BaseVector<int>* permutation) const;

    /// Return a permutation for saddle-point problems (zero diagonal entries),
    /// where all zero diagonal elements are mapped to the last block;
    /// the return size is the size of the first block
    virtual bool ZeroBlockPermutation(int& size, BaseVector<int>* permutation) const;

    /// Convert the matrix from another matrix (with different structure)
    virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat) = 0;

    /// Copy from another matrix
    virtual void CopyFrom(const BaseMatrix<ValueType>& mat) = 0;

    /// Copy to another matrix
    virtual void CopyTo(BaseMatrix<ValueType>* mat) const = 0;

    /// Async copy from another matrix
    virtual void CopyFromAsync(const BaseMatrix<ValueType>& mat);

    /// Copy to another matrix
    virtual void CopyToAsync(BaseMatrix<ValueType>* mat) const;

    /// Copy from CSR array (the matrix has to be allocated)
    virtual void CopyFromCSR(const int* row_offsets, const int* col, const ValueType* val);

    /// Copy to CSR array (the arrays have to be allocated)
    virtual void CopyToCSR(int* row_offsets, int* col, ValueType* val) const;

    /// Copy from COO array (the matrix has to be allocated)
    virtual void CopyFromCOO(const int* row, const int* col, const ValueType* val);

    /// Copy to COO array (the arrays have to be allocated)
    virtual void CopyToCOO(int* row, int* col, ValueType* val) const;

    /// Allocates and copies a host CSR matrix
    virtual void CopyFromHostCSR(
        const int* row_offset, const int* col, const ValueType* val, int nnz, int nrow, int ncol);

    /// Create a restriction matrix operator based on an int vector map
    virtual bool CreateFromMap(const BaseVector<int>& map, int n, int m);
    /// Create a restriction and prolongation matrix operator based on an int vector map
    virtual bool
    CreateFromMap(const BaseVector<int>& map, int n, int m, BaseMatrix<ValueType>* pro);

    /// Read matrix from MTX (Matrix Market Format) file
    virtual bool ReadFileMTX(const std::string filename);
    /// Write matrix to MTX (Matrix Market Format) file
    virtual bool WriteFileMTX(const std::string filename) const;

    /// Read matrix from CSR (ROCALUTION binary format) file
    virtual bool ReadFileCSR(const std::string filename);
    /// Write matrix to CSR (ROCALUTION binary format) file
    virtual bool WriteFileCSR(const std::string filename) const;

    /// Perform symbolic computation (structure only) of |this|^p
    virtual bool SymbolicPower(int p);

    /// Perform symbolic matrix-matrix multiplication (i.e. determine the structure),
    /// this = this*src
    virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& src);
    /// Multiply two matrices, this = A * B
    virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
    /// Perform symbolic matrix-matrix multiplication (i.e. determine the structure),
    /// this = A*B
    virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
    /// Perform numerical matrix-matrix multiplication (i.e. value computation),
    /// this = A*B
    virtual bool NumericMatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
    /// Multiply the matrix with diagonal matrix (stored in LocalVector),
    /// this=this*diag (right multiplication)
    virtual bool DiagonalMatrixMultR(const BaseVector<ValueType>& diag);
    /// Multiply the matrix with diagonal matrix (stored in LocalVector),
    /// this=diag*this (left multiplication)
    virtual bool DiagonalMatrixMultL(const BaseVector<ValueType>& diag);
    /// Perform matrix addition, this = alpha*this + beta*mat;
    /// if structure==false the structure of the matrix is not changed,
    /// if structure==true new data structure is computed
    virtual bool
    MatrixAdd(const BaseMatrix<ValueType>& mat, ValueType alpha, ValueType beta, bool structure);

    /// Perform ILU(0) factorization
    virtual bool ILU0Factorize(void);
    /// Perform LU factorization
    virtual bool LUFactorize(void);
    /// Perform ILU(t,m) factorization based on threshold and maximum
    /// number of elements per row
    virtual bool ILUTFactorize(double t, int maxrow);
    /// Perform ILU(p) factorization based on power (see power(q)-pattern method, D. Lukarski
    ///  "Parallel Sparse Linear Algebra for Multi-core and Many-core Platforms - Parallel Solvers
    ///  and
    /// Preconditioners", PhD Thesis, 2012, KIT)
    virtual bool ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat);

    /// Perform IC(0) factorization
    virtual bool ICFactorize(BaseVector<ValueType>* inv_diag);

    /// Analyse the structure (level-scheduling)
    virtual void LUAnalyse(void);
    /// Delete the analysed data (see LUAnalyse)
    virtual void LUAnalyseClear(void);
    /// Solve LU out = in; if level-scheduling algorithm is provided then the graph
    /// traversing is performed in parallel
    virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    /// Analyse the structure (level-scheduling)
    virtual void LLAnalyse(void);
    /// Delete the analysed data (see LLAnalyse)
    virtual void LLAnalyseClear(void);
    /// Solve LL^T out = in; if level-scheduling algorithm is provided then the graph
    // traversing is performed in parallel
    virtual bool LLSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual bool LLSolve(const BaseVector<ValueType>& in,
                         const BaseVector<ValueType>& inv_diag,
                         BaseVector<ValueType>* out) const;

    /// Analyse the structure (level-scheduling) L-part
    /// diag_unit == true the diag is 1;
    /// diag_unit == false the diag is 0;
    virtual void LAnalyse(bool diag_unit = false);
    /// Delete the analysed data (see LAnalyse) L-party
    virtual void LAnalyseClear(void);
    /// Solve L out = in; if level-scheduling algorithm is provided then the
    /// graph traversing is performed in parallel
    virtual bool LSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    /// Analyse the structure (level-scheduling) U-part;
    /// diag_unit == true the diag is 1;
    /// diag_unit == false the diag is 0;
    virtual void UAnalyse(bool diag_unit = false);
    /// Delete the analysed data (see UAnalyse) U-party
    virtual void UAnalyseClear(void);
    /// Solve U out = in; if level-scheduling algorithm is provided then the
    /// graph traversing is performed in parallel
    virtual bool USolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    /// Compute Householder vector
    virtual bool Householder(int idx, ValueType& beta, BaseVector<ValueType>* vec) const;
    /// QR Decomposition
    virtual bool QRDecompose(void);
    /// Solve QR out = in
    virtual bool QRSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    /// Invert this
    virtual bool Invert(void);

    /// Compute the spectrum approximation with Gershgorin circles theorem
    virtual bool Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

    /// Apply the matrix to vector, out = this*in;
    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const = 0;
    /// Apply and add the matrix to vector, out = out + scalar*this*in;
    virtual void ApplyAdd(const BaseVector<ValueType>& in,
                          ValueType scalar,
                          BaseVector<ValueType>* out) const = 0;

    /// Delete all entries abs(a_ij) <= drop_off;
    /// the diagonal elements are never deleted
    virtual bool Compress(double drop_off);

    /// Transpose the matrix
    virtual bool Transpose(void);

    /// Sort the matrix indices
    virtual bool Sort(void);

    // Return key for row, col and val
    virtual bool Key(long int& row_key, long int& col_key, long int& val_key) const;

    /// Replace a column vector of a matrix
    virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);

    /// Replace a column vector of a matrix
    virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);

    /// Extract values from a column of a matrix to a vector
    virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;

    /// Extract values from a row of a matrix to a vector
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

    /// Ruge Stüben coarsening
    virtual bool RugeStueben(ValueType eps,
                             BaseMatrix<ValueType>* prolong,
                             BaseMatrix<ValueType>* restrict) const;

    /// Factorized Sparse Approximate Inverse assembly for given system
    /// matrix power pattern or external sparsity pattern
    virtual bool FSAI(int power, const BaseMatrix<ValueType>* pattern);

    /// SParse Approximate Inverse assembly for given system matrix pattern
    virtual bool SPAI(void);

    /// Initial Pairwise Aggregation scheme
    virtual bool InitialPairwiseAggregation(ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    /// Initial Pairwise Aggregation scheme for split matrices
    virtual bool InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                            ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    /// Further Pairwise Aggregation scheme
    virtual bool FurtherPairwiseAggregation(ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    /// Further Pairwise Aggregation scheme for split matrices
    virtual bool FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                            ValueType beta,
                                            int& nc,
                                            BaseVector<int>* G,
                                            int& Gsize,
                                            int** rG,
                                            int& rGsize,
                                            int ordering) const;
    /// Build coarse operator for pairwise aggregation scheme
    virtual bool CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                 int nrow,
                                 int ncol,
                                 const BaseVector<int>& G,
                                 int Gsize,
                                 const int* rG,
                                 int rGsize) const;

    protected:
    /// Number of rows
    int nrow_;
    /// Number of columns
    int ncol_;
    /// Number of non-zero elements
    int nnz_;

    /// Backend descriptor (local copy)
    Rocalution_Backend_Descriptor local_backend_;

    friend class BaseVector<ValueType>;
    friend class HostVector<ValueType>;
    friend class AcceleratorVector<ValueType>;
    friend class HIPAcceleratorVector<ValueType>;
};

template <typename ValueType>
class HostMatrix : public BaseMatrix<ValueType>
{
    public:
    HostMatrix();
    virtual ~HostMatrix();
};

template <typename ValueType>
class AcceleratorMatrix : public BaseMatrix<ValueType>
{
    public:
    AcceleratorMatrix();
    virtual ~AcceleratorMatrix();

    /// Copy (accelerator matrix) from host matrix
    virtual void CopyFromHost(const HostMatrix<ValueType>& src) = 0;

    /// Async copy (accelerator matrix) from host matrix
    virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);

    /// Copy (accelerator matrix) to host matrix
    virtual void CopyToHost(HostMatrix<ValueType>* dst) const = 0;

    /// Async opy (accelerator matrix) to host matrix
    virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;
};

template <typename ValueType>
class HIPAcceleratorMatrix : public AcceleratorMatrix<ValueType>
{
    public:
    HIPAcceleratorMatrix();
    virtual ~HIPAcceleratorMatrix();
};

} // namespace rocalution

#endif // ROCALUTION_BASE_MATRIX_HPP_
