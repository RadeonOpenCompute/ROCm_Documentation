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

#ifndef ROCALUTION_GLOBAL_MATRIX_HPP_
#define ROCALUTION_GLOBAL_MATRIX_HPP_

#include "../utils/types.hpp"
#include "operator.hpp"
#include "parallel_manager.hpp"

namespace rocalution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class LocalVector;
template <typename ValueType>
class LocalMatrix;

/** \ingroup op_vec_module
  * \class GlobalMatrix
  * \brief GlobalMatrix class
  * \details
  * A GlobalMatrix is called global, because it can stay on a single or on multiple nodes
  * in a network. For this type of communication, MPI is used.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
template <typename ValueType>
class GlobalMatrix : public Operator<ValueType>
{
    public:
    GlobalMatrix();
    /** \brief Initialize a global matrix with a parallel manager */
    GlobalMatrix(const ParallelManager& pm);
    virtual ~GlobalMatrix();

    virtual IndexType2 GetM(void) const;
    virtual IndexType2 GetN(void) const;
    virtual IndexType2 GetNnz(void) const;
    virtual int GetLocalM(void) const;
    virtual int GetLocalN(void) const;
    virtual int GetLocalNnz(void) const;
    virtual int GetGhostM(void) const;
    virtual int GetGhostN(void) const;
    virtual int GetGhostNnz(void) const;

    /** \private */
    const LocalMatrix<ValueType>& GetInterior() const;
    /** \private */
    const LocalMatrix<ValueType>& GetGhost() const;

    virtual void MoveToAccelerator(void);
    virtual void MoveToHost(void);

    virtual void Info(void) const;

    /** \brief Return true if the matrix is ok (empty matrix is also ok) and false if
      * there is something wrong with the strcture or some of values are NaN
      */
    virtual bool Check(void) const;

    /** \brief Allocate CSR Matrix */
    void AllocateCSR(std::string name, int local_nnz, int ghost_nnz);
    /** \brief Allocate COO Matrix */
    void AllocateCOO(std::string name, int local_nnz, int ghost_nnz);

    virtual void Clear(void);

    /** \brief Set the parallel manager of a global vector */
    void SetParallelManager(const ParallelManager& pm);

    /** \brief Initialize a CSR matrix on the host with externally allocated data */
    void SetDataPtrCSR(int** local_row_offset,
                       int** local_col,
                       ValueType** local_val,
                       int** ghost_row_offset,
                       int** ghost_col,
                       ValueType** ghost_val,
                       std::string name,
                       int local_nnz,
                       int ghost_nnz);
    /** \brief Initialize a COO matrix on the host with externally allocated data */
    void SetDataPtrCOO(int** local_row,
                       int** local_col,
                       ValueType** local_val,
                       int** ghost_row,
                       int** ghost_col,
                       ValueType** ghost_val,
                       std::string name,
                       int local_nnz,
                       int ghost_nnz);

    /** \brief Initialize a CSR matrix on the host with externally allocated local data */
    void
    SetLocalDataPtrCSR(int** row_offset, int** col, ValueType** val, std::string name, int nnz);
    /** \brief Initialize a COO matrix on the host with externally allocated local data */
    void SetLocalDataPtrCOO(int** row, int** col, ValueType** val, std::string name, int nnz);

    /** \brief Initialize a CSR matrix on the host with externally allocated ghost data */
    void
    SetGhostDataPtrCSR(int** row_offset, int** col, ValueType** val, std::string name, int nnz);
    /** \brief Initialize a COO matrix on the host with externally allocated ghost data */
    void SetGhostDataPtrCOO(int** row, int** col, ValueType** val, std::string name, int nnz);

    /** \brief Leave a CSR matrix to host pointers */
    void LeaveDataPtrCSR(int** local_row_offset,
                         int** local_col,
                         ValueType** local_val,
                         int** ghost_row_offset,
                         int** ghost_col,
                         ValueType** ghost_val);
    /** \brief Leave a COO matrix to host pointers */
    void LeaveDataPtrCOO(int** local_row,
                         int** local_col,
                         ValueType** local_val,
                         int** ghost_row,
                         int** ghost_col,
                         ValueType** ghost_val);
    /** \brief Leave a local CSR matrix to host pointers */
    void LeaveLocalDataPtrCSR(int** row_offset, int** col, ValueType** val);
    /** \brief Leave a local COO matrix to host pointers */
    void LeaveLocalDataPtrCOO(int** row, int** col, ValueType** val);
    /** \brief Leave a CSR ghost matrix to host pointers */
    void LeaveGhostDataPtrCSR(int** row_offset, int** col, ValueType** val);
    /** \brief Leave a COO ghost matrix to host pointers */
    void LeaveGhostDataPtrCOO(int** row, int** col, ValueType** val);

    /** \brief Clone the entire matrix (values,structure+backend descr) from another
      * GlobalMatrix
      */
    void CloneFrom(const GlobalMatrix<ValueType>& src);
    /** \brief Copy matrix (values and structure) from another GlobalMatrix */
    void CopyFrom(const GlobalMatrix<ValueType>& src);

    /** \brief Convert the matrix to CSR structure */
    void ConvertToCSR(void);
    /** \brief Convert the matrix to MCSR structure */
    void ConvertToMCSR(void);
    /** \brief Convert the matrix to BCSR structure */
    void ConvertToBCSR(void);
    /** \brief Convert the matrix to COO structure */
    void ConvertToCOO(void);
    /** \brief Convert the matrix to ELL structure */
    void ConvertToELL(void);
    /** \brief Convert the matrix to DIA structure */
    void ConvertToDIA(void);
    /** \brief Convert the matrix to HYB structure */
    void ConvertToHYB(void);
    /** \brief Convert the matrix to DENSE structure */
    void ConvertToDENSE(void);
    /** \brief Convert the matrix to specified matrix ID format */
    void ConvertTo(unsigned int matrix_format);

    virtual void Apply(const GlobalVector<ValueType>& in, GlobalVector<ValueType>* out) const;
    virtual void ApplyAdd(const GlobalVector<ValueType>& in,
                          ValueType scalar,
                          GlobalVector<ValueType>* out) const;

    /** \brief Read matrix from MTX (Matrix Market Format) file */
    void ReadFileMTX(const std::string filename);
    /** \brief Write matrix to MTX (Matrix Market Format) file */
    void WriteFileMTX(const std::string filename) const;
    /** \brief Read matrix from CSR (ROCALUTION binary format) file */
    void ReadFileCSR(const std::string filename);
    /** \brief Write matrix to CSR (ROCALUTION binary format) file */
    void WriteFileCSR(const std::string filename) const;

    /** \brief Sort the matrix indices */
    void Sort(void);

    /** \brief Extract the inverse (reciprocal) diagonal values of the matrix into a
      * GlobalVector
      */
    void ExtractInverseDiagonal(GlobalVector<ValueType>* vec_inv_diag) const;

    /** \brief Scale all the values in the matrix */
    void Scale(ValueType alpha);

    /** \brief Initial Pairwise Aggregation scheme */
    void InitialPairwiseAggregation(ValueType beta,
                                    int& nc,
                                    LocalVector<int>* G,
                                    int& Gsize,
                                    int** rG,
                                    int& rGsize,
                                    int ordering) const;
    /** \brief Further Pairwise Aggregation scheme */
    void FurtherPairwiseAggregation(ValueType beta,
                                    int& nc,
                                    LocalVector<int>* G,
                                    int& Gsize,
                                    int** rG,
                                    int& rGsize,
                                    int ordering) const;
    /** \brief Build coarse operator for pairwise aggregation scheme */
    void CoarsenOperator(GlobalMatrix<ValueType>* Ac,
                         ParallelManager* pm,
                         int nrow,
                         int ncol,
                         const LocalVector<int>& G,
                         int Gsize,
                         const int* rG,
                         int rGsize) const;

    protected:
    virtual bool is_host_(void) const;
    virtual bool is_accel_(void) const;

    private:
    IndexType2 nnz_;

    LocalMatrix<ValueType> matrix_interior_;
    LocalMatrix<ValueType> matrix_ghost_;

    friend class GlobalVector<ValueType>;
    friend class LocalMatrix<ValueType>;
    friend class LocalVector<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_GLOBAL_MATRIX_HPP_
