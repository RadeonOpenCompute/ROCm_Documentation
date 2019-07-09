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

#ifndef ROCALUTION_PARALLEL_MANAGER_HPP_
#define ROCALUTION_PARALLEL_MANAGER_HPP_

#include "base_rocalution.hpp"
#include "../utils/types.hpp"

#include <string>
#include <complex>

namespace rocalution {

template <typename ValueType>
class GlobalMatrix;
template <typename ValueType>
class GlobalVector;

/** \ingroup backend_module
  * \brief Parallel Manager class
  * \details
  * The parallel manager class handles the communication and the mapping of the global
  * operators. Each global operator and vector need to be initialized with a valid
  * parallel manager in order to perform any operation. For many distributed simulations,
  * the underlying operator is already distributed. This information need to be passed to
  * the parallel manager.
  */
class ParallelManager : public RocalutionObj
{
    public:
    ParallelManager();
    ~ParallelManager();

    /** \brief Set the MPI communicator */
    void SetMPICommunicator(const void* comm);
    /** \brief Clear all allocated resources */
    void Clear(void);

    /** \brief Return the global size */
    IndexType2 GetGlobalSize(void) const;
    /** \brief Return the local size */
    int GetLocalSize(void) const;

    /** \brief Return the number of receivers */
    int GetNumReceivers(void) const;
    /** \brief Return the number of senders */
    int GetNumSenders(void) const;
    /** \brief Return the number of involved processes */
    int GetNumProcs(void) const;

    /** \brief Initialize the global size */
    void SetGlobalSize(IndexType2 size);
    /** \brief Initialize the local size */
    void SetLocalSize(int size);

    /** \brief Set all boundary indices of this ranks process */
    void SetBoundaryIndex(int size, const int* index);

    /** \brief Number of processes, the current process is receiving data from, array of
      * the processes, the current process is receiving data from and offsets, where the
      * boundary for process 'receiver' starts
      */
    void SetReceivers(int nrecv, const int* recvs, const int* recv_offset);

    /** \brief Number of processes, the current process is sending data to, array of the
      * processes, the current process is sending data to and offsets where the ghost
      * part for process 'sender' starts
      */
    void SetSenders(int nsend, const int* sends, const int* send_offset);

    /** \brief Mapping local to global */
    void LocalToGlobal(int proc, int local, int& global);
    /** \brief Mapping global to local */
    void GlobalToLocal(int global, int& proc, int& local);

    /** \brief Check sanity status of parallel manager */
    bool Status(void) const;

    /** \brief Read file that contains all relevant parallel manager data */
    void ReadFileASCII(const std::string filename);
    /** \brief Write file that contains all relevant parallel manager data */
    void WriteFileASCII(const std::string filename) const;

    private:
    const void* comm_;
    int rank_;
    int num_procs_;

    IndexType2 global_size_;
    int local_size_;

    // Number of total ids, the current process is receiving
    int recv_index_size_;
    // Number of total ids, the current process is sending
    int send_index_size_;

    // Number of processes, the current process receives data from
    int nrecv_;
    // Number of processes, the current process sends data to
    int nsend_;

    // Array of process ids, the current process receives data from
    int* recvs_;
    // Array of process ids, the current process sends data to
    int* sends_;

    // Array of offsets, the current process receives data from
    int* recv_offset_index_;
    // Array of offsets, the current process sends data to
    int* send_offset_index_;

    // Boundary index ids
    int* boundary_index_;

    friend class GlobalMatrix<double>;
    friend class GlobalMatrix<float>;
    friend class GlobalMatrix<std::complex<double>>;
    friend class GlobalMatrix<std::complex<float>>;
    friend class GlobalVector<double>;
    friend class GlobalVector<float>;
    friend class GlobalVector<std::complex<double>>;
    friend class GlobalVector<std::complex<float>>;
    friend class GlobalVector<int>;
};

} // namespace rocalution

#endif // ROCALUTION_PARALLEL_MANAGER_HPP_
