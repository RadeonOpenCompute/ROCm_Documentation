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

#include <mpi.h>
#include <rocalution.hpp>

using namespace rocalution;

template <typename ValueType>
void distribute_matrix(const MPI_Comm* comm,
                       LocalMatrix<ValueType>* lmat,
                       GlobalMatrix<ValueType>* gmat,
                       ParallelManager* pm)
{
    int rank;
    int num_procs;

    MPI_Comm_rank(*comm, &rank);
    MPI_Comm_size(*comm, &num_procs);

    size_t global_nrow = lmat->GetM();

    int* global_row_offset = NULL;
    int* global_col        = NULL;
    ValueType* global_val  = NULL;

    lmat->LeaveDataPtrCSR(&global_row_offset, &global_col, &global_val);

    // Compute local matrix sizes
    std::vector<int> local_size(num_procs);

    for(int i = 0; i < num_procs; ++i)
    {
        local_size[i] = global_nrow / num_procs;
    }

    if(global_nrow % num_procs != 0)
    {
        for(size_t i = 0; i < global_nrow % num_procs; ++i)
        {
            ++local_size[i];
        }
    }

    // Compute index offsets
    std::vector<int> index_offset(num_procs + 1);
    index_offset[0] = 0;
    for(int i = 0; i < num_procs; ++i)
    {
        index_offset[i + 1] = index_offset[i] + local_size[i];
    }

    // Read sub matrix - row_offset
    int local_nrow = local_size[rank];
    std::vector<int> local_row_offset(local_nrow + 1);

    for(int i = index_offset[rank], k = 0; k < local_nrow + 1; ++i, ++k)
    {
        local_row_offset[k] = global_row_offset[i];
    }

    free_host(&global_row_offset);

    // Read sub matrix - col and val
    int local_nnz = local_row_offset[local_nrow] - local_row_offset[0];
    std::vector<int> local_col(local_nnz);
    std::vector<ValueType> local_val(local_nnz);

    for(int i = local_row_offset[0], k = 0; k < local_nnz; ++i, ++k)
    {
        local_col[k] = global_col[i];
        local_val[k] = global_val[i];
    }

    free_host(&global_col);
    free_host(&global_val);

    // Shift row_offset entries
    int shift = local_row_offset[0];
    for(int i = 0; i < local_nrow + 1; ++i)
    {
        local_row_offset[i] -= shift;
    }

    int interior_nnz = 0;
    int ghost_nnz    = 0;
    int boundary_nnz = 0;
    int neighbors    = 0;

    std::vector<std::vector<int>> boundary(num_procs, std::vector<int>());
    std::vector<bool> neighbor(num_procs, false);
    std::vector<std::map<int, bool>> checked(num_procs, std::map<int, bool>());

    for(int i = 0; i < local_nrow; ++i)
    {
        for(int j = local_row_offset[i]; j < local_row_offset[i + 1]; ++j)
        {

            // Interior point
            if(local_col[j] >= index_offset[rank] && local_col[j] < index_offset[rank + 1])
            {
                ++interior_nnz;
            }
            else
            {
                // Boundary point above current process
                if(local_col[j] < index_offset[rank])
                {
                    // Loop over ranks above current process
                    for(int r = rank - 1; r >= 0; --r)
                    {
                        // Check if boundary belongs to rank r
                        if(local_col[j] >= index_offset[r] && local_col[j] < index_offset[r + 1])
                        {
                            // Add boundary point to rank r if it has not been added yet
                            if(!checked[r][i + index_offset[rank]])
                            {
                                boundary[r].push_back(i + index_offset[rank]);
                                neighbor[r] = true;
                                ++boundary_nnz;
                                checked[r][i + index_offset[rank]] = true;
                            }
                            ++ghost_nnz;
                            // Rank for current boundary point local_col[j] has been found
                            // Continue with next boundary point
                            break;
                        }
                    }
                }

                // boundary point below current process
                if(local_col[j] >= index_offset[rank + 1])
                {
                    // Loop over ranks above current process
                    for(int r = rank + 1; r < num_procs; ++r)
                    {
                        // Check if boundary belongs to rank r
                        if(local_col[j] >= index_offset[r] && local_col[j] < index_offset[r + 1])
                        {
                            // Add boundary point to rank r if it has not been added yet
                            if(!checked[r][i + index_offset[rank]])
                            {
                                boundary[r].push_back(i + index_offset[rank]);
                                neighbor[r] = true;
                                ++boundary_nnz;
                                checked[r][i + index_offset[rank]] = true;
                            }
                            ++ghost_nnz;
                            // Rank for current boundary point local_col[j] has been found
                            // Continue with next boundary point
                            break;
                        }
                    }
                }
            }
        }
    }

    for(int i = 0; i < num_procs; ++i)
    {
        if(neighbor[i] == true)
        {
            ++neighbors;
        }
    }

    std::vector<MPI_Request> mpi_req(neighbors * 2);
    int n = 0;
    // Array to hold boundary size for each interface
    std::vector<int> boundary_size(neighbors);

    // MPI receive boundary sizes
    for(int i = 0; i < num_procs; ++i)
    {
        // If neighbor receive from rank i is expected...
        if(neighbor[i] == true)
        {
            // Receive size of boundary from rank i to current rank
            MPI_Irecv(&(boundary_size[n]), 1, MPI_INT, i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }

    // MPI send boundary sizes
    for(int i = 0; i < num_procs; ++i)
    {
        // Send required if boundary for rank i available
        if(boundary[i].size() > 0)
        {
            int size = boundary[i].size();
            // Send size of boundary from current rank to rank i
            MPI_Isend(&size, 1, MPI_INT, i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }
    // Wait to finish communication
    MPI_Waitall(n - 1, &(mpi_req[0]), MPI_STATUSES_IGNORE);

    n = 0;
    // Array to hold boundary offset for each interface
    int k = 0;
    std::vector<int> recv_offset(neighbors + 1);
    std::vector<int> send_offset(neighbors + 1);
    recv_offset[0] = 0;
    send_offset[0] = 0;
    for(int i = 0; i < neighbors; ++i)
    {
        recv_offset[i + 1] = recv_offset[i] + boundary_size[i];
    }

    for(int i = 0; i < num_procs; ++i)
    {
        if(neighbor[i] == true)
        {
            send_offset[k + 1] = send_offset[k] + boundary[i].size();
            ++k;
        }
    }

    // Array to hold boundary for each interface
    std::vector<std::vector<int>> local_boundary(neighbors);
    for(int i = 0; i < neighbors; ++i)
    {
        local_boundary[i].resize(boundary_size[i]);
    }

    // MPI receive boundary
    for(int i = 0; i < num_procs; ++i)
    {
        // If neighbor receive from rank i is expected...
        if(neighbor[i] == true)
        {
            // Receive boundary from rank i to current rank
            MPI_Irecv(
                local_boundary[n].data(), boundary_size[n], MPI_INT, i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }

    // MPI send boundary
    for(int i = 0; i < num_procs; ++i)
    {
        // Send required if boundary for rank i is available
        if(boundary[i].size() > 0)
        {
            // Send boundary from current rank to rank i
            MPI_Isend(&(boundary[i][0]), boundary[i].size(), MPI_INT, i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }

    // Wait to finish communication
    MPI_Waitall(n - 1, &(mpi_req[0]), MPI_STATUSES_IGNORE);

    // Total boundary size
    int nnz_boundary = 0;
    for(int i = 0; i < neighbors; ++i)
    {
        nnz_boundary += boundary_size[i];
    }

    // Create local boundary index array
    k = 0;
    std::vector<int> bnd(boundary_nnz);

    for(int i = 0; i < num_procs; ++i)
    {
        for(unsigned int j = 0; j < boundary[i].size(); ++j)
        {
            bnd[k] = boundary[i][j] - index_offset[rank];
            ++k;
        }
    }

    // Create boundary index array
    std::vector<int> boundary_index(nnz_boundary);

    k = 0;
    for(int i = 0; i < neighbors; ++i)
    {
        for(int j = 0; j < boundary_size[i]; ++j)
        {
            boundary_index[k] = local_boundary[i][j];
            ++k;
        }
    }

    // Create map with boundary index relations
    std::map<int, int> boundary_map;

    for(int i = 0; i < nnz_boundary; ++i)
    {
        boundary_map[boundary_index[i]] = i;
    }

    // Build up ghost and interior matrix
    int* ghost_row       = new int[ghost_nnz];
    int* ghost_col       = new int[ghost_nnz];
    ValueType* ghost_val = new ValueType[ghost_nnz];

    memset(ghost_row, 0, sizeof(int) * ghost_nnz);
    memset(ghost_col, 0, sizeof(int) * ghost_nnz);
    memset(ghost_val, 0, sizeof(ValueType) * ghost_nnz);

    int* row_offset = new int[local_nrow + 1];
    int* col        = new int[interior_nnz];
    ValueType* val  = new ValueType[interior_nnz];

    memset(row_offset, 0, sizeof(int) * (local_nrow + 1));
    memset(col, 0, sizeof(int) * interior_nnz);
    memset(val, 0, sizeof(ValueType) * interior_nnz);

    row_offset[0] = 0;
    k             = 0;
    int l         = 0;
    for(int i = 0; i < local_nrow; ++i)
    {
        for(int j = local_row_offset[i]; j < local_row_offset[i + 1]; ++j)
        {

            // Boundary point -- create ghost part
            if(local_col[j] < index_offset[rank] || local_col[j] >= index_offset[rank + 1])
            {
                ghost_row[k] = i;
                ghost_col[k] = boundary_map[local_col[j]];
                ghost_val[k] = local_val[j];
                ++k;
            }
            else
            {
                // Interior point -- create interior part
                int c = local_col[j] - index_offset[rank];

                col[l] = c;
                val[l] = local_val[j];
                ++l;
            }
        }
        row_offset[i + 1] = l;
    }

    std::vector<int> recv(neighbors);
    std::vector<int> sender(neighbors);

    int nbc = 0;
    for(int i = 0; i < num_procs; ++i)
    {
        if(neighbor[i] == true)
        {
            recv[nbc]   = i;
            sender[nbc] = i;
            ++nbc;
        }
    }

    pm->SetMPICommunicator(comm);
    pm->SetGlobalSize(global_nrow);
    pm->SetLocalSize(local_size[rank]);
    pm->SetBoundaryIndex(boundary_nnz, bnd.data());
    pm->SetReceivers(neighbors, recv.data(), recv_offset.data());
    pm->SetSenders(neighbors, sender.data(), send_offset.data());

    gmat->SetParallelManager(*pm);
    gmat->SetLocalDataPtrCSR(&row_offset, &col, &val, "mat", interior_nnz);
    gmat->SetGhostDataPtrCOO(&ghost_row, &ghost_col, &ghost_val, "ghost", ghost_nnz);
}
