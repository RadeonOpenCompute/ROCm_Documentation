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
#include "base_rocalution.hpp"
#include "parallel_manager.hpp"

#include "../utils/log.hpp"
#include "../utils/allocate_free.hpp"

#include <vector>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

#ifdef SUPPORT_MULTINODE
#include <mpi.h>
#endif

namespace rocalution {

ParallelManager::ParallelManager()
{
#ifndef SUPPORT_MULTINODE
    LOG_INFO("Multinode support disabled");
    FATAL_ERROR(__FILE__, __LINE__);
#endif

    this->comm_      = NULL;
    this->rank_      = -1;
    this->num_procs_ = -1;

    this->global_size_ = 0;
    this->local_size_  = 0;

    this->recv_index_size_ = 0;
    this->send_index_size_ = 0;

    this->nrecv_ = 0;
    this->nsend_ = 0;

    this->recvs_ = NULL;
    this->sends_ = NULL;

    this->recv_offset_index_ = NULL;
    this->send_offset_index_ = NULL;

    this->boundary_index_ = NULL;

    // if new values are added, also put check into status function
}

ParallelManager::~ParallelManager() { this->Clear(); }

void ParallelManager::SetMPICommunicator(const void* comm)
{
    assert(comm != NULL);
    this->comm_ = comm;

#ifdef SUPPORT_MULTINODE
    MPI_Comm_rank(*(MPI_Comm*)this->comm_, &this->rank_);
    MPI_Comm_size(*(MPI_Comm*)this->comm_, &this->num_procs_);
#endif
}

void ParallelManager::Clear(void)
{
    this->global_size_ = 0;
    this->local_size_  = 0;

    if(this->nrecv_ > 0)
    {
        free_host(&this->recvs_);
        free_host(&this->recv_offset_index_);

        this->nrecv_ = 0;
    }

    if(this->nsend_ > 0)
    {
        free_host(&this->sends_);
        free_host(&this->send_offset_index_);

        this->nsend_ = 0;
    }

    if(this->recv_index_size_ > 0)
    {
        free_host(&this->boundary_index_);

        this->recv_index_size_ = 0;
    }

    if(this->send_index_size_ > 0)
    {
        this->send_index_size_ = 0;
    }
}

int ParallelManager::GetNumProcs(void) const
{
    assert(this->Status());

    return this->num_procs_;
}

void ParallelManager::SetGlobalSize(IndexType2 size)
{
    assert(size <= std::numeric_limits<IndexType2>::max());
    assert(size > 0);
    assert(size >= (IndexType2) this->local_size_);

    this->global_size_ = size;
}

void ParallelManager::SetLocalSize(int size)
{
    assert(size > 0);
    assert(size <= (IndexType2) this->global_size_);

    this->local_size_ = size;
}

IndexType2 ParallelManager::GetGlobalSize(void) const
{
    assert(this->Status());

    return this->global_size_;
}

int ParallelManager::GetLocalSize(void) const
{
    assert(this->Status());

    return this->local_size_;
}

int ParallelManager::GetNumReceivers(void) const
{
    assert(this->Status());

    return this->recv_index_size_;
}

int ParallelManager::GetNumSenders(void) const
{
    assert(this->Status());

    return this->send_index_size_;
}

void ParallelManager::SetBoundaryIndex(int size, const int* index)
{
    assert(size >= 0);
    assert(index != NULL);

    if(this->send_index_size_ != 0)
    {
        assert(this->send_index_size_ == size);
    }
    else
    {
        this->send_index_size_ = size;
    }

    allocate_host(size, &this->boundary_index_);

    for(int i = 0; i < size; ++i)
    {
        this->boundary_index_[i] = index[i];
    }
}

void ParallelManager::SetReceivers(int nrecv, const int* recvs, const int* recv_offset)
{
    assert(nrecv > 0);
    assert(recvs != NULL);
    assert(recv_offset != NULL);

    this->nrecv_ = nrecv;

    allocate_host(nrecv, &this->recvs_);
    allocate_host(nrecv + 1, &this->recv_offset_index_);

    this->recv_offset_index_[0] = 0;

    for(int i = 0; i < nrecv; ++i)
    {
        this->recvs_[i]                 = recvs[i];
        this->recv_offset_index_[i + 1] = recv_offset[i + 1];
    }

    this->recv_index_size_ = recv_offset[nrecv];
}

void ParallelManager::SetSenders(int nsend, const int* sends, const int* send_offset)
{
    assert(nsend > 0);
    assert(sends != NULL);
    assert(send_offset != NULL);

    this->nsend_ = nsend;

    allocate_host(nsend, &this->sends_);
    allocate_host(nsend + 1, &this->send_offset_index_);

    this->send_offset_index_[0] = 0;

    for(int i = 0; i < nsend; ++i)
    {
        this->sends_[i]                 = sends[i];
        this->send_offset_index_[i + 1] = send_offset[i + 1];
    }

    if(this->send_index_size_ != 0)
    {
        assert(this->send_index_size_ == send_offset[nsend]);
    }
    else
    {
        this->send_index_size_ = send_offset[nsend];
    }
}

bool ParallelManager::Status(void) const
{
    // clang-format off
    if(this->comm_ == NULL) return false;
    if(this->rank_ < 0) return false;
    if(this->global_size_ == 0) return false;
    if(this->local_size_ < 0) return false;
    if(this->nrecv_ < 0) return false;
    if(this->nsend_ < 0) return false;
    if(this->nrecv_ > 0 && this->recvs_ == NULL) return false;
    if(this->nsend_ > 0 && this->sends_ == NULL) return false;
    if(this->nrecv_ > 0 && this->recv_offset_index_ == NULL) return false;
    if(this->nsend_ > 0 && this->send_offset_index_ == NULL) return false;
    if(this->recv_index_size_ < 0) return false;
    if(this->send_index_size_ < 0) return false;
    if(this->recv_index_size_ > 0 && this->boundary_index_ == NULL) return false;
    // clang-format on

    return true;
}

void ParallelManager::WriteFileASCII(const std::string filename) const
{
    log_debug(this, "ParallelManager::WriteFileASCII()", filename);

    assert(this->Status());

    // Master rank writes the global headfile
    if(this->rank_ == 0)
    {
        std::ofstream headfile;

        LOG_INFO("WriteFileASCII: filename=" << filename << "; writing...");

        headfile.open((char*)filename.c_str(), std::ofstream::out);
        if(!headfile.is_open())
        {
            LOG_INFO("Cannot open ParallelManager file [write]: " << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        for(int i = 0; i < this->num_procs_; ++i)
        {
            std::ostringstream rs;
            rs << i;

            std::string name = filename + ".rank." + rs.str();

            headfile << name << "\n";
        }
    }

    std::ostringstream rs;
    rs << this->rank_;

    std::string name = filename + ".rank." + rs.str();
    std::ofstream file;

    file.open((char*)name.c_str(), std::ifstream::out);

    if(!file.is_open())
    {
        LOG_INFO("Cannot open ParallelManager file [write]:" << name);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "%% ROCALUTION MPI ParallelManager output %%" << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#RANK\n" << this->rank_ << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#GLOBAL_SIZE\n" << this->global_size_ << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#LOCAL_SIZE\n" << this->local_size_ << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#BOUNDARY_SIZE\n" << this->send_index_size_ << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#NUMBER_OF_RECEIVERS\n" << this->nrecv_ << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#NUMBER_OF_SENDERS\n" << this->nsend_ << std::endl;
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#RECEIVERS_RANK" << std::endl;
    for(int i = 0; i < this->nrecv_; ++i)
    {
        file << this->recvs_[i] << std::endl;
    }
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#SENDERS_RANK" << std::endl;
    for(int i = 0; i < this->nsend_; ++i)
    {
        file << this->sends_[i] << std::endl;
    }
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#RECEIVERS_INDEX_OFFSET" << std::endl;
    for(int i = 0; i < this->nrecv_ + 1; ++i)
    {
        file << this->recv_offset_index_[i] << std::endl;
    }
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#SENDERS_INDEX_OFFSET" << std::endl;
    for(int i = 0; i < this->nsend_ + 1; ++i)
    {
        file << this->send_offset_index_[i] << std::endl;
    }
    file << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    file << "#BOUNDARY_INDEX" << std::endl;
    for(int i = 0; i < this->send_index_size_; ++i)
    {
        file << this->boundary_index_[i] << std::endl;
    }

    file.close();

    LOG_INFO("WriteFileASCII: filename=" << name << "; done");
}

void ParallelManager::ReadFileASCII(const std::string filename)
{
    log_debug(this, "ParallelManager::ReadFileASCII()", filename);

    assert(this->comm_ != NULL);

    // Read header file
    std::ifstream headfile;

    LOG_INFO("ReadFileASCII: filename=" << filename << "; reading...");

    headfile.open((char*)filename.c_str(), std::ifstream::in);
    if(!headfile.is_open())
    {
        LOG_INFO("Cannot open ParallelManager file [read]: " << filename);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // Go to this ranks line in the headfile
    for(int i = 0; i < this->rank_; ++i)
    {
        headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string name;
    std::getline(headfile, name);

    headfile.close();

    // Extract directory containing the subfiles
    size_t found     = filename.find_last_of("\\/");
    std::string path = filename.substr(0, found + 1);

    name.erase(remove_if(name.begin(), name.end(), isspace), name.end());
    name = path + name;

    // Open the ranks corresponding file
    std::ifstream file;
    std::string line;

    file.open(name.c_str(), std::ifstream::in);
    if(!file.is_open())
    {
        LOG_INFO("Cannot open ParallelManager file [read]: " << name);
        FATAL_ERROR(__FILE__, __LINE__);
    }

    this->Clear();
    int rank = -1;

    while(!file.eof())
    {
        std::getline(file, line);

        if(line.find("#RANK") != std::string::npos)
        {
            file >> rank;
        }
        if(line.find("#GLOBAL_SIZE") != std::string::npos)
        {
            file >> this->global_size_;
        }
        if(line.find("#LOCAL_SIZE") != std::string::npos)
        {
            file >> this->local_size_;
        }
        if(line.find("#BOUNDARY_SIZE") != std::string::npos)
        {
            file >> this->send_index_size_;
        }
        if(line.find("#NUMBER_OF_RECEIVERS") != std::string::npos)
        {
            file >> this->nrecv_;
        }
        if(line.find("#NUMBER_OF_SENDERS") != std::string::npos)
        {
            file >> this->nsend_;
        }
        if(line.find("#RECEIVERS_RANK") != std::string::npos)
        {
            allocate_host(this->nrecv_, &this->recvs_);
            for(int i = 0; i < this->nrecv_; ++i)
            {
                file >> this->recvs_[i];
            }
        }
        if(line.find("#SENDERS_RANK") != std::string::npos)
        {
            allocate_host(this->nsend_, &this->sends_);
            for(int i = 0; i < this->nsend_; ++i)
            {
                file >> this->sends_[i];
            }
        }
        if(line.find("#RECEIVERS_INDEX_OFFSET") != std::string::npos)
        {
            assert(this->nrecv_ > -1);
            allocate_host(this->nrecv_ + 1, &this->recv_offset_index_);
            for(int i = 0; i < this->nrecv_ + 1; ++i)
            {
                file >> this->recv_offset_index_[i];
            }
        }
        if(line.find("#SENDERS_INDEX_OFFSET") != std::string::npos)
        {
            assert(this->nsend_ > -1);
            allocate_host(this->nsend_ + 1, &this->send_offset_index_);
            for(int i = 0; i < this->nsend_ + 1; ++i)
            {
                file >> this->send_offset_index_[i];
            }
        }
        if(line.find("#BOUNDARY_INDEX") != std::string::npos)
        {
            assert(this->send_index_size_ > -1);
            allocate_host(this->send_index_size_, &this->boundary_index_);
            for(int i = 0; i < this->send_index_size_; ++i)
            {
                file >> this->boundary_index_[i];
            }
        }
    }

    // Number of nnz we receive
    this->recv_index_size_ = this->recv_offset_index_[this->nrecv_];

    // Number of nnz we send == boundary size
    assert(this->send_index_size_ == this->send_offset_index_[this->nsend_]);

    file.close();

    assert(rank == this->rank_);

    if(this->Status() == false)
    {
        LOG_INFO("Incomplete ParallelManager file");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    LOG_INFO("ReadFileASCII: filename=" << filename << "; done");
}

} // namespace rocalution
