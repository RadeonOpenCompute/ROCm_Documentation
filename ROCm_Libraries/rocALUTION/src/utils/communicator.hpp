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

#ifndef RCOALUTION_UTILS_COMMUNICATOR_HPP_
#define ROCALUTION_UTILS_COMMUNICATOR_HPP_

#include <mpi.h>

namespace rocalution {

struct MRequest
{
    MPI_Request req;
};

// TODO make const what ever possible

template <typename ValueType>
void communication_allreduce_single_sum(ValueType local, ValueType* global, const void* comm);

template <typename ValueType>
void communication_async_recv(
    ValueType* buf, int count, int source, int tag, MRequest* request, const void* comm);

template <typename ValueType>
void communication_async_send(
    ValueType* buf, int count, int dest, int tag, MRequest* request, const void* comm);

void communication_syncall(int count, MRequest* requests);

} // namespace rocalution

#endif // ROCALUTION_UTILS_COMMUNICATOR_HPP_
