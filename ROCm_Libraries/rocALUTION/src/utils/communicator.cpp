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

#include "def.hpp"
#include "communicator.hpp"
#include "log_mpi.hpp"

#include <complex>

namespace rocalution {

template <>
void communication_allreduce_single_sum(double local, double* global, const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_DOUBLE, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(float local, float* global, const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_FLOAT, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(std::complex<double> local,
                                        std::complex<double>* global,
                                        const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(std::complex<float> local,
                                        std::complex<float>* global,
                                        const void* comm)
{
    // TODO check the type
    int status = MPI_Allreduce(&local, global, 1, MPI_COMPLEX, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(int local, int* global, const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_INT, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(unsigned int local, unsigned int* global, const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_UNSIGNED, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(long local, long* global, const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_LONG, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(unsigned long local,
                                        unsigned long* global,
                                        const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_UNSIGNED_LONG, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(long long local, long long* global, const void* comm)
{
    int status = MPI_Allreduce(&local, global, 1, MPI_LONG_LONG, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_allreduce_single_sum(unsigned long long local,
                                        unsigned long long* global,
                                        const void* comm)
{
    int status =
        MPI_Allreduce(&local, global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *(MPI_Comm*)comm);
    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_recv(
    double* buf, int count, int source, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Irecv(buf, count, MPI_DOUBLE, source, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_recv(
    float* buf, int count, int source, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Irecv(buf, count, MPI_FLOAT, source, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_recv(
    std::complex<double>* buf, int count, int source, int tag, MRequest* request, const void* comm)
{
    int status =
        MPI_Irecv(buf, count, MPI_DOUBLE_COMPLEX, source, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_recv(
    std::complex<float>* buf, int count, int source, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Irecv(buf, count, MPI_COMPLEX, source, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_recv(
    int* buf, int count, int source, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Irecv(buf, count, MPI_INT, source, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_send(
    double* buf, int count, int dest, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_send(
    float* buf, int count, int dest, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Isend(buf, count, MPI_FLOAT, dest, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_send(
    std::complex<double>* buf, int count, int dest, int tag, MRequest* request, const void* comm)
{
    int status =
        MPI_Isend(buf, count, MPI_DOUBLE_COMPLEX, dest, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_send(
    std::complex<float>* buf, int count, int dest, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Isend(buf, count, MPI_COMPLEX, dest, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template <>
void communication_async_send(
    int* buf, int count, int dest, int tag, MRequest* request, const void* comm)
{
    int status = MPI_Isend(buf, count, MPI_INT, dest, tag, *(MPI_Comm*)comm, &request->req);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

void communication_syncall(int count, MRequest* requests)
{
    int status = MPI_Waitall(count, &requests[0].req, MPI_STATUSES_IGNORE);

    CHECK_MPI_ERROR(status, __FILE__, __LINE__);
}

template void
communication_allreduce_single_sum<double>(double local, double* global, const void* comm);
template void
communication_allreduce_single_sum<float>(float local, float* global, const void* comm);

#ifdef SUPPORT_COMPLEX
template void communication_allreduce_single_sum<std::complex<double>>(std::complex<double> local,
                                                                       std::complex<double>* global,
                                                                       const void* comm);
template void communication_allreduce_single_sum<std::complex<float>>(std::complex<float> local,
                                                                      std::complex<float>* global,
                                                                      const void* comm);
#endif

template void communication_async_recv<double>(
    double* buf, int count, int source, int tag, MRequest* request, const void* comm);
template void communication_async_recv<float>(
    float* buf, int count, int source, int tag, MRequest* request, const void* comm);

#ifdef SUPPORT_COMPLEX
template void communication_async_recv<std::complex<double>>(
    std::complex<double>* buf, int count, int source, int tag, MRequest* request, const void* comm);
template void communication_async_recv<std::complex<float>>(
    std::complex<float>* buf, int count, int source, int tag, MRequest* request, const void* comm);
#endif

template void communication_async_send<double>(
    double* buf, int count, int dest, int tag, MRequest* request, const void* comm);
template void communication_async_send<float>(
    float* buf, int count, int dest, int tag, MRequest* request, const void* comm);

#ifdef SUPPORT_COMPLEX
template void communication_async_send<std::complex<double>>(
    std::complex<double>* buf, int count, int dest, int tag, MRequest* request, const void* comm);
template void communication_async_send<std::complex<float>>(
    std::complex<float>* buf, int count, int dest, int tag, MRequest* request, const void* comm);
#endif

} // namespace rocalution
