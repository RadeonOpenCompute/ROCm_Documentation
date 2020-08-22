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
#include "allocate_free.hpp"
#include "log.hpp"

#include <stdlib.h>
#include <string.h>
#include <complex>
#include <cstddef>

namespace rocalution {

//#define MEM_ALIGNMENT 64
//#define LONG_PTR size_t
//#define LONG_PTR long

template <typename DataType>
void allocate_host(int size, DataType** ptr)
{
    log_debug(0, "allocate_host()", "* begin", size, ptr);

    if(size > 0)
    {
        assert(*ptr == NULL);

        // *********************************************************
        // C++ style
        //    *ptr = new DataType[size];
        // *********************************************************

        // *********************************************************
        // C style
        // *ptr =  (DataType *) malloc(size*sizeof(DataType));
        // *********************************************************

        // *********************************************************
        // C style (zero-set)
        // *ptr = (DataType *) calloc(size, sizeof(DatatType));
        // *********************************************************

        // *********************************************************
        // Aligned allocation
        // total size = (size*datatype) + (alignment-1) + (void ptr)
        // void *non_aligned =  malloc(size*sizeof(DataType)+(MEM_ALIGNMENT-1)+sizeof(void*));
        // assert(non_aligned != NULL);

        // void *aligned = (void*)( ((LONG_PTR)(non_aligned)+MEM_ALIGNMENT+sizeof(void*) ) &
        // ~(MEM_ALIGNMENT-1) );
        // *((void**)aligned-1) = non_aligned;

        // *ptr = (DataType*) aligned;

        // LOG_INFO("A " << *ptr << " " <<  aligned << " " << non_aligned << " "<<  sizeof(DataType)
        // << " " << size);
        // *********************************************************

        // *********************************************************
        // C++ style and error handling

        *ptr = new(std::nothrow) DataType[size];

        if(!(*ptr))
        { // nullptr
            LOG_INFO("Cannot allocate memory");
            LOG_VERBOSE_INFO(2, "Size of the requested buffer = " << size * sizeof(DataType));
            FATAL_ERROR(__FILE__, __LINE__);
        }
        // *********************************************************

        assert(*ptr != NULL);
    }

    log_debug(0, "allocate_host()", "* end", *ptr);
}

template <typename DataType>
void free_host(DataType** ptr)
{
    log_debug(0, "free_host()", *ptr);

    assert(*ptr != NULL);

    // *********************************************************
    // C++ style
    delete[] * ptr;
    // *********************************************************

    // *********************************************************
    // C style
    // free(*ptr);
    // *********************************************************

    // *********************************************************
    // Aligned allocation
    //  free(*((void **)*ptr-1));
    // *********************************************************

    *ptr = NULL;
}

template <typename DataType>
void set_to_zero_host(int size, DataType* ptr)
{
    log_debug(0, "set_to_zero_host()", size, ptr);

    if(size > 0)
    {
        assert(ptr != NULL);

        memset(ptr, 0, size * sizeof(DataType));

        // for (int i=0; i<size; ++i)
        //   ptr[i] = DataType(0);
    }
}

template void allocate_host<float>(int size, float** ptr);
template void allocate_host<double>(int size, double** ptr);
#ifdef SUPPORT_COMPLEX
template void allocate_host<std::complex<float>>(int size, std::complex<float>** ptr);
template void allocate_host<std::complex<double>>(int size, std::complex<double>** ptr);
#endif
template void allocate_host<int>(int size, int** ptr);
template void allocate_host<unsigned int>(int size, unsigned int** ptr);
template void allocate_host<char>(int size, char** ptr);

template void free_host<float>(float** ptr);
template void free_host<double>(double** ptr);
#ifdef SUPPORT_COMPLEX
template void free_host<std::complex<float>>(std::complex<float>** ptr);
template void free_host<std::complex<double>>(std::complex<double>** ptr);
#endif
template void free_host<int>(int** ptr);
template void free_host<unsigned int>(unsigned int** ptr);
template void free_host<char>(char** ptr);

template void set_to_zero_host<float>(int size, float* ptr);
template void set_to_zero_host<double>(int size, double* ptr);
#ifdef SUPPORT_COMPLEX
template void set_to_zero_host<std::complex<float>>(int size, std::complex<float>* ptr);
template void set_to_zero_host<std::complex<double>>(int size, std::complex<double>* ptr);
#endif
template void set_to_zero_host<int>(int size, int* ptr);
template void set_to_zero_host<unsigned int>(int size, unsigned int* ptr);
template void set_to_zero_host<char>(int size, char* ptr);

} // namespace rocalution
