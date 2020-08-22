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

#ifndef ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
#define ROCALUTION_UTILS_ALLOCATE_FREE_HPP_

namespace rocalution {

/** \ingroup backend_module
  * \brief Allocate buffer on the host
  * \details
  * \p allocate_host allocates a buffer on the host.
  *
  * @param[in]
  * size    number of elements the buffer need to be allocated for
  * @param[out]
  * ptr     pointer to the position in memory where the buffer should be allocated,
  *         it is expected that \p *ptr == \p NULL
  *
  * \tparam DataType can be char, int, unsigned int, float, double, std::complex<float>
  *         or std::complex<double>.
  */
template <typename DataType>
void allocate_host(int size, DataType** ptr);

/** \ingroup backend_module
  * \brief Free buffer on the host
  * \details
  * \p free_host deallocates a buffer on the host. \p *ptr will be set to NULL after
  * successful deallocation.
  *
  * @param[inout]
  * ptr     pointer to the position in memory where the buffer should be deallocated,
  *         it is expected that \p *ptr != \p NULL
  *
  * \tparam DataType can be char, int, unsigned int, float, double, std::complex<float>
  *         or std::complex<double>.
  */
template <typename DataType>
void free_host(DataType** ptr);

/** \ingroup backend_module
  * \brief Set a host buffer to zero
  * \details
  * \p set_to_zero_host sets a host buffer to zero.
  *
  * @param[in]
  * size    number of elements
  * @param[inout]
  * ptr     pointer to the host buffer
  *
  * \tparam DataType can be char, int, unsigned int, float, double, std::complex<float>
  *         or std::complex<double>.
  */
template <typename DataType>
void set_to_zero_host(int size, DataType* ptr);

} // namespace rocalution

#endif // ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
