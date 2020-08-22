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

#ifndef ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_
#define ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_

#include <complex>

namespace rocalution {

/// Return absolute float value
float rocalution_abs(const float& val);
/// Return absolute double value
double rocalution_abs(const double& val);
/// Return absolute float value
float rocalution_abs(const std::complex<float>& val);
/// Return absolute double value
double rocalution_abs(const std::complex<double>& val);
/// Return absolute int value
int rocalution_abs(const int& val);

/// Return smallest positive floating point number
template <typename ValueType>
ValueType rocalution_eps(void);

/// Overloaded < operator for complex numbers
template <typename ValueType>
bool operator<(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);
/// Overloaded > operator for complex numbers
template <typename ValueType>
bool operator>(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);
/// Overloaded <= operator for complex numbers
template <typename ValueType>
bool operator<=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);
/// Overloaded >= operator for complex numbers
template <typename ValueType>
bool operator>=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);

} // namespace rocalution

#endif // ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_
