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
#include "math_functions.hpp"

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace rocalution {

float rocalution_abs(const float& val) { return std::fabs(val); }

double rocalution_abs(const double& val) { return std::fabs(val); }

float rocalution_abs(const std::complex<float>& val) { return std::abs(val); }

double rocalution_abs(const std::complex<double>& val) { return std::abs(val); }

int rocalution_abs(const int& val) { return abs(val); }

template <typename ValueType>
ValueType rocalution_eps(void)
{
    return std::numeric_limits<ValueType>::epsilon();
}

template <typename ValueType>
bool operator<(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return false;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() < rhs.real();
}

template <typename ValueType>
bool operator>(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return false;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() > rhs.real();
}

template <typename ValueType>
bool operator<=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return true;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() <= rhs.real();
}

template <typename ValueType>
bool operator>=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return true;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() >= rhs.real();
}

template double rocalution_eps(void);
template float rocalution_eps(void);
template std::complex<double> rocalution_eps(void);
template std::complex<float> rocalution_eps(void);

template bool operator<(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator<(const std::complex<double>& lhs, const std::complex<double>& rhs);

template bool operator>(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator>(const std::complex<double>& lhs, const std::complex<double>& rhs);

template bool operator<=(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator<=(const std::complex<double>& lhs, const std::complex<double>& rhs);

template bool operator>=(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator>=(const std::complex<double>& lhs, const std::complex<double>& rhs);

} // namespace rocalution
