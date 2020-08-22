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

#include "testing_fgmres.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, int, std::string, unsigned int> fgmres_tuple;

int fgmres_size[]            = {7, 63};
int fgmres_basis[]           = {20, 60};
std::string fgmres_precond[] = {
    "None", "Chebyshev", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int fgmres_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_fgmres : public testing::TestWithParam<fgmres_tuple>
{
    protected:
    parameterized_fgmres() {}
    virtual ~parameterized_fgmres() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_fgmres_arguments(fgmres_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.index   = std::get<1>(tup);
    arg.precond = std::get<2>(tup);
    arg.format  = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_fgmres, fgmres_float)
{
    Arguments arg = setup_fgmres_arguments(GetParam());
    ASSERT_EQ(testing_fgmres<float>(arg), true);
}

TEST_P(parameterized_fgmres, fgmres_double)
{
    Arguments arg = setup_fgmres_arguments(GetParam());
    ASSERT_EQ(testing_fgmres<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(fgmres,
                        parameterized_fgmres,
                        testing::Combine(testing::ValuesIn(fgmres_size),
                                         testing::ValuesIn(fgmres_basis),
                                         testing::ValuesIn(fgmres_precond),
                                         testing::ValuesIn(fgmres_format)));
