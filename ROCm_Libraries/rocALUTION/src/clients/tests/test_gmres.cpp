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

#include "testing_gmres.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, int, std::string, unsigned int> gmres_tuple;

int gmres_size[]            = {7, 63};
int gmres_basis[]           = {20, 60};
std::string gmres_precond[] = {
    "None", "Chebyshev", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int gmres_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_gmres : public testing::TestWithParam<gmres_tuple>
{
    protected:
    parameterized_gmres() {}
    virtual ~parameterized_gmres() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gmres_arguments(gmres_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.index   = std::get<1>(tup);
    arg.precond = std::get<2>(tup);
    arg.format  = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_gmres, gmres_float)
{
    Arguments arg = setup_gmres_arguments(GetParam());
    ASSERT_EQ(testing_gmres<float>(arg), true);
}

TEST_P(parameterized_gmres, gmres_double)
{
    Arguments arg = setup_gmres_arguments(GetParam());
    ASSERT_EQ(testing_gmres<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(gmres,
                        parameterized_gmres,
                        testing::Combine(testing::ValuesIn(gmres_size),
                                         testing::ValuesIn(gmres_basis),
                                         testing::ValuesIn(gmres_precond),
                                         testing::ValuesIn(gmres_format)));
