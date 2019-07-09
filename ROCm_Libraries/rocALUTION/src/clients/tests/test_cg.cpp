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

#include "testing_cg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> cg_tuple;

int cg_size[]            = {7, 63};
std::string cg_precond[] = {"None",
                            "Chebyshev",
                            "FSAI",
                            "SPAI",
                            "TNS",
                            "Jacobi",
                            "SGS",
                            "ILU",
                            "ILUT",
                            "IC",
                            "MCSGS",
                            "MCILU"};
unsigned int cg_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_cg : public testing::TestWithParam<cg_tuple>
{
    protected:
    parameterized_cg() {}
    virtual ~parameterized_cg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_cg_arguments(cg_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_cg, cg_float)
{
    Arguments arg = setup_cg_arguments(GetParam());
    ASSERT_EQ(testing_cg<float>(arg), true);
}

TEST_P(parameterized_cg, cg_double)
{
    Arguments arg = setup_cg_arguments(GetParam());
    ASSERT_EQ(testing_cg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(cg,
                        parameterized_cg,
                        testing::Combine(testing::ValuesIn(cg_size),
                                         testing::ValuesIn(cg_precond),
                                         testing::ValuesIn(cg_format)));
