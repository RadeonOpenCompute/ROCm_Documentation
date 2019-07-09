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

#include "testing_cr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> cr_tuple;

int cr_size[]            = {7, 63};
std::string cr_precond[] = {"None",
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
unsigned int cr_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_cr : public testing::TestWithParam<cr_tuple>
{
    protected:
    parameterized_cr() {}
    virtual ~parameterized_cr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_cr_arguments(cr_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    return arg;
}
/* TODO there _MIGHT_ be some issue with float accuracy
TEST_P(parameterized_cr, cr_float)
{
    Arguments arg = setup_cr_arguments(GetParam());
    ASSERT_EQ(testing_cr<float>(arg), true);
}
*/
TEST_P(parameterized_cr, cr_double)
{
    Arguments arg = setup_cr_arguments(GetParam());
    ASSERT_EQ(testing_cr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(cr,
                        parameterized_cr,
                        testing::Combine(testing::ValuesIn(cr_size),
                                         testing::ValuesIn(cr_precond),
                                         testing::ValuesIn(cr_format)));
