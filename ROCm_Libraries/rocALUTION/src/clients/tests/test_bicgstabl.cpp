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

#include "testing_bicgstabl.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int, int> bicgstabl_tuple;

int bicgstabl_size[]            = {7, 63};
std::string bicgstabl_precond[] = {
    "None", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int bicgstabl_format[] = {1, 2, 4, 5, 6, 7};
int bicgstabl_level[]           = {1, 2, 4};

class parameterized_bicgstabl : public testing::TestWithParam<bicgstabl_tuple>
{
    protected:
    parameterized_bicgstabl() {}
    virtual ~parameterized_bicgstabl() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bicgstabl_arguments(bicgstabl_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    arg.index   = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_bicgstabl, bicgstabl_float)
{
    Arguments arg = setup_bicgstabl_arguments(GetParam());
    ASSERT_EQ(testing_bicgstabl<float>(arg), true);
}

TEST_P(parameterized_bicgstabl, bicgstabl_double)
{
    Arguments arg = setup_bicgstabl_arguments(GetParam());
    ASSERT_EQ(testing_bicgstabl<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(bicgstabl,
                        parameterized_bicgstabl,
                        testing::Combine(testing::ValuesIn(bicgstabl_size),
                                         testing::ValuesIn(bicgstabl_precond),
                                         testing::ValuesIn(bicgstabl_format),
                                         testing::ValuesIn(bicgstabl_level)));
