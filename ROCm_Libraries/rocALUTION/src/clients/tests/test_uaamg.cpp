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

#include "testing_uaamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int> uaamg_tuple;

int uaamg_size[]             = {63, 134};
std::string uaamg_smoother[] = {"FSAI", "ILU"};
int uaamg_pre_iter[]         = {1, 2};
int uaamg_post_iter[]        = {1, 2};
int uaamg_cycle[]            = {0, 2};
int uaamg_scaling[]          = {0, 1};

unsigned int uaamg_format[] = {1, 6};

class parameterized_uaamg : public testing::TestWithParam<uaamg_tuple>
{
    protected:
    parameterized_uaamg() {}
    virtual ~parameterized_uaamg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_uaamg_arguments(uaamg_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.smoother    = std::get<1>(tup);
    arg.format      = std::get<2>(tup);
    arg.pre_smooth  = std::get<3>(tup);
    arg.post_smooth = std::get<4>(tup);
    arg.cycle       = std::get<5>(tup);
    arg.ordering    = std::get<6>(tup);
    return arg;
}

TEST_P(parameterized_uaamg, uaamg_float)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<float>(arg), true);
}

TEST_P(parameterized_uaamg, uaamg_double)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(uaamg,
                        parameterized_uaamg,
                        testing::Combine(testing::ValuesIn(uaamg_size),
                                         testing::ValuesIn(uaamg_smoother),
                                         testing::ValuesIn(uaamg_pre_iter),
                                         testing::ValuesIn(uaamg_post_iter),
                                         testing::ValuesIn(uaamg_format),
                                         testing::ValuesIn(uaamg_cycle),
                                         testing::ValuesIn(uaamg_scaling)));
