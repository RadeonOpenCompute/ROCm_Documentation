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

#include "testing_ruge_stueben_amg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int> rsamg_tuple;

int rsamg_size[]             = {63, 134};
std::string rsamg_smoother[] = {"ILU", "MCGS"};
int rsamg_pre_iter[]         = {1, 2};
int rsamg_post_iter[]        = {1, 2};
int rsamg_cycle[]            = {0, 1};
int rsamg_scaling[]          = {0, 1};

unsigned int rsamg_format[] = {1, 7};

class parameterized_ruge_stueben_amg : public testing::TestWithParam<rsamg_tuple>
{
    protected:
    parameterized_ruge_stueben_amg() {}
    virtual ~parameterized_ruge_stueben_amg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_arguments(rsamg_tuple tup)
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

TEST_P(parameterized_ruge_stueben_amg, ruge_stueben_amg_float)
{
    Arguments arg = setup_rsamg_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<float>(arg), true);
}

TEST_P(parameterized_ruge_stueben_amg, ruge_stueben_amg_double)
{
    Arguments arg = setup_rsamg_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(ruge_stueben_amg,
                        parameterized_ruge_stueben_amg,
                        testing::Combine(testing::ValuesIn(rsamg_size),
                                         testing::ValuesIn(rsamg_smoother),
                                         testing::ValuesIn(rsamg_pre_iter),
                                         testing::ValuesIn(rsamg_post_iter),
                                         testing::ValuesIn(rsamg_format),
                                         testing::ValuesIn(rsamg_cycle),
                                         testing::ValuesIn(rsamg_scaling)));
