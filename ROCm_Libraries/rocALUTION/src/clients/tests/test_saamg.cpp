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

#include "testing_saamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int> saamg_tuple;

int saamg_size[]             = {63, 134};
std::string saamg_smoother[] = {"FSAI", "SPAI"};
int saamg_pre_iter[]         = {1, 2};
int saamg_post_iter[]        = {1, 2};
int saamg_cycle[]            = {0, 2};
int saamg_scaling[]          = {0, 1};

unsigned int saamg_format[] = {1, 6};

class parameterized_saamg : public testing::TestWithParam<saamg_tuple>
{
    protected:
    parameterized_saamg() {}
    virtual ~parameterized_saamg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_saamg_arguments(saamg_tuple tup)
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

TEST_P(parameterized_saamg, saamg_float)
{
    Arguments arg = setup_saamg_arguments(GetParam());
    ASSERT_EQ(testing_saamg<float>(arg), true);
}

TEST_P(parameterized_saamg, saamg_double)
{
    Arguments arg = setup_saamg_arguments(GetParam());
    ASSERT_EQ(testing_saamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(saamg,
                        parameterized_saamg,
                        testing::Combine(testing::ValuesIn(saamg_size),
                                         testing::ValuesIn(saamg_smoother),
                                         testing::ValuesIn(saamg_pre_iter),
                                         testing::ValuesIn(saamg_post_iter),
                                         testing::ValuesIn(saamg_format),
                                         testing::ValuesIn(saamg_cycle),
                                         testing::ValuesIn(saamg_scaling)));
