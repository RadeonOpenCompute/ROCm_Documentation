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

#pragma once
#ifndef TESTING_LOCAL_MATRIX_HPP
#define TESTING_LOCAL_MATRIX_HPP

#include "utility.hpp"

#include <rocalution.hpp>
#include <gtest/gtest.h>

using namespace rocalution;

template <typename T>
void testing_local_matrix_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    LocalMatrix<T> mat1;
    LocalMatrix<T> mat2;
    LocalVector<T> vec1;
    LocalVector<int> int1;

    // null pointers
    int* null_int = nullptr;
    T* null_data  = nullptr;

    // Valid pointers
    int* vint = nullptr;
    T* vdata  = nullptr;

    allocate_host(safe_size, &vint);
    allocate_host(safe_size, &vdata);

    // ExtractSubMatrix, ExtractSubMatrices, Extract(Inverse)Diagonal, ExtractL/U
    {
        LocalMatrix<T>* mat_null   = nullptr;
        LocalMatrix<T>** mat_null2 = nullptr;
        LocalMatrix<T>*** pmat     = new LocalMatrix<T>**[1];
        pmat[0]                    = new LocalMatrix<T>*[1];
        pmat[0][0]                 = new LocalMatrix<T>;
        LocalVector<T>* null_vec   = nullptr;
        ASSERT_DEATH(mat1.ExtractSubMatrix(0, 0, safe_size, safe_size, mat_null),
                     ".*Assertion.*mat != NULL*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, null_int, vint, pmat),
                     ".*Assertion.*row_offset != NULL*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, vint, null_int, pmat),
                     ".*Assertion.*col_offset != NULL*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, vint, vint, nullptr),
                     ".*Assertion.*mat != NULL*");
        ASSERT_DEATH(mat1.ExtractSubMatrices(safe_size, safe_size, vint, vint, &mat_null2),
                     ".*Assertion.*mat != NULL*");
        ASSERT_DEATH(mat1.ExtractDiagonal(null_vec), ".*Assertion.*vec_diag != NULL*");
        ASSERT_DEATH(mat1.ExtractInverseDiagonal(null_vec), ".*Assertion.*vec_inv_diag != NULL*");
        ASSERT_DEATH(mat1.ExtractL(mat_null, true), ".*Assertion.*L != NULL*");
        ASSERT_DEATH(mat1.ExtractU(mat_null, true), ".*Assertion.*U != NULL*");
        delete pmat[0][0];
        delete[] pmat[0];
        delete[] pmat;
    }

    // CMK, RCMK, ConnectivityOrder, MultiColoring, MaximalIndependentSet, ZeroBlockPermutation
    {
        int val;
        LocalVector<int>* null_vec = nullptr;
        ASSERT_DEATH(mat1.CMK(null_vec), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(mat1.RCMK(null_vec), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(mat1.ConnectivityOrder(null_vec), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(mat1.MultiColoring(val, &vint, &int1), ".*Assertion.*size_colors == NULL*");
        ASSERT_DEATH(mat1.MultiColoring(val, &null_int, null_vec),
                     ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(mat1.MaximalIndependentSet(val, null_vec),
                     ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(mat1.ZeroBlockPermutation(val, null_vec), ".*Assertion.*permutation != NULL*");
    }

    // LSolve, USolve, LLSolve, LUSolve, QRSolve
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.LSolve(vec1, null_vec), ".*Assertion.*out != NULL*");
        ASSERT_DEATH(mat1.USolve(vec1, null_vec), ".*Assertion.*out != NULL*");
        ASSERT_DEATH(mat1.LLSolve(vec1, null_vec), ".*Assertion.*out != NULL*");
        ASSERT_DEATH(mat1.LLSolve(vec1, vec1, null_vec), ".*Assertion.*out != NULL*");
        ASSERT_DEATH(mat1.LUSolve(vec1, null_vec), ".*Assertion.*out != NULL*");
        ASSERT_DEATH(mat1.QRSolve(vec1, null_vec), ".*Assertion.*out != NULL*");
    }

    // ICFactorize, Householder
    {
        T val;
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.ICFactorize(null_vec), ".*Assertion.*inv_diag != NULL*");
        ASSERT_DEATH(mat1.Householder(0, val, null_vec), ".*Assertion.*vec != NULL*");
    }

    // CopyFrom functions
    {
        ASSERT_DEATH(mat1.UpdateValuesCSR(null_data), ".*Assertion.*val != NULL*");
        ASSERT_DEATH(mat1.CopyFromCSR(null_int, vint, vdata), ".*Assertion.*row_offsets != NULL*");
        ASSERT_DEATH(mat1.CopyFromCSR(vint, null_int, vdata), ".*Assertion.*col != NULL*");
        ASSERT_DEATH(mat1.CopyFromCSR(vint, vint, null_data), ".*Assertion.*val != NULL*");
        ASSERT_DEATH(mat1.CopyToCSR(null_int, vint, vdata), ".*Assertion.*row_offsets != NULL*");
        ASSERT_DEATH(mat1.CopyToCSR(vint, null_int, vdata), ".*Assertion.*col != NULL*");
        ASSERT_DEATH(mat1.CopyToCSR(vint, vint, null_data), ".*Assertion.*val != NULL*");
        ASSERT_DEATH(mat1.CopyFromCOO(null_int, vint, vdata), ".*Assertion.*row != NULL*");
        ASSERT_DEATH(mat1.CopyFromCOO(vint, null_int, vdata), ".*Assertion.*col != NULL*");
        ASSERT_DEATH(mat1.CopyFromCOO(vint, vint, null_data), ".*Assertion.*val != NULL*");
        ASSERT_DEATH(mat1.CopyToCOO(null_int, vint, vdata), ".*Assertion.*row != NULL*");
        ASSERT_DEATH(mat1.CopyToCOO(vint, null_int, vdata), ".*Assertion.*col != NULL*");
        ASSERT_DEATH(mat1.CopyToCOO(vint, vint, null_data), ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.CopyFromHostCSR(null_int, vint, vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != NULL*");
        ASSERT_DEATH(
            mat1.CopyFromHostCSR(vint, null_int, vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.CopyFromHostCSR(vint, vint, null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
    }

    // CreateFromMat
    {
        LocalMatrix<T>* null_mat = nullptr;
        ASSERT_DEATH(mat1.CreateFromMap(int1, safe_size, safe_size, null_mat),
                     ".*Assertion.*pro != NULL*");
    }

    // Apply(Add)
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.Apply(vec1, null_vec), ".*Assertion.*out != NULL*");
        ASSERT_DEATH(mat1.ApplyAdd(vec1, 1.0, null_vec), ".*Assertion.*out != NULL*");
    }

    // Row/Column manipulation
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat1.ExtractColumnVector(0, null_vec), ".*Assertion.*vec != NULL*");
        ASSERT_DEATH(mat1.ExtractRowVector(0, null_vec), ".*Assertion.*vec != NULL*");
    }

    // AMG
    {
        int val;
        LocalVector<int>* null_vec = nullptr;
        LocalMatrix<T>* null_mat   = nullptr;
        ASSERT_DEATH(mat1.AMGConnect(0.1, null_vec), ".*Assertion.*connections != NULL*");
        ASSERT_DEATH(mat1.AMGAggregate(int1, null_vec), ".*Assertion.*aggregates != NULL*");
        ASSERT_DEATH(mat1.AMGSmoothedAggregation(0.1, int1, int1, null_mat, &mat2),
                     ".*Assertion.*prolong != NULL*");
        ASSERT_DEATH(mat1.AMGSmoothedAggregation(0.1, int1, int1, &mat2, null_mat),
                     ".*Assertion.*restrict != NULL*");
        ASSERT_DEATH(mat1.AMGAggregation(int1, null_mat, &mat2), ".*Assertion.*prolong != NULL*");
        ASSERT_DEATH(mat1.AMGAggregation(int1, &mat2, null_mat), ".*Assertion.*restrict != NULL*");
        ASSERT_DEATH(mat1.RugeStueben(0.1, null_mat, &mat2), ".*Assertion.*prolong != NULL*");
        ASSERT_DEATH(mat1.RugeStueben(0.1, &mat2, null_mat), ".*Assertion.*restrict != NULL*");
        ASSERT_DEATH(mat1.InitialPairwiseAggregation(0.1, val, null_vec, val, &null_int, val, 0),
                     ".*Assertion.*G != NULL*");
        ASSERT_DEATH(mat1.InitialPairwiseAggregation(0.1, val, &int1, val, &vint, val, 0),
                     ".*Assertion.*rG == NULL*");
        ASSERT_DEATH(
            mat1.InitialPairwiseAggregation(mat2, 0.1, val, null_vec, val, &null_int, val, 0),
            ".*Assertion.*G != NULL*");
        ASSERT_DEATH(mat1.InitialPairwiseAggregation(mat2, 0.1, val, &int1, val, &vint, val, 0),
                     ".*Assertion.*rG == NULL*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(0.1, val, null_vec, val, &vint, val, 0),
                     ".*Assertion.*G != NULL*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(0.1, val, &int1, val, &null_int, val, 0),
                     ".*Assertion.*rG != NULL*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(mat2, 0.1, val, null_vec, val, &vint, val, 0),
                     ".*Assertion.*G != NULL*");
        ASSERT_DEATH(mat1.FurtherPairwiseAggregation(mat2, 0.1, val, &int1, val, &null_int, val, 0),
                     ".*Assertion.*rG != NULL*");
        ASSERT_DEATH(
            mat1.CoarsenOperator(null_mat, safe_size, safe_size, int1, safe_size, vint, safe_size),
            ".*Assertion.*Ac != NULL*");
        ASSERT_DEATH(
            mat1.CoarsenOperator(&mat2, safe_size, safe_size, int1, safe_size, null_int, safe_size),
            ".*Assertion.*rG != NULL*");
    }

    // SetDataPtr
    {
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&null_int, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&vint, &null_int, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&vint, &vint, &null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(nullptr, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCOO(&vint, nullptr, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(mat1.SetDataPtrCOO(&vint, &vint, nullptr, "", safe_size, safe_size, safe_size),
                     ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&null_int, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&vint, &null_int, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&vint, &vint, &null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(nullptr, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrCSR(&vint, nullptr, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(mat1.SetDataPtrCSR(&vint, &vint, nullptr, "", safe_size, safe_size, safe_size),
                     ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&null_int, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, &null_int, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, &vint, &null_data, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(nullptr, &vint, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*row_offset != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, nullptr, &vdata, "", safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrMCSR(&vint, &vint, nullptr, "", safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(&null_int, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(&vint, &null_data, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(nullptr, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*col != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrELL(&vint, nullptr, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(&null_int, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*offset != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(&vint, &null_data, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(nullptr, &vdata, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*offset != NULL*");
        ASSERT_DEATH(
            mat1.SetDataPtrDIA(&vint, nullptr, "", safe_size, safe_size, safe_size, safe_size),
            ".*Assertion.*val != NULL*");
        ASSERT_DEATH(mat1.SetDataPtrDENSE(&null_data, "", safe_size, safe_size),
                     ".*Assertion.*val != NULL*");
        ASSERT_DEATH(mat1.SetDataPtrDENSE(nullptr, "", safe_size, safe_size),
                     ".*Assertion.*val != NULL*");
    }

    // LeaveDataPtr
    {
        int val;
        ASSERT_DEATH(mat1.LeaveDataPtrCOO(&vint, &null_int, &null_data),
                     ".*Assertion.*row == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrCOO(&null_int, &vint, &null_data),
                     ".*Assertion.*col == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrCOO(&null_int, &null_int, &vdata),
                     ".*Assertion.*val == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrCSR(&vint, &null_int, &null_data),
                     ".*Assertion.*row_offset == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrCSR(&null_int, &vint, &null_data),
                     ".*Assertion.*col == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrCSR(&null_int, &null_int, &vdata),
                     ".*Assertion.*val == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrMCSR(&vint, &null_int, &null_data),
                     ".*Assertion.*row_offset == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrMCSR(&null_int, &vint, &null_data),
                     ".*Assertion.*col == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrMCSR(&null_int, &null_int, &vdata),
                     ".*Assertion.*val == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrELL(&null_int, &vdata, val), ".*Assertion.*val == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrELL(&vint, &null_data, val), ".*Assertion.*col == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrDIA(&null_int, &vdata, val), ".*Assertion.*val == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrDIA(&vint, &null_data, val), ".*Assertion.*offset == NULL*");
        ASSERT_DEATH(mat1.LeaveDataPtrDENSE(&vdata), ".*Assertion.*val == NULL*");
    }

    free_host(&vint);
    free_host(&vdata);

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_LOCAL_MATRIX_HPP
