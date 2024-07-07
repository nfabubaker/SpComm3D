#pragma once
#include <fstream>
#include <string>
#include <algorithm>
#include <gtest/gtest.h>

#include "mm.hpp"

void sort_rowmaj_coo(SpKernels::cooMat& M) {
    // Create a permutation vector
    std::vector<size_t> p(M.ii.size());
    std::iota(p.begin(), p.end(), 0);

    // Sort the permutation vector based on comparing the ii values
    std::sort(p.begin(), p.end(), [&](size_t i, size_t j) { 
        return std::tie(M.ii[i], M.jj[i]) < std::tie(M.ii[j], M.jj[j]); 
    });

    // Apply the permutation to ii, jj, and vv
    std::vector<idx_t> ii_sorted(M.ii.size());
    std::vector<idx_t> jj_sorted(M.jj.size());
    std::vector<real_t> vv_sorted(M.vv.size());

    for (size_t i = 0; i < M.ii.size(); i++) {
        ii_sorted[i] = M.ii[p[i]];
        jj_sorted[i] = M.jj[p[i]];
        vv_sorted[i] = M.vv[p[i]];
    }

    M.ii = ii_sorted;
    M.jj = jj_sorted;
    M.vv = vv_sorted;
}

void test_eq_mtx(const std::string& path_A, const std::string& path_B, const double tol = 1e-6) {
    // Check if mtx files exist
    std::ifstream file_A(path_A);
    std::ifstream file_B(path_B);
    EXPECT_TRUE(file_A.good());
    EXPECT_TRUE(file_B.good());

    // Read the matrices
    SpKernels::mm readerA(path_A);
    SpKernels::cooMat A;
    SpKernels::mm readerB(path_B);
    SpKernels::cooMat B;

    readerA.read_mm(path_A, A);
    readerB.read_mm(path_B, B);

    // assert that the matrices have the same dimensions
    ASSERT_EQ(A.gncols, B.gncols);
    ASSERT_EQ(A.gnrows, B.gnrows);
    ASSERT_EQ(A.gnnz, B.gnnz);

    // assert that the last value is non-zero (avoid wrong dimensions in mtx files)
    ASSERT_NE(A.vv[A.gnnz - 1], 0.0) << "Last value in A is zero, are dimensions correct in " << path_A << "?";
    ASSERT_NE(B.vv[B.gnnz - 1], 0.0) << "Last value in B is zero, are dimensions correct in " << path_B << "?";

    // sort the matrices in row-major order
    sort_rowmaj_coo(A);
    sort_rowmaj_coo(B);

    // compare each triplet
    for (size_t i = 0; i < A.gnnz; i++) {
        EXPECT_EQ(A.ii[i], B.ii[i]) << "Failed at index i=" << i << " for ii";
        EXPECT_EQ(A.jj[i], B.jj[i]) << "Failed at index i=" << i << " for jj";
        EXPECT_NEAR(A.vv[i], B.vv[i], tol) << "Failed at index i=" << i << " for vv";
    }
}