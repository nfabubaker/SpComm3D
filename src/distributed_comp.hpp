
#pragma once
#include "basic.hpp"
#include "comm.hpp"
namespace SpKernels{


    void dist_sddmm_spcomm(
            denseMatrix& A,
            denseMatrix& B,
            coo_mtx& S,
            SparseComm<real_t>& comm_pre,
            SparseComm<real_t>& comm_post,
            coo_mtx& C);
    void dist_sddmm_dcomm(
            denseMatrix& A,
            denseMatrix& B,
            coo_mtx& S,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            coo_mtx& C);
}
