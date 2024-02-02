#ifndef _DISTRIBUTED_COMP_H
#define _DISTRIBUTED_COMP_H
#include "basic.hpp"
#include "comm.hpp"
#include "denseComm.hpp"
#include "SparseMatrix.hpp"

using namespace SpKernels;
using namespace DComm;

void dist_spmm_spcomm(
            denseMatrix& X,
            cooMat& A,
            denseMatrix& Y,
            SparseComm<real_t>& comm_pre,
            SparseComm<real_t>& comm_post,
            MPI_Comm comm);

    void dist_spmm_dcomm(
            denseMatrix& X,
            denseMatrix& Y,
            cooMat& A,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            MPI_Comm comm);

    void dist_sddmm_spcomm(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            SparseComm<real_t>& comm_pre,
            DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm
            );
    void dist_sddmm_spcomm2(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            SparseComm<real_t>& comm_preA,
            SparseComm<real_t>& comm_preB,
            DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm
            );
    void dist_sddmm_spcomm3(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            SparseComm<real_t>& comm_preA,
            SparseComm<real_t>& comm_preB,
            DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm
            );
    void dist_sddmm_dcomm(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm
            );
#endif
