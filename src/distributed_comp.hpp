#ifndef _DISTRIBUTED_COMP_H
#define _DISTRIBUTED_COMP_H
#include "basic.hpp"
#include "comm.hpp"
namespace SpKernels{


void dist_spmm_spcomm(
            denseMatrix& X,
            coo_mtx& A,
            denseMatrix& Y,
            SparseComm<real_t>& comm_pre,
            SparseComm<real_t>& comm_post,
            MPI_Comm comm);

    void dist_spmm_dcomm(
            denseMatrix& X,
            denseMatrix& Y,
            coo_mtx& A,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            MPI_Comm comm);

    void dist_sddmm_spcomm(
            denseMatrix& A,
            denseMatrix& B,
            coo_mtx& S,
            SparseComm<real_t>& comm_pre,
            SparseComm<real_t>& comm_post,
            coo_mtx& C,
            MPI_Comm comm
            );
    void dist_sddmm_dcomm(
            denseMatrix& A,
            denseMatrix& B,
            coo_mtx& S,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            coo_mtx& C,
            MPI_Comm comm
            );
}
#endif
