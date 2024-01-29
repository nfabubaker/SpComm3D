#pragma once
#include "basic.hpp"
#include "comm.hpp"
#include "denseComm.hpp"


using namespace SpKernels;
using namespace DComm;
void setup_spmm(
        cooMat& Cloc,
        const idx_t f,
        const int c ,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        denseMatrix& Aloc,
        denseMatrix& Bloc,
        std::vector<int>& rpvec, 
        std::vector<int>& cpvec,
        SparseComm<real_t>& comm_expand,
        SparseComm<real_t>& comm_reduce
        );
void setup_3dsddmm(
        cooMat& Cloc,
        const idx_t f,
        const int c ,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        denseMatrix& Aloc,
        denseMatrix& Bloc,
        std::vector<int>& rpvec, 
        std::vector<int>& cpvec,
        SparseComm<real_t>& comm_expand,
        DenseComm& comm_reduce
        );
