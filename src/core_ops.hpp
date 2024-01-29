#ifndef _CORE_OPS_H
#define _CORE_OPS_H


#include "basic.hpp"
#include "SparseMatrix.hpp"


namespace SpKernels{
    void sddmm(denseMatrix& A, denseMatrix& B, cooMat& S, cooMat& C);
    void spmm(denseMatrix& X, cooMat& A, denseMatrix& Y);
}



#endif
