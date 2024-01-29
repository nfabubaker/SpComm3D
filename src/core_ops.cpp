#include "SparseMatrix.hpp"
#include "basic.hpp"

namespace SpKernels{
    void sddmm(denseMatrix& A, denseMatrix& B, cooMat& S, cooMat& C){
        idx_t k = A.n;
        for(idx_large_t i = 0; i < S.nnz; ++i){
            idx_t row, col;
            row = S.ii[i];
            col = S.jj[i];
            /*         const real_t *Ap = Aloc.data.data() + (row * f); const real_t *Bp = Bloc.data.data()+(col*f);
            */
            real_t vp =0.0;
            for (idx_t j = 0; j < k; j++){
                //vp += Ap[j]*Bp[j]; ++Ap; ++Bp;
                vp += A.at(row, j) * B.at(col, j);
            }
            C.vv.at(i) = vp * S.vv.at(i); 

        }
        
    }

    void spmm(denseMatrix& X, cooMat& A, denseMatrix& Y){
        idx_t k = X.n; 
        for(idx_large_t i = 0; i < A.nnz; ++i){
            idx_t row, col;
            row = A.ii[i];
            col = A.jj[i];
            real_t aval = A.vv[i];
            for(size_t j = 0; j < k; ++j){
                Y.at(row, j) += aval * X.at(col, j); 
            }
        }
    }
}



