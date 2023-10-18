#include "basic.hpp"

namespace SpKernels{
void sddmm(denseMatrix& A, denseMatrix& B, coo_mtx& S, coo_mtx& C){
    idx_t k = A.n;
    for(idx_t i = 0; i < S.lnnz; ++i){
        idx_t row, col;
        row = S.elms[i].row;
        col = S.elms[i].col;
        /*         const real_t *Ap = Aloc.data.data() + (row * f); const real_t *Bp = Bloc.data.data()+(col*f);
        */
        real_t vp =0.0;
        for (idx_t j = 0; j < k; j++){
            //vp += Ap[j]*Bp[j]; ++Ap; ++Bp;
            vp += A.at(row, j) * B.at(col, j);
        }
        C.elms.at(i).val = vp * S.elms.at(i).val; 

    }
    
}
}



