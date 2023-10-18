#ifndef _CORE_OPS_H
#define _CORE_OPS_H


#include "basic.hpp"

namespace SpKernels{
void sddmm(denseMatrix& A, denseMatrix& B, coo_mtx& S, coo_mtx& C);
}



#endif
