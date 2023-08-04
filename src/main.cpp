#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <vector>
#include "basic.hpp"



using namespace SpKernels; 

void communicate_pre(SparseComm &ch, idx_t f){

    ch.copy_to_sendbuff(f);
    /* perform sparse send/recv */
    ch.perform_sparse_comm(f);
    ch.copy_from_recvbuff(f);
}

void communicate_post(){

}

void multiply(real_t const * const Aloc, real_t const * const Bloc, coo_mtx * const Cloc, const idx_t f){
    for(idx_t i = 0; i < Cloc->nnz; ++i){ 
        idx_t row, col;
        row = Cloc->elms[i].row;
        col = Cloc->elms[i].col;
        Cloc->elms[i].val = 0.0;
        real_t const *Ap = Aloc + (row * f); real_t const *Bp = Bloc+(col*f);
        real_t &vp = Cloc->elms[i].val; 
        for (int j = 0; j < f; j++){ vp += Ap[j]*Bp[j]; ++Ap; ++Bp;}
    }
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    SparseComm comm_expand;
    SparseComm comm_reduce;
    setup_3dsddmm(comm_expand, comm_reduce);
    communicate_pre();
    multiply();
    communicate_post();

    return 0;
}
