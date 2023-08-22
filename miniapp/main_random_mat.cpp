#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <vector>
#include "../src/basic.hpp"



using namespace SpKernels; 

void communicate_pre(SparseComm<real_t> &ch){

    ch.copy_to_sendbuff();
    /* perform sparse send/recv */
    ch.perform_sparse_comm();
    ch.copy_from_recvbuff();
}

void communicate_post(SparseComm<real_t> &ch, idx_t lnnz){
    ch.copy_to_sendbuff();
    /* perform sparse send/recv */
    ch.perform_sparse_comm();
    /* reduce from recvBuff to Cloc */
    std::vector<real_t> reduced_vals(lnnz, 0.0);
    for(int i = 0; i < ch.inDegree; ++i){
        for(size_t j=0; j < ch.recvCount[i]; ++j){
            
        }
    }


}

void multiply(denseMatrix &Aloc, denseMatrix &Bloc, coo_mtx &Cloc){
    idx_t f = Aloc.n;
    for(idx_t i = 0; i < Cloc.lnnz; ++i){ 
        idx_t row, col;
        row = Cloc.elms[i].row;
        col = Cloc.elms[i].col;
        real_t const *Ap = Aloc.data.data() + (row * f); real_t const *Bp = Bloc.data.data()+(col*f);
        real_t vp =0.0;
        for (int j = 0; j < f; j++){ vp += Ap[j]*Bp[j]; ++Ap; ++Bp;}
        Cloc.elms[i].val *= vp; 
    }
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    SparseComm<real_t> comm_expand;
    SparseComm<real_t> comm_reduce;
    coo_mtx Cloc;
    idx_t f = 10; int c = 2;
    denseMatrix Aloc, Bloc;
    {
        coo_mtx C; 
        C.grows = 100; C.gcols = 100;
        if(rank == 0)
            C.self_generate_random(1000);
        setup_3dsddmm(C,f,c, comm, Cloc, Aloc, Bloc,  comm_expand, comm_reduce);
    }
    communicate_pre(comm_expand);
    multiply(Aloc, Bloc, Cloc);
    communicate_post(comm_reduce, Cloc.lnnz);

    return 0;
}
