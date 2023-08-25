#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <vector>
#include "../src/basic.hpp"
#include "mpi_proto.h"



using namespace SpKernels; 

void communicate_pre(SparseComm<real_t> &ch){

    ch.copy_to_sendbuff();
    /* perform sparse send/recv */
    ch.perform_sparse_comm();
    ch.copy_from_recvbuff();
}

void communicate_post(SparseComm<real_t> &ch, coo_mtx& Cloc){
    ch.copy_to_sendbuff();
    /* perform sparse send/recv */
    ch.perform_sparse_comm();
    /* reduce from recvBuff to Cloc */
    ch.SUM_from_recvbuff();
    for(size_t i = 0; i < Cloc.ownedNnz; ++i){
        idx_t lidx = Cloc.otl[i];
        Cloc.elms[lidx].val *= Cloc.owned[i];
    }


}

void multiply(denseMatrix &Aloc, denseMatrix &Bloc, coo_mtx &Cloc){
    idx_t f = Aloc.n;
    for(idx_t i = 0; i < Cloc.lnnz; ++i){ 
        idx_t row, col;
        row = Cloc.elms[i].row;
        col = Cloc.elms[i].col;
/*         const real_t *Ap = Aloc.data.data() + (row * f); const real_t *Bp = Bloc.data.data()+(col*f);
 */
        real_t vp =0.0;
        for (idx_t j = 0; j < f; j++){
            //vp += Ap[j]*Bp[j]; ++Ap; ++Bp;
            vp += Aloc.at(row, j) * Bloc.at(col, j);
        }
        Cloc.elms.at(i).val *= vp; 
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

    /* print A and B before & after comm: */
    if(rank == 0){
        std::cout << "A:" << std::endl;
        Aloc.printMatrix();
        std::cout << "B:" << std::endl;
        Bloc.printMatrix();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    communicate_pre(comm_expand);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Aloc:" << std::endl;
        Aloc.printMatrix();
        std::cout << "Bloc:" << std::endl;
        Bloc.printMatrix();
        std::cout << "Cloc:" << std::endl;
        Cloc.printMatrix();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    multiply(Aloc, Bloc, Cloc);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Cloc after comp:" << std::endl;
        Cloc.printMatrix();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    communicate_post(comm_reduce, Cloc);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Cloc after reduce:" << std::endl;
        Cloc.printMatrix();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}
