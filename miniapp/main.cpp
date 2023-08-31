#include <cstdlib>
#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <vector>
#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <getopt.h>




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


void process_args(int argc, char *argv[], idx_t& f, int& c, string& filename){
   int choice;
   while (1)
   {
       static struct option long_options[] =
       {
           /* Use flags like so:
           {"verbose",	no_argument,	&verbose_flag, 'V'}*/
           /* Argument styles: no_argument, required_argument, optional_argument */
           {"version", no_argument,	0,	'v'},
           {"help",	no_argument,	0,	'h'},
           
           {0,0,0,0}
       };
   
       int option_index = 0;
   
       /* Argument parameters:
           no_argument: " "
           required_argument: ":"
           optional_argument: "::" */
   
       choice = getopt_long( argc, argv, "vh:k:c:",
                   long_options, &option_index);
   
       if (choice == -1)
           break;
   
       switch( choice )
       {
           case 'k':
               f = atoi(optarg);
               break;
           case 'c':
               c = atoi(optarg);
               break;
           case 'v':
               printf("3D SDDMM version 1.0\n");
               break;
           case 'h':
               printf("3D SDDMM version 1.0\n");
               printf("usage: sddmm [-k <k value>] [-c <c value] /path/to/matrix");
               break;
   
           case '?':
               /* getopt_long will have already printed an error */
               break;
   
           default:
               /* Not sure how to get here... */
               exit( EXIT_FAILURE);
       }
   }
   
   /* Deal with non-option arguments here */
   if ( optind < argc )
   {
       filename = argv[optind];
/*        while ( optind < argc )
 *        {
 *            
 *        }
 */
   }
   else{
       printf("usage: sddmm [-k <k value>] [-c <c value] /path/to/matrix");
       exit(EXIT_FAILURE);
   }
   return;
    
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
    
    string filename;
    idx_t f; int c;
    process_args(argc, argv, f, c, filename);
    coo_mtx Cloc;
    
    denseMatrix Aloc, Bloc;
    {
        coo_mtx C;
        mm _mm(filename); 
        if(rank == 0)
            C = _mm.read_mm(filename); 
        setup_3dsddmm(C,f,c, comm, Cloc, Aloc, Bloc,  comm_expand, comm_reduce);
    }
    communicate_pre(comm_expand);
    multiply(Aloc, Bloc, Cloc);
    communicate_post(comm_reduce, Cloc);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Cloc after reduce:" << std::endl;
        Cloc.printMatrix();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}
