#include <cstdlib>
#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <vector>
#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include "../src/comm_stats.hpp"
#include <getopt.h>
#include <chrono>
#include "../src/distribute.hpp"







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
    string filename;
    idx_t f; int c;
    process_args(argc, argv, f, c, filename);
    std::string::size_type const p(filename.find_last_of('.'));
    std::string mtxName = filename.substr(0, p);
    mtxName = mtxName.substr(mtxName.find_last_of("/\\") +1);
    /* read & distribute the data */
    std::vector<idx_t> rpvec, cpvec;
    coo_mtx Cloc;

    {
        coo_mtx C;
        mm _mm(filename); 
        read_metadata;
        create_random_distribution;
        read_mm_MPI_IO(random_dist (input), Cloc(output));
        if(rank == 0)
            C = _mm.read_mm(filename); 
        
        
    }
    /* instance #1: sparse */
    {

        
        denseMatrix Aloc, Bloc;
        distribute3D_AB(Aloc, Bloc, rpvec2

        parallelTiming pt; 
        SparseComm<real_t> comm_expand;
        SparseComm<real_t> comm_reduce;
        setup_3dsddmm(C,f,c, comm, Cloc, Aloc, Bloc,  comm_expand, comm_reduce);
        auto start = chrono::high_resolution_clock::now();
        communicate_pre(comm_expand);
        auto stop = chrono::high_resolution_clock::now();
        pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
        start = chrono::high_resolution_clock::now();
        multiply(Aloc, Bloc, Cloc);
        stop = chrono::high_resolution_clock::now();
        pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
        start = chrono::high_resolution_clock::now();
        communicate_post(comm_reduce, Cloc);
        stop = chrono::high_resolution_clock::now();
        pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
        /*         MPI_Barrier(MPI_COMM_WORLD);
         *         if(rank == 0){
         *             std::cout << "Cloc after reduce:" << std::endl;
         *             Cloc.printMatrix();
         *         }
         */
        MPI_Barrier(MPI_COMM_WORLD);
        print_comm_stats_sparse(mtxName, comm_expand, f, c, pt, MPI_COMM_WORLD);
        print_comm_stats_sparse(mtxName, comm_reduce, f, c, pt, MPI_COMM_WORLD);
    }
    /* instance #2: dense */
    {
        parallelTiming pt;
        DenseComm comm_pre, comm_post;
        coo_mtx Cloc;
        denseMatrix Aloc, Bloc;
        {
            coo_mtx C;
            mm _mm(filename); 
            if(rank == 0)
                C = _mm.read_mm(filename); 
            setup_3dsddmm_bcast(C,f,c, comm, Cloc, Aloc, Bloc,  comm_pre, comm_post);
        }
        /* comm_pre */
        auto start = chrono::high_resolution_clock::now();
        comm_pre.perform_dense_comm();
        auto stop = chrono::high_resolution_clock::now();
        pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
        start = chrono::high_resolution_clock::now();
        multiply(Aloc, Bloc, Cloc);
        stop = chrono::high_resolution_clock::now();
        pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
        start = chrono::high_resolution_clock::now();
        comm_post.perform_dense_comm();
        stop = chrono::high_resolution_clock::now();
        pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
        for(size_t i = 0; i < Cloc.ownedNnz; ++i){
            idx_t lidx = Cloc.otl[i];
            Cloc.elms[lidx].val *= Cloc.owned[i];
        }
        /*         MPI_Barrier(MPI_COMM_WORLD);
         *         if(rank == 0){
         *             std::cout << "Cloc after reduce:" << std::endl;
         *             Cloc.printMatrix();
         *         }
         */
        MPI_Barrier(MPI_COMM_WORLD);
        print_comm_stats_dense(mtxName, comm_pre, f, c, pt, MPI_COMM_WORLD); 
        print_comm_stats_dense(mtxName, comm_post, f, c, pt, MPI_COMM_WORLD); 
    }
    MPI_Finalize();
    return 0;
}
