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
#include "distribute.hpp"
#include "distributed_comp.hpp"






using namespace SpKernels; 

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
    MPI_Comm comm = MPI_COMM_WORLD, xycomm, zcomm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    string filename;
    idx_t f; int c;
    process_args(argc, argv, f, c, filename);
    std::string::size_type const p(filename.find_last_of('.'));
    std::string mtxName = filename.substr(0, p);
    mtxName = mtxName.substr(mtxName.find_last_of("/\\") +1);
    std::vector<int> rpvec, cpvec;
    std:: vector<int> rpvec2D, cpvec2D;
    std::array<int, 3> dims = {0,0,c};
    std::array<int,3> zeroArr ={0,0,0};
    std::array<int,3> tdims ={0,0,0};
    MPI_Dims_create(size, 3, dims.data());
    MPI_Comm cartcomm;
    MPI_Cart_create(comm, 3, dims.data(), zeroArr.data(), 0, &cartcomm);   
    int X = dims[0], Y = dims[1], Z = dims[2];
    idx_t floc = f / Z;
    if(f % Z > 0){
        MPI_Cart_coords(cartcomm, rank, 3, tdims.data());
        int myzcoord = tdims[2];
        if(myzcoord < f% Z) ++floc;
    }
    coo_mtx Cloc, Sloc;
    Cloc.mtxName = mtxName;
    {
        coo_mtx S;
        mm _mm(filename); 
        if(rank == 0)
            S = _mm.read_mm(filename); 
        /* distribute C */
        distribute3D_C(S, Sloc, rpvec2D, cpvec2D, cartcomm, &zcomm); 
        /* copy C to S and reset values in S */
        Cloc = Sloc;
        for(idx_t i = 0; i < Sloc.lnnz; ++i){
            Cloc.elms.at(i).val = 0.0;
        }

    }
    std::array<int, 3> remaindims = {true, true, false};
    MPI_Cart_sub(cartcomm, remaindims.data(), &xycomm); 
    int myxyrank;
    MPI_Comm_rank(xycomm, &myxyrank);  
    /* distribute Aloc and Bloc  */
    distribute3D_AB_random(rpvec2D, cpvec2D, rpvec, cpvec, Cloc, f, xycomm);
    { /* distribute A,B and respect communication, setup sparse comm*/

        SparseComm<real_t> comm_expand;
        SparseComm<real_t> comm_reduce;
        denseMatrix Aloc, Bloc;
        /* prepare Aloc, Bloc according to local dims of Cloc */
        // split the 3D mesh communicator to 2D slices 

        Aloc.m = Cloc.lrows; Aloc.n = floc;
        Bloc.m = Cloc.lcols; Bloc.n = floc;
        Aloc.data.resize(Aloc.m * Aloc.n, 1);
        Bloc.data.resize(Bloc.m * Bloc.n, 1);
        setup_3dsddmm(Cloc, f, c, xycomm, zcomm, Aloc, Bloc, rpvec, cpvec, comm_expand, comm_reduce); 
        dist_sddmm_spcomm(Aloc, Bloc, Sloc, comm_expand, comm_reduce, Cloc);
    }
    /* instance #2: dense */
    {
        DenseComm comm_pre, comm_post;
        coo_mtx Cloc;
        denseMatrix Aloc, Bloc;
        std::vector<idx_t> gtlR(Cloc.grows, -1), gtlC(Cloc.gcols, -1), ltgR, ltgC;
        create_AB_Bcast(Cloc, floc, rpvec, cpvec, xycomm, Aloc, Bloc,
                gtlR, gtlC, ltgR, ltgC);
        /* re-map local rows/cols in Cloc */
        for(auto& el : Cloc.elms){
            idx_t lrid, lcid;
            lrid = el.row; 
            lcid = el.col;
            el.row = gtlR[Cloc.ltgR[lrid]];
            el.col = gtlC[Cloc.ltgC[lcid]];
        }
        setup_3dsddmm_bcast(Cloc,f,c, Aloc, Bloc, rpvec, cpvec, xycomm, zcomm,  comm_pre, comm_post);
        dist_sddmm_dcomm(Aloc, Bloc, Sloc, comm_pre, comm_post, Cloc);
        /* re-map local rows/cols in Cloc */
        for(auto& el : Cloc.elms){
            idx_t lrid, lcid;
            lrid = el.row; 
            lcid = el.col;
            el.row = Cloc.gtlR[ltgR[lrid]];
            el.col = Cloc.gtlC[ltgC[lcid]];
        }
    }
    MPI_Finalize();
    return 0;
}
