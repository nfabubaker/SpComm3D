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
#include "parallel_io.hpp"


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

void print_numerical_sum(coo_mtx& C, MPI_Comm zcomm, MPI_Comm worldcomm){
    real_t sum = 0.0;
    int myzrank, myworldrank;
    MPI_Comm_rank(zcomm, &myzrank);
    MPI_Comm_rank(worldcomm, &myworldrank);
    for(size_t i = 0; i < C.lnnz; ++i){
        if(C.owners.at(i) == myzrank) 
            sum += C.elms.at(i).val;
    }
    printf("Numerical sum at p%d = %.2f\n", myworldrank, sum );
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
    std::array<int, 3> remaindims = {true, true, false};
    MPI_Cart_sub(cartcomm, remaindims.data(), &xycomm); 
    int myxyrank;
    MPI_Comm_rank(xycomm, &myxyrank);  
    remaindims = {false, false, true};
    MPI_Cart_sub(cartcomm, remaindims.data(), &zcomm); 
    coo_mtx Aloc;
    Aloc.mtxName = mtxName;
    {
        std:: vector<int> rpvec2D, cpvec2D;
        /* distribute C */
        read_bin_parallel_distribute(filename, Aloc, rpvec2D, cpvec2D,
                cartcomm,xycomm, zcomm);

    
    Aloc.rank = rank;
    MPI_Comm_rank(zcomm, &Aloc.zrank);
    /* distribute Aloc and Bloc  */
    //distribute3D_AB_random(rpvec2D, cpvec2D, rpvec, cpvec, Cloc, f, xycomm);
    distribute3D_AB_respect_communication(rpvec2D, cpvec2D,
            rpvec, cpvec,
            Aloc, f, xycomm, zcomm, cartcomm);
    }
    { /* distribute A,B and respect communication, setup sparse comm*/

        SparseComm<real_t> comm_expand;
        SparseComm<real_t> comm_reduce;
        denseMatrix Xloc, Yloc;
        /* prepare Aloc, Bloc according to local dims of Cloc */
        // split the 3D mesh communicator to 2D slices 

        Yloc.m = Aloc.lrows; Yloc.n = floc;
        Xloc.m = Aloc.lcols; Xloc.n = floc;
        Xloc.data.resize(Xloc.m * Xloc.n, myxyrank+1);
        Yloc.data.resize(Yloc.m * Yloc.n, myxyrank+1);
        setup_spmm(Aloc, f, c, xycomm, zcomm, Xloc, Yloc, rpvec, cpvec, comm_expand, comm_reduce); 

        dist_spmm_spcomm(Xloc, Aloc, Yloc, comm_expand, comm_reduce, cartcomm);
 //       print_numerical_sum(Cloc, zcomm, cartcomm);
    }
    /* instance #2: dense */
    {
        for(auto& elm : Aloc.elms) elm.val = 0.0;
        DenseComm comm_pre, comm_post;
        denseMatrix Xloc, Yloc;
        std::unordered_map<idx_t, idx_t> gtlR, gtlC;
        std::vector<idx_t> mapY(Aloc.lrows), mapX(Aloc.lcols);
        create_AB_Bcast(Aloc, floc, rpvec, cpvec, xycomm, Yloc, Xloc);
        std::vector<idx_t> mapYI(Yloc.m), mapXI(Xloc.m);
        setup_3dspmm_bcast(Aloc,f,c, Xloc, Yloc, rpvec, cpvec,
                xycomm, zcomm,  comm_pre, comm_post, mapX, mapY);
        for(idx_t i = 0; i < Aloc.lrows; ++i) mapYI[mapY[i]] = i;
        for(idx_t i = 0; i < Aloc.lcols; ++i) mapXI[mapX[i]] = i;
        // re-map local rows/cols in Cloc 
        for(size_t i = 0; i < Aloc.lnnz; ++i){
            idx_t lrid, lcid;
            auto& el = Aloc.elms.at(i);
            lrid = el.row; 
            lcid = el.col;
/*             el.row = gtlR[Cloc.ltgR[lrid]];
 *             el.col = gtlC[Cloc.ltgC[lcid]];
 */
            el.row = mapY[lrid];
            el.col = mapX[lcid];
        }
        dist_spmm_dcomm(Xloc, Yloc, Aloc, comm_pre, comm_post, cartcomm);
        // re-map local rows/cols in Cloc ///
        for(size_t i = 0; i < Aloc.lnnz; ++i){
            idx_t lrid, lcid;
            auto& el = Aloc.elms.at(i);
            lrid = el.row; 
            lcid = el.col;
            el.row = mapYI[lrid];
            el.col = mapXI[lcid];
        }

    //print_numerical_sum(Aloc, zcomm, cartcomm);
    }


    MPI_Finalize();
    return 0;
}
