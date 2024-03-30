#include <mpi.h>
#include <numeric>
#include <sstream>
#include <sys/types.h>
#include <vector>
#include "basic.hpp"
#include "SparseMatrix.hpp"
#include "comm.hpp"
#include "comm_stats.hpp"
#include <getopt.h>
#include <chrono>
#include "denseComm.hpp"
#include "distribute.hpp"
#include "distributed_comp.hpp"
#include "parallel_io.hpp"
#include "comm_setup.hpp"





#define NUM_ITER 5
using namespace SpKernels; 

void vals_from_str(string str, vector<idx_t>& vals){
    stringstream ss(str);
    string fi;
    while(getline(ss, fi, ',')){
        vals.push_back(stoi(fi));
    }
}

void process_args(int argc, char *argv[], std::vector<idx_t>& fvals, int& c, string& filename){
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
                vals_from_str(string(optarg), fvals);
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

void print_numerical_sum(cooMat& C, MPI_Comm zcomm, MPI_Comm worldcomm){
    real_t sum = 0.0;
    int myzrank, myworldrank, zsize;
    MPI_Comm_rank(zcomm, &myzrank);
    MPI_Comm_size(zcomm, &zsize);
    MPI_Comm_rank(worldcomm, &myworldrank);
    //idx_t owned_startIdx, owned_endIdx;
    //idx_t no_pp = C.nnz / zsize;
    //idx_t no_pp_r = C.nnz % zsize;
    //owned_startIdx = (myzrank < no_pp_r ? myzrank * (no_pp+1) : no_pp_r*no_pp+ (myzrank-no_pp_r)*no_pp); 
    //owned_endIdx = owned_startIdx+(myzrank < no_pp_r ? no_pp+1: no_pp);
    //for(idx_large_t i = owned_startIdx; i < owned_endIdx; ++i) sum += C.vv[i];
    sum = std::accumulate(C.ownedVals.begin(), C.ownedVals.end(), 0.0);

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
    int c;
    vector<idx_t> fvals;
    process_args(argc, argv, fvals, c, filename);
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
    std::array<int, 3> remaindims = {true, true, false};
    MPI_Cart_sub(cartcomm, remaindims.data(), &xycomm); 
    int myxyrank;
    MPI_Comm_rank(xycomm, &myxyrank);  
    remaindims = {false, false, true};
    MPI_Cart_sub(cartcomm, remaindims.data(), &zcomm); 
    cooMat Sloc;
    Sloc.mtxName = mtxName;
    {
        std:: vector<int> rpvec2D, cpvec2D;
        /* distribute C */
        read_bin_parallel_distribute_coo(filename, Sloc, rpvec2D, cpvec2D,
                cartcomm,xycomm, zcomm);
        Sloc.localizeIndices();
        MPI_Comm_rank(zcomm, &Sloc.zrank);
        Sloc.rank = rank;
        /* distribute Yloc and Xloc  */
        //distribute3D_AB_random(rpvec2D, cpvec2D, rpvec, cpvec, Cloc, f, xycomm);
        distribute3D_AB_respect_communication(rpvec2D, cpvec2D,
                rpvec, cpvec,
                Sloc, xycomm, zcomm, cartcomm);
    }
    for(auto f : fvals){
        idx_t floc = f / Z;
        if(f % Z > 0){
            MPI_Cart_coords(cartcomm, rank, 3, tdims.data());
            int myzcoord = tdims[2];
            if(myzcoord < f% Z) ++floc;
        }
        { /* distribute A,B and respect communication, setup sparse comm*/
            SparseComm<real_t> comm_expand, comm_reduce;
            denseMatrix Yloc, Xloc;
            /* prepare Yloc, Xloc according to local dims of Cloc */
            // split the 3D mesh communicator to 2D slices 

            Yloc.m = Sloc.nrows; Yloc.n = floc;
            Xloc.m = Sloc.ncols; Xloc.n = floc;
            Yloc.data.resize(Yloc.m * Yloc.n, myxyrank+1);
            Xloc.data.resize(Xloc.m * Xloc.n, myxyrank+1);
            setup_spmm(Sloc, f, c, xycomm, zcomm, Xloc, Yloc, rpvec, cpvec, comm_expand, comm_reduce); 

            for(int i = 0; i < NUM_ITER; ++i)
                dist_spmm_spcomm(Xloc, Sloc, Yloc, comm_expand, comm_reduce, cartcomm);
            //       print_numerical_sum(Cloc, zcomm, cartcomm);
        }
        /* instance #2: dense */
        {
            DenseComm comm_pre, comm_post;
            denseMatrix Yloc, Xloc;
            std::unordered_map<idx_t, idx_t> gtlR, gtlC;
            std::vector<idx_t> mapY(Sloc.nrows), mapX(Sloc.ncols);
            create_AB_Bcast(Sloc, floc, rpvec, cpvec, xycomm, Yloc, Xloc);
            std::vector<idx_t> mapYI(Yloc.m), mapXI(Xloc.m);
            setup_3dspmm_bcast(Sloc,f,c, Xloc, Yloc, rpvec, cpvec,
                    xycomm, zcomm,  comm_pre, comm_post, mapX, mapY);
            for(idx_t i = 0; i < Sloc.nrows; ++i) mapYI[mapY[i]] = i;
            for(idx_t i = 0; i < Sloc.ncols; ++i) mapXI[mapX[i]] = i;
            // re-map local rows/cols in Cloc 
            for(size_t i = 0; i < Sloc.nnz; ++i){
                Sloc.ii[i] = mapY[Sloc.ii[i]];
                Sloc.jj[i] = mapX[Sloc.jj[i]];
            }
            for(int i = 0; i < NUM_ITER; ++i)
                dist_spmm_dcomm(Xloc, Yloc, Sloc, comm_pre, comm_post, cartcomm);
            // re-map local rows/cols in Cloc ///
            for(size_t i = 0; i < Sloc.nnz; ++i){
                Sloc.ii[i] = mapYI[Sloc.ii[i]];
                Sloc.jj[i] = mapXI[Sloc.jj[i]];
            }

            //            print_numerical_sum(Cloc, zcomm, cartcomm);

        //    MPI_Comm_free(&comm_pre.commX);
          //  MPI_Comm_free(&comm_pre.commY);
        }
    }

    MPI_Comm_free(&xycomm);
    MPI_Comm_free(&zcomm);
    MPI_Comm_free(&cartcomm);
    MPI_Finalize();
    return 0;
}

