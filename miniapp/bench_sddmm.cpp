#include <mpi.h>
#include <numeric>
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
    cooMat Cloc, Sloc;
    Cloc.mtxName = mtxName;
    Sloc.mtxName = mtxName;
    {
        std:: vector<int> rpvec2D, cpvec2D;
        /* distribute C */
        read_bin_parallel_distribute_coo(filename, Sloc, rpvec2D, cpvec2D,
                cartcomm,xycomm, zcomm);
        Sloc.localizeIndices();
        /* copy C to S and reset values in S */
        Cloc = Sloc;


        Cloc.rank = rank;
        MPI_Comm_rank(zcomm, &Cloc.zrank);
        Sloc.rank = rank;
        /* distribute Aloc and Bloc  */
        //distribute3D_AB_random(rpvec2D, cpvec2D, rpvec, cpvec, Cloc, f, xycomm);
        distribute3D_AB_respect_communication(rpvec2D, cpvec2D,
                rpvec, cpvec,
                Cloc, f, xycomm, zcomm, cartcomm);
    }
    { /* distribute A,B and respect communication, setup sparse comm*/
        for(auto& elm : Cloc.vv) elm = 0.0;
        SparseComm<real_t> comm_expand;
        DenseComm comm_reduce;
        denseMatrix Aloc, Bloc;
        /* prepare Aloc, Bloc according to local dims of Cloc */
        // split the 3D mesh communicator to 2D slices 

        Aloc.m = Cloc.nrows; Aloc.n = floc;
        Bloc.m = Cloc.ncols; Bloc.n = floc;
        Aloc.data.resize(Aloc.m * Aloc.n, myxyrank+1);
        Bloc.data.resize(Bloc.m * Bloc.n, myxyrank+1);
        setup_3dsddmm(Cloc, f, c, xycomm, zcomm, Aloc, Bloc, rpvec, cpvec, comm_expand, comm_reduce); 

        dist_sddmm_spcomm(Aloc, Bloc, Sloc, comm_expand, comm_reduce, Cloc, cartcomm);
        print_numerical_sum(Cloc, zcomm, cartcomm);
    }
    { /* sparse sddmm instance#2: no recv buffer*/  
        for(auto& elm : Cloc.vv) elm = 0.0;
        std::fill(Cloc.ownedVals.begin(), Cloc.ownedVals.end(), 0.0);
        SparseComm<real_t> comm_expandA;
        SparseComm<real_t> comm_expandB;
        DenseComm comm_reduce;
        denseMatrix Aloc, Bloc;
        /* prepare Aloc, Bloc according to local dims of Cloc */
        // split the 3D mesh communicator to 2D slices 

        Aloc.m = Cloc.nrows; Aloc.n = floc;
        Bloc.m = Cloc.ncols; Bloc.n = floc;
        Aloc.data.resize(Aloc.m * Aloc.n, myxyrank+1);
        Bloc.data.resize(Bloc.m * Bloc.n, myxyrank+1);
        std::vector<idx_t> mapA(Aloc.m, -1), mapB(Bloc.m, -1), mapAI(Aloc.m), mapBI(Bloc.m);
        setup_3dsddmm_NoRecvBuffer(Cloc, f, c, xycomm, zcomm, Aloc, Bloc, rpvec, cpvec, comm_expandA, comm_expandB, comm_reduce, mapA, mapB); 
        for(auto &elm : mapA) if(elm == (idx_t) -1) cout <<"ERROR mapA!" <<endl;
        for(auto &elm : mapB) if(elm == (idx_t) -1) cout <<"ERROR mapB!" <<endl;
        for(idx_t i = 0; i < Aloc.m; ++i) mapAI[mapA[i]] = i;
        for(idx_t i = 0; i < Bloc.m; ++i) mapBI[mapB[i]] = i;

        Sloc.ReMapIndices(mapA, mapB);
        Cloc.ReMapIndices(mapA, mapB);
        dist_sddmm_spcomm2(Aloc, Bloc, Sloc, comm_expandA, comm_expandB, comm_reduce, Cloc, cartcomm);
        Sloc.ReMapIndices(mapAI, mapBI);
        Cloc.ReMapIndices(mapAI, mapBI);
        print_numerical_sum(Cloc, zcomm, cartcomm);
    }
    { /* sparse sddmm instance#3: no buffers*/  
        for(auto& elm : Cloc.vv) elm = 0.0;
        std::fill(Cloc.ownedVals.begin(), Cloc.ownedVals.end(), 0.0);
        SparseComm<real_t> comm_expandA;
        SparseComm<real_t> comm_expandB;
        DenseComm comm_reduce;
        denseMatrix Aloc, Bloc;
        /* prepare Aloc, Bloc according to local dims of Cloc */
        // split the 3D mesh communicator to 2D slices 

        Aloc.m = Cloc.nrows; Aloc.n = floc;
        Bloc.m = Cloc.ncols; Bloc.n = floc;
        Aloc.data.resize(Aloc.m * Aloc.n, myxyrank+1);
        Bloc.data.resize(Bloc.m * Bloc.n, myxyrank+1);
        std::vector<idx_t> mapA(Aloc.m, -1), mapB(Bloc.m, -1), mapAI(Aloc.m), mapBI(Bloc.m);
        setup_3dsddmm_NoBuffers(Cloc, f, c, xycomm, zcomm, Aloc, Bloc, rpvec, cpvec, comm_expandA, comm_expandB, comm_reduce, mapA, mapB); 
        for(auto &elm : mapA) if(elm == (idx_t) -1) cout <<"ERROR mapA!" <<endl;
        for(auto &elm : mapB) if(elm == (idx_t) -1) cout <<"ERROR mapB!" <<endl;
        for(idx_t i = 0; i < Aloc.m; ++i) mapAI[mapA[i]] = i;
        for(idx_t i = 0; i < Bloc.m; ++i) mapBI[mapB[i]] = i;

        Sloc.ReMapIndices(mapA, mapB);
        Cloc.ReMapIndices(mapA, mapB);
        dist_sddmm_spcomm3(Aloc, Bloc, Sloc, comm_expandA, comm_expandB, comm_reduce, Cloc, cartcomm);
        Sloc.ReMapIndices(mapAI, mapBI);
        Cloc.ReMapIndices(mapAI, mapBI);
        print_numerical_sum(Cloc, zcomm, cartcomm);
    }
    /* instance #2: dense */
    {
        for(auto& elm : Cloc.vv) elm = 0.0;
        std::fill(Cloc.ownedVals.begin(), Cloc.ownedVals.end(), 0.0);
        fill(Cloc.vv.begin(), Cloc.vv.end(), 0);
        DenseComm comm_pre, comm_post;
        denseMatrix Aloc, Bloc;
        std::unordered_map<idx_t, idx_t> gtlR, gtlC;
        std::vector<idx_t> mapA(Cloc.nrows), mapB(Cloc.ncols);
        create_AB_Bcast(Cloc, floc, rpvec, cpvec, xycomm, Aloc, Bloc);
        std::vector<idx_t> mapAI(Aloc.m), mapBI(Bloc.m);
        setup_3dsddmm_bcast(Cloc,f,c, Aloc, Bloc, rpvec, cpvec,
                xycomm, zcomm,  comm_pre, comm_post, mapA, mapB);
        for(idx_t i = 0; i < Cloc.nrows; ++i) mapAI[mapA[i]] = i;
        for(idx_t i = 0; i < Cloc.ncols; ++i) mapBI[mapB[i]] = i;
        // re-map local rows/cols in Cloc 
        for(size_t i = 0; i < Cloc.nnz; ++i){
            Cloc.ii[i] = Sloc.ii[i] = mapA[Cloc.ii[i]];
            Cloc.jj[i] = Sloc.jj[i] = mapB[Cloc.jj[i]];
        }
        dist_sddmm_dcomm(Aloc, Bloc, Sloc, comm_pre, comm_post, Cloc, cartcomm);
        // re-map local rows/cols in Cloc ///
        for(size_t i = 0; i < Cloc.nnz; ++i){
            Cloc.ii[i] = Sloc.ii[i] = mapAI[Cloc.ii[i]];
            Cloc.jj[i] = Sloc.jj[i] = mapBI[Cloc.jj[i]];
        }

            print_numerical_sum(Cloc, zcomm, cartcomm);
        
        MPI_Comm_free(&comm_pre.commX);
        MPI_Comm_free(&comm_pre.commY);
    }

    MPI_Comm_free(&xycomm);
    MPI_Comm_free(&zcomm);
    MPI_Comm_free(&cartcomm);
    MPI_Finalize();
    return 0;
}
