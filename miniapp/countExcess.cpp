
#include <cstdlib>
#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <vector>
#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <getopt.h>
#include <chrono>
#include "distribute.hpp"
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

        choice = getopt_long( argc, argv, "vh:k:c:i:",
                long_options, &option_index);

        if (choice == -1)
            break;

        switch( choice )
        {
            case 'k':
                f = atoi(optarg);
                break;
            case 'i':
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
    coo_mtx Sloc;
    Sloc.mtxName = mtxName;
    {
        std:: vector<int> rpvec2D, cpvec2D;
        /* distribute C */
        read_bin_parallel_distribute(filename, Sloc, rpvec2D, cpvec2D,
                cartcomm,xycomm, zcomm);
     
    Sloc.rank = rank;
    MPI_Comm_rank(zcomm, &Sloc.zrank);

    /* distribute Aloc and Bloc  */
    distribute3D_AB_respect_communication(rpvec2D, cpvec2D,
            rpvec, cpvec,
            Sloc, xycomm, zcomm, cartcomm);
    }
    idx_t lrowsCnt=0, lcolsCnt=0;
    MPI_Cart_coords(xycomm, myxyrank, 2, tdims.data());
    int myxcoord = tdims[0];
    int myycoord = tdims[1];
    for(size_t i=0; i < Sloc.grows; ++i){
        MPI_Cart_coords(xycomm, rpvec[i], 2, tdims.data());
        if(tdims[0] == myxcoord) lrowsCnt++;
        
    }
    for(size_t i=0; i < Sloc.gcols; ++i){ 
        MPI_Cart_coords(xycomm, cpvec[i], 2, tdims.data());
        if(tdims[1] == myycoord) lcolsCnt++; 
    }
    idx_t lcntsg=0, totcntsg=0;
    lrowsCnt += lcolsCnt;
    idx_t lcnts = Sloc.lrows + Sloc.lcols;
    MPI_Reduce(&lrowsCnt, &totcntsg, 1, MPI_IDX_T, MPI_SUM, 0, comm);
    MPI_Reduce(&lcnts, &lcntsg, 1, MPI_IDX_T, MPI_SUM, 0, comm);

    if( rank == 0)
        std::cout << mtxName <<" " << size << " "<< X << " " << Y<< " " << Z << " " << lcntsg << " " << totcntsg << " " << totcntsg - lcntsg << endl;

    MPI_Finalize();
    return 0;
}
