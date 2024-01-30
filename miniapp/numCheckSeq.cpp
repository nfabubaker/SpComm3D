#include <numeric>
#include <stdlib.h>
#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <getopt.h>
#include <algorithm>
#include "SparseMatrix.hpp"
#include <math.h>
#include "core_ops.hpp"






using namespace SpKernels;

void process_args(int argc, char *argv[], int& N, idx_t& f, int& c, idx_t& input_dim, double& nnzDensity, string& filename){
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

        choice = getopt_long( argc, argv, "vh:k:c:n:m:i:d:",
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
            case 'n':
                N = atoi(optarg);
                break;
            case 'i':
                input_dim = atoi(optarg);
                break;
            case 'd':
                nnzDensity = atof(optarg);
                break;
            case 'm':
                filename = optarg;
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

    string filename;
    idx_t f, input_dim;
    int N, c;
    double nnzDensity;
    process_args(argc, argv, N, f, c, input_dim, nnzDensity, filename);
    std::string::size_type const p(filename.find_last_of('.'));
    std::string mtxName = filename.substr(0, p);
    mtxName = mtxName.substr(mtxName.find_last_of("/\\") +1);

    cooMat C, S;

    fprintf(stderr, "Hallo! mtxName=%s f=%d\n", mtxName.c_str(), f);
    mm _mm(filename);
    _mm.read_mm(filename, S);
    C = S;
    denseMatrix A, B;
    A.m = S.gnrows; A.n = f;
    B.m = S.gncols; B.n = f;
    fprintf(stderr, "mtxName=%s f=%d nrows=%u ncols=%d nnz=%lu\n", mtxName.c_str(), f, C.gnrows, C.gncols, C.gnnz );
    A.data.resize(A.m *A.n);
    B.data.resize(B.m *B.n);
    for(size_t i = 0; i < A.m; ++i)
        for(size_t j = 0; j < A.n; ++j) A.at(i, j) = i/(real_t)1000;
    for(size_t i = 0; i < B.m; ++i)
        for(size_t j = 0; j < B.n; ++j) B.at(i, j) = i/(real_t)1000;

    sddmm(A, B, S, C);
    real_t resSum = accumulate(C.vv.begin(), C.vv.end(), 0.0);
    cout << resSum << endl;
    return 0;
}

