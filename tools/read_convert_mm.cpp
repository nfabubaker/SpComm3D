#include "basic.hpp"
#include "mm.hpp"
#include <bits/getopt_core.h>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <algorithm>
#include <ios>
#include <pthread.h>
#include <string>
#include "SparseMatrix.hpp"

using namespace SpKernels;

void parse_arguments(int argc, char* argv[], int& X, int& Y, int&Z, string& inFN, string& outFN){
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
    
        choice = getopt_long( argc, argv, "vh:o:z:",
                    long_options, &option_index);
    
        if (choice == -1)
            break;
    
        switch( choice )
        {
            case 'v':
                
                break;
    
            case 'h':
                
                break;
            case 'o':
               outFN = optarg;
                break;
            case 'z':
               Z = atoi(optarg);
                break;
    
            case '?':
                /* getopt_long will have already printed an error */
                break;
    
            default:
                /* Not sure how to get here... */
                exit(EXIT_FAILURE);
        }
    }
    
    /* Deal with non-option arguments here */
    if ( optind < argc)
    {
        inFN = argv[optind];
    }
    else{
        fprintf(stderr, "Error, call as rd [optional arguments] /path/to/matrix X Y");
        exit(EXIT_FAILURE);
    }
    
}

void convert_to_binary(SpKernels::cooMat& C, string outputFN){
    ofstream ofile(outputFN, ios_base::out);
    ofile.write((char *) &C.gnrows, sizeof(idx_t)); 
    ofile.write((char *) &C.gncols, sizeof(idx_t)); 
    ofile.write((char *) &C.gnnz, sizeof(idx_large_t)); 
    for(idx_large_t i = 0; i < C.gnnz; ++i){
       ofile.write((char *) &C.ii[i], sizeof(idx_t));
       ofile.write((char *) &C.jj[i], sizeof(idx_t));
       ofile.write((char *) &C.vv[i], sizeof(real_t));
    }
    
    ofile.close();

}

void test_binary_serial(string inFN, SpKernels::cooMat& C){
    ifstream infile(inFN, ios::out | ios::binary);
    if(!infile){
        cout << "cannot open file!" << endl;
        exit(EXIT_FAILURE);
    }
    infile.read((char *) &C.gnrows, sizeof(idx_t));
    infile.read((char *) &C.gncols, sizeof(idx_t));
    infile.read((char *) &C.gnnz, sizeof(idx_large_t));
    C.ii.resize(C.gnnz);
    C.jj.resize(C.gnnz);
    C.vv.resize(C.gnnz);
    for(idx_large_t i = 0; i < C.gnnz; ++i){
        infile.read((char *) &C.ii[i], sizeof(idx_t));
        infile.read((char *) &C.jj[i], sizeof(idx_t));
        infile.read((char *) &C.vv[i], sizeof(real_t));
    }
    infile.close();

}


int main(int argc, char *argv[])
{
    string inFN, outFN = "";
    int X, Y, Z;

    parse_arguments(argc, argv, X, Y, Z, inFN, outFN);
    if(outFN == ""){
        outFN = inFN.substr(0, inFN.find_last_of(".")) + ".bin";
    }

    {
        SpKernels::mm _mm(inFN);
        SpKernels::cooMat C;
        _mm.read_mm(inFN, C);
        convert_to_binary(C, outFN);
    }
/*     {
 * 
 *         SpKernels::cooMat C;
 *         test_binary_serial(outFN, C);
 *         cout << C.gnrows << C.gncols << C.gnnz << endl;
 *         C.printMatrix(20);
 *     }
 */

    
    return 0;
}
