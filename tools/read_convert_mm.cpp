#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <bits/getopt_core.h>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <algorithm>
#include <ios>
#include <string>



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

void convert_to_binary(SpKernels::coo_mtx& C, string outputFN){
    ofstream ofile(outputFN, ios_base::out);
    ofile.write((char *) &C.grows, sizeof(idx_t)); 
    ofile.write((char *) &C.gcols, sizeof(idx_t)); 
    ofile.write((char *) &C.gnnz, sizeof(idx_t)); 
    for(SpKernels::triplet& m : C.elms){
       ofile.write((char *) &m, sizeof(SpKernels::triplet));
    }
    
    ofile.close();

}

void test_binary_serial(string inFN, SpKernels::coo_mtx& C){
    ifstream infile(inFN, ios::out | ios::binary);
    if(!infile){
        cout << "cannot open file!" << endl;
        exit(EXIT_FAILURE);
    }
    infile.read((char *) &C.grows, sizeof(idx_t));
    infile.read((char *) &C.gcols, sizeof(idx_t));
    infile.read((char *) &C.gnnz, sizeof(idx_t));
    C.elms.resize(C.gnnz);
    for(size_t i = 0; i < C.gnnz; ++i){
        infile.read((char *) &C.elms[i], sizeof(SpKernels::triplet));
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
        SpKernels::coo_mtx C;
        C = _mm.read_mm(inFN);
        convert_to_binary(C, outFN);
    }
    {

        SpKernels::coo_mtx C;
        test_binary_serial(outFN, C);
        cout << C.grows << C.gcols << C.gnnz << endl;
        C.printMatrix(C.gnnz);

    }
    
    return 0;
}
