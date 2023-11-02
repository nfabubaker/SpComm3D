#include "../src/mm.hpp"
#include <algorithm>
#include <getopt.h>



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


int main(int argc, char *argv[])
{
    string inFN, outFN = "";
    int X, Y, Z;

    parse_arguments(argc, argv, X, Y, Z, inFN, outFN);
    {
        SpKernels::mm _mm(inFN);
        SpKernels::coo_mtx C;
        C = _mm.read_mm(inFN);
        vector<bool> rowsF(C.grows, false), colsF(C.gcols, false);
        for(auto& el : C.elms){
            rowsF.at(el.row) = true; 
            colsF.at(el.col) = true;
        }
        idx_t emptyRows = count(rowsF.begin(), rowsF.end(), false);
        idx_t emptyCols = count(colsF.begin(), colsF.end(), false);
        cout << "empty rows: "<<emptyRows << " empty cols: " << emptyCols << endl;
    }
    
    return 0;
}

