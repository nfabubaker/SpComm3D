#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <bits/getopt_core.h>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <algorithm>
#include <ios>



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
    if ( optind < argc && optind+2 < argc)
    {
        inFN = argv[optind];
        X = atoi(argv[optind+1]);
        Y = atoi(argv[optind+2]);
    }
    else{
        fprintf(stderr, "Error, call as rd [optional arguments] /path/to/matrix X Y");
        exit(EXIT_FAILURE);
    }
    
}


void distr_2D(SpKernels::coo_mtx& C, string outputFN, int X, int Y){
    vector<int> rpvec2D, cpvec2D;
    std::vector<idx_t> frontMeshCnts;
    int N = X*Y;

    rpvec2D.resize(C.grows); cpvec2D.resize(C.gcols);
    std::vector<idx_t> row_space(C.grows), col_space(C.gcols);
    for(size_t i = 0; i < C.grows; ++i) row_space[i] = i;
    for(size_t i = 0; i < C.gcols; ++i) col_space[i] = i;
    std::random_shuffle(row_space.begin(), row_space.end());
    std::random_shuffle(col_space.begin(), col_space.end());
    /* assign Rows/Cols with RR */
    for(size_t i = 0, p=0; i < C.grows; ++i)
        rpvec2D[row_space[i]] = (p++) % X; 
    for(size_t i = 0, p=0; i < C.gcols; ++i) 
        cpvec2D[col_space[i]] = (p++) % Y; 

    std::vector<std::vector<idx_t>> mesh2DCnt(X,
            std::vector<idx_t> (Y, 0));
    /* now rows/cols are divided, now distribute nonzeros */
    /* first: count nnz per 2D block */
    for(auto& t : C.elms){
        mesh2DCnt[rpvec2D[t.row]][cpvec2D[t.col]]++; 
    }
    std::vector<std::vector<SpKernels::triplet>> M;
    M.resize(N);
    std::vector<int> tcnts(N,0);
    for(int i=0; i < N; ++i) 
        M[i].resize(mesh2DCnt[i/Y][i%Y]);

    for(auto& t : C.elms){
        int p = rpvec2D[t.row]*Y +  cpvec2D[t.col];
        M[p][tcnts[p]++] = t;
    }

    ofstream ofile(outputFN, ios_base::out);
    for(int i = 0; i < N; ++i) {
        idx_t msize = M[i].size();
        ofile.write((char *) &msize, sizeof(idx_t)); 
    }
    for(std::vector<SpKernels::triplet>& m : M)
       ofile.write((char *) m.data(), sizeof(idx_t) * m.size());
    
    ofile.close();
    

}


int main(int argc, char *argv[])
{
    string inFN, outFN = "";
    int X, Y, Z;

    parse_arguments(argc, argv, X, Y, Z, inFN, outFN);
    if(outFN == ""){
        outFN = inFN.substr(0, inFN.find_last_of(".")) + ".bin";
    }

    SpKernels::mm _mm(inFN);
    SpKernels::coo_mtx C;
    C = _mm.read_mm(inFN);
    distr_2D(C, outFN, X, Y);
    
    return 0;
}
