#include <cmath>
#include <stdlib.h>
#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <getopt.h>
#include <algorithm>
#include "../src/Mesh3D.hpp"




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


void findXY(const int N, int& X, int& Y){
   int sqrtN = sqrt(N); 
   X = sqrtN; Y = sqrtN;
   for(int i = sqrtN; i >= 1; --i) {
       if(N % i == 0){ X = i; Y = N/i;}
   }
}

void dist_C(coo_mtx&  ){
}

int main(int argc, char *argv[])
{

    string filename;
    std::string::size_type const p(filename.find_last_of('.'));
    std::string mtxName = filename.substr(0, p);
    mtxName = mtxName.substr(mtxName.find_last_of("/\\") +1);
    idx_t f;
    int c, N;
    coo_mtx C;
    process_args(argc, argv, f, c, filename);
    
    mm _mm(filename); 
    C = _mm.read_mm(filename);
    

    /* distribute C */
        
    int X, Y, Z;
    Z = c;
    findXY(N/c, X, Y);
    Mesh3D mesh(X, Y, Z);
    vector<int> rpvec2D, cpvec2D;
    std::vector<idx_t> frontMeshCnts;
        
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
            frontMeshCnts.resize(N, 0);
            for(int i = 0; i < X; ++i){
                for (int j= 0; j < Y; ++j){
                    frontMeshCnts[mesh.getRankFromCoords(i, j, 0)] = mesh2DCnt[i][j];
                }
            }
            std::vector<std::vector<triplet>> M;
            M.resize(N);
            std::vector<int> tcnts(N,0);
            for(int i=0; i < N; ++i) 
                if(frontMeshCnts[i] > 0) 
                    M[i].resize(frontMeshCnts[i]);

            for(auto& t : C.elms){
                int p = mesh.getRankFromCoords(rpvec2D[t.row], cpvec2D[t.col], 0);
                M[p][tcnts[p]++] = t;
            }
    
    
    return 0;
}

