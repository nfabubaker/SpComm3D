#include <numeric>
#include <stdlib.h>
#include "../src/basic.hpp"
#include "../src/mm.hpp"
#include <getopt.h>
#include <algorithm>
#include "../src/Mesh3D.hpp"
#include <math.h>





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
//    if ( optind < argc )
//    {
//        //filename = argv[optind];
//        /*        while ( optind < argc )
//         *        {
//         *            
//         *        }
//         */
//    }
//    else{
//        printf("usage: sddmm [-k <k value>] [-c <c value] /path/to/matrix");
//        exit(EXIT_FAILURE);
//    }
    return;

}


void findXY(const int N, int& X, int& Y){
    int sqrtN = sqrt(N); 
    X = sqrtN; Y = sqrtN;
    for(int i = sqrtN; i >= 1; --i) {
        if(N % i == 0){ X = i; Y = N/i; break;}
    }
}


void serialSim_instance(coo_mtx& C, int N, idx_t f, idx_t input_dim, int c, double nnzDensity, string& mtxName)
{
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

    /* now distribute both A & B  */
    std::vector<size_t> cntsPer2Drow(X,0);
    std::vector<size_t> cntsPer2Dcol(Y,0);
    std::vector<int> rpvec(C.grows), cpvec(C.gcols);
    std::vector<idx_t> ownerRCnts(N, 0), ownerCCnts(N, 0);
    for(size_t i=0; i < C.grows; ++i){
        int rowid2D = rpvec2D[i];
        int p = mesh.getRankFromCoords(rowid2D, (int)cntsPer2Drow[rowid2D]++ % Y, 0);
        rpvec[i] = p;
        ownerRCnts[p]++;
    }
    for(size_t i=0; i < C.gcols; ++i){
        int colid2D = cpvec2D[i];
        int p = mesh.getRankFromCoords((int)cntsPer2Dcol[colid2D]++ % X, colid2D, 0);
        cpvec[i] = p;
        ownerCCnts[p]++;
    }
    typedef struct _plocal{
        vector<idx_t> sendCnt;
        vector<idx_t> recvCnt;
    } plocal;

    std::vector<plocal> ALLP(N);
    for(auto& pl : ALLP){
        pl.sendCnt.resize(N, 0);
        pl.recvCnt.resize(N, 0);
    }
    std::vector<bool> rowFlags(C.grows), colFlags(C.gcols);
    for(int p = 0; p < N/Z; ++p){
        /* go over each processor's nonzeros */
        std::fill(rowFlags.begin(), rowFlags.end(), false);
        std::fill(colFlags.begin(), colFlags.end(), false);
        for(size_t i = 0; i < M[p].size(); ++i){
            idx_t row, col;
            row = M[p][i].row; col = M[p][i].col;
            rowFlags[row] = true; colFlags[col] = true;
        }
        for(size_t row = 0; row < C.grows; ++row){
            int rowOwner;
            rowOwner = rpvec[row];
            if(rowFlags[row] == true && rowOwner != p){
                ALLP[p].recvCnt[rowOwner]++;
                ALLP[rowOwner].sendCnt[p]++;
            }
        }
        for(size_t col = 0; col < C.gcols; ++col){
            int colOwner = cpvec[col];
            if(colFlags[col] == true && colOwner != p){
                ALLP[p].recvCnt[colOwner]++;
                ALLP[colOwner].sendCnt[p]++;
            }
        }
    }
    
    idx_t maxSMsg = 0, maxRMsg = 0, maxSVol=0, maxRVol=0,
          totSMsg = 0, totRMsg = 0, totSVol = 0, totRVol = 0;
    for(int p = 0; p < N/Z; ++p){
        idx_t mySMsgs, myRMsgs, mySVol, myRVol;
        mySMsgs = std::count_if(ALLP[p].sendCnt.begin(), ALLP[p].sendCnt.end(), [](int i){return i > 0;} );
        myRMsgs = std::count_if(ALLP[p].recvCnt.begin(), ALLP[p].recvCnt.end(), [](int i){return i > 0;});
        mySVol = std::accumulate(ALLP[p].sendCnt.begin(), ALLP[p].sendCnt.end(), 0.0);
        myRVol = std::accumulate(ALLP[p].recvCnt.begin(), ALLP[p].recvCnt.end(), 0.0);
        if(mySMsgs > maxSMsg) maxSMsg = mySMsgs;
        if(myRMsgs > maxRMsg) maxRMsg = myRMsgs;
        if(mySVol > maxSVol) maxSVol = mySVol;
        if(myRVol > maxRVol) maxRVol = myRVol;
        totSMsg += mySMsgs; totRMsg += myRMsgs;
        totSVol += mySVol; totRVol += myRVol;
    }
    totSVol *= Z; totRVol*=Z; totSMsg*=Z; totRMsg*=Z;
    printf("%s %lu %lu %lu %d %dx%dx%d %d %d S %.2f %.2f %.2f %.2f %lu %lu %lu %lu\n",
            mtxName.c_str(),C.grows, C.gcols, C.gnnz, N,X,Y,Z, f,c, 
            totSMsg/(float)N, totSVol/(float)N, totRMsg/(float)N, totRVol/(float)N,
            maxSMsg, maxSVol, maxRMsg, maxRVol);

    /* now Dense stats */
    maxSMsg = 0, maxRMsg = 0, maxSVol=0, maxRVol=0,
            totSMsg = 0, totRMsg = 0, totSVol = 0, totRVol = 0;
    for(int p = 0; p < N/Z; ++p){
        idx_t mySMsgs, myRMsgs, mySVol, myRVol;
        mySMsgs = (X+Y) - 2; 
        myRMsgs = (X + Y) - 2;
        mySVol = (ownerRCnts[p]*(Y-1)) + (ownerCCnts[p]*(X-1));
        myRVol = (cntsPer2Dcol[mesh.getYCoord(p)] - ownerCCnts[p]) + (cntsPer2Drow[mesh.getXCoord(p)] - ownerRCnts[p]);
        if(mySMsgs > maxSMsg) maxSMsg = mySMsgs;
        if(myRMsgs > maxRMsg) maxRMsg = myRMsgs;
        if(mySVol > maxSVol) maxSVol = mySVol;
        if(myRVol > maxRVol) maxRVol = myRVol;
        totSMsg += mySMsgs; totRMsg += myRMsgs;
        totSVol += mySVol; totRVol += myRVol;
    }
    totSVol *= Z; totRVol*=Z; totSMsg*=Z; totRMsg*=Z;
    printf("%s %lu %lu %lu %d %dx%dx%d %d %d B %.2f %.2f %.2f %.2f %lu %lu %lu %lu\n",
            mtxName.c_str(),C.grows, C.gcols, C.gnnz, N, X,Y,Z, f,c, 
            totSMsg/(float)N, totSVol/(float)N, totRMsg/(float)N, totRVol/(float)N,
            maxSMsg, maxSVol, maxRMsg, maxRVol);
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
/*     coo_mtx C;
 *     mm _mm(filename); 
 *     if(filename.empty() == true){
 *         C.grows = input_dim;
 *         C.gcols = input_dim;
 *         C.gnnz = nnzDensity * input_dim * input_dim;
 *         C.self_generate_random(C.gnnz);
 *     }
 *     else
 *         C = _mm.read_mm(filename);
 */

    for(double nnzD = 1e-7; nnzD <= 1e-3; nnzD*=10.0){
        coo_mtx C;
        C.grows = input_dim;
        C.gcols = input_dim;
        C.gnnz = nnzD * input_dim * input_dim;
        C.self_generate_random(C.gnnz);
        mtxName = "SYNTH";
        for(int K = 64; K<= 32768; K*=2)
            serialSim_instance(C, K, f, input_dim,c, nnzD, mtxName);
    }


    return 0;
}

