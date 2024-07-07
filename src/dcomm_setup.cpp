
#include "basic.hpp"
#include "denseComm.hpp"
#include "mpi.h"
#include "SparseMatrix.hpp"
#include <array>


using namespace DComm;
using namespace std;
void DComm::setup_3dspmm_bcast(
        cooMat& Aloc,
        const idx_t f,
        const int c,
        denseMatrix& Xloc,
        denseMatrix& Yloc, 
        std::vector<int>& rpvec, 
        std::vector<int>& cpvec,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        DenseComm& comm_pre,
        DenseComm& comm_post,
        std::vector<idx_t>& mapX, 
        std::vector<idx_t>& mapY
        )
{

    std::array<int, 2> dims, tarr1, tarr2;
    MPI_Comm xcomm, ycomm;
    std::array<int, 2> remdim = {false, true};
    MPI_Cart_sub(xycomm, remdim.data(), &xcomm); 
    remdim = {true, false};
    MPI_Cart_sub(xycomm, remdim.data(), &ycomm); 
    MPI_Cart_get(xycomm, 2, dims.data(), tarr1.data(), tarr2.data()); 
    int X = dims[0], Y = dims[1], myxcoord = tarr2[0], myycoord = tarr2[1];   


    /* setup Bcast comm */
    comm_pre.OP = 0; comm_pre.commYflag = true;
    comm_post.OP = 1; comm_post.commXflag = true;
    comm_post.commX = xcomm;
    comm_pre.commY = ycomm;
    comm_post.rqstsX.resize(Y); comm_pre.rqstsY.resize(X);
    comm_post.outDegreeX = dims[1]; 
    comm_pre.outDegreeY = dims[0]; 
    comm_post.bufferptrX = Yloc.data.data();
    comm_pre.bufferptrY = Xloc.data.data();
    comm_post.bcastXcnt.resize(comm_post.outDegreeX, 0);
    comm_pre.bcastYcnt.resize(comm_pre.outDegreeY,0);
    comm_post.bcastXdisp.resize(comm_post.outDegreeX+1, 0);
    comm_pre.bcastYdisp.resize(comm_pre.outDegreeY+1,0);
    vector<int> rpvecX(Aloc.nrows), cpvecY(Aloc.ncols);
    idx_t tt = 0;
    for(size_t i = 0; i < Aloc.nrows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[Aloc.ltgR[i]], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            rpvecX[tt++] = tarr1[1];
        }
    }
    tt = 0;
    for(size_t i = 0; i < Aloc.ncols; ++i){
        MPI_Cart_coords(xycomm, cpvec[Aloc.ltgC[i]], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            cpvecY[tt++] = tarr1[0];
        }
    }
    for(size_t i = 0; i < Aloc.gnrows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[i], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            comm_post.bcastXcnt[tarr1[1]]++;
        }
    }
    for(size_t i = 0; i < Aloc.gncols; ++i){
        MPI_Cart_coords(xycomm, cpvec[i], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            comm_pre.bcastYcnt[tarr1[0]]++;
        }
    }

    for(size_t i = 1; i < Y+1; ++i) {
        comm_post.bcastXdisp[i] = comm_post.bcastXdisp[i-1] + comm_post.bcastXcnt[i-1];} 
    for(size_t i = 1; i < X+1; ++i) {
        comm_pre.bcastYdisp[i] = comm_pre.bcastYdisp[i-1] + comm_pre.bcastYcnt[i-1];} 

    for (size_t i = 0; i < Aloc.nrows; ++i) mapY[i] = comm_post.bcastXdisp[rpvecX[i]]++; 
    for (size_t i = 0; i < Aloc.ncols; ++i) mapX[i] = comm_pre.bcastYdisp[cpvecY[i]]++; 
/*     for (size_t i = 0; i < Cloc.grows; ++i)
 *         if(Cloc.gtlR[i] != -1) Cloc.gtlR.at(i) = mapA[Cloc.gtlR.at(i)]; 
 *     for (size_t i = 0; i < Cloc.gcols; ++i)
 *         if(Cloc.gtlC[i] != -1) Cloc.gtlC.at(i) = mapB[Cloc.gtlC.at(i)]; 
 */

    std::fill(comm_post.bcastXdisp.begin(), comm_post.bcastXdisp.end(), 0);
    std::fill(comm_pre.bcastYdisp.begin(), comm_pre.bcastYdisp.end(), 0);
    for(size_t i = 1; i < Y+1; ++i) {
        comm_post.bcastXdisp[i] = comm_post.bcastXdisp[i-1] + comm_post.bcastXcnt[i-1];} 
    for(size_t i = 1; i < X+1; ++i) {
        comm_pre.bcastYdisp[i] = comm_pre.bcastYdisp[i-1] + comm_pre.bcastYcnt[i-1];} 

    for(size_t i = 0; i < Y; ++i) {
        comm_post.bcastXcnt[i] *= Yloc.n;
        comm_post.bcastXdisp[i] *= Yloc.n;
    }
    for(size_t i = 0; i < X; ++i) {
        comm_pre.bcastYcnt[i] *= Xloc.n;
        comm_pre.bcastYdisp[i] *= Xloc.n;
    }
}

void DComm::setup_3dsddmm_bcast(
        cooMat& Cloc,
        const idx_t f,
        const int c,
        denseMatrix& Aloc,
        denseMatrix& Bloc, 
        std::vector<int>& rpvec, 
        std::vector<int>& cpvec,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        DenseComm& comm_pre,
        DenseComm& comm_post,
        std::vector<idx_t>& mapA, 
        std::vector<idx_t>& mapB
        )
{

    std::array<int, 2> dims, tarr1, tarr2;
    MPI_Comm xcomm, ycomm;
    std::array<int, 2> remdim = {false, true};
    MPI_Cart_sub(xycomm, remdim.data(), &xcomm); 
    remdim = {true, false};
    MPI_Cart_sub(xycomm, remdim.data(), &ycomm); 
    MPI_Cart_get(xycomm, 2, dims.data(), tarr1.data(), tarr2.data()); 
    int X = dims[0], Y = dims[1], myxcoord = tarr2[0], myycoord = tarr2[1];   


    /* setup Bcast comm */
    comm_pre.OP = 0;
    comm_pre.commXflag = true; comm_pre.commYflag = true;
    comm_pre.commX = xcomm; comm_pre.commY = ycomm;
    comm_pre.rqstsX.resize(Y); comm_pre.rqstsY.resize(X);
    comm_pre.outDegreeX = dims[1]; comm_pre.outDegreeY = dims[0]; 
    comm_pre.bufferptrX = Aloc.data.data();
    comm_pre.bufferptrY = Bloc.data.data();
    comm_pre.bcastXcnt.resize(comm_pre.outDegreeX, 0);
    comm_pre.bcastYcnt.resize(comm_pre.outDegreeY,0);
    comm_pre.bcastXdisp.resize(comm_pre.outDegreeX+1, 0);
    comm_pre.bcastYdisp.resize(comm_pre.outDegreeY+1,0);
    vector<int> rpvecX(Cloc.nrows), cpvecY(Cloc.ncols);
    idx_t tt = 0;
    for(size_t i = 0; i < Cloc.nrows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[Cloc.ltgR[i]], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            rpvecX[tt++] = tarr1[1];
        }
    }
    tt = 0;
    for(size_t i = 0; i < Cloc.ncols; ++i){
        MPI_Cart_coords(xycomm, cpvec[Cloc.ltgC[i]], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            cpvecY[tt++] = tarr1[0];
        }
    }
    for(size_t i = 0; i < Cloc.gnrows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[i], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            comm_pre.bcastXcnt[tarr1[1]]++;
        }
    }
    for(size_t i = 0; i < Cloc.gncols; ++i){
        MPI_Cart_coords(xycomm, cpvec[i], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            comm_pre.bcastYcnt[tarr1[0]]++;
        }
    }

    for(size_t i = 1; i < Y+1; ++i) {
        comm_pre.bcastXdisp[i] = comm_pre.bcastXdisp[i-1] + comm_pre.bcastXcnt[i-1];} 
    for(size_t i = 1; i < X+1; ++i) {
        comm_pre.bcastYdisp[i] = comm_pre.bcastYdisp[i-1] + comm_pre.bcastYcnt[i-1];} 

    for (size_t i = 0; i < Cloc.nrows; ++i) mapA[i] = comm_pre.bcastXdisp[rpvecX[i]]++; 
    for (size_t i = 0; i < Cloc.ncols; ++i) mapB[i] = comm_pre.bcastYdisp[cpvecY[i]]++; 
/*     for (size_t i = 0; i < Cloc.grows; ++i)
 *         if(Cloc.gtlR[i] != -1) Cloc.gtlR.at(i) = mapA[Cloc.gtlR.at(i)]; 
 *     for (size_t i = 0; i < Cloc.gcols; ++i)
 *         if(Cloc.gtlC[i] != -1) Cloc.gtlC.at(i) = mapB[Cloc.gtlC.at(i)]; 
 */

    std::fill(comm_pre.bcastXdisp.begin(), comm_pre.bcastXdisp.end(), 0);
    std::fill(comm_pre.bcastYdisp.begin(), comm_pre.bcastYdisp.end(), 0);
    for(size_t i = 1; i < Y+1; ++i) {
        comm_pre.bcastXdisp[i] = comm_pre.bcastXdisp[i-1] + comm_pre.bcastXcnt[i-1];} 
    for(size_t i = 1; i < X+1; ++i) {
        comm_pre.bcastYdisp[i] = comm_pre.bcastYdisp[i-1] + comm_pre.bcastYcnt[i-1];} 

    for(size_t i = 0; i < Y; ++i) {
        comm_pre.bcastXcnt[i] *= Aloc.n;
        comm_pre.bcastXdisp[i] *= Aloc.n;
    }
    for(size_t i = 0; i < X; ++i) {
        comm_pre.bcastYcnt[i] *= Bloc.n;
        comm_pre.bcastYdisp[i] *= Bloc.n;
    }
    /* now set up for redue_scatter */
    int myzrank, zsize;
    MPI_Comm_size(zcomm, &zsize);
    MPI_Comm_rank(zcomm, &myzrank);
    comm_post.commXflag = true; comm_post.commYflag = true;
    comm_post.commX = zcomm;
    comm_post.OP = 1;
    comm_post.bufferptrX = Cloc.vv.data();
    comm_post.bufferptrY = Cloc.ownedVals.data();
    comm_post.bcastXcnt.resize(zsize,0);
    comm_post.bcastXdisp.resize(zsize,0);
    idx_t nnz_pp = Cloc.nnz / zsize;
    idx_t nnz_pp_r = Cloc.nnz % zsize;
    for(int i =0; i < zsize; ++i)
        comm_post.bcastXcnt[i] = nnz_pp + (i< nnz_pp_r ? 1:0);
    for(int i =1; i < zsize; ++i)
        comm_post.bcastXdisp[i] = comm_post.bcastXdisp[i-1] + comm_post.bcastXcnt[i-1];
    assert(Cloc.ownedVals.size() == comm_post.bcastXcnt[myzrank]); 
    
    /* re-localize C nonzeros */
/*     for(size_t i = 0; i < Cloc.lnnz; ++i){
 *         idx_t row = Cloc.elms.at(i).row;
 *         Cloc.elms.at(i).row = Cloc.gtlR.at(row);
 *         idx_t col = Cloc.elms.at(i).col;
 *         Cloc.elms.at(i).col = Cloc.gtlC.at(col);
 *     }
 */


}
