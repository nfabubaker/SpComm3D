#include "basic.hpp"
#include "comm.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <streambuf>
#include <unordered_map>
#include "denseComm.hpp"
#include "distribute.hpp"
#include "mpi.h"
#include "SparseMatrix.hpp"



using namespace SpKernels;
using namespace DComm;
using namespace std;


void setup_spmm_side(denseMatrix& Dloc, cooMat& Sloc, int side, idx_t dimSize, unordered_map<idx_t, idx_t>& gtlD, vector<idx_t>& ltgD, vector<int>& pvec, SparseComm<real_t>& comm_handle,  MPI_Comm comm){

    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    /* step1: decide my rows and my cols (global) */
    vector<bool> myCols(dimSize, false);
    vector<idx_t> recvCount(size, 0);
    vector<idx_t> sendCount(size, 0);
    for(idx_large_t i = 0; i < Sloc.nnz; ++i){
        if(side == 0)
            myCols.at(ltgD.at(Sloc.ii[i])) = true;
        else
            myCols.at(ltgD.at(Sloc.jj[i])) = true;
    }

    /* step2: decide which rows/cols I will recv from which processor */
    for (size_t i = 0; i < dimSize; ++i) if(myCols.at(i)) recvCount.at(pvec.at(i))++; 

    /* exchange recv info, now each processor knows send count for each other processor */
    MPI_Alltoall(recvCount.data(), 1, MPI_IDX_T, sendCount.data(), 1, MPI_IDX_T, comm);
    idx_t totSendCnt = 0, totRecvCnt = 0;
    totSendCnt = accumulate(sendCount.begin(), sendCount.end(), totSendCnt);
    totSendCnt -= sendCount[myrank];
    totRecvCnt = accumulate(recvCount.begin(), recvCount.end(), totRecvCnt);
    totRecvCnt -= recvCount[myrank];

    /* exchange row/col IDs */
    SparseComm<idx_t> esc(1); /* esc: short for expand setup comm */
    esc.commT = SparseComm<idx_t>::P2P;
    esc.commP = comm;
    for(int i = 0; i < size; ++i){ 
        if(sendCount.at(i) > 0 && i != myrank) esc.inDegree++;
        if(recvCount.at(i) > 0 && i != myrank) esc.outDegree++;
    }
    esc.sendBuff.resize(totRecvCnt); esc.recvBuff.resize(totSendCnt);
    esc.inSet.resize(esc.inDegree); esc.outSet.resize(esc.outDegree);
    esc.recvCount.resize(esc.inDegree, 0); esc.sendCount.resize(esc.outDegree,0);
    esc.recvDisp.resize(esc.inDegree+2, 0); esc.sendDisp.resize(esc.outDegree+2,0);

    vector<int> gtlR(size,-1), gtlS(size,-1);
    int tcnt=0;
    for (int i = 0 ; i < size; ++i) {
        if(sendCount.at(i) > 0 && i != myrank ){ 
            gtlR.at(i) = tcnt; 
            esc.inSet.at(tcnt) = i;
            esc.recvCount.at(tcnt++) = sendCount.at(i);
        }
    }
    assert(tcnt == esc.inDegree);
    tcnt = 0;
    for (int i = 0; i < size; ++i) {
        if(recvCount.at(i) > 0 && i != myrank ){
            gtlS.at(i) = tcnt; 
            esc.outSet.at(tcnt) = i;
            esc.sendCount.at(tcnt++) = recvCount.at(i);
        }
    }
    assert(tcnt == esc.outDegree);
    for (int i = 0; i < esc.outDegree; ++i) { esc.sendDisp.at(i+2) = esc.sendDisp.at(i+1) + esc.sendCount.at(i);}
    for (int i = 0; i < esc.inDegree; ++i) { esc.recvDisp.at(i+1) = esc.recvDisp.at(i) + esc.recvCount.at(i);}

    /* Tell processors what rows/cols you want from them */
    /* 1 - determine what to send */
    for ( size_t i = 0; i < dimSize; ++i) {
        int ploc = (pvec[i] == -1 ? -1 : gtlR.at(pvec.at(i))); 
        if(ploc != -1 && myCols.at(i)){ 
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    esc.perform_sparse_comm(false, false);

    /* now recv from each processor available in recvBuff */
    idx_t f = Dloc.n;
    comm_handle.init(f, esc.outDegree, esc.inDegree, totSendCnt, totRecvCnt, SparseComm<real_t>::P2P, comm, MPI_COMM_NULL); 
    comm_handle.inSet = esc.outSet;
    comm_handle.outSet = esc.inSet;
    for(int i=0; i < esc.inDegree; ++i) {
        comm_handle.sendCount.at(i) = (esc.recvCount.at(i))*f;
        comm_handle.sendDisp.at(i+1) = (esc.recvDisp.at(i+1))*f;
    }
    for(int i=0; i < esc.outDegree; ++i) {
        comm_handle.recvCount.at(i) = (esc.sendCount.at(i))*f;
        comm_handle.recvDisp.at(i+1) = (esc.sendDisp.at(i+1))*f;
    }
    if(comm_handle.commT == SparseComm<real_t>::NEIGHBOR){
        MPI_Dist_graph_create_adjacent(comm, comm_handle.inDegree, comm_handle.inSet.data(),
                MPI_WEIGHTS_EMPTY, comm_handle.outDegree, comm_handle.outSet.data(), MPI_WEIGHTS_EMPTY, MPI_INFO_NULL, 0, &comm_handle.commN);
    }

    /* prepare expand sendbuff based on the row/col IDs I recv 
     * (what I send is what other processors tell me to send)*/
    for ( size_t i = 0;  i < esc.inDegree; ++ i) {
        idx_t d = esc.recvDisp.at(i);
        for (size_t j = 0; j < esc.recvCount.at(i) ; j++) {
            idx_t idx = esc.recvBuff.at(d+j);
            assert(gtlD.at(idx) < (idx_t)-1); 
            comm_handle.sendptr.at(d+j) = &Dloc.data.at(gtlD.at(idx) * f);
        }
    }
    /* prepare expand recvbuff based on row/col IDs I send
     * (what I recv is what I tell other processors to send) */
    for(size_t i = 0; i < esc.outDegree; ++i){
        idx_t d = esc.sendDisp.at(i);
        for (size_t j = 0; j < esc.sendCount.at(i); j++) {
            idx_t idx = esc.sendBuff.at(d+j);
            assert(gtlD.at(idx) < (idx_t) -1);
            comm_handle.recvptr.at(d+j) = &Dloc.data.at(gtlD.at(idx) * f);
        }
    }
    /* exchange row/col ids */
/*     for (int i = 0; i < esc.inDegree; ++i) {
 * 
 *     }
 */
    return;

ERR_EXIT:
    MPI_Finalize();
    exit(EXIT_FAILURE);
}

void setup_3dsddmm_expand(denseMatrix& Aloc, denseMatrix& Bloc, cooMat& Cloc, vector<int>& rpvec, vector<int>& cpvec, SparseComm<real_t>& comm_expand,  MPI_Comm comm){

    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    /* step1: decide my rows and my cols (global) */
    vector<bool> myRows(Cloc.gnrows,false), myCols(Cloc.gncols, false);
    vector<idx_t> recvCount(size, 0);
    vector<idx_t> sendCount(size, 0);
    for(idx_large_t i = 0; i < Cloc.nnz; ++i){
        myRows.at(Cloc.ltgR.at(Cloc.ii[i])) = true;
        myCols.at(Cloc.ltgC.at(Cloc.jj[i])) = true;
    }

    /* step2: decide which rows/cols I will recv from which processor */
    for (size_t i = 0; i < Cloc.gnrows; ++i) if(myRows.at(i)) recvCount.at(rpvec.at(i))++; 
    for (size_t i = 0; i < Cloc.gncols; ++i) if(myCols.at(i)) recvCount.at(cpvec.at(i))++; 

    /* exchange recv info, now each processor knows send count for each other processor */
    MPI_Alltoall(recvCount.data(), 1, MPI_IDX_T, sendCount.data(), 1, MPI_IDX_T, comm);
    idx_t totSendCnt = 0, totRecvCnt = 0;
    totSendCnt = accumulate(sendCount.begin(), sendCount.end(), totSendCnt);
    totSendCnt -= sendCount[myrank];
    totRecvCnt = accumulate(recvCount.begin(), recvCount.end(), totRecvCnt);
    totRecvCnt -= recvCount[myrank];

    /* exchange row/col IDs */
    SparseComm<idx_t> esc(2); /* esc: short for expand setup comm */
    esc.commT = SparseComm<idx_t>::P2P;
    esc.commP = comm;
    for(int i = 0; i < size; ++i){ 
        if(sendCount.at(i) > 0 && i != myrank) esc.inDegree++;
        if(recvCount.at(i) > 0 && i != myrank) esc.outDegree++;
    }
    esc.sendBuff.resize(totRecvCnt*2); esc.recvBuff.resize(totSendCnt*2);
    esc.inSet.resize(esc.inDegree); esc.outSet.resize(esc.outDegree);
    esc.recvCount.resize(esc.inDegree, 0); esc.sendCount.resize(esc.outDegree,0);
    esc.recvDisp.resize(esc.inDegree+2, 0); esc.sendDisp.resize(esc.outDegree+2,0);

    vector<int> gtlR(size,-1), gtlS(size,-1);
    int tcnt=0;
    for (int i = 0 ; i < size; ++i) {
        if(sendCount.at(i) > 0 && i != myrank ){ 
            gtlR.at(i) = tcnt; 
            esc.inSet.at(tcnt) = i;
            esc.recvCount.at(tcnt++) = sendCount.at(i)*2;
        }
    }
    assert(tcnt == esc.inDegree);
    tcnt = 0;
    for (int i = 0; i < size; ++i) {
        if(recvCount.at(i) > 0 && i != myrank ){
            gtlS.at(i) = tcnt; 
            esc.outSet.at(tcnt) = i;
            esc.sendCount.at(tcnt++) = recvCount.at(i)*2;
        }
    }
    assert(tcnt == esc.outDegree);
    for (int i = 0; i < esc.outDegree; ++i) { esc.sendDisp.at(i+2) = esc.sendDisp.at(i+1) + esc.sendCount.at(i);}
    for (int i = 0; i < esc.inDegree; ++i) { esc.recvDisp.at(i+1) = esc.recvDisp.at(i) + esc.recvCount.at(i);}

    /* Tell processors what rows/cols you want from them */
    /* 1 - determine what to send */
    for ( size_t i = 0; i < Cloc.gnrows; ++i) {
        int ploc = (rpvec[i] == -1 ? -1 : gtlR.at(rpvec.at(i))); 
        if( ploc != -1 && myRows.at(i)) {
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 0;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    for ( size_t i = 0; i < Cloc.gncols; ++i) {
        int ploc = (cpvec[i] == -1 ? -1 : gtlR.at(cpvec.at(i))); 
        if(ploc != -1 && myCols.at(i)){ 
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 1;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    esc.perform_sparse_comm(false, false);

    /* now recv from each processor available in recvBuff */
    idx_t f = Aloc.n;
    comm_expand.init(f, esc.outDegree, esc.inDegree, totSendCnt, totRecvCnt, SparseComm<real_t>::P2P, comm, MPI_COMM_NULL); 
    comm_expand.inSet = esc.outSet;
    comm_expand.outSet = esc.inSet;
    for(int i=0; i < esc.inDegree; ++i) {
        comm_expand.sendCount.at(i) = (esc.recvCount.at(i)/2)*f;
        comm_expand.sendDisp.at(i+1) = (esc.recvDisp.at(i+1)/2)*f;
    }
    for(int i=0; i < esc.outDegree; ++i) {
        comm_expand.recvCount.at(i) = (esc.sendCount.at(i)/2)*f;
        comm_expand.recvDisp.at(i+1) = (esc.sendDisp.at(i+1)/2)*f;
    }
    if(comm_expand.commT == SparseComm<real_t>::NEIGHBOR){
        MPI_Dist_graph_create_adjacent(comm, comm_expand.inDegree, comm_expand.inSet.data(),
                MPI_WEIGHTS_EMPTY, comm_expand.outDegree, comm_expand.outSet.data(), MPI_WEIGHTS_EMPTY, MPI_INFO_NULL, 0, &comm_expand.commN);
    }

    /* prepare expand sendbuff based on the row/col IDs I recv 
     * (what I send is what other processors tell me to send)*/
    for ( size_t i = 0;  i < esc.inDegree; ++ i) {
        idx_t d = esc.recvDisp.at(i);
        for (size_t j = 0; j < esc.recvCount.at(i) ; j+=2) {
            idx_t side = esc.recvBuff.at(d+j);
            idx_t idx = esc.recvBuff.at(d+j+1);
            if(side == 0){ 
                assert(Cloc.gtlR.at(idx) < (idx_t)-1 );
                comm_expand.sendptr.at((d/2)+(j/2)) = 
                    &Aloc.data.at(Cloc.gtlR.at(idx) * f);
            }
            else if(side == 1) { 
                assert(Cloc.gtlC.at(idx) < (idx_t)-1); 
                comm_expand.sendptr.at((d/2)+(j/2)) = &Bloc.data.at(Cloc.gtlC.at(idx) * f);}
            else{
                fprintf(stderr, "Error in comm setup: side neither 0 nor 1\n");
                goto ERR_EXIT;
            }
        }
    }
    /* prepare expand recvbuff based on row/col IDs I send
     * (what I recv is what I tell other processors to send) */
    for(size_t i = 0; i < esc.outDegree; ++i){
        idx_t d = esc.sendDisp.at(i);
        for (size_t j = 0; j < esc.sendCount.at(i); j+=2) {
            idx_t side = esc.sendBuff.at(d+j);
            idx_t idx = esc.sendBuff.at(d+j+1);
            if(side == 0){ assert(Cloc.gtlR.at(idx) < (idx_t)-1); comm_expand.recvptr.at((d/2)+(j/2)) = &Aloc.data.at(Cloc.gtlR.at(idx) * f);}
            else if(side == 1) { assert(Cloc.gtlC.at(idx) < (idx_t) -1); comm_expand.recvptr.at((d/2)+(j/2)) = &Bloc.data.at(Cloc.gtlC.at(idx) * f);}
            else{
                fprintf(stderr, "Error in comm setup: side neither 0 nor 1\n");
                goto ERR_EXIT;
            }
        }
    }
    /* exchange row/col ids */
    for (int i = 0; i < esc.inDegree; ++i) {

    }
    return;

ERR_EXIT:
    MPI_Finalize();
    exit(EXIT_FAILURE);
}

/* replaced with reduce_scatter */
//void setup_3dsddmm_reduce(cooMat& Cloc, SparseComm<real_t>& comm_reduce, MPI_Comm comm){
//
//    int myrank, size;
//    MPI_Comm_rank(comm, &myrank);
//    MPI_Comm_size(comm, &size);
//    vector<idx_t> RecvCntPP(size, 0), SendCntPP(size, 0);    
//    /* I recv "my owned nnz" from each processor that is not me */
//    idx_t ownedCount = 0;
//    for(idx_large_t i =0; i < Cloc.nnz; ++i){
//        if(Cloc.owners.at(i) == myrank) 
//            ownedCount++;
//    }
//    for(int i = 0; i < size; ++i){
//        if(i != myrank) RecvCntPP[i] = ownedCount;
//    }
//    MPI_Alltoall(RecvCntPP.data(), 1, MPI_IDX_T, SendCntPP.data(),1,
//            MPI_IDX_T, comm); 
//    SparseComm<idx_t> rsc(1); /* short for reduce setup comm */
//    vector<int> gtlR(size, -1), gtlS(size, -1);
//    idx_t totRecvCnt = 0, totSendCnt = 0;
//    for(int i=0; i < size; ++i){
//        if(RecvCntPP.at(i) > 0 && i != myrank) {
//            rsc.outSet.push_back(i);
//            gtlR.at(i) = rsc.outDegree++;
//            totRecvCnt += RecvCntPP.at(i);
//        }
//        if(SendCntPP.at(i) > 0 && i != myrank) {
//            rsc.inSet.push_back(i);
//            gtlS.at(i) = rsc.inDegree++;
//            totSendCnt += SendCntPP.at(i);
//        }
//    }
//
//    /* I will communicate the indices of the local nonzeros */
//    rsc.commP = comm;
//    rsc.recvBuff.resize(totSendCnt);
//    rsc.sendBuff.resize(totRecvCnt);
//    rsc.recvCount.resize(rsc.inDegree);
//    rsc.sendCount.resize(rsc.outDegree);
//    rsc.recvDisp.resize(rsc.inDegree+2, 0);
//    rsc.sendDisp.resize(rsc.outDegree+2, 0);
//    for(int i = 0; i < rsc.inDegree; ++i){
//        int p = rsc.inSet.at(i);
//        rsc.recvCount.at(i) = SendCntPP.at(p);
//        rsc.recvDisp.at(i+1) = rsc.recvDisp.at(i) + rsc.recvCount.at(i); 
//    }
//    for(int i = 0; i < rsc.outDegree; ++i){
//        int p = rsc.outSet.at(i);
//        rsc.sendCount.at(i) = RecvCntPP[p];
//        rsc.sendDisp.at(i+2) = rsc.sendDisp.at(i+1) + rsc.sendCount.at(i); 
//    }
//    /* prepare send buffer: telling others what I want them to send me */
//    for(idx_large_t  i = 0; i < Cloc.nnz; ++i){
//        int p = Cloc.owners.at(i);
//        if(p == myrank){
//            for(int j = 0; j < rsc.outDegree; ++j)
//                rsc.sendBuff.at(rsc.sendDisp.at(j+1)++) = i;
//        }
//
//    }
//    rsc.perform_sparse_comm(false, false);
//
//    comm_reduce.init(1, rsc.outDegree, rsc.inDegree, totSendCnt, totRecvCnt, SparseComm<real_t>::P2P, comm, MPI_COMM_NULL);
//    comm_reduce.inSet = rsc.outSet;
//    comm_reduce.outSet = rsc.inSet;
//    comm_reduce.recvCount = rsc.sendCount;
//    comm_reduce.sendCount = rsc.recvCount;
//    comm_reduce.recvDisp = rsc.sendDisp;
//    comm_reduce.sendDisp = rsc.recvDisp;
//    if(comm_reduce.commT == SparseComm<real_t>::NEIGHBOR){
//        MPI_Dist_graph_create_adjacent(comm, comm_reduce.inDegree, comm_reduce.inSet.data(),
//                MPI_WEIGHTS_EMPTY, comm_reduce.outDegree, comm_reduce.outSet.data(), MPI_WEIGHTS_EMPTY, MPI_INFO_NULL, 0, &comm_reduce.commN);
//    }
//
//    /* go over recvd data to check what will I send per processor */
//    for(int i = 0; i < rsc.inDegree; ++i){
//        idx_t disp = rsc.recvDisp.at(i);
//        for(idx_t j =0 ; j < rsc.recvCount.at(i); ++j){
//            comm_reduce.sendptr.at(disp+j) = &Cloc.vv[rsc.recvBuff.at(disp+j)]; 
//        }
//    }
//    /* FIXME the problem here is that recv will not consider the original
//     * value of Cloc.elms.at(i).val --> this should be preserved first*/
//    for(int i = 0; i < rsc.outDegree; ++i){
//        idx_t disp = rsc.sendDisp.at(i);
//        for(idx_t j =0 ; j < rsc.sendCount.at(i); ++j){
//            comm_reduce.recvptr.at(disp+j) = &(Cloc.vv[rsc.sendBuff.at(disp+j)]); 
//        }
//    }
//}

void setup_3dsddmm_reduce_scatter(cooMat& Cloc, DComm::DenseComm& comm_post, MPI_Comm zcomm){
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
}


void setup_3dsddmm(
        cooMat& Cloc,
        const idx_t f,
        const int c ,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        denseMatrix& Aloc,
        denseMatrix& Bloc,
        std::vector<int>& rpvec,
        std::vector<int>& cpvec,
        SparseComm<real_t>& comm_expand,
        DComm::DenseComm& comm_post
        )
{
    setup_3dsddmm_expand(Aloc, Bloc, Cloc, rpvec, cpvec, comm_expand, xycomm);
    setup_3dsddmm_reduce_scatter(Cloc, comm_post, zcomm);
    //setup_3dsddmm_reduce(Cloc, comm_reduce, zcomm);

}
void setup_spmm(
        cooMat& Cloc,
        const idx_t f,
        const int c ,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        denseMatrix& Xloc,
        denseMatrix& Yloc,
        std::vector<int>& rpvec,
        std::vector<int>& cpvec,
        SparseComm<real_t>& comm_expand,
        SparseComm<real_t>& comm_reduce
        )
{
    setup_spmm_side(Xloc, Cloc, 1, Cloc.gncols, Cloc.gtlC, Cloc.ltgC, cpvec, comm_expand, xycomm);
    setup_spmm_side(Yloc, Cloc, 0, Cloc.gnrows, Cloc.gtlR, Cloc.ltgR, rpvec, comm_reduce, xycomm);

    /* swap send and recv data in comm_reduce */

    swap(comm_reduce.recvBuff, comm_reduce.sendBuff);
    swap(comm_reduce.recvDisp, comm_reduce.sendDisp);
    swap(comm_reduce.recvCount, comm_reduce.sendCount);
    swap(comm_reduce.recvptr, comm_reduce.sendptr);
    swap(comm_reduce.outDegree, comm_reduce.inDegree);
    swap(comm_reduce.outSet, comm_reduce.inSet);

}

