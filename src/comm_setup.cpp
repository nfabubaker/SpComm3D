#include "basic.hpp"
#include "comm.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <streambuf>
#include <unordered_map>
#include "distribute.hpp"
#include "mpi.h"


using namespace SpKernels;
using namespace std;


void setup_spmm_side(denseMatrix& Dloc, coo_mtx& Sloc, int side, idx_t dimSize, unordered_map<idx_t, idx_t>& gtlD, vector<idx_t>& ltgD, vector<int>& pvec, SparseComm<real_t>& comm_handle,  MPI_Comm comm){

    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    /* step1: decide my rows and my cols (global) */
    vector<bool> myCols(dimSize, false);
    vector<idx_t> recvCount(size, 0);
    vector<idx_t> sendCount(size, 0);
    for (auto& elm : Sloc.elms) {
        if(side == 0)
            myCols.at(ltgD.at(elm.row)) = true;
        else
            myCols.at(ltgD.at(elm.col)) = true;
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

void setup_3dsddmm_expand(denseMatrix& Aloc, denseMatrix& Bloc, coo_mtx& Cloc, vector<int>& rpvec, vector<int>& cpvec, SparseComm<real_t>& comm_expand,  MPI_Comm comm){

    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    /* step1: decide my rows and my cols (global) */
    vector<bool> myRows(Cloc.grows,false), myCols(Cloc.gcols, false);
    vector<idx_t> recvCount(size, 0);
    vector<idx_t> sendCount(size, 0);
    for (auto& elm : Cloc.elms) {
        myRows.at(Cloc.ltgR.at(elm.row)) = true;
        myCols.at(Cloc.ltgC.at(elm.col)) = true;
    }

    /* step2: decide which rows/cols I will recv from which processor */
    for (size_t i = 0; i < Cloc.grows; ++i) if(myRows.at(i)) recvCount.at(rpvec.at(i))++; 
    for (size_t i = 0; i < Cloc.gcols; ++i) if(myCols.at(i)) recvCount.at(cpvec.at(i))++; 

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
    for ( size_t i = 0; i < Cloc.grows; ++i) {
        int ploc = (rpvec[i] == -1 ? -1 : gtlR.at(rpvec.at(i))); 
        if( ploc != -1 && myRows.at(i)) {
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 0;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    for ( size_t i = 0; i < Cloc.gcols; ++i) {
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


void setup_3dsddmm_reduce(coo_mtx& Cloc, SparseComm<real_t>& comm_reduce, MPI_Comm comm){

    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    vector<idx_t> RecvCntPP(size, 0), SendCntPP(size, 0);    
    /* I recv "my owned nnz" from each processor that is not me */
    idx_t ownedCount = 0;
    for(size_t i =0; i < Cloc.lnnz; ++i){
        if(Cloc.owners.at(i) == myrank) 
            ownedCount++;
    }
    for(int i = 0; i < size; ++i){
        if(i != myrank) RecvCntPP[i] = ownedCount;
    }
    MPI_Alltoall(RecvCntPP.data(), 1, MPI_IDX_T, SendCntPP.data(),1,
            MPI_IDX_T, comm); 
    SparseComm<idx_t> rsc(1); /* short for reduce setup comm */
    vector<int> gtlR(size, -1), gtlS(size, -1);
    idx_t totRecvCnt = 0, totSendCnt = 0;
    for(int i=0; i < size; ++i){
        if(RecvCntPP.at(i) > 0 && i != myrank) {
            rsc.outSet.push_back(i);
            gtlR.at(i) = rsc.outDegree++;
            totRecvCnt += RecvCntPP.at(i);
        }
        if(SendCntPP.at(i) > 0 && i != myrank) {
            rsc.inSet.push_back(i);
            gtlS.at(i) = rsc.inDegree++;
            totSendCnt += SendCntPP.at(i);
        }
    }

    /* I will communicate the indices of the local nonzeros */
    rsc.commP = comm;
    rsc.recvBuff.resize(totSendCnt);
    rsc.sendBuff.resize(totRecvCnt);
    rsc.recvCount.resize(rsc.inDegree);
    rsc.sendCount.resize(rsc.outDegree);
    rsc.recvDisp.resize(rsc.inDegree+2, 0);
    rsc.sendDisp.resize(rsc.outDegree+2, 0);
    for(int i = 0; i < rsc.inDegree; ++i){
        int p = rsc.inSet.at(i);
        rsc.recvCount.at(i) = SendCntPP.at(p);
        rsc.recvDisp.at(i+1) = rsc.recvDisp.at(i) + rsc.recvCount.at(i); 
    }
    for(int i = 0; i < rsc.outDegree; ++i){
        int p = rsc.outSet.at(i);
        rsc.sendCount.at(i) = RecvCntPP[p];
        rsc.sendDisp.at(i+2) = rsc.sendDisp.at(i+1) + rsc.sendCount.at(i); 
    }
    /* prepare send buffer: telling others what I want them to send me */
    for(size_t  i = 0; i < Cloc.lnnz; ++i){
        int p = Cloc.owners.at(i);
        if(p == myrank){
            for(int j = 0; j < rsc.outDegree; ++j)
                rsc.sendBuff.at(rsc.sendDisp.at(j+1)++) = i;
        }

    }
    rsc.perform_sparse_comm(false, false);

    comm_reduce.init(1, rsc.outDegree, rsc.inDegree, totSendCnt, totRecvCnt, SparseComm<real_t>::P2P, comm, MPI_COMM_NULL);
    comm_reduce.inSet = rsc.outSet;
    comm_reduce.outSet = rsc.inSet;
    comm_reduce.recvCount = rsc.sendCount;
    comm_reduce.sendCount = rsc.recvCount;
    comm_reduce.recvDisp = rsc.sendDisp;
    comm_reduce.sendDisp = rsc.recvDisp;
    if(comm_reduce.commT == SparseComm<real_t>::NEIGHBOR){
        MPI_Dist_graph_create_adjacent(comm, comm_reduce.inDegree, comm_reduce.inSet.data(),
                MPI_WEIGHTS_EMPTY, comm_reduce.outDegree, comm_reduce.outSet.data(), MPI_WEIGHTS_EMPTY, MPI_INFO_NULL, 0, &comm_reduce.commN);
    }

    /* go over recvd data to check what will I send per processor */
    for(int i = 0; i < rsc.inDegree; ++i){
        idx_t disp = rsc.recvDisp.at(i);
        for(idx_t j =0 ; j < rsc.recvCount.at(i); ++j){
            comm_reduce.sendptr.at(disp+j) = &Cloc.elms[rsc.recvBuff.at(disp+j)].val; 
        }
    }
    /* FIXME the problem here is that recv will not consider the original
     * value of Cloc.elms.at(i).val --> this should be preserved first*/
    for(int i = 0; i < rsc.outDegree; ++i){
        idx_t disp = rsc.sendDisp.at(i);
        for(idx_t j =0 ; j < rsc.sendCount.at(i); ++j){
            comm_reduce.recvptr.at(disp+j) = &(Cloc.elms[rsc.sendBuff.at(disp+j)].val); 
        }
    }
    Cloc.ownedNnz = 0;
    for(size_t i = 0; i < Cloc.lnnz; ++i){
        if(Cloc.owners[i] == myrank){ 
            Cloc.otl.push_back(i);
            Cloc.ownedNnz++;
            Cloc.owned.push_back(Cloc.elms[i].val);
        }
    }

}


void SpKernels::setup_3dsddmm(
        coo_mtx& Cloc,
        const idx_t f,
        const int c ,
        const MPI_Comm xycomm,
        const MPI_Comm zcomm,
        denseMatrix& Aloc,
        denseMatrix& Bloc,
        std::vector<int>& rpvec,
        std::vector<int>& cpvec,
        SparseComm<real_t>& comm_expand,
        SparseComm<real_t>& comm_reduce
        )
{
    setup_3dsddmm_expand(Aloc, Bloc, Cloc, rpvec, cpvec, comm_expand, xycomm);
    setup_3dsddmm_reduce(Cloc, comm_reduce, zcomm);

}
void SpKernels::setup_spmm(
        coo_mtx& Cloc,
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
    setup_spmm_side(Xloc, Cloc, 1, Cloc.gcols, Cloc.gtlC, Cloc.ltgC, cpvec, comm_expand, xycomm);
    setup_spmm_side(Yloc, Cloc, 0, Cloc.grows, Cloc.gtlR, Cloc.ltgR, rpvec, comm_reduce, xycomm);

    /* swap send and recv data in comm_reduce */

    swap(comm_reduce.recvBuff, comm_reduce.sendBuff);
    swap(comm_reduce.recvDisp, comm_reduce.sendDisp);
    swap(comm_reduce.recvCount, comm_reduce.sendCount);
    swap(comm_reduce.recvptr, comm_reduce.sendptr);
    swap(comm_reduce.outDegree, comm_reduce.inDegree);
    swap(comm_reduce.outSet, comm_reduce.inSet);

}
void SpKernels::setup_3dspmm_bcast(
        coo_mtx& Aloc,
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
    vector<int> rpvecX(Aloc.lrows), cpvecY(Aloc.lcols);
    idx_t tt = 0;
    for(size_t i = 0; i < Aloc.lrows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[Aloc.ltgR[i]], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            rpvecX[tt++] = tarr1[1];
        }
    }
    tt = 0;
    for(size_t i = 0; i < Aloc.lcols; ++i){
        MPI_Cart_coords(xycomm, cpvec[Aloc.ltgC[i]], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            cpvecY[tt++] = tarr1[0];
        }
    }
    for(size_t i = 0; i < Aloc.grows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[i], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            comm_post.bcastXcnt[tarr1[1]]++;
        }
    }
    for(size_t i = 0; i < Aloc.gcols; ++i){
        MPI_Cart_coords(xycomm, cpvec[i], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            comm_pre.bcastYcnt[tarr1[0]]++;
        }
    }

    for(size_t i = 1; i < Y+1; ++i) {
        comm_post.bcastXdisp[i] = comm_post.bcastXdisp[i-1] + comm_post.bcastXcnt[i-1];} 
    for(size_t i = 1; i < X+1; ++i) {
        comm_pre.bcastYdisp[i] = comm_pre.bcastYdisp[i-1] + comm_pre.bcastYcnt[i-1];} 

    for (size_t i = 0; i < Aloc.lrows; ++i) mapY[i] = comm_post.bcastXdisp[rpvecX[i]]++; 
    for (size_t i = 0; i < Aloc.lcols; ++i) mapX[i] = comm_pre.bcastYdisp[cpvecY[i]]++; 
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

void SpKernels::setup_3dsddmm_bcast(
        coo_mtx& Cloc,
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
    vector<int> rpvecX(Cloc.lrows), cpvecY(Cloc.lcols);
    idx_t tt = 0;
    for(size_t i = 0; i < Cloc.lrows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[Cloc.ltgR[i]], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            rpvecX[tt++] = tarr1[1];
        }
    }
    tt = 0;
    for(size_t i = 0; i < Cloc.lcols; ++i){
        MPI_Cart_coords(xycomm, cpvec[Cloc.ltgC[i]], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            cpvecY[tt++] = tarr1[0];
        }
    }
    for(size_t i = 0; i < Cloc.grows; ++i){
        /* first, identify if the row belongs to my group*/
        MPI_Cart_coords(xycomm, rpvec[i], 2, tarr1.data());
        if(tarr1[0]  == myxcoord){
            comm_pre.bcastXcnt[tarr1[1]]++;
        }
    }
    for(size_t i = 0; i < Cloc.gcols; ++i){
        MPI_Cart_coords(xycomm, cpvec[i], 2, tarr1.data());
        if(tarr1[1]  == myycoord){ 
            comm_pre.bcastYcnt[tarr1[0]]++;
        }
    }

    for(size_t i = 1; i < Y+1; ++i) {
        comm_pre.bcastXdisp[i] = comm_pre.bcastXdisp[i-1] + comm_pre.bcastXcnt[i-1];} 
    for(size_t i = 1; i < X+1; ++i) {
        comm_pre.bcastYdisp[i] = comm_pre.bcastYdisp[i-1] + comm_pre.bcastYcnt[i-1];} 

    for (size_t i = 0; i < Cloc.lrows; ++i) mapA[i] = comm_pre.bcastXdisp[rpvecX[i]]++; 
    for (size_t i = 0; i < Cloc.lcols; ++i) mapB[i] = comm_pre.bcastYdisp[cpvecY[i]]++; 
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
    /* now set up for Allreduce */
    comm_post.commXflag = true; comm_post.commYflag = true;
    comm_post.commX = zcomm;
    comm_post.OP = 1;
    comm_post.ClocPtr = Cloc.elms.data();
    comm_post.lnnz = Cloc.lnnz;
    comm_post.reduceBuffer.resize(Cloc.lnnz);
    Cloc.ownedNnz = 0;
    int myzrank;
    MPI_Comm_rank(zcomm, &myzrank);
    for(size_t i = 0; i < Cloc.lnnz; ++i){
        if(Cloc.owners[i] == myzrank){ 
            Cloc.otl.push_back(i);
            Cloc.ownedNnz++;
            Cloc.owned.push_back(Cloc.elms[i].val);
        }
    }
    /* re-localize C nonzeros */
/*     for(size_t i = 0; i < Cloc.lnnz; ++i){
 *         idx_t row = Cloc.elms.at(i).row;
 *         Cloc.elms.at(i).row = Cloc.gtlR.at(row);
 *         idx_t col = Cloc.elms.at(i).col;
 *         Cloc.elms.at(i).col = Cloc.gtlC.at(col);
 *     }
 */


}

