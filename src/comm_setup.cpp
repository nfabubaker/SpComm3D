#include "basic.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <streambuf>
#include "distribute.hpp"
#include "mpi.h"


using namespace SpKernels;
using namespace std;


void setup_3dsddmm_expand(denseMatrix& Aloc, denseMatrix& Bloc, coo_mtx& Cloc, vector<int>& rpvec, vector<int>& cpvec, SparseComm<real_t>& comm_expand,  MPI_Comm comm){

    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    /* step1: decide my rows and my cols (global) */
    vector<bool> myRows(Cloc.grows,false), myCols(Cloc.gcols, false);
    vector<idx_t> recvCount(size, 0);
    vector<idx_t> sendCount(size, 0);
    for (int i = 0; i < Cloc.lnnz; ++i) { myRows.at(Cloc.elms.at(i).row) = true; myCols.at(Cloc.elms.at(i).col) = true; }

    /* step2: decide which rows/cols I will recv from which processor */
    for (size_t i = 0; i < Cloc.grows; ++i) if(myRows.at(i)) recvCount.at(rpvec.at(i))++; 
    for (size_t i = 0; i < Cloc.gcols; ++i) if(myCols.at(i)) recvCount.at(cpvec.at(i))++; 

    /* exchange recv info, now each processor knows send count for each other processor */
    MPI_Alltoall(recvCount.data(), 1, MPI_IDX_T, sendCount.data(), 1, MPI_IDX_T, comm);
    idx_t totSendCnt = 0, totRecvCnt = 0;
    totSendCnt = accumulate(sendCount.begin(), sendCount.end(), totSendCnt); totSendCnt -= sendCount[myrank];
    totRecvCnt = accumulate(recvCount.begin(), recvCount.end(), totRecvCnt); totRecvCnt -= recvCount[myrank];

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
        int ploc = gtlR.at(rpvec.at(i)); 
        if( ploc != -1 && myRows.at(i)) {
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 0;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    for ( size_t i = 0; i < Cloc.gcols; ++i) {
        int ploc = gtlR.at(cpvec.at(i));
        if(ploc != -1 && myCols.at(i)){ 
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 1;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    esc.perform_sparse_comm();

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
    for(size_t i =0; i < Cloc.lnnz; ++i){
        RecvCntPP.at(Cloc.owners.at(i))++;
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
        if(p != myrank)
            rsc.sendBuff.at(rsc.sendDisp.at(gtlR.at(p)+1)++) = i;

    }
    rsc.perform_sparse_comm();

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
            comm_reduce.recvptr.at(disp+j) = &Cloc.elms[rsc.sendBuff.at(disp+j)].val; 
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


void localize_C_indices(coo_mtx& Cloc){
    for(size_t i = 0; i < Cloc.lnnz; ++i){
        idx_t row = Cloc.elms.at(i).row;
        Cloc.elms.at(i).row = Cloc.gtlR.at(row);
        idx_t col = Cloc.elms.at(i).col;
        Cloc.elms.at(i).col = Cloc.gtlC.at(col);
    }
}

void SpKernels::setup_3dsddmm(coo_mtx& C, const idx_t f, const int c , const MPI_Comm comm, coo_mtx& Cloc, denseMatrix& Aloc, denseMatrix& Bloc, 
        SparseComm<real_t>& comm_expand, SparseComm<real_t>& comm_reduce){
    MPI_Comm xycomm, zcomm;
    vector<int> rpvec(C.grows), cpvec(C.gcols); 
    distribute3D(C, f, c, comm, Cloc, Aloc, Bloc, rpvec, cpvec,
            &xycomm, &zcomm);
    setup_3dsddmm_expand(Aloc, Bloc, Cloc, rpvec, cpvec, comm_expand, xycomm);
    setup_3dsddmm_reduce(Cloc, comm_reduce, zcomm);
    localize_C_indices(Cloc);

}
