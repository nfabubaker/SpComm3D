#include "basic.hpp"
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
    for (int i = 0 ; i < size; ++i) { if(sendCount.at(i) > 0 && i != myrank ){ gtlR.at(i) = tcnt;
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
    for (int i = 2; i <= esc.outDegree+1; ++i) { esc.sendDisp.at(i) = esc.sendDisp.at(i-1) + esc.sendCount.at(i-2);}
    for (int i = 1; i <= esc.inDegree; ++i) { esc.recvDisp.at(i) = esc.recvDisp.at(i-1) + esc.recvCount.at(i-1);}

    /* Tell processors what rows/cols you want from them */
    /* 1 - determine what to send */
    for ( size_t i = 0; i < Cloc.grows; ++i) {
        int ploc = gtlS.at(rpvec.at(i)); 
            if( ploc > -1 && myRows.at(i)) {
                esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 0;
                esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
            }
    }
    for ( size_t i = 0; i < Cloc.gcols; ++i) {
        int ploc = gtlS.at(cpvec.at(i));
            if(ploc > -1 && myCols.at(i)){ 
                esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 1;
                esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
            }
    }
    esc.perform_sparse_comm();

    /* now recv from each processor available in recvBuff */
    idx_t f = Aloc.n;
    comm_expand.dataUnitSize = f;
    comm_expand.sendBuff.resize(totSendCnt * f); 
    comm_expand.sendptr.resize(totSendCnt); 
    comm_expand.recvBuff.resize(totRecvCnt * f); 
    comm_expand.recvptr.resize(totRecvCnt); 
    comm_expand.commP = comm;
    if(comm_expand.commT == SparseComm<real_t>::NEIGHBOR){
        MPI_Dist_graph_create_adjacent(comm, comm_expand.inDegree, comm_expand.inSet.data(),
                MPI_WEIGHTS_EMPTY, comm_expand.outDegree, comm_expand.outSet.data(), MPI_WEIGHTS_EMPTY, MPI_INFO_NULL, 0, &comm_expand.commN);
    }

    /* prepare expand sendbuff based on the row/col IDs I recv 
     * (what I send is what other processors tell me to send)*/
    for ( size_t i = 0;  i < esc.inDegree; ++ i) {
        idx_t d = esc.recvDisp.at(i);
        for (size_t j = 0; j < esc.recvCount.at(i); j+=2) {
            idx_t side = esc.recvBuff.at(d+j);
            idx_t idx = esc.recvBuff.at(d+j+1);
            if(side == 0) comm_expand.sendptr.at(d+(j/2)) = &Aloc.data.at(Aloc.gtl.at(idx) * f);
            else if(side == 1) comm_expand.sendptr.at(d+(j/2)) = &Bloc.data.at(Bloc.gtl.at(idx) * f);
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
            if(side == 0) comm_expand.recvptr.at(d+(j/2)) = &Aloc.data.at(Aloc.gtl.at(idx) * f);
            else if(side == 1) comm_expand.recvptr.at(d+(j/2)) = &Bloc.data.at(Bloc.gtl.at(idx) * f);
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
        if(RecvCntPP[i] > 0 && i != myrank) {
            rsc.inSet.push_back(i);
            gtlR[i] = rsc.inDegree++;
            totRecvCnt += RecvCntPP[i];
        }
        if(SendCntPP[i] > 0 && i != myrank) {
            rsc.outSet.push_back(i);
            gtlS[i] = rsc.outDegree++;
            totSendCnt += SendCntPP[i];
        }
    }

    /* I will communicate the indices of the local nonzeros */
    rsc.recvBuff.resize(totRecvCnt);
    rsc.sendBuff.resize(totSendCnt);
    rsc.recvCount.resize(rsc.inDegree);
    rsc.sendCount.resize(rsc.outDegree);
    rsc.recvDisp.resize(rsc.inDegree+2, 0);
    rsc.sendDisp.resize(rsc.outDegree+2, 0);
    for(int i = 0; i < rsc.inDegree; ++i){
        int p = rsc.inSet[i];
        rsc.recvCount[i] = RecvCntPP[p];
        if( i < rsc.inDegree-1 )
            rsc.recvDisp[i+1] = rsc.recvDisp[i] + rsc.recvCount[i]; 
    }
    for(int i = 0; i < rsc.outDegree; ++i){
        int p = rsc.outSet[i];
        rsc.sendCount[i] = SendCntPP[p];
        if( i < rsc.outDegree-1 )
            rsc.sendDisp[i+1] = rsc.sendDisp[i] + rsc.sendCount[i]; 
    }
    /* prepare send buffer: telling others what I want them to send me */
    for(size_t  i = 0; i < Cloc.lnnz; ++i){
        int p = Cloc.owners.at(i);
        if(p != myrank)
            rsc.sendBuff.at(rsc.sendDisp.at(gtlS.at(p)+1)++) = i;

    }
    rsc.perform_sparse_comm();

    comm_reduce.inSet.assign(rsc.outSet.begin(), rsc.outSet.end());
    comm_reduce.inDegree = rsc.outDegree;
    comm_reduce.outSet.assign(rsc.inSet.begin(), rsc.inSet.end());
    comm_reduce.outDegree = rsc.inDegree;
    comm_reduce.recvCount = rsc.sendCount;
    comm_reduce.sendCount = rsc.recvCount;
    comm_reduce.recvDisp = rsc.sendDisp;
    comm_reduce.sendDisp = rsc.recvDisp;
    comm_reduce.sendBuff.resize(totRecvCnt);
    comm_reduce.recvBuff.resize(totSendCnt);
    comm_reduce.commP = comm;
    if(comm_reduce.commT == SparseComm<real_t>::NEIGHBOR){
        MPI_Dist_graph_create_adjacent(comm, comm_reduce.inDegree, comm_reduce.inSet.data(),
                MPI_WEIGHTS_EMPTY, comm_reduce.outDegree, comm_reduce.outSet.data(), MPI_WEIGHTS_EMPTY, MPI_INFO_NULL, 0, &comm_reduce.commN);
    }

    /* go over recvd data to check what will I send per processor */
    for(int i = 0; i < rsc.inDegree; ++i){
        idx_t disp = rsc.recvDisp[i];
        for(idx_t j =0 ; j < rsc.recvCount[i]; ++j){
            comm_reduce.sendptr[disp+j] = &Cloc.elms[rsc.recvBuff[disp+j]].val; 
        }
    }
    /* FIXME the problem here is that recv will not consider the original
     * value of Cloc.elms[i].val --> this should be preserved first*/
    for(int i = 0; i < rsc.outDegree; ++i){
        idx_t disp = rsc.sendDisp[i];
        for(idx_t j =0 ; j < rsc.sendCount[i]; ++j){
            comm_reduce.recvptr[disp+j] = &Cloc.elms[rsc.sendBuff[disp+j]].val; 
        }
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

}
