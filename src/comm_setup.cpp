#include "basic.hpp"
#include <cstdlib>
#include <numeric>
#include <streambuf>
#include "distribute.hpp"


using namespace SpKernels;
using namespace std;


void setup_3dsddmm_expand(denseMatrix& Aloc, denseMatrix& Bloc, coo_mtx& Cloc, vector<int>& rpvec, vector<int>& cpvec, SparseComm<real_t>& comm_expand, SparseComm<real_t>& comm_reduce, MPI_Comm comm){
   
    int myrank, size, inDegree=0, outDegree=0;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);
    /* step1: decide my rows and my cols (global) */
    vector<bool> myRows(Cloc.grows,false), myCols(Cloc.gcols, false);
    vector<idx_t> recvCount(size, 0);
    vector<idx_t> sendCount(size, 0);
    for (int i = 0; i < Cloc.lnnz; ++i) { myRows.at(Cloc.elms[i].row) = true; myCols.at(Cloc.elms[i].col) = true; }

    /* step2: decide which rows/cols I will recv from which processor */
    for (size_t i = 0; i < Cloc.grows; ++i) if(myRows[i]) recvCount.at(rpvec[i])++; 
    for (size_t i = 0; i < Cloc.gcols; ++i) if(myCols[i]) recvCount.at(cpvec[i])++; 

    /* exchange recv info, now each processor knows send count for each other processor */
    MPI_Alltoall(recvCount.data(), 1, MPI_IDX_T, sendCount.data(), 1, MPI_IDX_T, comm);
    idx_t totSendCnt = 0, totRecvCnt = 0;
    totSendCnt = accumulate(sendCount.begin(), sendCount.end(), totSendCnt); totSendCnt -= sendCount[myrank];
    totRecvCnt = accumulate(recvCount.begin(), recvCount.end(), totRecvCnt); totRecvCnt -= recvCount[myrank];

    /* exchange row/col IDs */
    SparseComm<idx_t> esc; /* esc: short for expand setup comm */
    for(int i = 0; i < size; ++i){ 
        if(sendCount[i] > 0 && i != myrank) esc.inDegree++;
        if(recvCount[i] > 0 && i != myrank) esc.outDegree++;
    }
    esc.sendBuff.resize(totRecvCnt); esc.recvBuff.resize(totSendCnt);
    esc.inSet.resize(inDegree); esc.outSet.resize(outDegree);
    esc.recvCount.resize(esc.inDegree, 0); esc.sendCount.resize(esc.outDegree,0);
    esc.recvDisp.resize(esc.inDegree+2, 0); esc.sendDisp.resize(esc.outDegree+2,0);

    vector<int> gtlR(size,-1), gtlS(size,-1);
    for (int i = 0, tcnt =0; i < size; ++i) { if(sendCount[i] > 0 && i != myrank ){ gtlR[i] = tcnt; esc.recvCount[tcnt++] = sendCount[i]*2;} }
    for (int i = 0, tcnt=0; i < size; ++i) { if(sendCount[i] > 0 && i != myrank ){ gtlS[i] = tcnt; esc.sendCount[tcnt++] = recvCount[i]*2;} }
    for (int i = 2; i <= outDegree+1; ++i) { esc.sendDisp.at(i) = esc.sendDisp.at(i-1) + esc.sendCount.at(i-1);}
    for (int i = 1; i <= inDegree; ++i) { esc.recvDisp.at(i) = esc.recvDisp.at(i-1) + esc.recvCount.at(i-1);}

    /* Tell processors what rows/cols you want from them */
    /* 1 - determine what to send */
    for ( size_t i = 0; i < Cloc.grows; ++i) {
        int ploc = gtlR[rpvec[i]]; 
        if(myRows[i]) {
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 0;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    for ( size_t i = 0; i < Cloc.gcols; ++i) {
        int ploc = gtlR[cpvec[i]];
        if(myCols[i]){ 
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = 1;
            esc.sendBuff.at(esc.sendDisp.at(ploc+1)++) = i;
        }
    }
    esc.perform_sparse_comm(2);

    /* now recv from each processor available in recvBuff */
    idx_t f = Aloc.n;
    comm_expand.sendBuff.resize(totSendCnt * f); 
    comm_expand.sendptr.resize(totSendCnt); 
    comm_expand.recvBuff.resize(totRecvCnt * f); 
    comm_expand.recvptr.resize(totRecvCnt); 
    
    /* prepare expand sendbuff based on the row/col IDs I recv 
     * (what I send is what other processors tell me to send)*/
    for ( size_t i = 0;  i < esc.inDegree; ++ i) {
        idx_t d = esc.recvDisp[i];
        for (size_t j = 0; j < esc.recvCount[i]; j+=2) {
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
        idx_t d = esc.sendDisp[i];
        for (size_t j = 0; j < esc.sendCount[i]; j+=2) {
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
    for (int i = 0; i < inDegree; ++i) {
    
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
    SparseComm<idx_t> rsc; /* short for reduce setup comm */
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
    rsc.perform_sparse_comm(1);

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

void SpKernels::setup_3dsddmm(coo_mtx& C, coo_mtx&Cloc, denseMatrix& Aloc, denseMatrix& Bloc, 
        SparseComm<real_t>& comm_expand, SparseComm<real_t>& comm_reduce, MPI_Comm comm, idx_t f, int c){
    MPI_Comm xycomm, zcomm;
    vector<int> rpvec(C.grows), cpvec(C.gcols); 
    distribute3D(C, Cloc, Aloc, Bloc, rpvec, cpvec, &xycomm, &zcomm, comm, f, c); 
}
