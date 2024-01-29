#include "basic.hpp"
#include "comm.hpp"
#include "denseComm.hpp"
#include "mpi.h"
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>

using namespace DComm;
using namespace SpKernels;

void get_comm_stats(std::string mtxName, std::string algName, idx_t f, int c, MPI_Comm comm, idx_large_t mySendMsg, idx_large_t mySendVol, idx_large_t myRecvMsg, idx_large_t myRecvVol, std::string& retStr){    
    idx_large_t totSendVol, maxSendVol, totRecvVol, maxRecvVol, totSendMsg, totRecvMsg, maxSendMsg, maxRecvMsg;
    MPI_Reduce(&mySendVol, &totSendVol, 1, MPI_IDX_LARGE_T, MPI_SUM, 0, comm);
    MPI_Reduce(&myRecvVol, &totRecvVol, 1, MPI_IDX_LARGE_T, MPI_SUM, 0, comm);
    MPI_Reduce(&mySendMsg, &totSendMsg, 1, MPI_IDX_LARGE_T, MPI_SUM, 0, comm);
    MPI_Reduce(&myRecvMsg, &totRecvMsg, 1, MPI_IDX_LARGE_T, MPI_SUM, 0, comm);

    MPI_Reduce(&mySendVol, &maxSendVol, 1, MPI_IDX_LARGE_T, MPI_MAX, 0, comm);
    MPI_Reduce(&myRecvVol, &maxRecvVol, 1, MPI_IDX_LARGE_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mySendMsg, &maxSendMsg, 1, MPI_IDX_LARGE_T, MPI_MAX, 0, comm);
    MPI_Reduce(&myRecvMsg, &maxRecvMsg, 1, MPI_IDX_LARGE_T, MPI_MAX, 0, comm);
    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);

/*     std::array<int, 3> tarr1, tarr2, tarr3;
 *     MPI_Cart_get(comm, 3, tarr1.data(), tarr2.data(), tarr3.data()); 
 */
    if(myrank == 0){
        char buff[1024];
            sprintf(buff,"%s %s %d %u %d %.2f %.2f %.2f %.2f %lu %lu %lu %lu",
                mtxName.c_str(), algName.c_str(), size, f,c, 
                totSendMsg/(float)size, totSendVol/(float)size, totRecvMsg/(float)size, totRecvVol/(float)size,
                maxSendMsg, maxSendVol, maxRecvMsg, maxRecvVol);
            
        retStr = buff;
    }
}

void get_timing_stats(idx_t mycomm1time, idx_t mycomm2time, idx_t mycomptime, idx_t mytottime,  MPI_Comm comm, std::string& retStr){
    idx_t gcomm1Time, gcomm2Time, gcompTime, gtotTime;
    MPI_Reduce(&mycomm1time, &gcomm1Time, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mycomm2time, &gcomm2Time, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mycomptime, &gcompTime, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mytottime, &gtotTime, 1, MPI_IDX_T, MPI_MAX, 0, comm);

    int myrank;
    MPI_Comm_rank(comm, &myrank);
    if(myrank == 0){
        char buff[1024];
        std::ostringstream ss; 
        //sprintf(buff,"%u %u %u", gcomm1Time, gcomm2Time, gcompTime);
        ss << gcomm1Time<<" " << gcomm2Time <<" "<< gcompTime <<" " << gtotTime;
        retStr = ss.str();
    }
}

void print_comm_stats_sparse(std::string mtxName, std::string algName, SparseComm<real_t>& SpComm, idx_t f, parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm){

    idx_large_t mySendVol, myRecvVol, mySendMsg, myRecvMsg;
    mySendVol = std::accumulate(SpComm.sendCount.begin(), SpComm.sendCount.end(), 0.0); 
    myRecvVol = std::accumulate(SpComm.recvCount.begin(), SpComm.recvCount.end(), 0.0);
    mySendMsg = SpComm.outDegree;
    myRecvMsg = SpComm.inDegree;
    int myrank; 
    MPI_Comm_rank(comm, &myrank);
    std::string stats_str, times_str;
    get_comm_stats(mtxName,algName, f, Z, comm, mySendMsg, mySendVol, myRecvMsg, myRecvVol, stats_str);
    get_timing_stats(pt.comm1Time, pt.comm2Time, pt.compTime, pt.totalTime, comm, times_str);
    if(myrank == 0){
        printf("%s %s\n",stats_str.c_str(), times_str.c_str());
    }
}
void print_comm_stats_dense(std::string mtxName,std::string algName, DenseComm& DComm, idx_t f,
        parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm){

    idx_large_t mySendVol, myRecvVol, mySendMsg, myRecvMsg;
  
    if(DComm.OP == 1 && DComm.commXflag && DComm.commYflag){
        int myrankinZcomm;
        MPI_Comm_rank(DComm.commX, &myrankinZcomm);
        mySendVol = DComm.bcastXdisp.back() + DComm.bcastXcnt.back(); 
        myRecvVol = DComm.bcastXcnt[myrankinZcomm]; 
        myRecvMsg = 1; 
        mySendMsg = 1;

    }
    else{
        int myrankinXcomm, myrankinYcomm;
        if (DComm.commXflag) MPI_Comm_rank(DComm.commX, &myrankinXcomm);
        if (DComm.commYflag) MPI_Comm_rank(DComm.commY, &myrankinYcomm);
        idx_large_t allvol = 0;
        if(DComm.commXflag) allvol += std::accumulate(DComm.bcastXcnt.begin(), DComm.bcastXcnt.end(), 0.0);
        if(DComm.commYflag) allvol +=std::accumulate(DComm.bcastYcnt.begin(), DComm.bcastYcnt.end(), 0.0); 
        mySendVol = (DComm.commXflag? DComm.bcastXcnt[myrankinXcomm]:0) + (DComm.commYflag?DComm.bcastYcnt[myrankinYcomm]:0); 
        myRecvVol = allvol - mySendVol; 
        myRecvMsg = 0;
        myRecvMsg += (DComm.commXflag?DComm.outDegreeX:0);
        myRecvMsg += (DComm.commYflag ? DComm.outDegreeY: 0); 
        mySendMsg = 1; 
    }
    int myrank; 
    MPI_Comm_rank(comm, &myrank);
    std::string stats_str, times_str;
    get_comm_stats(mtxName, algName, f, Z, comm, mySendMsg, mySendVol, myRecvMsg, myRecvVol, stats_str);
    get_timing_stats(pt.comm1Time, pt.comm2Time, pt.compTime, pt.totalTime, comm, times_str);
    if(myrank == 0){
        printf("%s %s\n",stats_str.c_str(), times_str.c_str());
    }
}

