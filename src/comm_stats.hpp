
#include "comm.hpp"
#include <cstdint>
#include <numeric>
#include <string>

using namespace SpKernels;



void get_comm_stats(std::string mtxName, idx_t f, int c, MPI_Comm comm, idx_t mySendMsg, idx_t mySendVol, idx_t myRecvMsg, idx_t myRecvVol, std::string& retStr){    
    idx_t totSendVol, maxSendVol, totRecvVol, maxRecvVol, totSendMsg, totRecvMsg, maxSendMsg, maxRecvMsg;
    MPI_Reduce(&mySendVol, &totSendVol, 1, MPI_IDX_T, MPI_SUM, 0, comm);
    MPI_Reduce(&myRecvVol, &totRecvVol, 1, MPI_IDX_T, MPI_SUM, 0, comm);
    MPI_Reduce(&mySendMsg, &totSendMsg, 1, MPI_IDX_T, MPI_SUM, 0, comm);
    MPI_Reduce(&myRecvMsg, &totRecvMsg, 1, MPI_IDX_T, MPI_SUM, 0, comm);

    MPI_Reduce(&mySendVol, &maxSendVol, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&myRecvVol, &maxRecvVol, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mySendMsg, &maxSendMsg, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&myRecvMsg, &maxRecvMsg, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    int myrank, size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &size);

/*     std::array<int, 3> tarr1, tarr2, tarr3;
 *     MPI_Cart_get(comm, 3, tarr1.data(), tarr2.data(), tarr3.data()); 
 */
    if(myrank == 0){
        char buff[512];
        if(sizeof(idx_t) == sizeof(uint32_t))
            sprintf(buff,"%s %d %d %d %.2f %.2f %.2f %.2f %u %u %u %u",
                mtxName.c_str(), size, f,c, 
                totSendMsg/(float)size, totSendVol/(float)size, totRecvMsg/(float)size, totRecvVol/(float)size,
                maxSendMsg, maxSendVol, maxRecvMsg, maxRecvVol);
        else 
            sprintf(buff,"%s %d %d %d %.2f %.2f %.2f %.2f %lu %lu %lu %lu",
                mtxName.c_str(), size, f,c, 
                totSendMsg/(float)size, totSendVol/(float)size, totRecvMsg/(float)size, totRecvVol/(float)size,
                maxSendMsg, maxSendVol, maxRecvMsg, maxRecvVol);
            
        retStr = buff;
    }
}

void get_timing_stats(idx_t mycomm1time, idx_t mycomm2time, idx_t mycomptime,  MPI_Comm comm, std::string& retStr){
    uint32_t gcomm1Time, gcomm2Time, gcompTime;
    MPI_Reduce(&mycomm1time, &gcomm1Time, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mycomm2time, &gcomm2Time, 1, MPI_IDX_T, MPI_MAX, 0, comm);
    MPI_Reduce(&mycomptime, &gcompTime, 1, MPI_IDX_T, MPI_MAX, 0, comm);

    int myrank;
    MPI_Comm_rank(comm, &myrank);
    if(myrank == 0){
        char buff[512];
        sprintf(buff,"%u %u %u", gcomm1Time, gcomm2Time, gcompTime);
        retStr = buff;
    }
}

void print_comm_stats_sparse(std::string mtxName, SparseComm<real_t>& SpComm, idx_t f, parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm){

    idx_t mySendVol, myRecvVol, mySendMsg, myRecvMsg;
    mySendVol = std::accumulate(SpComm.sendCount.begin(), SpComm.sendCount.end(), 0.0); 
    myRecvVol = std::accumulate(SpComm.recvCount.begin(), SpComm.recvCount.end(), 0.0);
    mySendMsg = SpComm.outDegree;
    myRecvMsg = SpComm.inDegree;
    int myrank; 
    MPI_Comm_rank(comm, &myrank);
    std::string stats_str, times_str;
    get_comm_stats(mtxName, f, Z, comm, mySendMsg, mySendVol, myRecvMsg, myRecvVol, stats_str);
    get_timing_stats(pt.comm1Time, pt.comm2Time, pt.compTime, comm, times_str);
    if(myrank == 0){
        printf("%s %s\n",stats_str.c_str(), times_str.c_str());
    }
}
void print_comm_stats_dense(std::string mtxName, DenseComm& DComm, idx_t f,
        parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm){

    idx_t mySendVol, myRecvVol, mySendMsg, myRecvMsg;
  
    if(DComm.OP == 0){
        int myrankinXcomm, myrankinYcomm;
        MPI_Comm_rank(DComm.commX, &myrankinXcomm);
        MPI_Comm_rank(DComm.commY, &myrankinYcomm);
        idx_t allvol = std::accumulate(DComm.bcastXcnt.begin(), DComm.bcastXcnt.end(), 0.0) +
        std::accumulate(DComm.bcastYcnt.begin(), DComm.bcastYcnt.end(), 0.0); 
        mySendVol = DComm.bcastXcnt[myrankinXcomm] + DComm.bcastYcnt[myrankinYcomm]; 
        myRecvVol = allvol - mySendVol; 
        myRecvMsg = DComm.outDegreeX + DComm.outDegreeY; 
        mySendMsg = 1; 
    }
    else if(DComm.OP == 1){
        int myrankinZcomm;
        MPI_Comm_rank(DComm.commX, &myrankinZcomm);
        mySendVol = DComm.lnnz; 
        myRecvVol = DComm.lnnz; 
        myRecvMsg = 1; 
        mySendMsg = 1;

    }
    int myrank; 
    MPI_Comm_rank(comm, &myrank);
    std::string stats_str, times_str;
    get_comm_stats(mtxName, f, Z, comm, mySendMsg, mySendVol, myRecvMsg, myRecvVol, stats_str);
    get_timing_stats(pt.comm1Time, pt.comm2Time, pt.compTime, comm, times_str);
    if(myrank == 0){
        printf("%s %s\n",stats_str.c_str(), times_str.c_str());
    }
}

