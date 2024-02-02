#ifndef _COMM_STATS_H
#define _COMM_STATS_H
#include "comm.hpp"
#include "denseComm.hpp"

#include <cstdint>
#include <numeric>
#include <string>
using namespace DComm;
using namespace SpKernels;

void get_comm_stats(std::string mtxName, std::string algName, idx_t f, int c, MPI_Comm comm, idx_t mySendMsg, idx_t mySendVol, idx_t myRecvMsg, idx_t myRecvVol, std::string& retStr);

void get_timing_stats(idx_t mycomm1time, idx_t mycomm2time, idx_t mycomptime,  MPI_Comm comm, std::string& retStr);

void print_comm_stats_sparse(std::string mtxName, std::string algName,SparseComm<real_t>& SpComm, idx_t f, parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm);
void print_comm_stats_sparse2(std::string mtxName, std::string algName,SparseComm<real_t>& SpComm1, SparseComm<real_t> SpComm2, idx_t f, parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm);
void print_comm_stats_dense(std::string mtxName, std::string algName, DenseComm& DComm, idx_t f,
        parallelTiming& pt, int X, int Y, int Z, MPI_Comm comm);
#endif
