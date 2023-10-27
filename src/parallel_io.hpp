#ifndef _PARALLEL_IO_H
#define _PARALLEL_IO_H
#include <mpi.h>
#include <iostream>

#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include "comm.hpp"

#include "basic.hpp"
#include "mpio.h"
using namespace std;
namespace SpKernels{
    void read_bin_parallel_distribute(std::string filename, coo_mtx& Cloc, std::vector<int> rpvec2D, std::vector<int> cpvec2D, MPI_Comm xycomm, MPI_Comm zcomm ){
        int myxyrank, myzrank, xysize, zsize;
        MPI_Comm_rank( xycomm, &myxyrank);
        MPI_Comm_size( xycomm, &xysize);
        MPI_Comm_rank( zcomm, &myzrank);
        MPI_Comm_size( zcomm, &zsize);

        /* create new datatype for coo_mtx */
        const int nitems = 3;
        int myrank, nprocs, blocklengths[3] = {1, 1, 1};
        MPI_Datatype types[3] = {MPI_IDX_T, MPI_IDX_T, MPI_REAL_T};
        MPI_Datatype mpi_s_type;
        MPI_Aint offsets[3];

        offsets[0] = offsetof(triplet, row);
        offsets[1] = offsetof(triplet, col);
        offsets[2] = offsetof(triplet, val);
        MPI_Type_create_struct(nitems, blocklengths, offsets, types, 
                &mpi_s_type);
        MPI_Type_commit(&mpi_s_type);
        /* get XY comm 
         * if zcomm = 0 pefrorm the read
         * */
        if(myzrank == 0){
            MPI_File file;
            MPI_File_open(xycomm, filename.data(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
            MPI_File_read(file, &Cloc.grows, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
            MPI_File_read(file, &Cloc.gcols, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
            MPI_File_read(file, &Cloc.gnnz, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
            idx_t nnz_pp = Cloc.gnnz / xysize;
            idx_t nnz_ppR = Cloc.gnnz % xysize;
            idx_t my_nnz = (myxyrank < nnz_ppR ? nnz_pp+1 : nnz_pp);
            idx_t my_nnz_start = 
                myxyrank > nnz_ppR ?
                (nnz_pp+1)* nnz_ppR + (nnz_pp *(myxyrank-nnz_ppR))
                :
                (nnz_pp+1)*myxyrank;
            idx_t my_nnz_end = my_nnz_start + (myxyrank >= nnz_ppR ? nnz_pp:nnz_pp+1);
            assert(my_nnz_end-my_nnz_start == my_nnz);
            MPI_File_seek(file, my_nnz_start*sizeof(triplet), MPI_SEEK_CUR);
            for(size_t i = 0; i < my_nnz; ++i)
                MPI_File_read(file, &Cloc.elms[i], 1, mpi_s_type, MPI_STATUS_IGNORE); 
            int trank;
            std::vector<vector<triplet>> sendLists(xysize);
            std::array<int, 2> coords;
            for(auto& elm: Cloc.elms){
                idx_t rid = elm.row;
                idx_t cid = elm.col;
                coords = {rpvec2D[rid], cpvec2D[cid]}; 

                MPI_Cart_rank(xycomm, coords.data(), &trank);
                sendLists.at(trank).push_back(elm);
            }
            vector<idx_t> sendCnts(xysize,0), recvCnts(xysize,0),
                sendDisps(xysize+1,0), recvDisps(xysize+1,0);
            for(int i=0; i < xysize; ++i) sendCnts[i] = sendLists[i].size();
            for(int i=1; i <= xysize; ++i) 
                sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
            MPI_Alltoall(sendCnts.data(), 1, MPI_IDX_T, 
                    recvCnts.data(), 1, MPI_IDX_T, xycomm);
            for(int i=1; i <= xysize; ++i) 
                recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];

            Cloc.lnnz = accumulate(recvCnts.begin(), recvCnts.end(), 0.0); 
            Cloc.elms.resize(Cloc.lnnz);
            /* flattern SendLists and perform alltoall */
            vector<triplet> sendBuff;
            for(auto& v : sendLists){
                sendBuff.insert(sendBuff.end(), v.begin(), v.end());
            }
            MPI_Alltoallv(sendBuff.data(), (const int*)sendCnts.data(),
                    (const int*)sendDisps.data(), mpi_s_type, Cloc.elms.data(), 
                    (const int*) recvCnts.data(), (const int*)recvDisps.data(), 
                    mpi_s_type, xycomm);

        }
        MPI_Bcast(&Cloc.lnnz, 1, MPI_IDX_T, 0, zcomm);
        if(myzrank > 0) Cloc.elms.resize(Cloc.lnnz);
        MPI_Bcast(Cloc.elms.data(), Cloc.lnnz, mpi_s_type, 0, zcomm);
    }


#endif
