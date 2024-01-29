#ifndef _PARALLEL_IO_H
#define _PARALLEL_IO_H
#include <cstdint>
#include <mpi.h>
#include <iostream>

#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include "SparseMatrix.hpp"
#include "comm.hpp"

#include "basic.hpp"
#include "mpio.h"
#include <algorithm>

using namespace std;
namespace SpKernels{
    void read_bin_parallel_distribute(
            std::string filename,
            std::vector<triplet>& myelms,
            idx_t& grows,
            idx_t& gcols,
            idx_large_t& gnnz,
            idx_large_t& lnnz,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            MPI_Comm worldcomm,
            MPI_Comm xycomm,
            MPI_Comm zcomm){
        int myworldrank, worldsize, myxyrank, myzrank, xysize, zsize;
        int X,Y,Z;
        std::array<int, 3> dims, t1, t2;
        MPI_Cart_get(worldcomm, 3, dims.data(), t1.data(), t2.data());
        X = dims[0]; Y=dims[1]; Z=dims[2];
        MPI_Comm_rank( worldcomm, &myworldrank);
        MPI_Comm_size( worldcomm, &worldsize);
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
        MPI_File file;
        MPI_File_open(xycomm, filename.data(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        MPI_File_read(file, &grows, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
        MPI_File_read(file, &gcols, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
        MPI_File_read(file, &gnnz, 1, MPI_IDX_LARGE_T, MPI_STATUS_IGNORE);
        if(myxyrank == 0){
            rpvec2D.resize(grows); cpvec2D.resize(gcols);
            std::vector<idx_t> row_space(grows), col_space(gcols);
            for(size_t i = 0; i < grows; ++i) row_space[i] = i;
            for(size_t i = 0; i < gcols; ++i) col_space[i] = i;
            std::random_shuffle(row_space.begin(), row_space.end());
            std::random_shuffle(col_space.begin(), col_space.end());
            /* assign Rows/Cols with RR */
            for(size_t i = 0, p=0; i < grows; ++i)
                rpvec2D[row_space[i]] = (p++) % X; 
            for(size_t i = 0, p=0; i < gcols; ++i) 
                cpvec2D[col_space[i]] = (p++) % Y; 
        }
        if(myxyrank!=0){ rpvec2D.resize(grows); cpvec2D.resize(gcols);}
        MPI_Bcast(rpvec2D.data(), grows, MPI_INT, 0, xycomm);
        MPI_Bcast(cpvec2D.data(), gcols, MPI_INT, 0, xycomm);

        idx_large_t nnz_pp = gnnz / xysize;
        idx_large_t nnz_ppR = gnnz % xysize;
        idx_large_t my_nnz = (myxyrank < nnz_ppR ? nnz_pp+1 : nnz_pp);
        idx_large_t my_nnz_start = 
            myxyrank > nnz_ppR ?
            (nnz_pp+1)* nnz_ppR + (nnz_pp *(myxyrank-nnz_ppR))
            :
            (nnz_pp+1)*myxyrank;
        idx_large_t my_nnz_end = my_nnz_start + (myxyrank >= nnz_ppR ? nnz_pp:nnz_pp+1);
        assert(my_nnz_end-my_nnz_start == my_nnz);
        MPI_File_seek(file, my_nnz_start*sizeof(triplet), MPI_SEEK_CUR);
        myelms.resize(my_nnz);
        for(size_t i = 0; i < my_nnz; ++i)
            MPI_File_read(file, &myelms[i], 1, mpi_s_type, MPI_STATUS_IGNORE); 
        int trank;
        std::vector<vector<triplet>> sendLists(xysize);
        std::array<int, 2> coords;
        for(auto& elm: myelms){
            idx_t rid = elm.row;
            idx_t cid = elm.col;
            coords = {rpvec2D[rid], cpvec2D[cid]}; 

            MPI_Cart_rank(xycomm, coords.data(), &trank);
            sendLists.at(trank).push_back(elm);
        }
        vector<int> sendCnts(xysize,0), recvCnts(xysize,0),
            sendDisps(xysize+1,0), recvDisps(xysize+1,0);
        for(int i=0; i < xysize; ++i) 
            sendCnts[i] = sendLists[i].size();
        for(int i=1; i <= xysize; ++i) 
            sendDisps[i] = sendDisps[i-1] + sendCnts[i-1];
        MPI_Alltoall(sendCnts.data(), 1, MPI_INT, 
                recvCnts.data(), 1, MPI_INT, xycomm);
        for(int i=1; i <= xysize; ++i) 
            recvDisps[i] = recvDisps[i-1] + recvCnts[i-1];

        lnnz = accumulate(recvCnts.begin(), recvCnts.end(), 0.0); 
        myelms.resize(lnnz);
        /* flattern SendLists and perform alltoall */
        vector<triplet> sendBuff;
        for(auto& v : sendLists){
            sendBuff.insert(sendBuff.end(), v.begin(), v.end());
        }
        MPI_Alltoallv(sendBuff.data(), (const int*)sendCnts.data(),
                (const int*)sendDisps.data(), mpi_s_type, myelms.data(), 
                (const int*) recvCnts.data(), (const int*)recvDisps.data(), 
                mpi_s_type, xycomm);
        MPI_File_close(&file);    
    }
    void read_bin_parallel_distribute_csr(std::string filename, csrMat& Sloc, std::vector<int>& rpvec2D, std::vector<int>& cpvec2D, MPI_Comm worldcomm, MPI_Comm xycomm, MPI_Comm zcomm ){
    }
    void read_bin_parallel_distribute_coo(std::string filename, cooMat& Sloc, std::vector<int>& rpvec2D, std::vector<int>& cpvec2D, MPI_Comm worldcomm, MPI_Comm xycomm, MPI_Comm zcomm ){
        int myxyrank, xysize, myzrank, zsize;
        MPI_Comm_rank( xycomm, &myxyrank);
        MPI_Comm_size( xycomm, &xysize);
        MPI_Comm_rank( zcomm, &myzrank);
        MPI_Comm_size( zcomm, &zsize);
        idx_t grows, gcols;
        idx_large_t gnnz, lnnz;
        if(myzrank == 0){
            std::vector<triplet> myelms;
            read_bin_parallel_distribute(filename, myelms, grows, gcols, gnnz, lnnz, 
                    rpvec2D, cpvec2D, worldcomm,xycomm,zcomm); 
            Sloc.ii.resize(lnnz);
            Sloc.jj.resize(lnnz);
            Sloc.vv.resize(lnnz);
            for(size_t i = 0; i < lnnz; ++i){
                Sloc.ii[i] = myelms[i].row; Sloc.jj[i] = myelms[i].col;
                Sloc.vv[i] = myelms[i].val;
            }
        }
        MPI_Bcast(&lnnz, 1, MPI_IDX_LARGE_T, 0, zcomm);
        if(myzrank > 0){ 
            Sloc.ii.resize(lnnz);
            Sloc.jj.resize(lnnz);
            Sloc.vv.resize(lnnz);
        }
        MPI_Bcast(Sloc.ii.data(), lnnz, MPI_IDX_T, 0, zcomm);
        MPI_Bcast(Sloc.jj.data(), lnnz, MPI_IDX_T, 0, zcomm);
        MPI_Bcast(Sloc.vv.data(), lnnz, MPI_REAL_T, 0, zcomm);
        MPI_Bcast(&grows, 1, MPI_IDX_T, 0, zcomm);
        MPI_Bcast(&gcols, 1, MPI_IDX_T, 0, zcomm);
        MPI_Bcast(&gnnz, 1, MPI_IDX_LARGE_T, 0, zcomm);
        Sloc.gnrows = grows; Sloc.gncols = gcols; Sloc.gnnz = gnnz; Sloc.nnz = Sloc.ii.size();
        if(myzrank > 0){ rpvec2D.resize(grows); cpvec2D.resize(gcols);}
        MPI_Bcast(rpvec2D.data(), grows, MPI_INT, 0, zcomm);
        MPI_Bcast(cpvec2D.data(), gcols, MPI_INT, 0, zcomm);
        idx_t nnz_pp_z = lnnz / zsize;
        idx_t nnz_pp_z_r = lnnz % zsize;
        Sloc.ownedNnz = (myzrank < nnz_pp_z_r?nnz_pp_z+1:nnz_pp_z);
        Sloc.ownedVals.resize(Sloc.ownedNnz, 0.0);
         
        /* assign nnz owners per 2D block */
/*         idx_t nnz_pp_z = lnnz / zsize;
 *         idx_t nnz_pp_z_r = lnnz % zsize;
 *         idx_t tcnt = 0, tp = 0;
 *         for(size_t i = 0; i < lnnz; ++i){
 *             Sloc.owners[i] = tp;
 *             idx_t target_nnz = (tp < nnz_pp_z_r) ? nnz_pp_z+1:nnz_pp_z;
 *             if(++tcnt == target_nnz){ ++tp; tcnt =0;}
 *         }
 */
        /*         if(myzrank == 0 ){
         *             int t = 0;
         *             for(size_t i = 0; i < lnnz; ++i){
         *                 Sloc.owners[i] = t++ % zsize;
         *             }
         *         }
         */
    }
}

#endif
