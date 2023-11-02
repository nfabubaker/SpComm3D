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
#include <algorithm>

using namespace std;
namespace SpKernels{
    void read_bin_parallel_distribute(std::string filename, coo_mtx& Sloc, std::vector<int>& rpvec2D, std::vector<int>& cpvec2D, MPI_Comm worldcomm, MPI_Comm xycomm, MPI_Comm zcomm ){
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
        MPI_File_read(file, &Sloc.grows, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
        MPI_File_read(file, &Sloc.gcols, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
        MPI_File_read(file, &Sloc.gnnz, 1, MPI_IDX_T, MPI_STATUS_IGNORE);
        if(myworldrank == 0){
            rpvec2D.resize(Sloc.grows); cpvec2D.resize(Sloc.gcols);
            std::vector<idx_t> row_space(Sloc.grows), col_space(Sloc.gcols);
            for(size_t i = 0; i < Sloc.grows; ++i) row_space[i] = i;
            for(size_t i = 0; i < Sloc.gcols; ++i) col_space[i] = i;
            std::random_shuffle(row_space.begin(), row_space.end());
            std::random_shuffle(col_space.begin(), col_space.end());
            /* assign Rows/Cols with RR */
            for(size_t i = 0, p=0; i < Sloc.grows; ++i)
                rpvec2D[row_space[i]] = (p++) % X; 
            for(size_t i = 0, p=0; i < Sloc.gcols; ++i) 
                cpvec2D[col_space[i]] = (p++) % Y; 
        }
        if(myworldrank!=0){ rpvec2D.resize(Sloc.grows); cpvec2D.resize(Sloc.gcols);}
        MPI_Bcast(rpvec2D.data(), Sloc.grows, MPI_INT, 0, worldcomm);
        MPI_Bcast(cpvec2D.data(), Sloc.gcols, MPI_INT, 0, worldcomm);

        if(myzrank == 0){
            idx_t nnz_pp = Sloc.gnnz / xysize;
            idx_t nnz_ppR = Sloc.gnnz % xysize;
            idx_t my_nnz = (myxyrank < nnz_ppR ? nnz_pp+1 : nnz_pp);
            idx_t my_nnz_start = 
                myxyrank > nnz_ppR ?
                (nnz_pp+1)* nnz_ppR + (nnz_pp *(myxyrank-nnz_ppR))
                :
                (nnz_pp+1)*myxyrank;
            idx_t my_nnz_end = my_nnz_start + (myxyrank >= nnz_ppR ? nnz_pp:nnz_pp+1);
            assert(my_nnz_end-my_nnz_start == my_nnz);
            MPI_File_seek(file, my_nnz_start*sizeof(triplet), MPI_SEEK_CUR);
            Sloc.elms.resize(my_nnz);
            for(size_t i = 0; i < my_nnz; ++i)
                MPI_File_read(file, &Sloc.elms[i], 1, mpi_s_type, MPI_STATUS_IGNORE); 
            int trank;
            std::vector<vector<triplet>> sendLists(xysize);
            std::array<int, 2> coords;
            for(auto& elm: Sloc.elms){
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

            Sloc.lnnz = accumulate(recvCnts.begin(), recvCnts.end(), 0.0); 
            Sloc.elms.resize(Sloc.lnnz);
            /* flattern SendLists and perform alltoall */
            vector<triplet> sendBuff;
            for(auto& v : sendLists){
                sendBuff.insert(sendBuff.end(), v.begin(), v.end());
            }
            MPI_Alltoallv(sendBuff.data(), (const int*)sendCnts.data(),
                    (const int*)sendDisps.data(), mpi_s_type, Sloc.elms.data(), 
                    (const int*) recvCnts.data(), (const int*)recvDisps.data(), 
                    mpi_s_type, xycomm);
            MPI_File_close(&file);    

        }
        else{
            MPI_File_close(&file);    
        }
        MPI_Bcast(&Sloc.lnnz, 1, MPI_IDX_T, 0, zcomm);
        if(myzrank > 0) Sloc.elms.resize(Sloc.lnnz);
        MPI_Bcast(Sloc.elms.data(), Sloc.lnnz, mpi_s_type, 0, zcomm);
        Sloc.owners.resize(Sloc.lnnz);
        /* assign nnz owners per 2D block */
        if(myzrank == 0 ){
            int t = 0;
            for(size_t i = 0; i < Sloc.lnnz; ++i){
                Sloc.owners[i] = t++ % zsize;
            }
        }
        MPI_Bcast(Sloc.owners.data(), Sloc.lnnz, MPI_INT, 0, zcomm);
        Sloc.ownedNnz = 0;
        for(size_t i = 0; i < Sloc.lnnz; ++i){ 
            if(Sloc.owners[i] == myzrank) Sloc.ownedNnz++;
        }
        Sloc.ltgR.clear(); Sloc.ltgC.clear(); 
        Sloc.gtlR.resize(Sloc.grows, -1); Sloc.gtlC.resize(Sloc.gcols, -1);
        Sloc.lrows = 0; Sloc.lcols = 0;
        for(size_t i = 0; i < Sloc.lnnz; ++i){
            idx_t rid = Sloc.elms.at(i).row;
            idx_t cid = Sloc.elms.at(i).col;
            if( Sloc.gtlR[rid] == -1){
                Sloc.gtlR[rid] = Sloc.lrows++;
                Sloc.ltgR.push_back(rid);
            }
            if( Sloc.gtlC[cid] == -1){
                Sloc.gtlC[cid] = Sloc.lcols++;
                Sloc.ltgC.push_back(cid);
            }
        }
        /* now localize C indices */
        for(size_t i = 0; i < Sloc.lnnz; ++i){
            idx_t row = Sloc.elms.at(i).row;
            Sloc.elms.at(i).row = Sloc.gtlR.at(row);
            idx_t col = Sloc.elms.at(i).col;
            Sloc.elms.at(i).col = Sloc.gtlC.at(col);
        }
    }
}

#endif
