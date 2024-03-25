
#include <algorithm>
#include <array>
#include <cstddef>
#include <cassert>
#include <numeric>
#include "SparseMatrix.hpp"
#include "basic.hpp"
#include "comm.hpp"
#include "Mesh3D.hpp"
#include "mpi.h"
#include "distribute.hpp"



namespace SpKernels {

    inline int get_pid_from_indx(idx_t g_index, idx_t units_per_processor, idx_t units_remainder)
    {

        idx_t first_few_cnt = (units_per_processor+1) * units_remainder;
        if(g_index > first_few_cnt)
            return units_remainder + (g_index-first_few_cnt)/units_per_processor;
        else return g_index/(units_per_processor+1);
    }

    void partition_respect_comm(idx_t const dimsize, std::vector<int>& pvec,
            int vtype,
            const std::vector<int>& pvec2D, const std::vector<idx_t> units_to_partition , MPI_Comm world_comm){
        int worldsize, myworldrank;
        MPI_Comm_size(world_comm, &worldsize);
        MPI_Comm_rank(world_comm, &myworldrank);

        idx_t M = dimsize; 
        idx_t rows_pp = M/worldsize, rows_ppR = M%worldsize;
       
        idx_t myrowsstart = 
            myworldrank > rows_ppR ?
            (rows_pp+1)* rows_ppR + (rows_pp *(myworldrank-rows_ppR))
            :
            (rows_pp+1)*myworldrank;
        idx_t myrowsend = myrowsstart + (myworldrank >= rows_ppR ? rows_pp:rows_pp+1);

        idx_t myrowscount = (myworldrank < rows_ppR ? rows_pp+1: rows_pp);
        assert(myrowsend-myrowsstart == myrowscount);
        /* for my local rows, assign them to their responsible owners */
        std::vector<std::vector<idx_t>> sendBuff(worldsize);
        for(size_t i = 0; i < units_to_partition.size(); ++i){
           idx_t gid = units_to_partition[i]; 
           int where_to_send = get_pid_from_indx(gid, rows_pp, rows_ppR); 
           sendBuff[where_to_send].push_back(gid);
        }

        SparseComm<idx_t> sc;
        std::vector<int> per_processor_ssize(worldsize, 0),
            per_processor_rsize(worldsize, 0);
        int nsendto=0, nrecvfrom = 0;
        idx_t totalSendCnt=0, totalRecvCnt = 0;
        for(int i = 0; i < worldsize; ++i)
            if(sendBuff[i].size() > 0 && i != myworldrank){
                per_processor_ssize[i] = sendBuff[i].size();
                nsendto++; 
                totalSendCnt += per_processor_ssize[i]; 
            }
       MPI_Alltoall(per_processor_ssize.data(), 1, MPI_INT, 
               per_processor_rsize.data(), 1, MPI_INT, world_comm);  
       
       /* now prepare recv buffer */
       for(int i = 0; i < worldsize; ++i){ 
           if(per_processor_rsize[i] > 0 && i != myworldrank){
               nrecvfrom++;
               totalRecvCnt += per_processor_rsize[i];
           }
       }
       sc.init(1, nsendto, nrecvfrom, totalSendCnt, totalRecvCnt,
               SparseComm<idx_t>::P2P, world_comm, MPI_COMM_NULL, true, true);
       
       idx_t idx1 = 0, idx2=0;
        for(int i = 0; i < worldsize; ++i){
            idx_t scnt = per_processor_ssize[i];
            idx_t rcnt = per_processor_rsize[i];
            if(scnt > 0 && i != myworldrank){
                sc.sendCount[idx1] = sc.sendDisp[idx1+1] = scnt;
                sc.outSet[idx1++] = i;
            }
            if(rcnt > 0 && i != myworldrank){
                sc.recvCount[idx2] = sc.recvDisp[idx2+1] = rcnt;
                sc.inSet[idx2++] = i;
            }
        }
        for(int i = 0; i < sc.outDegree; ++i)
            sc.sendDisp[i+1] += sc.sendDisp[i]; 
        for(int i = 0; i < sc.inDegree; ++i)
            sc.recvDisp[i+1] += sc.recvDisp[i]; 
        idx1 = 0; 
       for(int i = 0; i < worldsize; ++i){
            auto& v = sendBuff.at(i);
           if(i != myworldrank && v.size() > 0){
            for(auto& e : v) sc.sendBuff[idx1++] = e; 
           }
       }

       
       sc.perform_sparse_comm(false, false);
       
       /* I'm responsible for rows from ii to jj  */
       std::vector<std::vector<int>> rows_sets(myrowsend - myrowsstart);

       /* add received entries to my row sets */
       for(int i = 0; i < sc.inDegree; ++i){
            int p = sc.inSet[i];
           for(idx_t j = sc.recvDisp[i]; j < sc.recvDisp[i] + sc.recvCount[i];++j){
               idx_t rv = sc.recvBuff[j];
               assert(rv >= myrowsstart && rv < myrowsend);
               rows_sets.at(rv-myrowsstart).push_back(p);
           }
       }
       /* include also my rows in the row sets */
       for(auto& e : sendBuff[myworldrank]){
           assert( e >= myrowsstart && e < myrowsend);
          rows_sets.at(e - myrowsstart).push_back(myworldrank); 
       }


       /* pick randomly*/
       std::vector<int> lpvec(myrowsend - myrowsstart, -1);
      

       std::array<int, 3> dims, tarr1, tarr2;
       /* get the dims */
       MPI_Cart_get(world_comm, 3, dims.data(), tarr1.data(), tarr2.data()); 
       int X = dims[0], Y = dims[1];
       int currDimSize = (vtype == 0 ? Y : X);
       int tcnt = 0;
       for(idx_t i = 0; i < (myrowsend - myrowsstart); ++i){
            if(rows_sets[i].size() == 0){
                if(vtype ==0){
                    tarr1[0] = pvec2D[i+myrowsstart];
                    tarr1[1] = tcnt++ % currDimSize;
                }
                else{
                    tarr1[1] = pvec2D[i+myrowsstart];
                    tarr1[0] = tcnt++ % currDimSize;
                }
                MPI_Cart_rank(world_comm, tarr1.data(), &lpvec[i]);
                continue;
            }
            std::random_shuffle(rows_sets[i].begin(), rows_sets[i].end());
            int rdn = rand() % rows_sets[i].size();
            lpvec[i] = rows_sets[i].at(rdn);
            MPI_Cart_coords(world_comm, lpvec[i], 3, dims.data()); 
            assert(dims[vtype] == pvec2D[i+myrowsstart]);
       }
       std::vector<int> AG_sendcnts(worldsize, 0), disps(worldsize+1, 0);
       for(int i = 0; i < worldsize; ++i){
           disps[i+1] = disps[i] + (i < rows_ppR? (rows_pp+1): rows_pp);
           AG_sendcnts[i] = (i < rows_ppR ? rows_pp+1 : rows_pp);
       }

       MPI_Allgatherv(lpvec.data(), (myrowsend-myrowsstart), MPI_INT,
               pvec.data(), AG_sendcnts.data(), disps.data(), MPI_INT, world_comm); 
            
    }
    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_respect_communication(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            cooMat& Cloc,
            MPI_Comm cartXYcomm,
            MPI_Comm zcomm,
            MPI_Comm world_comm)
    {
        rpvec.resize(Cloc.gnrows, -1); cpvec.resize(Cloc.gncols, -1);
        
        partition_respect_comm(Cloc.gnrows, rpvec, 0, rpvec2D, Cloc.ltgR, cartXYcomm);
        partition_respect_comm(Cloc.gncols, cpvec, 1, cpvec2D, Cloc.ltgC, cartXYcomm);


        /* send in rpvec and cpvec in Z directio */
        MPI_Bcast(rpvec.data(), Cloc.gnrows, MPI_INT, 0, zcomm);
        MPI_Bcast(cpvec.data(), Cloc.gncols, MPI_INT, 0, zcomm);

        /* check the partition: get X and Y for each pvec entry and compare it with
         * rpvec2D and cpvec2D*/
        int rank, size, X,Y;
        std::array<int, 3> dims, tdims;
        MPI_Comm_size(world_comm, &size);
        MPI_Comm_rank(world_comm, &rank);
        //MPI_Cart_coords(cartXYcomm, rank, 3,dims.data()); 
        //MPI_Cart_get(cartXYcomm, 2, dims.data(), t1.data(), t2.data());
        //X = dims[0]; Y=dims[1]; 
        for(size_t i = 0; i < Cloc.gnrows; ++i){
            if(rpvec[i] != -1){
                MPI_Cart_coords(cartXYcomm, rpvec[i], 2, dims.data());
                assert(dims[0] == rpvec2D[i]);
            }

        }
        for(size_t i = 0; i < Cloc.gncols; ++i){
            if(cpvec[i] != -1){
                MPI_Cart_coords(cartXYcomm, cpvec[i], 2, dims.data());
                assert(dims[1] == cpvec2D[i]);
            }
        }
        

    }

    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_random(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            cooMat& Cloc,
            MPI_Comm cartXYcomm)
    {
        idx_t M = Cloc.gnrows, N = Cloc.gncols;
        rpvec.resize(M); cpvec.resize(N);
        int rank, size, X,Y;
        std::array<int, 2> dims, t1, t2;
        MPI_Comm_size(cartXYcomm, &size);
        MPI_Comm_rank(cartXYcomm, &rank);
        MPI_Cart_coords(cartXYcomm, rank, 2,dims.data()); 
        MPI_Cart_get(cartXYcomm, 2, dims.data(), t1.data(), t2.data());
        X = dims[0]; Y=dims[1]; 
        if(rank == 0){
            std::vector<size_t> cntsPer2Drow(X,0);
            std::vector<size_t> cntsPer2Dcol(Y,0);
            for(size_t i=0; i < M; ++i){
                int rowid2D = rpvec2D[i];
                dims = {rowid2D, (int)cntsPer2Drow[rowid2D]++ % Y};
                MPI_Cart_rank(cartXYcomm, dims.data(), &rpvec[i]); 
            }
            for(size_t i=0; i < N; ++i){
                int colid2D = cpvec2D[i];
                dims = {(int)cntsPer2Dcol[colid2D]++ % X, colid2D};
                MPI_Cart_rank(cartXYcomm, dims.data(), &cpvec[i]); 
            }

        }
        MPI_Bcast(rpvec.data(), M, MPI_INT, 0, cartXYcomm);
        MPI_Bcast(cpvec.data(), N, MPI_INT, 0, cartXYcomm);

    }


    void create_AB_Bcast(cooMat& Cloc, idx_t floc, 
            std::vector<int>& rpvec, std::vector<int>& cpvec,
            MPI_Comm xycomm, denseMatrix& Aloc, denseMatrix& Bloc)
    {
        int myxyrank;
        std::array<int,3> tdims ={0,0,0};
        MPI_Comm_rank(xycomm, &myxyrank);
        MPI_Cart_coords(xycomm, myxyrank, 2, tdims.data());
        int myxcoord = tdims[0];
        int myycoord = tdims[1];
        Aloc.m = 0; Bloc.m = 0;
        for(size_t i=0; i < Cloc.gnrows; ++i){
            MPI_Cart_coords(xycomm, rpvec[i], 2, tdims.data());
            if(tdims[0] == myxcoord){ 
                Aloc.m++;
            }
        }
        for(size_t i=0; i < Cloc.gncols; ++i){ 
            MPI_Cart_coords(xycomm, cpvec[i], 2, tdims.data());
            if(tdims[1] == myycoord){ 
                Bloc.m++; 
            }
        }
        Aloc.n = floc;
        Bloc.n = floc;
        Aloc.data.resize(Aloc.m * Aloc.n, myxyrank+1);
        Bloc.data.resize(Bloc.m * Bloc.n, myxyrank+1);
    }
}
