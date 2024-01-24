
#include <algorithm>
#include <array>
#include <cstddef>
#include <cassert>
#include <numeric>
#include "basic.hpp"
#include "comm.hpp"
#include "Mesh3D.hpp"
#include "mpi.h"


namespace SpKernels {
    void distribute3D_C(
            coo_mtx& C,
            coo_mtx& Cloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            MPI_Comm cartcomm,
            MPI_Comm *zcomm)
    {
        int rank, size, X,Y,Z;
        std::array<int, 3> dims, t1, t2;
        MPI_Comm_size(cartcomm, &size);
        MPI_Comm_rank(cartcomm, &rank);

        MPI_Cart_get(cartcomm, 3, dims.data(), t1.data(), t2.data());
        X = dims[0]; Y=dims[1]; Z=dims[2];
        std::vector<idx_t> frontMeshCnts;
        if(rank == 0){
            rpvec2D.resize(C.grows); cpvec2D.resize(C.gcols);
            std::vector<idx_t> row_space(C.grows), col_space(C.gcols);
            for(size_t i = 0; i < C.grows; ++i) row_space[i] = i;
            for(size_t i = 0; i < C.gcols; ++i) col_space[i] = i;
            std::random_shuffle(row_space.begin(), row_space.end());
            std::random_shuffle(col_space.begin(), col_space.end());
            /* assign Rows/Cols with RR */
            for(size_t i = 0, p=0; i < C.grows; ++i)
                rpvec2D[row_space[i]] = (p++) % X; 
            for(size_t i = 0, p=0; i < C.gcols; ++i) 
                cpvec2D[col_space[i]] = (p++) % Y; 

            std::vector<std::vector<idx_t>> mesh2DCnt(X,
                    std::vector<idx_t> (Y, 0));
            /* now rows/cols are divided, now distribute nonzeros */
            /* first: count nnz per 2D block */
            for(auto& t : C.elms){
                mesh2DCnt[rpvec2D[t.row]][cpvec2D[t.col]]++; 
            }
            frontMeshCnts.resize(size, 0);
            for(int i = 0; i < X; ++i){
                for (int j= 0; j < Y; ++j){
                    dims = {i, j, 0};
                    int trank;MPI_Cart_rank(cartcomm, dims.data(), &trank);
                    frontMeshCnts[trank] = mesh2DCnt[i][j];
                }
            }
        }

        /* communicate local and global dims */
        std::array<idx_t, 3> tarr;
        if(rank == 0) tarr[0] = C.grows; tarr[1] = C.gcols; tarr[2]=C.gnnz;
        MPI_Bcast(tarr.data(), 3, MPI_IDX_T, 0, cartcomm); 
        Cloc.grows = tarr[0]; Cloc.gcols = tarr[1]; Cloc.gnnz = tarr[2];
        if(rank!=0){ rpvec2D.resize(Cloc.grows); cpvec2D.resize(Cloc.gcols);}
        MPI_Bcast(rpvec2D.data(), Cloc.grows, MPI_INT, 0, cartcomm);
        MPI_Bcast(cpvec2D.data(), Cloc.gcols, MPI_INT, 0, cartcomm);
        MPI_Scatter(frontMeshCnts.data(), 1, MPI_IDX_T, &Cloc.lnnz, 1,
                MPI_IDX_T, 0, cartcomm);

        assert(Cloc.lnnz >= 0);
        if(Cloc.lnnz > 0)Cloc.elms.resize(Cloc.lnnz);
        std::vector<std::vector<triplet>> M;
        if(rank == 0){
            M.resize(size);
            std::vector<int> tcnts(size,0);
            for(int i=0; i < size; ++i) 
                if(frontMeshCnts[i] > 0) 
                    M[i].resize(frontMeshCnts[i]);

            for(auto& t : C.elms){
                int p;
                dims = {rpvec2D[t.row], cpvec2D[t.col], 0};
                MPI_Cart_rank(cartcomm, dims.data(), &p);
                M[p][tcnts[p]++] = t;
            }
            for(size_t i = 0; i < Cloc.lnnz; ++i) Cloc.elms.at(i) = M[0].at(i);


        }


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
        /* send data to processors in the forntal mesh */
        if(rank == 0){ for(int i =1; i < size; ++i) if(frontMeshCnts[i] > 0){
            MPI_Send(M[i].data(), frontMeshCnts[i], mpi_s_type, i,
                    777, cartcomm); } }
        else{ if(Cloc.lnnz > 0) MPI_Recv(Cloc.elms.data(), Cloc.lnnz, mpi_s_type, 0, 
                777, cartcomm, MPI_STATUS_IGNORE); }
        /* splitt comm on the Z-axis */
        int zrank, zsize;
        //MPI_Comm_split(cartcomm, rank % Z, rank, zcomm);
        std::array<int, 3> remaindims = {false, false, true};
        MPI_Cart_sub(cartcomm, remaindims.data(), zcomm); 
        MPI_Comm_rank( *zcomm, &zrank);
        MPI_Comm_size( *zcomm, &zsize);
        MPI_Bcast(&Cloc.lnnz, 1, MPI_IDX_T, 0, *zcomm);
        if(zrank > 0) Cloc.elms.resize(Cloc.lnnz);
        MPI_Bcast(Cloc.elms.data(), Cloc.lnnz, mpi_s_type, 0, *zcomm);
        Cloc.owners.resize(Cloc.lnnz);
        /* assign nnz owners per 2D block */
        if(zrank == 0 ){
            int t = 0;
            for(size_t i = 0; i < Cloc.lnnz; ++i){
                Cloc.owners[i] = t++ % zsize;
            }
        }
        MPI_Bcast(Cloc.owners.data(), Cloc.lnnz, MPI_INT, 0, *zcomm);
        for(size_t i = 0; i < Cloc.lnnz; ++i){ if(Cloc.owners[i] == zrank) Cloc.ownedNnz++;}
        Cloc.ltgR.clear(); Cloc.ltgC.clear(); 
        Cloc.lrows = 0; Cloc.lcols = 0;
        for(size_t i = 0; i < Cloc.lnnz; ++i){
            idx_t rid = Cloc.elms.at(i).row;
            idx_t cid = Cloc.elms.at(i).col;
            if( Cloc.gtlR.find(rid) == Cloc.gtlR.end()){
                Cloc.gtlR[rid] = Cloc.lrows++;
                Cloc.ltgR.push_back(rid);
            }
            if(Cloc.gtlC.find(cid) == Cloc.gtlC.end()){
                Cloc.gtlC[cid] = Cloc.lcols++;
                Cloc.ltgC.push_back(cid);
            }
        }
        /* now localize C indices */
        for(size_t i = 0; i < Cloc.lnnz; ++i){
            idx_t row = Cloc.elms.at(i).row;
            Cloc.elms.at(i).row = Cloc.gtlR.at(row);
            idx_t col = Cloc.elms.at(i).col;
            Cloc.elms.at(i).col = Cloc.gtlC.at(col);
        }
    }

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
               SparseComm<idx_t>::P2P, world_comm, MPI_COMM_NULL);
       
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
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm,
            MPI_Comm zcomm,
            MPI_Comm world_comm)
    {
        rpvec.resize(Cloc.grows, -1); cpvec.resize(Cloc.gcols, -1);
        
        partition_respect_comm(Cloc.grows, rpvec, 0, rpvec2D, Cloc.ltgR, cartXYcomm);
        partition_respect_comm(Cloc.gcols, cpvec, 1, cpvec2D, Cloc.ltgC, cartXYcomm);


        /* send in rpvec and cpvec in Z directio */
        MPI_Bcast(rpvec.data(), Cloc.grows, MPI_INT, 0, zcomm);
        MPI_Bcast(cpvec.data(), Cloc.gcols, MPI_INT, 0, zcomm);

        /* check the partition: get X and Y for each pvec entry and compare it with
         * rpvec2D and cpvec2D*/
        int rank, size, X,Y;
        std::array<int, 3> dims, tdims;
        MPI_Comm_size(world_comm, &size);
        MPI_Comm_rank(world_comm, &rank);
        //MPI_Cart_coords(cartXYcomm, rank, 3,dims.data()); 
        //MPI_Cart_get(cartXYcomm, 2, dims.data(), t1.data(), t2.data());
        //X = dims[0]; Y=dims[1]; 
        for(size_t i = 0; i < Cloc.grows; ++i){
            if(rpvec[i] != -1){
                MPI_Cart_coords(cartXYcomm, rpvec[i], 2, dims.data());
                assert(dims[0] == rpvec2D[i]);
            }

        }
        for(size_t i = 0; i < Cloc.gcols; ++i){
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
            coo_mtx& Cloc,
            MPI_Comm cartXYcomm)
    {
        idx_t M = Cloc.grows, N = Cloc.gcols;
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


    //    void distribute3D(coo_mtx& C, idx_t f, int c, MPI_Comm world_comm, coo_mtx& Cloc, denseMatrix& Aloc, 
    //            denseMatrix& Bloc, std::vector<int>& rpvec, 
    //            std::vector<int>& cpvec, MPI_Comm* xycomm, MPI_Comm* zcomm ){
    //
    //        /*
    //         * Distribute C and fill rpvec2D, cpvec2D
    //         */
    //        int myrank, size;
    //        MPI_Comm_size(world_comm, &size);
    //        MPI_Comm_rank(world_comm, &myrank);
    //        std::array<int, 3> dims = {0,0,c};
    //        std::array<int,3> zeroArr ={0,0,0};
    //        std::array<int,3> tdims ={0,0,0};
    //        MPI_Dims_create(size, 3, dims.data());
    //        MPI_Comm cartcomm;
    //        MPI_Cart_create(world_comm, 3, dims.data(), zeroArr.data(), 0, &cartcomm);   
    //        int X = dims[0], Y = dims[1], Z = dims[2];
    //        idx_t floc = f / Z;
    //        if(f % Z > 0){
    //            MPI_Cart_coords(cartcomm, myrank, 3, tdims.data());
    //            int myzcoord = tdims[2];
    //            if(myzcoord < f% Z) ++floc;
    //        }
    //
    //        distribute3D_C(C, Cloc, rpvec2D, cpvec2D, cartcomm, zcomm);
    //        rpvec.resize(Cloc.grows); cpvec.resize(Cloc.gcols);
    //
    //        /* prepare Aloc, Bloc according to local dims of Cloc */
    //        // split the 3D mesh communicator to 2D slices 
    //        std::array<int, 3> remaindims = {true, true, false};
    //        MPI_Cart_sub(cartcomm, remaindims.data(), xycomm); 
    //        int myxyrank;
    //        MPI_Comm_rank(*xycomm, &myxyrank);  
    //        /* distribute Aloc and Bloc  */
    //        distribute3D_AB(Aloc, Bloc, rpvec2D, cpvec2D, rpvec, cpvec,
    //                Cloc, f,  *xycomm); 
    //        /* update C info */
    //        for(size_t i =0; i < Cloc.grows; ++i) 
    //            if(rpvec.at(i) == myxyrank && Cloc.gtlR.at(i) == -1) Cloc.gtlR.at(i) = Cloc.lrows++;
    //        for(size_t i =0; i < Cloc.gcols; ++i) 
    //            if(cpvec.at(i) == myxyrank && Cloc.gtlC.at(i) == -1) Cloc.gtlC.at(i) = Cloc.lcols++;
    //        Aloc.m = Cloc.lrows; Aloc.n = floc;
    //        Bloc.m = Cloc.lcols; Bloc.n = floc;
    //        Aloc.data.resize(Aloc.m * Aloc.n, myrank+1);
    //        Bloc.data.resize(Bloc.m * Bloc.n, myrank+1);
    //        for(size_t i = 0; i < Cloc.grows; ++i) assert(rpvec[i] >= 0 && rpvec[i] <= size/Z);
    //        for(size_t i = 0; i < Cloc.gcols; ++i) assert(cpvec[i] >= 0 && cpvec[i] <= size/Z);
    //    }

    void create_AB_Bcast(coo_mtx& Cloc, idx_t floc, 
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
        for(size_t i=0; i < Cloc.grows; ++i){
            MPI_Cart_coords(xycomm, rpvec[i], 2, tdims.data());
            if(tdims[0] == myxcoord){ 
                Aloc.m++;
            }
        }
        for(size_t i=0; i < Cloc.gcols; ++i){ 
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
