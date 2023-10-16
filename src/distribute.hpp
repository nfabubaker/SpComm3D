

#include "basic.hpp"
#include "comm.hpp"
#include "Mesh3D.hpp"
#include "mpi.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cassert>




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
        Cloc.gtlR.resize(Cloc.grows, -1); Cloc.gtlC.resize(Cloc.gcols, -1);
        Cloc.lrows = 0; Cloc.lcols = 0;
        for(size_t i = 0; i < Cloc.lnnz; ++i){
            idx_t rid = Cloc.elms.at(i).row;
            idx_t cid = Cloc.elms.at(i).col;
            if( Cloc.gtlR[rid] == -1){
                Cloc.gtlR[rid] = Cloc.lrows++;
                Cloc.ltgR.push_back(rid);
            }
            if( Cloc.gtlC[cid] == -1){
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


    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_respect_communication(
            denseMatrix& Aloc,
            denseMatrix& Bloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm)
    {
    }
    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_random(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm)
    {
        idx_t M = Cloc.grows, N = Cloc.gcols;
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
            MPI_Comm xycomm, denseMatrix& Aloc, denseMatrix& Bloc,
            std::vector<idx_t> gtlR, std::vector<idx_t> gtlC,
            std::vector<idx_t> ltgR, std::vector<idx_t> ltgC
            )
    {
        int myxyrank;
        std::array<int,3> tdims ={0,0,0};
        MPI_Cart_coords(xycomm, myxyrank, 2, tdims.data());
        int myxcoord = tdims[0];
        int myycoord = tdims[1];
        Aloc.m = 0; Bloc.m = 0;
        for(size_t i=0; i < Cloc.grows; ++i){
            MPI_Cart_coords(xycomm, rpvec[i], 2, tdims.data());
            if(tdims[0] == myxcoord){ 
                ltgR.push_back(i);
                gtlR.at(i) = Aloc.m++;
            }
        }
        for(size_t i=0; i < Cloc.gcols; ++i){ 
            MPI_Cart_coords(xycomm, cpvec[i], 2, tdims.data());
            if(tdims[1] == myycoord){ 
                ltgC.push_back(i);
                gtlC.at(i) = Bloc.m++; 
            }
        }
        Aloc.n = floc;
        Bloc.n = floc;
        Aloc.data.resize(Aloc.m * Aloc.n, 1);
        Bloc.data.resize(Bloc.m * Bloc.n, 1);
    }
}
