

#include "basic.hpp"
#include "Mesh3D.hpp"
#include "mpi.h"
#include "mpi_proto.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cassert>




namespace SpKernels {
    void distribute3D_C(
            coo_mtx& C,
            Mesh3D& mesh3d,
            coo_mtx& Cloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            MPI_Comm comm,
            MPI_Comm *zcomm)
    {
        int rank, size;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        std::vector<idx_t> frontMeshCnts;
        if(rank == 0){
            std::vector<idx_t> row_space(C.grows), col_space(C.gcols);
            for(size_t i = 0; i < C.grows; ++i) row_space[i] = i;
            for(size_t i = 0; i < C.gcols; ++i) col_space[i] = i;
            std::random_shuffle(row_space.begin(), row_space.end());
            std::random_shuffle(col_space.begin(), col_space.end());
            /* assign Rows/Cols with RR */
            for(size_t i = 0, p=0; i < C.grows; ++i)
                rpvec2D[row_space[i]] = (p++) % mesh3d.getX(); 
            for(size_t i = 0, p=0; i < C.gcols; ++i) 
                cpvec2D[col_space[i]] = (p++) % mesh3d.getY(); 

            std::vector<std::vector<idx_t>> mesh2DCnt(mesh3d.getX(),
                    std::vector<idx_t> (mesh3d.getY(), 0));
            /* now rows/cols are divided, now distribute nonzeros */
            /* first: count nnz per 2D block */
            for(auto& t : C.elms){
                mesh2DCnt[rpvec2D[t.row]][cpvec2D[t.col]]++; 
            }
            frontMeshCnts.resize(size, 0);
            for(size_t i = 0; i < mesh3d.getX(); ++i){
                for (size_t j= 0; j < mesh3d.getY(); ++j){
                    frontMeshCnts[mesh3d.getRankFromCoords(i, j, 0)] 
                        = mesh2DCnt[i][j];
                }
            }
        }

        /* communicate local and global dims */
        std::array<idx_t, 3> tarr;
        if(rank == 0) tarr[0] = C.grows; tarr[1] = C.gcols; tarr[2]=C.gnnz;
        MPI_Bcast(tarr.data(), 3, MPI_IDX_T, 0, comm); 
        Cloc.grows = tarr[0]; Cloc.gcols = tarr[1]; Cloc.gnnz = tarr[2];
        MPI_Bcast(rpvec2D.data(), Cloc.grows, MPI_INT, 0, comm);
        MPI_Bcast(cpvec2D.data(), Cloc.gcols, MPI_INT, 0, comm);
        MPI_Scatter(frontMeshCnts.data(), 1, MPI_IDX_T, &Cloc.lnnz, 1,
                MPI_IDX_T, 0, comm);

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
                int p = mesh3d.getRankFromCoords(rpvec2D[t.row], cpvec2D[t.col], 0);
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
        if(rank == 0){ for(int i =0; i < size; ++i) if(frontMeshCnts[i] > 0){
            MPI_Send(M[i].data(), frontMeshCnts[i], mpi_s_type, i,
                    777, comm); } }
        else{ if(Cloc.lnnz > 0) MPI_Recv(Cloc.elms.data(), Cloc.lnnz, mpi_s_type, 0, 
                777, comm, MPI_STATUS_IGNORE); }
        /* splitt comm on the Z-axis */
        int zrank, zsize;
        MPI_Comm_split(comm, rank % mesh3d.getMZ(), rank, zcomm);
        MPI_Comm_rank( *zcomm, &zrank);
        MPI_Comm_size( *zcomm, &zsize);
        MPI_Bcast(&Cloc.lnnz, 1, MPI_IDX_T, 0, *zcomm);
        if(zrank > 0) Cloc.elms.resize(Cloc.lnnz);
        MPI_Bcast(Cloc.elms.data(), Cloc.lnnz, mpi_s_type, 0, *zcomm);
        Cloc.owners.resize(Cloc.lnnz);
        /* assign nnz owners per 2D block */
        if(zrank == 0 ){
            int t = 0;
            int *tarr;
            tarr = mesh3d.getAllCoords(rank);
            int X = tarr[0]; int Y = tarr[1];
            for(size_t i = 0; i < Cloc.lnnz; ++i){
                /*                 int pp = t++ % zsize;
                 *                 Cloc.owners[i] = mesh3d.getRankFromCoords(X, Y, pp);
                 */
                Cloc.owners[i] = t++ % zsize;
            }
        }
        MPI_Bcast(Cloc.owners.data(), Cloc.lnnz, MPI_INT, 0, *zcomm);
        for(size_t i = 0; i < Cloc.lnnz; ++i){ if(Cloc.owners[i] == zrank) Cloc.ownedNnz++;}

    }


    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB(

            /*             denseMatrix& A,
             *             denseMatrix& B,
             We assume random A&B for now, therefore we only distribute indices
             */
            Mesh3D& mesh3d,
            denseMatrix& Aloc,
            denseMatrix& Bloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            const coo_mtx& Cloc,
            const idx_t f,
            int zcoord,
            MPI_Comm comm)
    {
        idx_t M = Cloc.grows, N = Cloc.gcols;
        Aloc.m = Cloc.grows; Aloc.n = f;
        Bloc.m = Cloc.gcols; Bloc.n = f;
        int rank, size;
        MPI_Comm_rank( comm, &rank);
        MPI_Comm_size( comm, &size);
        if(rank == 0){
            int X = mesh3d.getX(), Y=mesh3d.getY();
            std::vector<size_t> cntsPer2Drow(X,0);
            std::vector<size_t> cntsPer2Dcol(Y,0);
            for(size_t i=0; i < M; ++i){
                int rowid2D = rpvec2D[i];
                rpvec[i] = mesh3d.getRankFromCoords(rowid2D, 
                        cntsPer2Drow[rowid2D]++ % Y, 0); 
            }
            for(size_t i=0; i < N; ++i){
                int colid2D = cpvec2D[i];
                cpvec[i] = mesh3d.getRankFromCoords(
                        cntsPer2Dcol[colid2D]++ % X,
                        colid2D, 0); 
            }

        }
        MPI_Bcast(rpvec.data(), M, MPI_INT, 0, comm);
        MPI_Bcast(cpvec.data(), N, MPI_INT, 0, comm);
        Aloc.m = 0; Aloc.n = f;
        Bloc.m = 0; Bloc.n = f;
        for(size_t i = 0; i < M; ++i) if(rpvec[i] == mesh3d.getRank())
            Aloc.m++;
        for(size_t i = 0; i < N; ++i) if(cpvec[i] == mesh3d.getRank())
            Bloc.m++;
        Aloc.data.resize(Aloc.m * Aloc.n, 1);
        Bloc.data.resize(Bloc.m * Bloc.n, 1);
    }


    void distribute3D(coo_mtx& C, idx_t f, int c, MPI_Comm world_comm, coo_mtx& Cloc, denseMatrix& Aloc, 
            denseMatrix& Bloc, std::vector<int>& rpvec, 
            std::vector<int>& cpvec, MPI_Comm* xycomm, MPI_Comm* zcomm ){

        std:: vector<int> rpvec2D(C.grows), cpvec2D(C.gcols);
        /*
         * Distribute C and fill rpvec2D, cpvec2D
         */
        int myrank, size;
        MPI_Comm_size(world_comm, &size);
        MPI_Comm_rank(world_comm, &myrank);
        std::array<int, 3> dims = {0,0,c};
        MPI_Dims_create(size, 3, dims.data());
        Mesh3D mesh3d(dims[0], dims[1], dims[2], world_comm);
        idx_t floc = f / mesh3d.getZ();
        if(f % mesh3d.getZ() > 0){
            int myzcoord = mesh3d.getZCoord(mesh3d.getRank());
            if(myzcoord < f%mesh3d.getZ()) ++floc;
        }

        distribute3D_C(C, mesh3d, Cloc, rpvec2D, cpvec2D, world_comm, zcomm);
        // split the 3D mesh communicator to 2D slices 
        int color = mesh3d.getRank() / mesh3d.getMZ();
        MPI_Comm_split(mesh3d.getComm(), color, mesh3d.getRank(), xycomm); 
        /* distribute Aloc and Bloc  */
        distribute3D_AB(mesh3d, Aloc, Bloc, rpvec2D, cpvec2D, rpvec, cpvec,
                Cloc, f, color, *xycomm); 
        for(size_t i = 0; i < Cloc.grows; ++i) assert(rpvec[i] >= 0 && rpvec[i] <= size/mesh3d.getZ());
        for(size_t i = 0; i < Cloc.gcols; ++i) assert(cpvec[i] >= 0 && cpvec[i] <= size/mesh3d.getZ());
    }
}
