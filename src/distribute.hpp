

#include "basic.hpp"
#include "Mesh3D.hpp"
#include "mpi.h"
#include <algorithm>
#include <array>
#include <cstddef>



namespace SpKernels {
    void distribute3D_C(
            coo_mtx& C,
            Mesh3D& mesh3d,
            coo_mtx& Cloc,
            MPI_Comm comm)
    {
        int rank, size;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        std::vector<int> rpvec(C.grows, -1), cpvec(C.gcols, -1); 
        std::vector<idx_t> row_space(C.grows), col_space(C.gcols);
        for(size_t i = 0; i < C.grows; ++i) row_space[i] = i;
        for(size_t i = 0; i < C.gcols; ++i) col_space[i] = i;
        std::random_shuffle(row_space.begin(), row_space.end());
        std::random_shuffle(col_space.begin(), col_space.end());
        /* assign Rows/Cols with RR */
        for(size_t i = 0, p=0; i < C.grows; ++i)
            rpvec[row_space[i]] = (p++) % mesh3d.getX(); 
        for(size_t i = 0, p=0; i < C.gcols; ++i) 
            cpvec[col_space[i]] = (p++) % mesh3d.getY(); 

        std::vector<std::vector<idx_t>> mesh2DCnt(mesh3d.getX(),
                std::vector<idx_t> (mesh3d.getY(), 0));
        /* now rows/cols are divided, now distribute nonzeros */
        /* first: count nnz per 2D block */
        for(auto& t : C.elms){
            mesh2DCnt[rpvec[t.row]][cpvec[t.col]]++; 
        }
        std::vector<idx_t> frontMeshCnts(size, 0);
        for(size_t i = 0; i < mesh3d.getX(); ++i){
            for (size_t j= 0; j < mesh3d.getY(); ++j){
                frontMeshCnts[mesh3d.getRankFromCoords(i, j, 0)] 
                    = mesh2DCnt[i][j];
            }

        }

        /* communicate local and global dims */
        std::array<idx_t, 3> tarr;
        if(rank == 0) tarr[0] = C.grows; tarr[1] = C.gcols; tarr[2]=C.gnnz;
        MPI_Bcast(tarr.data(), 3, MPI_IDX_T, 0, comm); 
        Cloc.grows = tarr[0]; Cloc.gcols = tarr[1]; Cloc.gnnz = tarr[2];
        
        MPI_Scatter(frontMeshCnts.data(), 1, MPI_IDX_T, &Cloc.lnnz, 1,
                MPI_IDX_T, 0, comm);

        std::vector<int> tcnts(size,0);
        std::vector<std::vector<triplet>> M(size);
        for(int i=0; i < size; ++i) 
            if(frontMeshCnts[i] > 0) 
                M[i].resize(frontMeshCnts[i]);
       
        for(auto& t : C.elms){
            int p = mesh3d.getRankFromCoords(rpvec[t.row], cpvec[t.col], 0);
            M[p][tcnts[p]++] = t;
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
        MPI_Comm zcomm;
        int zrank, zsize;
        MPI_Comm_split(comm, rank % mesh3d.getMZ(), rank, &zcomm);
        MPI_Comm_rank( zcomm, &zrank);
        MPI_Comm_size( zcomm, &zsize);
        MPI_Bcast(&Cloc.lnnz, 1, MPI_IDX_T, 0, zcomm);
        MPI_Bcast(Cloc.elms.data(), Cloc.lnnz, mpi_s_type, 0, zcomm);
        /* assign nnz owners per 2D block */
        if(zrank == 0 ){
            int t = 0;
            int *tarr;
            tarr = mesh3d.getAllCoords(rank);
            int X = tarr[0]; int Y = tarr[1];
           for(size_t i = 0; i < Cloc.lnnz; ++i){
                int pp = t++ % zrank;
                Cloc.owners[i] = mesh3d.getRankFromCoords(X, Y, pp);
           }
        }
        MPI_Bcast(Cloc.owners.data(), Cloc.lnnz, MPI_INT, 0, zcomm);
    }
    void distribute3D_AB(
            
/*             denseMatrix& A,
 *             denseMatrix& B,
    We assume random A&B for now, therefore we only distribute indices
 */
            Mesh3D& mesh3d,
            denseMatrix& Aloc,
            denseMatrix& Bloc,
            const coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm comm)
    {
        Aloc.m = Cloc.grows; Aloc.n = f;
        Bloc.m = Cloc.gcols; Bloc.n = f;
        

        
    }
}
