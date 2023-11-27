#ifndef _DISTRIBUTE_H
#define _DISTRIBUTE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cassert>
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
            MPI_Comm *zcomm);
    


    /* this function operates on 2D mesh (communicator split over z) */
/*     void distribute3D_AB_respect_communication(
 *             denseMatrix& Aloc,
 *             denseMatrix& Bloc,
 *             std::vector<int>& rpvec2D,
 *             std::vector<int>& cpvec2D,
 *             std::vector<int>& rpvec,
 *             std::vector<int>& cpvec,
 *             coo_mtx& Cloc,
 *             const idx_t f,
 *             MPI_Comm cartXYcomm)
 *     {
 *     }
 */
    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_random(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm);
    
    void distribute3D_AB_respect_communication(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm,
            MPI_Comm zcomm,
            MPI_Comm world_comm);


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
            MPI_Comm xycomm, denseMatrix& Aloc, denseMatrix& Bloc);
}
#endif
