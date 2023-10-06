#pragma once
#include "comm.hpp"

namespace SpKernels {
    void distribute3D_C(
            coo_mtx& C,
            coo_mtx& Cloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            MPI_Comm cartcomm,
            MPI_Comm *zcomm);
    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_random(

            /*             denseMatrix& A,
             *             denseMatrix& B,
             We assume random A&B for now, therefore we only distribute indices
             */
            denseMatrix& Aloc,
            denseMatrix& Bloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm);
    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_respect_communication(

            /*             denseMatrix& A,
             *             denseMatrix& B,
             We assume random A&B for now, therefore we only distribute indices
             */
            denseMatrix& Aloc,
            denseMatrix& Bloc,
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            coo_mtx& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm)

}
