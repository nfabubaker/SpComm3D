#ifndef _DISTRIBUTE_H
#define _DISTRIBUTE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cassert>
#include "comm.hpp"
#include "Mesh3D.hpp"
#include "mpi.h"
#include "SparseMatrix.hpp"


namespace SpKernels {
    /* this function operates on 2D mesh (communicator split over z) */
    void distribute3D_AB_random(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            cooMat& Cloc,
            const idx_t f,
            MPI_Comm cartXYcomm);
    
    void distribute3D_AB_respect_communication(
            std::vector<int>& rpvec2D,
            std::vector<int>& cpvec2D,
            std::vector<int>& rpvec,
            std::vector<int>& cpvec,
            cooMat& Cloc,
            MPI_Comm cartXYcomm,
            MPI_Comm zcomm,
            MPI_Comm world_comm);

    void create_AB_Bcast(cooMat& Cloc, idx_t floc, 
            std::vector<int>& rpvec, std::vector<int>& cpvec,
            MPI_Comm xycomm, denseMatrix& Aloc, denseMatrix& Bloc);
}
#endif
