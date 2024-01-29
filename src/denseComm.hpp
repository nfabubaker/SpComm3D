#pragma once
#include "../src/basic.hpp"
#include "SparseMatrix.hpp"
#include "comm.hpp"


#include <mpi.h>
#include <vector>
using namespace SpKernels;

namespace DComm {
    typedef struct _denseComm{
        MPI_Comm commX, commY;
        int OP; /*  0 = Bcast, 1- Allreduce */
        bool commXflag = false, commYflag = false; 
        real_t *bufferptrX, *bufferptrY;
        int outDegreeX = 0, outDegreeY = 0;
        std::vector<idx_t > bcastXcnt, bcastYcnt, bcastXdisp, bcastYdisp;
        std::vector<MPI_Request> rqstsX, rqstsY;
        std::vector<real_t> reduceBuffer;
        void perform_dense_comm(){

            if(this->OP == 0){/* bcast */
                std::vector<MPI_Status> sttsX(outDegreeX), sttsY(outDegreeY);
                if(commXflag)
                    for(int i =0; i < outDegreeX; ++i)
                        MPI_Ibcast(bufferptrX + bcastXdisp[i], bcastXcnt[i], MPI_REAL_T, i,commX, &rqstsX[i]);
                if(commYflag)
                    for(int i =0; i < outDegreeY; ++i)
                        MPI_Ibcast(bufferptrY + bcastYdisp[i], bcastYcnt[i], MPI_REAL_T, i,commY, &rqstsY[i]);
                if(commXflag) MPI_Waitall(outDegreeX, rqstsX.data(), sttsX.data());
                if(commYflag) MPI_Waitall(outDegreeY, rqstsY.data(), sttsY.data());
            }
            else if (this->OP == 1){
                if(commXflag && commYflag){
                    //int myzrank;
                    //MPI_Comm_rank(commX, &myzrank);
                    //MPI_Reduce_scatter(bufferptrX, bufferptrX + bcastXdisp[myzrank], (const int*) bcastXcnt.data(), MPI_REAL_T, MPI_SUM, commX);
                    MPI_Reduce_scatter(bufferptrX, bufferptrY, (const int*) bcastXcnt.data(), MPI_REAL_T, MPI_SUM, commX);
                }
                else if(commXflag){
                    MPI_Allreduce(MPI_IN_PLACE, bufferptrX, bcastXdisp[outDegreeX], MPI_REAL_T, MPI_SUM, commX);
                }
                else if(commYflag){
                    MPI_Allreduce(MPI_IN_PLACE, bufferptrY, bcastYdisp[outDegreeY], MPI_REAL_T, MPI_SUM, commY);
                }
            }
        }

    } DenseComm;


    void setup_3dspmm_bcast(
            cooMat& Aloc,
            const idx_t f,
            const int c,
            denseMatrix& Xloc,
            denseMatrix& Yloc, 
            std::vector<int>& rpvec, 
            std::vector<int>& cpvec,
            const MPI_Comm xycomm,
            const MPI_Comm zcomm,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            std::vector<idx_t>& mapX, 
            std::vector<idx_t>& mapY
            );
    void setup_3dsddmm_bcast(
            cooMat& Cloc,
            const idx_t f,
            const int c,
            denseMatrix& Aloc,
            denseMatrix& Bloc, 
            std::vector<int>& rpvec, 
            std::vector<int>& cpvec,
            const MPI_Comm xycomm,
            const MPI_Comm zcomm,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            std::vector<idx_t>& mapA, 
            std::vector<idx_t>& mapB
            );
}
