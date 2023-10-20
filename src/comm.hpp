#ifndef _COMM_H
#define _COMM_H
#include "../src/basic.hpp"
#include <mpi.h>
#include <vector>
#define MPI_IDX_T MPI_UNSIGNED_LONG
#define MPI_REAL_T MPI_DOUBLE
namespace SpKernels {

    typedef struct _denseComm{
        MPI_Comm commX, commY;
        int OP; /*  0 = Bcast, 1- Allreduce */
        real_t *bufferptrX, *bufferptrY;
        int outDegreeX, outDegreeY;
        std::vector<idx_t > bcastXcnt, bcastYcnt, bcastXdisp, bcastYdisp;
        std::vector<MPI_Request> rqstsX, rqstsY;
        std::vector<real_t> reduceBuffer;
        triplet *ClocPtr;
        idx_t lnnz;
        void perform_dense_comm(){
            if(this->OP == 0){/* bcast */
                std::vector<MPI_Status> sttsX(outDegreeX), sttsY(outDegreeY);
                for(int i =0; i < outDegreeX; ++i)
                    MPI_Ibcast(bufferptrX + bcastXdisp[i], bcastXcnt[i], MPI_REAL_T, i,commX, &rqstsX[i]);
                for(int i =0; i < outDegreeY; ++i)
                    MPI_Ibcast(bufferptrY + bcastYdisp[i], bcastYcnt[i], MPI_REAL_T, i,commY, &rqstsY[i]);
                MPI_Waitall(outDegreeX, rqstsX.data(), sttsX.data());
                MPI_Waitall(outDegreeY, rqstsY.data(), sttsY.data());
            }
            else if (this->OP == 1){
                for(size_t i = 0; i < lnnz; ++i )
                    reduceBuffer[i]  = ClocPtr[i].val;
                MPI_Allreduce(MPI_IN_PLACE, reduceBuffer.data(), lnnz, MPI_REAL_T, MPI_SUM, commX);
                for(size_t i = 0; i < lnnz; ++i )
                    ClocPtr[i].val = reduceBuffer[i];
            }
        }

    } DenseComm;


    template <typename T>
        class SparseComm
        {
            public:
                /* data */
                MPI_Comm cartcomm;
                MPI_Comm commP, commN;
                int inDegree, outDegree;
                idx_t dataUnitSize;
                std::vector<int> inSet, outSet;
                std::vector<int> sendCount, recvCount;
                std::vector<int> sendDisp, recvDisp;
                std::vector<T> sendBuff, recvBuff;  
                /* csr-like DS to facilitate moving data between send/recv buffers and local storage */ 
                std::vector<T *> sendptr, recvptr;
                enum comm_T {P2P, NEIGHBOR};
                comm_T commT = P2P;
                SparseComm () : SparseComm(1){}
                SparseComm ( idx_t unitSize): dataUnitSize(unitSize){
                    inDegree = 0; outDegree = 0;
                    commP = MPI_COMM_NULL; commN = MPI_COMM_NULL;
                }
                void init(idx_t unitSize, int inDegree, int outDegree, 
                        idx_t totSendCnt, idx_t totRecvCnt,
                        comm_T commt, MPI_Comm commP, MPI_Comm commN){
                    this->dataUnitSize = unitSize;
                    this->inDegree = outDegree;
                    this->outDegree = inDegree;
                    this->inSet.resize(inDegree);
                    this->outSet.resize(outDegree);
                    this->sendBuff.resize(totSendCnt * unitSize); 
                    this->sendptr.resize(totSendCnt); 
                    this->recvBuff.resize(totRecvCnt * unitSize); 
                    this->recvptr.resize(totRecvCnt); 
                    this->sendCount.resize(outDegree);
                    this->recvCount.resize(inDegree);
                    this->sendDisp.resize(outDegree+1);
                    this->recvDisp.resize(inDegree+1);
                    this->commT = commt;
                    this->commP = commP;
                    this->commN = commN;
                }
                virtual ~SparseComm (){};//{if(commT == P2P) MPI_Comm_free(&this->commP); else if(commT == NEIGHBOR) MPI_Comm_free(&this->commN); }
                                         //virtual ~SparseComm (){if(commT == P2P) MPI_Comm_free(&this->commP); else if(commT == NEIGHBOR) MPI_Comm_free(&this->commN); }
        std::vector<T *> get_sendptr(){return this->sendptr;}
        void copy_to_sendbuff(){
            /* copy to sendBuff */
            size_t idx = 0;
            size_t totSendCnt = std::accumulate(sendCount.begin(), sendCount.end(), 0);
            assert(totSendCnt/dataUnitSize == sendptr.size());
            for (size_t i = 0; i < sendptr.size() ; ++i) {
                if(this->dataUnitSize > 1){
                    T *p = sendptr[i];
                    for (size_t j = 0; j < this->dataUnitSize ; ++j)
                        sendBuff[idx++] = p[j];
                }
                else sendBuff[idx++] = *sendptr[i];
            }
        }
        void perform_sparse_comm(bool copyflag = true){
            if(copyflag) copy_to_sendbuff();
            /* TODO implement more efficient Irecv .. etc */
            if(commT == P2P){ 
                if(commP == MPI_COMM_NULL) goto ERR_EXIT;
                int i,j;
                MPI_Request *rqsts = new MPI_Request[inDegree];
                for(i =0; i < inDegree; ++i)
                    MPI_Irecv(recvBuff.data() + recvDisp[i],
                            recvCount[i], mpi_get_type(), inSet[i] ,
                            77, commP, &rqsts[i]);
                for(i =0; i < outDegree; ++i) 
                    MPI_Send(sendBuff.data() + sendDisp[i],
                            sendCount[i], mpi_get_type(), outSet[i] ,
                            77, commP);
                MPI_Waitall(inDegree, rqsts, MPI_STATUSES_IGNORE);
                delete [] rqsts;
            }
            else if (commT == NEIGHBOR){
                if(commN == MPI_COMM_NULL) goto ERR_EXIT;
                MPI_Neighbor_alltoallv(sendBuff.data(), 
                        sendCount.data(), sendDisp.data(), 
                        mpi_get_type(), recvBuff.data(),
                        recvCount.data(), recvDisp.data(),
                        mpi_get_type(), commN);
            }
            if(copyflag) copy_from_recvbuff();
            return;

ERR_EXIT:
            fprintf(stderr, "error: commP/N is NULL\n");
            exit(EXIT_FAILURE); 
            MPI_Finalize();

        }
        void copy_from_recvbuff(){
            /* copy from recvBuff */
            size_t idx = 0;
            for (size_t i = 0; i < recvptr.size() ; ++i) {
                if(this->dataUnitSize > 1){ 
                    T *p = recvptr[i];
                    for (size_t i = 0; i < this->dataUnitSize ; ++i)
                        *p++ = recvBuff[idx++];
                }
                else *recvptr[i] = recvBuff[idx++];
            }
        }
        void SUM_from_recvbuff(){
            /* copy from recvBuff */
            size_t idx = 0;
            for (size_t i = 0; i < recvptr.size() ; ++i) {
                if(this->dataUnitSize > 1){ 
                    T *p = recvptr[i];
                    for (size_t i = 0; i < this->dataUnitSize ; ++i)
                        *p++ = recvBuff[idx++];
                }
                else *recvptr[i] += recvBuff[idx++];
            }
        }


        [[nodiscard]] constexpr MPI_Datatype mpi_get_type() noexcept{
            MPI_Datatype mpiT = MPI_DATATYPE_NULL;
            if constexpr (std::is_same_v<T, real_t>){ mpiT = MPI_REAL_T; }
            else if constexpr (std::is_same_v<T, idx_t>){ mpiT = MPI_IDX_T; }
            return mpiT; 
        }
            private:
        };

    /* 3d comm setup 
     * 
     * */
    void setup_3dsddmm(
            coo_mtx& Cloc,
            const idx_t f,
            const int c ,
            const MPI_Comm xycomm,
            const MPI_Comm zcomm,
            denseMatrix& Aloc,
            denseMatrix& Bloc,
            std::vector<int>& rpvec, 
            std::vector<int>& cpvec,
            SparseComm<real_t>& comm_expand,
            SparseComm<real_t>& comm_reduce
            );
    void setup_3dsddmm_bcast(
            coo_mtx& Cloc,
            const idx_t f,
            const int c,
            denseMatrix& Aloc,
            denseMatrix& Bloc, 
            std::vector<int>& rpvec, 
            std::vector<int>& cpvec,
            const MPI_Comm xycomm,
            const MPI_Comm zcomm,
            DenseComm& comm_pre,
            DenseComm& comm_post
            );
}
#endif
