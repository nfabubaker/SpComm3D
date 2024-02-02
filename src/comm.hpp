#ifndef _COMM_H
#define _COMM_H
#include "../src/basic.hpp"
#include "SparseMatrix.hpp"
#include <memory>
#include <mpi.h>
#include <vector>
#define MPI_IDX_T MPI_UNSIGNED
#define MPI_IDX_LARGE_T MPI_UNSIGNED_LONG
#define MPI_REAL_T MPI_DOUBLE
namespace SpKernels {
    template <typename T>
        class Msg
        {
            public:
                T* buff;
                int srcDst;
                idx_t size;
        };
    template <typename T>
        class SpComM
        {
            public:
                std::vector<Msg<T>> sendMsgs;
                std::vector<Msg<T>> recvMsgs;
        };

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
                int TAG;
                T *sendBuffPtr, *recvBuffPtr;  
                std::vector<T> sendBuff, recvBuff;  
                std::vector<MPI_Datatype> sendDataTypes;
                std::vector<MPI_Datatype> recvDataTypes;
                std::vector<std::vector<idx_t>> sDTdisps;
                std::vector<std::vector<idx_t>> sDTblen;

                /* csr-like DS to facilitate moving data between send/recv buffers and local storage */ 
                std::vector<T *> sendptr, recvptr;
                enum comm_T {P2P, NEIGHBOR};
                comm_T commT = P2P;
                MPI_Request *rqsts;
                SparseComm () : SparseComm(1){}
                SparseComm ( idx_t unitSize): dataUnitSize(unitSize){
                    inDegree = 0; outDegree = 0;
                    commP = MPI_COMM_NULL; commN = MPI_COMM_NULL;
                    rqsts = NULL;
                    TAG = 77;
                }
                void init(idx_t unitSize, int inDegree, int outDegree, 
                        idx_t totSendCnt, idx_t totRecvCnt,
                        comm_T commt, MPI_Comm commP, MPI_Comm commN, bool allocSendBuff, bool allocRecvBuff){
                    this->dataUnitSize = unitSize;
                    this->inDegree = outDegree;
                    this->outDegree = inDegree;
                    this->inSet.resize(inDegree);
                    this->outSet.resize(outDegree);
                    if(allocSendBuff){
                        this->sendBuff.resize(totSendCnt * unitSize); 
                        this->sendBuffPtr = this->sendBuff.data();
                        this->sendptr.resize(totSendCnt); 
                    }
                    if(allocRecvBuff){
                        this->recvBuff.resize(totRecvCnt * unitSize); 
                        this->recvBuffPtr = this->recvBuff.data();
                        this->recvptr.resize(totRecvCnt); 
                    }
                    this->sendCount.resize(outDegree);
                    this->recvCount.resize(inDegree);
                    this->sendDisp.resize(outDegree+1);
                    this->recvDisp.resize(inDegree+1);
                    this->commT = commt;
                    this->commP = commP;
                    this->commN = commN;
                    this->rqsts = NULL;
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
                        sendBuffPtr[idx++] = p[j];
                }
                else sendBuffPtr[idx++] = *sendptr[i];
            }
        }

        void issue_Irecvs(){

            int i;
            rqsts = new MPI_Request[inDegree];
            for(i =0; i < inDegree; ++i)
                MPI_Irecv(recvBuffPtr + recvDisp[i],
                        recvCount[i], mpi_get_type(), inSet[i] ,
                        TAG, commP, &rqsts[i]);
        }
        void issue_Sends(bool with_DT = false){
            int i; 
            if(with_DT){
                for(i =0; i < outDegree; ++i) 
                    MPI_Send(sendBuffPtr, 1, sendDataTypes[i], outSet[i], TAG, commP);
            }
            else{ 
            for(i =0; i < outDegree; ++i) 
                MPI_Send(sendBuffPtr + sendDisp[i],
                        sendCount[i], mpi_get_type(), outSet[i] ,
                        TAG, commP);
            }
        }

        void issue_Waitall(){
            if(rqsts == NULL){ 
                fprintf(stderr, "error in Waitall: no requests allocated\n");
                exit(EXIT_FAILURE); 
                MPI_Finalize();
            }
            else{
                MPI_Waitall(inDegree, rqsts, MPI_STATUSES_IGNORE);
                delete [] rqsts;
            }

        }
        void perform_sparse_comm(bool sendcopyflag = true, bool recvcopyflag=true){
            /* TODO implement more efficient Irecv .. etc */
            if(commT == P2P){ 
                if(commP == MPI_COMM_NULL) goto ERR_EXIT;
                this->issue_Irecvs();
                if(sendcopyflag) copy_to_sendbuff();
                this->issue_Sends();
                this->issue_Waitall();
            }
            else if (commT == NEIGHBOR){
                if(commN == MPI_COMM_NULL) goto ERR_EXIT;
                MPI_Neighbor_alltoallv(sendBuffPtr, 
                        sendCount.data(), sendDisp.data(), 
                        mpi_get_type(), recvBuffPtr,
                        recvCount.data(), recvDisp.data(),
                        mpi_get_type(), commN);
            }
            if(recvcopyflag) copy_from_recvbuff();
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
                    for (size_t j = 0; j < this->dataUnitSize ; ++j)
                        p[j] = recvBuffPtr[idx++];
                }
                else *recvptr[i] = recvBuffPtr[idx++];
            }
        }
        void SUM_from_recvbuff(){
            /* copy from recvBuff */
            size_t idx = 0;
            for (size_t i = 0; i < recvptr.size() ; ++i) {
                if(this->dataUnitSize > 1){ 
                    T *p = recvptr[i];
                    for (size_t j = 0; j < this->dataUnitSize ; ++j)
                        p[j] += recvBuffPtr[idx++];
                }
                else *recvptr[i] += recvBuffPtr[idx++];
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
}
#endif
