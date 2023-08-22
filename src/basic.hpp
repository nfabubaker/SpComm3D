#pragma once

#include <type_traits>
#include <vector>
#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <unordered_set>



#define idx_t uint32_t
#define real_t double
#define MPI_IDX_T MPI_UNSIGNED
#define MPI_REAL_T MPI_DOUBLE


namespace SpKernels {


    typedef struct _triplet{
        idx_t row;
        idx_t col;
        real_t val;
    } triplet;

    typedef struct _denseMatrix{
        idx_t m, n;
        /* data is an mxn row-major matrix*/
        std::vector<real_t> data;
        std::vector<idx_t> ltg;
        std::vector<idx_t> gtl;
        inline real_t at(idx_t x, idx_t y){return data.at(x*n + y);}
    } denseMatrix;

    typedef struct _coo_mtx{
        idx_t lrows, lcols, lnnz, ownedNnz, grows, gcols, gnnz;
        std::vector<int> owners; /* owner per local nnz */
        std::vector<triplet> elms;
        void addEntry(idx_t row, idx_t col, real_t val){
            triplet entry = {row, col, val};
            this->elms.push_back(entry);
        }

        void printMatrix(){
            for (const triplet& t : elms) 
                std::cout << t.row << " " << t.col << " " << t.val << std::endl; 
        }
        void self_generate_random(idx_t nnz){
            srand(static_cast<unsigned int>(time(nullptr)));
            std::unordered_set<idx_t> usedIndices;

            for (int i = 0; i < nnz; ++i) {
                int row, col;
                do {
                    row = rand() % this->grows;
                    col = rand() % this->gcols;
                } while (usedIndices.count(row * gcols + col) > 0); // Check for duplicate indices

                usedIndices.insert(row * gcols + col);

                real_t value = static_cast<real_t>(rand()) / RAND_MAX; // Random value between 0 and 1
                this->addEntry(row, col, value);
            }
        }
    } coo_mtx;


    template <typename T>
        class SparseComm
        {
            public:
                /* data */
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
                }
                virtual ~SparseComm (){MPI_Comm_free(&this->commP); if(commT == NEIGHBOR) MPI_Comm_free(&this->commN); }
                std::vector<T *> get_sendptr(){return this->sendptr;}
                void copy_to_sendbuff(){
                    /* copy to sendBuff */
                    size_t idx = 0;
                    for (size_t i = 0; i < sendptr.size() ; ++i) {
                        if(this->dataUnitSize > 1){ T *p = sendptr[i]; for (size_t i = 0; i < this->dataUnitSize ; ++i) sendBuff[idx++] = *p++;}
                        else sendBuff[idx++] = *sendptr[i];
                    }
                }
                void perform_sparse_comm(){
                    /* TODO implement more efficient Irecv .. etc */
                    if(commT == P2P){ 
                        int i,j;
                        for(i =0; i < inDegree; ++i) MPI_Send(sendBuff.data() + sendDisp[i], sendCount[i], mpi_get_type(), outSet[i] , 77, commP);
                        for(i =0; i < outDegree; ++i) MPI_Recv(recvBuff.data() + recvDisp[i], recvCount[i], mpi_get_type(), inSet[i] , 88, commP, MPI_STATUS_IGNORE);
                    }
                    else{
                        MPI_Neighbor_alltoallv(sendBuff.data(), sendCount.data(), sendDisp.data(), mpi_get_type(), recvBuff.data(), recvCount.data(), recvDisp.data(), mpi_get_type(), commN); }


                }
                void copy_from_recvbuff(){
                    /* copy from recvBuff */
                    size_t idx = 0;
                    for (size_t i = 0; i < recvptr.size() ; ++i) {
                        if(this->dataUnitSize > 1){ 
                            real_t *p = recvptr[i];
                            for (size_t i = 0; i < this->dataUnitSize ; ++i)
                                *p++ = recvBuff[idx++];
                        }
                        else *recvptr[i] = recvBuff[idx++];
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
    void setup_3dsddmm(coo_mtx& C, const idx_t f, const int c , const MPI_Comm comm, coo_mtx& Cloc, denseMatrix& Aloc, denseMatrix& Bloc, 
            SparseComm<real_t>& comm_expand, SparseComm<real_t>& comm_reduce);
}

