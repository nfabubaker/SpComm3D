#include "comm.hpp"
#include <chrono>
#include "distributed_comp.hpp"
#include "core_ops.hpp"
#include "comm_stats.hpp"




using namespace std;

namespace SpKernels{
void dist_sddmm_spcomm(
            denseMatrix& A,
            denseMatrix& B,
            coo_mtx& S,
            SparseComm<real_t>& comm_pre,
            SparseComm<real_t>& comm_post,
            coo_mtx& C){

            parallelTiming pt; 
            auto start = chrono::high_resolution_clock::now();
            comm_pre.perform_sparse_comm();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            if(C.rank == 0){
                std::cout << "Aloc before sddmm (sp):" << std::endl;
                //C.printOwnedMatrix(10);
                A.printMatrix(10);
            }
            start = chrono::high_resolution_clock::now();
            sddmm(A,B,S,C);
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            if(C.rank == 0){
                std::cout << "Cloc after sddmm:" << std::endl;
                C.printOwnedMatrix(10);
            }
            start = chrono::high_resolution_clock::now();
            comm_post.perform_sparse_comm(true,false);
            comm_post.SUM_from_recvbuff();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            if(C.rank == 0){
                std::cout << "Cloc after sparse reduce:" << std::endl;
                C.printOwnedMatrix(10);
            }
            /*         MPI_Barrier(MPI_COMM_WORLD);
             *         if(rank == 0){
             *             std::cout << "Cloc after reduce:" << std::endl;
             *             Cloc.printOwnedMatrix(10);
             *         }
             */
            MPI_Barrier(MPI_COMM_WORLD);
            
            print_comm_stats_sparse(C.mtxName, comm_pre, A.n, pt,0,0,0, MPI_COMM_WORLD);
            print_comm_stats_sparse(C.mtxName, comm_post, A.n, pt,0,0,0, MPI_COMM_WORLD);


}
    void dist_sddmm_dcomm(
            denseMatrix& A,
            denseMatrix& B,
            coo_mtx& S,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            coo_mtx& C){
            parallelTiming pt; 
            /* comm_pre */
            auto start = chrono::high_resolution_clock::now();
            comm_pre.perform_dense_comm();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            if(C.rank == 0){
                std::cout << "Aloc before sddmm (dense):" << std::endl;
                //C.printOwnedMatrix(10);
                A.printMatrix(10);
            }
            if(C.rank == 0){
                std::cout << "Cloc before sddmm:" << std::endl;
                C.printOwnedMatrix(10);
            }
            start = chrono::high_resolution_clock::now();
            sddmm(A, B,S, C);
            if(C.rank == 0){
                std::cout << "Cloc after sddmm:" << std::endl;
                C.printOwnedMatrix(10);
            }
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            start = chrono::high_resolution_clock::now();
            comm_post.perform_dense_comm();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            if(C.rank == 0){
                std::cout << "Cloc after dense comm:" << std::endl;
                C.printOwnedMatrix(10);
            }
/*             for(size_t i = 0; i < Cloc.ownedNnz; ++i){
 *                 idx_t lidx = Cloc.otl[i];
 *                 Cloc.elms[lidx].val *= Cloc.owned[i];
 *             }
 */
            /*         MPI_Barrier(MPI_COMM_WORLD);
             *         if(rank == 0){
             *             std::cout << "Cloc after reduce:" << std::endl;
             *             Cloc.printOwnedMatrix(10);
             *         }
             */
            MPI_Barrier(MPI_COMM_WORLD);
            print_comm_stats_dense(C.mtxName, comm_pre, A.n,pt, 0,0,0, MPI_COMM_WORLD); 
            print_comm_stats_dense(C.mtxName, comm_post, A.n, pt, 0,0,0, MPI_COMM_WORLD); 

    }
}
