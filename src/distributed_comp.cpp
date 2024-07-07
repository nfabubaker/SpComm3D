#include "comm.hpp"
#include <chrono>
#include "denseComm.hpp"
#include "distributed_comp.hpp"
#include "core_ops.hpp"
#include "comm_stats.hpp"
#include <array>




using namespace std;

using namespace SpKernels;
using namespace DComm;


void dist_spmm_spcomm(

            denseMatrix& X,
            cooMat& A,
            denseMatrix& Y,
            SparseComm<real_t>& comm_pre,
            SparseComm<real_t>& comm_post,
            MPI_Comm comm){
            int Xd,Yd,Zd;
            std::array<int, 3> dims, t1, t2;
            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
            Xd = dims[0]; Yd=dims[1]; Zd=dims[2];

            parallelTiming pt = {0}; 
            auto totStart = chrono::high_resolution_clock::now();
            auto start = chrono::high_resolution_clock::now();
            comm_pre.issue_Irecvs();
            auto start2 = chrono::high_resolution_clock::now();
            comm_pre.copy_to_sendbuff();
            auto stop2 = chrono::high_resolution_clock::now();
            pt.commcpyTime = chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
            comm_pre.issue_Sends();
            comm_pre.issue_Waitall();
            start2 = chrono::high_resolution_clock::now();
            comm_pre.copy_from_recvbuff();
            stop2 = chrono::high_resolution_clock::now();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            pt.commcpyTime+= chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
/*             if(C.rank == 0){
 *                 std::cout << "Aloc before sddmm (sp):" << std::endl;
 *                 //C.printOwnedMatrix(10);
 *                 A.printMatrix(10);
 *             }
 */
            start = chrono::high_resolution_clock::now();
            spmm(X,A,Y);
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            start = chrono::high_resolution_clock::now();
            comm_post.issue_Irecvs();
            start2 = chrono::high_resolution_clock::now();
            comm_pre.copy_from_recvbuff();
            stop2 = chrono::high_resolution_clock::now();
            pt.commcpyTime += chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
            comm_post.issue_Sends();
            comm_post.issue_Waitall();
            start2 = chrono::high_resolution_clock::now();
            comm_post.SUM_from_recvbuff();
            stop2 = chrono::high_resolution_clock::now();
            pt.commcpyTime += chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            auto totStop = chrono::high_resolution_clock::now(); 
            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
            MPI_Barrier(MPI_COMM_WORLD);
            
            print_comm_stats_sparse(A.mtxName, "sparse spmm PRE", comm_pre, X.n, pt,Xd,Yd,Zd, MPI_COMM_WORLD);
            print_comm_stats_sparse(A.mtxName, "sparse spmm POST",comm_post, X.n, pt,Xd,Yd,Zd, MPI_COMM_WORLD);
}

//void dist_spmm_spcomm(
//            denseMatrix& X,
//            cooMat& A,
//            denseMatrix& Y,
//            SparseComm<real_t>& comm_pre,
//            SparseComm<real_t>& comm_post,
//            MPI_Comm comm){
//            int dX,dY,dZ;
//            std::array<int, 3> dims, t1, t2;
//            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
//            dX = dims[0]; dY=dims[1]; dZ=dims[2];
//
//            parallelTiming pt = {0}; 
//            auto totStart = chrono::high_resolution_clock::now();
//            auto start = chrono::high_resolution_clock::now();
//            comm_pre.perform_sparse_comm(true, false);
//            auto stop = chrono::high_resolution_clock::now();
//            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
//            start = chrono::high_resolution_clock::now();
//            comm_pre.copy_from_recvbuff();
//            stop = chrono::high_resolution_clock::now();
//            pt.commcpyTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
///*             if(C.rank == 0){
// *                 std::cout << "Aloc before sddmm (sp):" << std::endl;
// *                 //C.printOwnedMatrix(10);
// *                 A.printMatrix(10);
// *             }
// */
//            start = chrono::high_resolution_clock::now();
//            spmm(X, A, Y);
//            stop = chrono::high_resolution_clock::now();
//            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
///*             if(C.rank == 0){
// *                 std::cout << "Cloc after sddmm:" << std::endl;
// *                 C.printOwnedMatrix(10);
// *             }
// */
//            start = chrono::high_resolution_clock::now();
//            comm_post.perform_sparse_comm(true,false);
//            stop = chrono::high_resolution_clock::now();
//            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
//            start = chrono::high_resolution_clock::now();
//            comm_post.SUM_from_recvbuff();
//            stop = chrono::high_resolution_clock::now();
//            pt.commcpyTime += chrono::duration_cast<chrono::milliseconds>(stop-start).count();
//            auto totStop = chrono::high_resolution_clock::now(); 
//            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
///*             if(C.rank == 0){
// *                 std::cout << "Cloc after sparse reduce:" << std::endl;
// *                 C.printOwnedMatrix(10);
// *             }
// */
//            /*         MPI_Barrier(MPI_COMM_WORLD);
//             *         if(rank == 0){
//             *             std::cout << "Cloc after reduce:" << std::endl;
//             *             Cloc.printOwnedMatrix(10);
//             *         }
//             */
//
//            
//            print_comm_stats_sparse(A.mtxName, "sparse spmm PRE", comm_pre, X.n, pt,dX,dY,dZ, MPI_COMM_WORLD);
//            print_comm_stats_sparse(A.mtxName,"sparse spmm POST",comm_post, X.n, pt,dX,dY,dZ, MPI_COMM_WORLD);
//}
void dist_sddmm_spcomm(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            SparseComm<real_t>& comm_pre,
            DComm::DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm){
            int X,Y,Z;
            std::array<int, 3> dims, t1, t2;
            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
            X = dims[0]; Y=dims[1]; Z=dims[2];

            parallelTiming pt = {0}; 
            auto totStart = chrono::high_resolution_clock::now();
            auto start = chrono::high_resolution_clock::now();
            comm_pre.issue_Irecvs();
            auto start2 = chrono::high_resolution_clock::now();
            comm_pre.copy_to_sendbuff();
            auto stop2 = chrono::high_resolution_clock::now();
            pt.commcpyTime = chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
            comm_pre.issue_Sends();
            comm_pre.issue_Waitall();
            start2 = chrono::high_resolution_clock::now();
            comm_pre.copy_from_recvbuff();
            stop2 = chrono::high_resolution_clock::now();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            pt.commcpyTime+= chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
/*             if(C.rank == 0){
 *                 std::cout << "Aloc before sddmm (sp):" << std::endl;
 *                 //C.printOwnedMatrix(10);
 *                 A.printMatrix(10);
 *             }
 */
            start = chrono::high_resolution_clock::now();
            sddmm(A,B,S,C);
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            start = chrono::high_resolution_clock::now();
           
            comm_post.perform_dense_comm();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            auto totStop = chrono::high_resolution_clock::now(); 
            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
            MPI_Barrier(MPI_COMM_WORLD);
            
            print_comm_stats_sparse(C.mtxName, "sparse sddmm PRE", comm_pre, A.n, pt,X,Y,Z, MPI_COMM_WORLD);
            print_comm_stats_dense(C.mtxName, "sparse sddmm POST",comm_post, A.n, pt,X,Y,Z, MPI_COMM_WORLD);


}
void dist_sddmm_spcomm3(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            SparseComm<real_t>& comm_preA,
            SparseComm<real_t>& comm_preB,
            DComm::DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm){
            int X,Y,Z;
            std::array<int, 3> dims, t1, t2;
            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
            X = dims[0]; Y=dims[1]; Z=dims[2];

            parallelTiming pt = {0}; 
            auto totStart = chrono::high_resolution_clock::now();
            auto start = chrono::high_resolution_clock::now();
            comm_preA.issue_Irecvs();
            comm_preB.issue_Irecvs();
            comm_preA.issue_Sends(true);
            comm_preB.issue_Sends(true);
            comm_preA.issue_Waitall();
            comm_preB.issue_Waitall();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            start = chrono::high_resolution_clock::now();
            sddmm(A,B,S,C);
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            start = chrono::high_resolution_clock::now();
           
            comm_post.perform_dense_comm();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            auto totStop = chrono::high_resolution_clock::now(); 
            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
            MPI_Barrier(MPI_COMM_WORLD);
            
            print_comm_stats_sparse2(C.mtxName, "sparseNB sddmm PRE", comm_preA, comm_preB, A.n, pt,X,Y,Z, MPI_COMM_WORLD);
            print_comm_stats_dense(C.mtxName, "sparseNB sddmm POST",comm_post, A.n, pt,X,Y,Z, MPI_COMM_WORLD);


}
void dist_sddmm_spcomm2(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            SparseComm<real_t>& comm_preA,
            SparseComm<real_t>& comm_preB,
            DComm::DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm){
            int X,Y,Z;
            std::array<int, 3> dims, t1, t2;
            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
            X = dims[0]; Y=dims[1]; Z=dims[2];

            parallelTiming pt = {0}; 
            auto totStart = chrono::high_resolution_clock::now();
            auto start = chrono::high_resolution_clock::now();
            comm_preA.issue_Irecvs();
            comm_preB.issue_Irecvs();
            auto start2 = chrono::high_resolution_clock::now();
            comm_preA.copy_to_sendbuff();
            comm_preB.copy_to_sendbuff();
            auto stop2 = chrono::high_resolution_clock::now();
            pt.commcpyTime = chrono::duration_cast<chrono::milliseconds>(stop2-start2).count();
            comm_preA.issue_Sends();
            comm_preB.issue_Sends();
            comm_preA.issue_Waitall();
            comm_preB.issue_Waitall();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            start = chrono::high_resolution_clock::now();
            sddmm(A,B,S,C);
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            start = chrono::high_resolution_clock::now();
           
            comm_post.perform_dense_comm();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count();
            auto totStop = chrono::high_resolution_clock::now(); 
            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
            MPI_Barrier(MPI_COMM_WORLD);
            
            print_comm_stats_sparse2(C.mtxName, "sparseNRB sddmm PRE", comm_preA, comm_preB, A.n, pt,X,Y,Z, MPI_COMM_WORLD);
            print_comm_stats_dense(C.mtxName, "sparseNRB sddmm POST",comm_post, A.n, pt,X,Y,Z, MPI_COMM_WORLD);


}
    void dist_spmm_dcomm(
            denseMatrix& X,
            denseMatrix& Y,
            cooMat& A,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            MPI_Comm comm){
            int dX,dY,dZ;
            std::array<int, 3> dims, t1, t2;
            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
            dX = dims[0]; dY=dims[1]; dZ=dims[2];
            parallelTiming pt = {0}; 
            /* comm_pre */
            auto totStart = chrono::high_resolution_clock::now();
            auto start = chrono::high_resolution_clock::now();
            comm_pre.perform_dense_comm();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
/*             if(C.rank == 0){
 *                 std::cout << "Aloc before sddmm (dense):" << std::endl;
 *                 //C.printOwnedMatrix(10);
 *                 A.printMatrix(10);
 *             }
 *             if(C.rank == 0){
 *                 std::cout << "Cloc before sddmm:" << std::endl;
 *                 C.printOwnedMatrix(10);
 *             }
 */
            start = chrono::high_resolution_clock::now();
            spmm(X, A, Y);
/*             if(C.rank == 0){
 *                 std::cout << "Cloc after sddmm:" << std::endl;
 *                 C.printOwnedMatrix(10);
 *             }
 */
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            start = chrono::high_resolution_clock::now();
            comm_post.perform_dense_comm();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            auto totStop = chrono::high_resolution_clock::now(); 
            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
/*             if(C.rank == 0){
 *                 std::cout << "Cloc after dense comm:" << std::endl;
 *                 C.printOwnedMatrix(10);
 *             }
 */
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
            MPI_Barrier(comm);
            print_comm_stats_dense(A.mtxName,"dense spmm PRE", comm_pre, X.n,pt, dX,dY,dZ, comm); 
            print_comm_stats_dense(A.mtxName, "dense spmm POST", comm_post, X.n, pt, dX,dY,dZ, comm); 

    }
    void dist_sddmm_dcomm(
            denseMatrix& A,
            denseMatrix& B,
            cooMat& S,
            DenseComm& comm_pre,
            DenseComm& comm_post,
            cooMat& C,
            MPI_Comm comm){
            int X,Y,Z;
            std::array<int, 3> dims, t1, t2;
            MPI_Cart_get(comm, 3, dims.data(), t1.data(), t2.data());
            X = dims[0]; Y=dims[1]; Z=dims[2];
            parallelTiming pt = {0}; 
            /* comm_pre */
            auto totStart = chrono::high_resolution_clock::now();
            auto start = chrono::high_resolution_clock::now();
            comm_pre.perform_dense_comm();
            auto stop = chrono::high_resolution_clock::now();
            pt.comm1Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
/*             if(C.rank == 0){
 *                 std::cout << "Aloc before sddmm (dense):" << std::endl;
 *                 //C.printOwnedMatrix(10);
 *                 A.printMatrix(10);
 *             }
 *             if(C.rank == 0){
 *                 std::cout << "Cloc before sddmm:" << std::endl;
 *                 C.printOwnedMatrix(10);
 *             }
 */
            start = chrono::high_resolution_clock::now();
            sddmm(A, B,S, C);
/*             if(C.rank == 0){
 *                 std::cout << "Cloc after sddmm:" << std::endl;
 *                 C.printOwnedMatrix(10);
 *             }
 */
            stop = chrono::high_resolution_clock::now();
            pt.compTime = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            start = chrono::high_resolution_clock::now();
            comm_post.perform_dense_comm();
            stop = chrono::high_resolution_clock::now();
            pt.comm2Time = chrono::duration_cast<chrono::milliseconds>(stop-start).count(); 
            auto totStop = chrono::high_resolution_clock::now(); 
            pt.totalTime = chrono::duration_cast<chrono::milliseconds>(totStop-totStart).count();
/*             if(C.rank == 0){
 *                 std::cout << "Cloc after dense comm:" << std::endl;
 *                 C.printOwnedMatrix(10);
 *             }
 */
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
            MPI_Barrier(comm);
            print_comm_stats_dense(C.mtxName,"dense sddmm PRE", comm_pre, A.n,pt, X,Y,Z, comm); 
            print_comm_stats_dense(C.mtxName, "dense sddmm POST", comm_post, A.n, pt, X,Y,Z, comm); 

    }

