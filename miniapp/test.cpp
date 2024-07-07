#include <mpi.h>
#include <array>
#include <vector>
#include <string>
#include "basic.hpp"
#include "SparseMatrix.hpp"
#include "comm.hpp"
#include "comm_stats.hpp"
#include <getopt.h>
#include <chrono>
#include "denseComm.hpp"
#include "distribute.hpp"
#include "distributed_comp.hpp"
#include "parallel_io.hpp"
#include "comm_setup.hpp"

using namespace std;

template <typename T>
void printfvec(vector<T> &vec) {
    for (auto &v : vec) {
        cout << v << " ";
    }
    printf("\n");
}

template <typename T>
void printfmap(unordered_map<T, T> &map) {
    for (auto &m : map) {
        cout <<  "(" << m.first << "," << m.second << ") ";
    }
    printf("\n");
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD, xycomm, zcomm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    string filename;
    if (argc == 2) {
        filename = argv[1];
    } else {
        if (rank == 0) {
            printf("Usage: %s filename\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    std::string::size_type const p(filename.find_last_of('.'));
    std::string mtxName = filename.substr(0, p);
    mtxName = mtxName.substr(mtxName.find_last_of("/\\") +1);

    std::vector<int> rpvec, cpvec;
    std::array<int, 3> dims = {0,0,0};
    std::array<int,3> zeroArr ={0,0,0};
    std::array<int,3> tdims ={0,0,0};
    MPI_Dims_create(size, 3, dims.data());

    MPI_Comm cartcomm;
    MPI_Cart_create(comm, 3, dims.data(), zeroArr.data(), 0, &cartcomm);   

    int X = dims[0], Y = dims[1], Z = dims[2];

    std::array<int, 3> remaindims = {true, true, false};
    MPI_Cart_sub(cartcomm, remaindims.data(), &xycomm);

    int myxyrank;
    MPI_Comm_rank(xycomm, &myxyrank);  

    remaindims = {false, false, true};
    MPI_Cart_sub(cartcomm, remaindims.data(), &zcomm); 
    cooMat Sloc;
    Sloc.mtxName = mtxName;
    {
        std:: vector<int> rpvec2D, cpvec2D;
        /* distribute C */
        read_bin_parallel_distribute_coo(filename, Sloc, rpvec2D, cpvec2D,
                cartcomm,xycomm, zcomm);
        Sloc.localizeIndices();
        MPI_Comm_rank(zcomm, &Sloc.zrank);
        Sloc.rank = rank;
    }

    // output the matrix info on rank 0
    if (rank == 0) {
        printf("Matrix %s: %d x %d, %ld nnz\n", Sloc.mtxName.c_str(), Sloc.gnrows, Sloc.gncols, Sloc.gnnz);
        printf("local to global row mapping:\n");
        printfvec(Sloc.ltgR);
        printf("local to global col mapping:\n");
        printfvec(Sloc.ltgC);
        printf("global to local row mapping:\n");
        printfmap(Sloc.gtlR);
        printf("global to local col mapping:\n");
        printfmap(Sloc.gtlC);
        printf("local matrix:\n");
        Sloc.printMatrix();
    }
}