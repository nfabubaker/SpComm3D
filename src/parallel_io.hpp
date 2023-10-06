#pragma once
#include <mpi.h>
#include <iostream>

#include <fstream>
#include <sstream>
#include <string>
#include "Mesh3D.hpp"



#include "basic.hpp"
#include "mpio.h"
using namespace std;
namespace SpKernels{
    void read_mm_MPI_IO(Mesh3D& mesh, std::string filename, idx_t gnnz, coo_mtx &Cloc, MPI_Comm comm){
       int myrank, size;
       MPI_Comm_rank(comm, &myrank);
       MPI_File file;
       MPI_Status status;
       idx_t lnnz;

       MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
       /* everyone (on the front mesh) reads their own nnz data */
       if(mesh.getZCoord(myrank) == 0){
           
           MPI_File_read_at(file, myrank, &lnnz, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);  

       }
       

    }

}


