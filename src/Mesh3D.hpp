#pragma once

#include "basic.hpp"
#include "mpi.h"
#include <cstdlib>


namespace SpKernels{
    class Mesh3D
    {
        public:
            Mesh3D (int X, int Y, int Z, MPI_Comm comm=MPI_COMM_WORLD){
                int size, rank;
                MPI_Comm_size(comm, &size);
                MPI_Comm_rank(comm, &rank);
                if(X*Y*Z != size) {
                    fprintf(stderr, "ERROR: mult of dims is not equal"
                           " to size");
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                this->comm = comm;
                this->rank = rank;
                this->size = size;
                this->X = X; this->MX = 1;
                this->Y = Y; this->MY = X;
                this->Z = Z; this->MZ = MY * Y;
            }
            virtual ~Mesh3D ();
            int getX(){return X;}
            int getY(){return Y;}
            int getZ(){return Z;}
            int getMX(){return MX;}
            int getMY(){return MY;}
            int getMZ(){return MZ;}
            int* getAllCoords(int rank){
                static int coords[3];
                coords[2] = rank/MZ;
                int res = rank - (MZ * coords[2]);
                coords[1] = res / MY;
                res -= MY * coords[1];
                coords[0] = res / MX;
                return coords;
            }
            int getRankFromCoords(int x, int y, int z){
                return MX*x + MY*y + MZ*z;
            }

            int getXCoord(int rank){ return getAllCoords(rank)[0];}
            int getYCoord(int rank){ return getAllCoords(rank)[1];}
            int getZCoord(int rank){ return getAllCoords(rank)[2];}
            std::vector<int> neighborsZ(int rank){
                std::vector<int> NZ;
                int *coords = getAllCoords(rank);
                for(size_t i=0; i < Z; ++i) NZ.push_back(getRankFromCoords(coords[0], coords[1], i));
                return NZ;
            }
            std::vector<int> neighborsY(int rank){
                std::vector<int> NY;
                int *coords = getAllCoords(rank);
                for(size_t i=0; i < Y; ++i) NY.push_back(getRankFromCoords(coords[0], i, coords[2]));
                return NY;
            }
            std::vector<int> neighborsX(int rank){
                std::vector<int> NX;
                int *coords = getAllCoords(rank);
                for(size_t i=0; i < X; ++i) NX.push_back(getRankFromCoords(i, coords[1], coords[2]));
                return NX;
            }
            int getRank() {return rank;}
            int getSize() {return size;}
            MPI_Comm getComm() {return comm;}
        private:
            /* data */
            int X,Y,Z;
            int MX, MY, MZ;
            int size, rank;
            MPI_Comm comm;
            
    };

}
