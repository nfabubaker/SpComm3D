#pragma once

#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <cassert>


#define idx_t uint64_t
#define real_t double

typedef struct _parallelTiming{
    idx_t comm1Time, comm2Time, compTime;
} parallelTiming;

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
        void printMatrix(){
            for (size_t i = 0; i < m; ++i) 
                for (size_t j = 0; j < n; ++j) 
                    std::cout << i << " " << j << " " << at(i,j) << std::endl; 
        }
    } denseMatrix;

    typedef struct _coo_mtx{
        std::string mtxName;
        idx_t lrows, lcols, lnnz, ownedNnz, grows, gcols, gnnz;
        std::vector<idx_t> ltgR, gtlR, ltgC, gtlC, lto, otl;
        std::vector<real_t> owned;
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

/*                 real_t value = static_cast<real_t>(rand()) / RAND_MAX; // Random value between 0 and 1
 */
                real_t value = 1.0;
                this->addEntry(row, col, value);
            }
        }
    } coo_mtx;

}

