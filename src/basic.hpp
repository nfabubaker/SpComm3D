#ifndef _BASIC_H
#define _BASIC_H

#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <cassert>
#include <unordered_map>

namespace SpKernels {

#define idx_t uint32_t
#define real_t double
#define idx_large_t uint64_t

typedef struct _parallelTiming{
    idx_t comm1Time, comm2Time, commcpyTime, compTime, totalTime;
} parallelTiming;

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
        inline real_t& at(idx_t x, idx_t y){return data.at(x*n + y);}
        void printMatrix(idx_t count =0 ){
            idx_t nrows = m;
            if(count != 0) nrows = count / n;
            for (size_t i = 0; i < nrows; ++i) 
                for (size_t j = 0; j < n; ++j) 
                    std::cout << i << " " << j << " " << at(i,j) << std::endl; 
        }
    } denseMatrix;
}

#endif
