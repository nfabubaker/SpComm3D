#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_set>
#include "basic.hpp"


namespace SpKernels{

    template<typename T, typename IdxT>
        class SparseMatrix
        {
            public:
                std::vector<IdxT> ltgR, ltgC;
                std::vector<T> ownedVals;
                std::unordered_map<IdxT, IdxT> gtlR, gtlC;
                virtual void localizeIndices(){};
                virtual void delocalizeIndices(){};
                virtual void ReMapIndices(std::vector<IdxT> mapR, std::vector<IdxT>mapC){};
                std::string mtxName;
                int rank, xyrank, zrank;
                IdxT nrows, ncols, ownedNnz, gnrows, gncols;
                idx_large_t nnz, gnnz;
            private:
                /* data */

        };

    template<typename T, typename IdxT>
        class COOMatrix : public SparseMatrix<T, IdxT>
    {
        public:
            std::vector<IdxT> ii, jj;
            std::vector<T> vv;
            void addEntry(IdxT row, IdxT col, T val){
                ii.push_back(row);
                jj.push_back(col);
                vv.push_back(val);
            }

            void printMatrix(int count = 0){
                IdxT nr = (count == 0 ? this->nnz : count); 
                for (size_t i = 0; i < nr; ++i) 
                    std::cout << ii[i] << " " << jj[i] << " " << vv[i] << std::endl; 
            }

            void self_generate_random(IdxT nnz){
                srand(static_cast<unsigned int>(time(nullptr)));
                std::unordered_set<IdxT> usedIndices;

                for (int i = 0; i < nnz; ++i) {
                    int row, col;
                    do {
                        row = rand() % this->gnrows;
                        col = rand() % this->gncols;
                    } while (usedIndices.count(row * this->gncols + col) > 0); // Check for duplicate indices

                    usedIndices.insert(row * this->gncols + col);

                    /*                 T value = static_cast<T>(rand()) / RAND_MAX; // Random value between 0 and 1
                    */
                    T value = 1.0;
                    this->addEntry(row, col, value);
                }
            }

            void localizeIndices(){
                this->ltgR.clear(); this->ltgC.clear(); 
                //this->gtlR.resize(this->grows, -1); this->gtlC.resize(this->gcols, -1);
                this->nrows = 0; this->ncols = 0;
                for(idx_large_t i = 0; i < this->nnz; ++i){
                    IdxT rid = ii.at(i);
                    IdxT cid = jj.at(i);
                    if( this->gtlR.find(rid) == this->gtlR.end()){
                        this->gtlR[rid] = this->nrows++;
                    }
                    if( this->gtlC.find(cid) == this->gtlC.end()){
                        this->gtlC[cid] = this->ncols++;
                    }
                }
                this->ltgR.resize(this->nrows);
                this->ltgC.resize(this->ncols);
                for(auto& it : this->gtlR)
                    this->ltgR[it.second] = it.first;
                for(auto& it : this->gtlC)
                    this->ltgC[it.second] = it.first;


                /* now localize C indices */
                for(size_t i = 0; i < this->nnz; ++i){
                    IdxT row = ii.at(i);
                    ii.at(i) = this->gtlR.at(row);
                    IdxT col = jj.at(i);
                    jj.at(i) = this->gtlC.at(col);
                }

            }
            void ReMapIndices(std::vector<IdxT> mapR, std::vector<IdxT>mapC)
            {
                for(idx_large_t i = 0; i < this->nnz; ++i){
                    ii[i] =  mapR[ii[i]];
                    jj[i] =  mapC[jj[i]];
                }
            }


    };
    template<typename T, typename IdxT>
        class CSRMatrix : public SparseMatrix<T, IdxT>
    {
        public:
            std::vector<IdxT> rowPtr, colIdx;
            std::vector<T> val;

            void build_from_COO(COOMatrix<T, IdxT>& coomtx){
                rowPtr.resize(coomtx.nrows+2);
                colIdx.resize(coomtx.nnz);
                val.resize(coomtx.nnz);
                /* 1-cnt nnz per row category */
                std::fill(rowPtr.begin(), rowPtr.end(),0);
                for(size_t i =0; i < coomtx.nnz; ++i) rowPtr[coomtx.ii[i] +2]++;
                for(size_t i =2; i < coomtx.nrows+2; ++i) rowPtr[i] += rowPtr[i-1];
                for(size_t i =0; i < coomtx.nnz; ++i){
                    colIdx[rowPtr[coomtx.ii+1]] = coomtx.jj[i];
                    val[rowPtr[coomtx.ii+1]++] = coomtx.vv[i];
                }
            }
            void build_from_triplet(std::vector<SpKernels::triplet> &elms){
            }
    };
    using SpMat = SparseMatrix<real_t, idx_t>;
    using cooMat = COOMatrix<real_t, idx_t>;
    using csrMat = CSRMatrix<real_t, idx_t>;
}
