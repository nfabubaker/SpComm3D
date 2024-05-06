#ifndef _MM_H
#define _MM_H

#include "basic.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include "SparseMatrix.hpp"

using namespace std;
namespace SpKernels{
    class mm
    {
        public:
            mm (string FN){
                _FN = FN;
            }
            void read_mm(std::string FN, cooMat& _mtx){
                // initialize symmetry info vars to false
                bool use_pfile = false, write_ovec=false, mm_pattern=false, mm_symm=false, mm_complex=false;

                // declare matrix dimension variables
                idx_t _nrows, _ncols, _nnz;
                
                string firstline;
                ifstream mtxfile(FN.c_str());

                // get first line
                getline(mtxfile, firstline);

                // obtain symmetry information and matrix dimension
                if(firstline.find("pattern") != string::npos) mm_pattern = true;
                if(firstline.find("symmetric") != string::npos) mm_symm = true;
                if(firstline.find("complex") != string::npos) mm_complex = true;
                while(mtxfile.peek() == '%') mtxfile.ignore(2048, '\n');
                mtxfile >> _nrows >> _ncols >> _nnz;

                // DEBUG output
                //printf("nrows=%d ncols=%d nnz=%d nparts=%d mm_pattern is %d mm_symm is %d\n", nrows, ncols, nnz, nparts, mm_pattern, mm_symm);
                
                if(mm_symm) _nnz*=2;

                // shape matrix to match input dimension, set corresponding meber vars in return mtx class
                _mtx.ii.resize(_nnz);
                _mtx.jj.resize(_nnz);
                _mtx.vv.resize(_nnz);
                _mtx.gnnz = _mtx.nnz = _nnz;
                _mtx.gncols = _mtx.ncols = _ncols;
                _mtx.gnrows = _mtx.nrows= _nrows;

                // COO coordinate and value vectors
                vector<idx_t>& _ii = _mtx.ii;
                vector<idx_t>& _jj = _mtx.jj;
                vector<real_t>& _vv = _mtx.vv;

                // value index (different from col/row index due to very large matrices having
                // much larger nnz than #rows / #columns
                idx_large_t i;
                idx_large_t hnnz = (mm_symm)? _nnz/2:_nnz;
                real_t tt;

                // if the input matrix has nonzeros that are not 1.0
                if(!mm_pattern)
                    for (i = 0; i < hnnz; ++i){ 
                        if(!mm_complex)
                            mtxfile >> _ii[i] >> _jj[i] >> _vv[i]; 
                        else
                            mtxfile >> _ii[i] >> _jj[i] >> _vv[i] >> tt; 
                        // convert to 0 indexing
                        _ii[i]--;
                        _jj[i]--;

                        // need to set two entries for symmetric matrices
                        if(mm_symm){
                            _ii[i+(_nnz/2)] = _jj[i];
                            _jj[i+(_nnz/2)] = _ii[i];
                            _vv[i+(_nnz/2)]= _vv[i];
                        }
                    }
                // case when all nonzero values are 1
                else{
                    for (i = 0; i < hnnz; ++i){ 
                        mtxfile >> _ii[i] >> _jj[i];
                        _ii[i]--;
                        _jj[i]--;
                        _vv[i]= 1.0;
                        if(mm_symm){
                            _ii[i+(_nnz/2)] = _jj[i];
                            _jj[i+(_nnz/2)] = _ii[i];
                            _vv[i+(_nnz/2)] = 1.0;
                        }
                    }
                }
                mtxfile.close();
            }
            virtual ~mm (){};
        private:
            /*  private data */
            string _FN;
    };
}

#endif
