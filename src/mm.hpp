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
                bool use_pfile = false, write_ovec=false, mm_pattern=false, mm_symm=false, mm_complex=false;
                idx_t _nrows, _ncols, _nnz;
                string firstline;
                ifstream mtxfile(FN.c_str());
                getline(mtxfile, firstline);
                if(firstline.find("pattern") != string::npos) mm_pattern = true;
                if(firstline.find("symmetric") != string::npos) mm_symm = true;
                if(firstline.find("complex") != string::npos) mm_complex = true;
                while(mtxfile.peek() == '%') mtxfile.ignore(2048, '\n');
                mtxfile >> _nrows >> _ncols >> _nnz;
                //printf("nrows=%d ncols=%d nnz=%d nparts=%d mm_pattern is %d mm_symm is %d\n", nrows, ncols, nnz, nparts, mm_pattern, mm_symm);
                if(mm_symm) _nnz*=2;
                _mtx.ii.resize(_nnz);
                _mtx.jj.resize(_nnz);
                _mtx.vv.resize(_nnz);
                _mtx.gnnz = _mtx.nnz = _nnz;
                _mtx.gncols = _mtx.ncols = _ncols;
                _mtx.gnrows = _mtx.nrows= _nrows;
                vector<idx_t>& _ii = _mtx.ii;
                vector<idx_t>& _jj = _mtx.jj;
                vector<real_t>& _vv = _mtx.vv;
                idx_large_t i;
                idx_large_t hnnz = (mm_symm)? _nnz/2:_nnz;
                real_t tt;
                if(!mm_pattern)
                    for (i = 0; i < hnnz; ++i){ 
                        if(!mm_complex)
                            mtxfile >> _ii[i] >> _jj[i] >> _vv[i]; 
                        else
                            mtxfile >> _ii[i] >> _jj[i] >> _vv[i] >> tt; 
                        _ii[i]--;
                        _jj[i]--;
                        if(mm_symm){
                            _ii[i+(_nnz/2)] = _jj[i];
                            _jj[i+(_nnz/2)] = _ii[i];
                            _vv[i+(_nnz/2)]= _vv[i];
                        }
                    }

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
