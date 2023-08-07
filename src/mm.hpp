#pragma once

#include "basic.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
namespace SpKernels{
    class mm
    {
        public:
            mm (string FN){
                _FN = FN;
            }
            coo_mtx& read_mm(std::string FN){
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
                _mtx.elms.resize(_nnz);
                _mtx.gnnz = _mtx.lnnz = _nnz;
                _mtx.gcols = _mtx.lcols = _ncols;
                _mtx.grows = _mtx.lrows= _nrows;
                vector<SpKernels::triplet>& _M = _mtx.elms;
                idx_t i;
                idx_t hnnz = (mm_symm)? _nnz/2:_nnz;
                real_t tt;
                if(!mm_pattern)
                    for (i = 0; i < hnnz; ++i){ 
                        if(!mm_complex)
                            mtxfile >> _M[i].row >> _M[i].col >> _M[i].val; 
                        else
                            mtxfile >> _M[i].row >> _M[i].col >> _M[i].val >> tt; 
                        _M[i].row--;
                        _M[i].col--;
                        if(mm_symm){
                            _M[i+(_nnz/2)].row = _M[i].col;
                            _M[i+(_nnz/2)].col = _M[i].row;
                            _M[i+(_nnz/2)].val= _M[i].val;
                        }
                    }

                else{
                    for (i = 0; i < hnnz; ++i){ 
                        mtxfile >> _M[i].row >> _M[i].col;
                        _M[i].row--;
                        _M[i].col--;
                        _M[i].val= 1.0;
                        if(mm_symm){
                            _M[i+(_nnz/2)].row = _M[i].col;
                            _M[i+(_nnz/2)].col = _M[i].row;
                            _M[i+(_nnz/2)].val= 1.0;
                        }
                    }
                }
                mtxfile.close();
            return _mtx;
            }
            virtual ~mm (){};
        private:
            /*  private data */
            SpKernels::coo_mtx _mtx;
            string _FN;
    };
}

