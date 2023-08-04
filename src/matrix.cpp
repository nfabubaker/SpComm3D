#include <complex>
#include <cstddef>
#include <stdlib.h>

template<typename T>
class csrMtx {
    public:
        size_t nr, nc, nnz;
    private:
        size_t *rptr;
        size_t *colidx;
        T *vals;
}


