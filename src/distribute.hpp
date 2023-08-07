

#include "basic.hpp"
#include "Mesh3D.hpp"
#include <algorithm>


namespace SpKernels {

    void distribute3D(denseMatrix& A,
            denseMatrix& B,
            coo_mtx& C,
            Mesh3D& mesh3d,
            MPI_Comm comm)
    {
        int rank, size;
        MPI_Comm_size(comm, &size);
        std::vector<int> rpvec(C.grows, -1), cpvec(C.gcols, -1); 
        std::vector<idx_t> row_space(C.grows), col_space(C.gcols);
        for(size_t i = 0; i < C.grows; ++i) row_space[i] = i;
        for(size_t i = 0; i < C.gcols; ++i) col_space[i] = i;
        std::random_shuffle(row_space.begin(), row_space.end());
        std::random_shuffle(col_space.begin(), col_space.end());
        /* assign Rows/Cols with RR */
        for(size_t i = 0, p=0; i < C.grows; ++i)
            rpvec[row_space[i]] = (p++) % mesh3d.getX(); 
        for(size_t i = 0, p=0; i < C.gcols; ++i) 
            cpvec[col_space[i]] = (p++) % mesh3d.getY(); 

        std::vector<std::vector<idx_t>> mesh2DCnt(mesh3d.getX(),
                std::vector<idx_t> (mesh3d.getY(), 0));
        /* now rows/cols are divided, now distribute nonzeros */
        /* first: count nnz per 2D block */
        for(auto& t : C.elms){
            mesh2DCnt[rpvec[t.row]][cpvec[t.col]]++; 
        }
        /* tell processors what you'll send them */
        /* now create comm groups ?  */

        
    }
}
