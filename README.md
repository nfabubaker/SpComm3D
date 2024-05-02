# SpComm3D: A Framework for Enabling Sparse Communication in 3D Sparse Kernels  

### Overview  
SpComm3D is an MPI-based framework for building distributed-memeory 3D sparse kernels for CPUs and GPUs. The main feature of SpComm3D is that it efficiently uses sparsity-aware communication in 3D sparse kernels instead of sparsity-agnostic communication.

Currently, SpComm3D contains sparsity-aware implementations for 3D algorithms of two important ML sparse kernels that occur frequently in Graph/Hypergraph Neural Networks:
- Sampled Dense-Dense Matrix Multiplication (SDDMM)
- Sparse Matrix-Dense Matrix Multiplication (SpMM)

### To build conv2bin tool:  
```
make conv2bin
```

### To make benchSddmm:  
``` 
make benchSddmm  
```


### Authors  
Nabil Abubaker, Torsten Hoefler

Citation:  

```
@misc{abubaker2024spcomm3d,
      title={SpComm3D: A Framework for Enabling Sparse Communication in 3D Sparse Kernels}, 
      author={Nabil Abubaker and Torsten Hoefler},
      year={2024},
      eprint={2404.19638},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
