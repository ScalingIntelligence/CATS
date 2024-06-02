import importlib
import os.path as osp
import torch
from .flash_gemv import (
    fuse_gemv_cmp, 
    fuse_gemv_flag, 
    fuse_gemv_flag_gemv, 
    fuse_gemv_flag_gemv_gemv, 
    fuse_gemv_gemv_gemv, 
    fuse_gemv_flag_batch, 
    fuse_gemv_flag_local, 
    fuse_flag_gemv_local, 
    atomic_gemv, 
    flag_gemv_gemv_atomic, 
    flag_gemv_gemv, 
    flag_gemv_gemv_inner, 
    flag_gemv_gemv_inner_fp32, 
    flag_gemv_gemv_inner_bf16, 
)
from .kernels import (
    gather_gemv_elemul_flag_3d,
    gather_transposed_gemv_flag_3d,
    mistral_mlp_partial_sparse,
    mistral_mlp_sparse_direct_index_2d
)

__version__ = '0.0.1'

library = '_C'
spec = importlib.machinery.PathFinder().find_spec(
    library, [osp.dirname(__file__)])
if spec is not None:
    torch.ops.load_library(spec.origin)
else:
    raise ImportError(f"Could not find module '{library}' in "
                      f'{osp.dirname(__file__)}')

def flag_gemv_gemv_triton(x, Wgate, Wup, Wdownt, threshold):
    act_fn = torch.nn.SiLU()
    x_1 = act_fn(torch.matmul(x, Wgate))
    flags = torch.abs(x_1) > threshold
    x = gather_gemv_elemul_flag_3d(x, x_1, Wup, flags)
    return gather_transposed_gemv_flag_3d(x, Wdownt, flags)

def gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = gather_gemv_elemul_flag_3d(x, x_1, Wup, flags)
    return gather_transposed_gemv_flag_3d(x, Wdownt, flags)

def flag_gemv_gemv_dejavu(x, Wgate, Wup, Wdownt, threshold):
    act_fn = torch.nn.SiLU()
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return mistral_mlp_partial_sparse(x, x_1[0][idx], Wup, Wdownt, idx)

def flag_gemv_gemv_fuse_dejavu(x, Wgate, Wup, Wdownt, threshold):
    act_fn = torch.nn.SiLU()
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return mistral_mlp_sparse_direct_index_2d(x, x_1[0][idx], Wup, Wdownt, idx)

__all__ = [fuse_gemv_cmp, 
           fuse_gemv_flag, 
           fuse_gemv_flag_gemv, 
           fuse_gemv_flag_gemv_gemv, 
           fuse_gemv_gemv_gemv, 
           fuse_gemv_flag_batch, 
           fuse_gemv_flag_local, 
           fuse_flag_gemv_local, 
           atomic_gemv, 
           flag_gemv_gemv_atomic, 
           flag_gemv_gemv, 
           flag_gemv_gemv_inner, 
           flag_gemv_gemv_inner_fp32, 
           flag_gemv_gemv_inner_bf16, 
           flag_gemv_gemv_triton, 
           gemv_gemv_triton,
           flag_gemv_gemv_fuse_dejavu,
           flag_gemv_gemv_dejavu,
        ]
