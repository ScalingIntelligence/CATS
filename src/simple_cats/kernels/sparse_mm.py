import torch

from src.simple_cats.kernels.triton_kernel.mm_kernels import (
    gather_gemv_elemul_flag_3d,
    gather_transposed_gemv_flag_3d,
)


def gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = gather_gemv_elemul_flag_3d(x, x_1, Wup, flags)
    return gather_transposed_gemv_flag_3d(x, Wdownt, flags)


def gemv_triton(x, W, threshold):
    return gather_transposed_gemv_flag_3d(x, W, torch.abs(x) > threshold)
