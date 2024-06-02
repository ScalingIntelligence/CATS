import torch

def fuse_gemv_cmp(input, weight, output, idx, threshold):
    return torch.ops.flash_gemv.fuse_gemv_cmp(input, weight, output, idx, threshold)

def fuse_gemv_flag(input, weight, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag(input, weight, output, flag, threshold)

def fuse_gemv_flag_gemv(input, wgate, wup, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag_gemv(input, wgate, wup, output, flag, threshold)

def fuse_gemv_flag_gemv_gemv(input, wgate, wup, wdown, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag_gemv_gemv(input, wgate, wup, wdown, output, flag, threshold)

def fuse_gemv_gemv_gemv(input, wgate, wup, wdown, output, threshold):
    return torch.ops.flash_gemv.fuse_gemv_gemv_gemv(input, wgate, wup, wdown, output, threshold)

def fuse_gemv_flag_batch(input, weight, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag_batch(input, weight, output, flag, threshold)

def fuse_gemv_flag_local(input, weight, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag_local(input, weight, output, flag, threshold)

def fuse_flag_gemv_local(input, weight, output, flag):
    return torch.ops.flash_gemv.fuse_flag_gemv_local(input, weight, output, flag)

def atomic_gemv(input, weight, output, flag):
    return torch.ops.flash_gemv.atomic_gemv(input, weight, output, flag)

def flag_gemv_gemv_atomic(input, wgate, wup, wdown, output, threshold):
    return torch.ops.flash_gemv.flag_gemv_gemv_atomic(input, wgate, wup, wdown, output, threshold)

def flag_gemv_gemv(input, wgate, wup, wdown, output, threshold):
    return torch.ops.flash_gemv.flag_gemv_gemv(input, wgate, wup, wdown, output, threshold)

def flag_gemv_gemv_inner(input, wgate, wup, wdown, threshold):
    return torch.ops.flash_gemv.flag_gemv_gemv_inner(input, wgate, wup, wdown, threshold)

def flag_gemv_gemv_inner_fp32(input, wgate, wup, wdown, threshold):
    return torch.ops.flash_gemv.flag_gemv_gemv_inner_fp32(input, wgate, wup, wdown, threshold)

def flag_gemv_gemv_inner_bf16(input, wgate, wup, wdown, threshold):
    return torch.ops.flash_gemv.flag_gemv_gemv_inner_bf16(input, wgate, wup, wdown, threshold)
