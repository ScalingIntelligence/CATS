from typing import Optional

import torch
import triton
import triton.language as tl

# https://github.com/FMInference/DejaVu/tree/master/Dejavu/src/ops/triton
@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))

@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return (x_ * x_).to(x.dtype)

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def apply_activation(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == "relu":
        x = relu(x)
    elif ACTIVATION == "squared_relu":
        x = squared_relu(x)
    elif ACTIVATION == "swish":
        x = silu(x)
    return x

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)

@triton.jit
def gather_gemv_elemul_flag_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    X_1,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Kernel for computing Y = A[IDX, :] @ X) * X_1, where A is a
    dense matrix with M rows and N columns.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, N)
    - Input X_1 has shape (BATCHSIZE, M)
    - A has shape (M, N)
    - IDX has shape (M), where M is the flag for non-zero rows in A
    - Output has shape (BATCHSIZE, M)
    """
    # EVEN_N is asserted to be true
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A and B
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X_1 = X_1 + rm
    X = X + rn
    
    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        i_mask = idx[:, None]
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=i_mask, other=0.0)
            x0 = tl.load(X)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        acc_0 = acc0 * x1_0.to(tl.float32)
    elif BATCHSIZE == 2:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        x1_1 = tl.load(X_1 + M, mask=idx_1, other=0.0)
        i_mask = (idx | idx_1)[:, None]
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
            x0_0 = tl.load(X)
            x0_1 = tl.load(X + N)
            acc0 += tl.sum(a * x0_0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a * x0_1.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        acc_0 = acc0 * x1_0.to(tl.float32)
        acc_1 = acc1 * x1_1.to(tl.float32)

    elif BATCHSIZE == 3:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        idx_2 = tl.load(IDX + 2 * M, mask=rm < M, other=0) > 0
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        x1_1 = tl.load(X_1 + M, mask=idx_1, other=0.0)
        x1_2 = tl.load(X_1 + 2 * M, mask=idx_2, other=0.0)
        i_mask = (idx | idx_1 | idx_2)[:, None]
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
            x0_0 = tl.load(X)
            x0_1 = tl.load(X + N)
            x0_2 = tl.load(X + 2 * N)
            acc0 += tl.sum(a * x0_0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a * x0_1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a * x0_2.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        acc_0 = acc0 * x1_0.to(tl.float32)
        acc_1 = acc1 * x1_1.to(tl.float32)
        acc_2 = acc2 * x1_2.to(tl.float32)    
    elif BATCHSIZE == 4:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        idx_2 = tl.load(IDX + 2 * M, mask=rm < M, other=0) > 0
        idx_3 = tl.load(IDX + 3 * M, mask=rm < M, other=0) > 0
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        x1_1 = tl.load(X_1 + M, mask=idx_1, other=0.0)
        x1_2 = tl.load(X_1 + 2 * M, mask=idx_2, other=0.0)
        x1_3 = tl.load(X_1 + 3 * M, mask=idx_3, other=0.0)
        i_mask = (idx | idx_1 | idx_2 | idx_3)[:, None]
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
            x0_0 = tl.load(X)
            x0_1 = tl.load(X + N)
            x0_2 = tl.load(X + 2 * N)
            x0_3 = tl.load(X + 3 * N)
            acc0 += tl.sum(a * x0_0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a * x0_1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a * x0_2.to(tl.float32)[None, :], 1)
            acc3 += tl.sum(a * x0_3.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        acc_0 = acc0 * x1_0.to(tl.float32)
        acc_1 = acc1 * x1_1.to(tl.float32)
        acc_2 = acc2 * x1_2.to(tl.float32)
        acc_3 = acc3 * x1_3.to(tl.float32)

    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    Y = Y + rm
    # acc = acc0 * x1
    tl.store(Y, acc_0, mask=rm < M)
    if BATCHSIZE >= 2:
        tl.store(Y + M, acc_1, mask=rm < M)
    if BATCHSIZE >= 3:
        tl.store(Y + 2 * M, acc_2, mask=rm < M)
    if BATCHSIZE >= 4:
        tl.store(Y + 3 * M, acc_3, mask=rm < M)

def gather_gemv_elemul_flag_3d(
    x: torch.Tensor,
    x_1: torch.Tensor,
    wup: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = activation(x @ wgate[idx, :].T) * (x @ wup[idx, :].T).
    :param x: input tensor, (batch, N)
    :param x_1: input tensor, (batch, Z)
    :param wup: up weigth matrix, (Z, N)
    :param idx: flags, (Z,)
    :return: result tensor, (batch, N)
    """
    Z, N = wup.shape
    beam_width, seq_len, _ = x.shape
    # assert x.shape == (batch, N)
    # assert x_1.shape == (batch, Z)
    assert seq_len == 1
    assert beam_width >= 1 and beam_width <= 4
    x = x.contiguous()
    x_1 = x_1.contiguous()
    if wup.stride(1) > 1:
        wup = wup.contiguous()
    assert (
        x.dtype == wup.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {wup.dtype}"

    output = torch.empty(beam_width, seq_len, Z, device=x.device, dtype=torch.float32)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(Z, META["BLOCK_M"]),)  # noqa

    gather_gemv_elemul_flag_kernel[grid](
        output,  # data ptrs
        wup,
        x,
        x_1,
        idx,
        Z,  # shapes
        N,
        Z // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        wup.stride(0),  # strides
        beam_width,  # Can't use kwargs because auto-tuner requires args
    )
    return output.to(x.dtype)

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_flag_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    
    if BATCHSIZE == 1:
        a = tl.load(A, mask=idx[:, None], other=0.0)
        x0 = tl.load(X)#, mask=idx, other=0.0) # if flag_gemv is correct, this will be unnecessary.
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
    elif BATCHSIZE == 2:
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        i_mask = (idx | idx_1)[:, None]
        a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
        x0 = tl.load(X)#, mask=idx, other=0.0)
        x0_1 = tl.load(X + M)#, mask=idx_1, other=0.0)
        acc0 = tl.sum(a * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a * x0_1.to(tl.float32)[:, None], 0)
    elif BATCHSIZE == 3:
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        idx_2 = tl.load(IDX + 2 * M, mask=rm < M, other=0) > 0
        i_mask = (idx | idx_1 | idx_2)[:, None]
        a = tl.load(A, mask=i_mask, other=0.0)
        x0 = tl.load(X)#, mask=idx, other=0.0)
        x0_1 = tl.load(X + M)#, mask=idx_1, other=0.0)
        x0_2 = tl.load(X + 2 * M)#, mask=idx_2, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x0_1.to(tl.float32)[:, None], 0)
        acc2 = tl.sum(a.to(tl.float32) * x0_2.to(tl.float32)[:, None], 0)
    elif BATCHSIZE == 4:
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        idx_2 = tl.load(IDX + 2 * M, mask=rm < M, other=0) > 0
        idx_3 = tl.load(IDX + 3 * M, mask=rm < M, other=0) > 0
        i_mask = (idx | idx_1 | idx_2 | idx_3)[:, None]
        a = tl.load(A, mask=i_mask, other=0.0)
        x0 = tl.load(X)#, mask=idx, other=0.0)
        x0_1 = tl.load(X + M)#, mask=idx_1, other=0.0)
        x0_2 = tl.load(X + 2 * M)#, mask=idx_2, other=0.0)
        x0_3 = tl.load(X + 3 * M)#, mask=idx_3, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x0_1.to(tl.float32)[:, None], 0)
        acc2 = tl.sum(a.to(tl.float32) * x0_2.to(tl.float32)[:, None], 0)
        acc3 = tl.sum(a.to(tl.float32) * x0_3.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)
    if BATCHSIZE >= 2:
        tl.atomic_add(Y + N, acc1, mask=rn < N)
    if BATCHSIZE >= 3:
        tl.atomic_add(Y + 2 * N, acc2, mask=rn < N)
    if BATCHSIZE >= 4:
        tl.atomic_add(Y + 3 * N, acc3, mask=rn < N)

def gather_transposed_gemv_flag_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()

    output = torch.empty(
        beam_width,
        seq_len,
        N,
        device=x.device,
        dtype=torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(Z, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa

    kernel = gather_transposed_gemv_flag_atomicadd_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        beam_width,  # can't use kwargs because auto-tuner requires args
    )
    return output.to(dtype=weight.dtype)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_gemv_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    BIAS,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = ACTIVATION(A[IDX, :] @ X + BIAS[IDX]), where A is a dense matrix with
    Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, N)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - BIAS has shape (Z,)
    - Output has shape (BATCHSIZE, M)
    """
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (idx[:, None] * stride_am + rn[None, :])
    X = X + rn
    if (
        HAS_BIAS
    ):  # This load is slow because it's not coalesced, so we kick it off early
        bias = tl.load(BIAS + idx, mask=rm < M, other=0.0).to(tl.float32)

    # We're copying-pasting code because it's hard to get Triton to batch things
    # without massive slowdown.
    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
    elif BATCHSIZE == 2:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            x1 = tl.load(X + N) if EVEN_N else tl.load(X + N, mask=rn < n, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
            acc1 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        acc1 = apply_activation(acc1, ACTIVATION=ACTIVATION)
    elif BATCHSIZE == 3:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            x1 = tl.load(X + N) if EVEN_N else tl.load(X + N, mask=rn < n, other=0.0)
            x2 = (
                tl.load(X + 2 * N)
                if EVEN_N
                else tl.load(X + 2 * N, mask=rn < n, other=0.0)
            )
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
            acc1 += bias
            acc2 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        acc1 = apply_activation(acc1, ACTIVATION=ACTIVATION)
        acc2 = apply_activation(acc2, ACTIVATION=ACTIVATION)
    elif BATCHSIZE == 4:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            x1 = tl.load(X + N) if EVEN_N else tl.load(X + N, mask=rn < n, other=0.0)
            x2 = (
                tl.load(X + 2 * N)
                if EVEN_N
                else tl.load(X + 2 * N, mask=rn < n, other=0.0)
            )
            x3 = (
                tl.load(X + 3 * N)
                if EVEN_N
                else tl.load(X + 3 * N, mask=rn < n, other=0.0)
            )
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[None, :], 1)
            acc3 += tl.sum(a.to(tl.float32) * x3.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
            acc1 += bias
            acc2 += bias
            acc3 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        acc1 = apply_activation(acc1, ACTIVATION=ACTIVATION)
        acc2 = apply_activation(acc2, ACTIVATION=ACTIVATION)
        acc3 = apply_activation(acc3, ACTIVATION=ACTIVATION)

    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    Y = Y + rm
    tl.store(Y, acc0, mask=rm < M)
    if BATCHSIZE >= 2:
        tl.store(Y + M, acc1, mask=rm < M)
    if BATCHSIZE >= 3:
        tl.store(Y + 2 * M, acc2, mask=rm < M)
    if BATCHSIZE >= 4:
        tl.store(Y + 3 * M, acc3, mask=rm < M)


def gather_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "id",
) -> torch.Tensor:
    """
    Compute y = activation(x @ weight[idx, :].T + bias[idx]).
    :param x: input tensor, (batch, N)
    :param weight: weight matrix, (Z, N)
    :param idx: indices, (M,)
    :param bias: bias, (Z,)
    :return: result tensor, (batch, M)
    """
    assert activation in ["id", "relu", "squared_relu", "swish"]
    Z, N = weight.shape
    batch, _ = x.shape
    assert x.shape == (batch, N)
    assert batch in [1, 2, 3, 4]
    (M,) = idx.shape
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()
    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:

        bias = bias.contiguous()
        assert bias.shape == (Z,), "Incompatible dimensions in between weight and bias"
        assert (
            x.dtype == bias.dtype
        ), f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"

    output = torch.empty(batch, M, device=x.device, dtype=x.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa

    gather_gemv_kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        bias,
        M,  # shapes
        N,
        M // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        weight.stride(0),  # strides
        batch,  # Can't use kwargs because auto-tuner requires args
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
    )

    return output

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    BIAS,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (idx[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    if HAS_BIAS:
        BIAS = BIAS + rn
    a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < N, other=0.0)
    if BATCHSIZE == 1:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
    elif BATCHSIZE == 2:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        x1 = tl.load(X + M, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
                acc1 += bias
    elif BATCHSIZE == 3:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        x1 = tl.load(X + M, mask=rm < M, other=0.0)
        x2 = tl.load(X + 2 * M, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
        acc2 = tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
                acc1 += bias
                acc2 += bias
    elif BATCHSIZE == 4:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        x1 = tl.load(X + M, mask=rm < M, other=0.0)
        x2 = tl.load(X + 2 * M, mask=rm < M, other=0.0)
        x3 = tl.load(X + 3 * M, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
        acc2 = tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
        acc3 = tl.sum(a.to(tl.float32) * x3.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
                acc1 += bias
                acc2 += bias
                acc3 += bias
    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)
    if BATCHSIZE >= 2:
        tl.atomic_add(Y + N, acc1, mask=rn < N)
    if BATCHSIZE >= 3:
        tl.atomic_add(Y + 2 * N, acc2, mask=rn < N)
    if BATCHSIZE >= 4:
        tl.atomic_add(Y + 3 * N, acc3, mask=rn < N)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    BIAS,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_n = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + rm
    IDX = IDX + rm

    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            a = (
                tl.load(A_ptr)
                if EVEN_N
                else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            )
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
    elif BATCHSIZE == 2:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            x1 = tl.load(X + M, mask=rm < m, other=0.0)
            a = (
                tl.load(A_ptr)
                if EVEN_N
                else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            )
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
            acc1 += bias
    elif BATCHSIZE == 3:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            x1 = tl.load(X + M, mask=rm < m, other=0.0)
            x2 = tl.load(X + 2 * M, mask=rm < m, other=0.0)
            a = (
                tl.load(A_ptr)
                if EVEN_N
                else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            )
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
            acc1 += bias
            acc2 += bias
    elif BATCHSIZE == 4:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            x1 = tl.load(X + M, mask=rm < m, other=0.0)
            x2 = tl.load(X + 2 * M, mask=rm < m, other=0.0)
            x3 = tl.load(X + 3 * M, mask=rm < m, other=0.0)
            a = (
                tl.load(A_ptr)
                if EVEN_N
                else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            )
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
            acc3 += tl.sum(a.to(tl.float32) * x3.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
            acc1 += bias
            acc2 += bias
            acc3 += bias

    # write back result
    Y = Y + rn
    tl.store(Y, acc0, mask=rn < N)
    if BATCHSIZE >= 2:
        tl.store(Y + N, acc1, mask=rn < N)
    if BATCHSIZE >= 3:
        tl.store(Y + 2 * N, acc2, mask=rn < N)
    if BATCHSIZE >= 4:
        tl.store(Y + 3 * N, acc3, mask=rn < N)


def gather_transposed_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x + bias.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    (M,) = idx.shape
    batch, _ = x.shape
    assert x.shape == (
        batch,
        M,
    )
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()
    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:
        bias = bias.contiguous()
        assert bias.shape == (N,), "Incompatible dimensions in between weight and bias"
        assert (
            x.dtype == bias.dtype
        ), f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"

    # kernel_type = "deterministic"
    kernel_type = "atomicadd"  # This always seems to be faster for now

    output = torch.empty(
        batch,
        N,
        device=x.device,
        dtype=x.dtype if kernel_type == "deterministic" else torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    if kernel_type == "deterministic":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)  # noqa
    else:
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )  # noqa

    kernel = (
        gather_transposed_gemv_kernel
        if kernel_type == "deterministic"
        else gather_transposed_gemv_atomicadd_kernel
    )
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        bias,
        M,  # shapes
        N,
        M // 1024,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        batch,  # can't use kwargs because auto-tuner requires args
        HAS_BIAS=bias is not None,
    )

    return output.to(dtype=x.dtype)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BEAMSIZE"],
)

@triton.jit
def gather_gemv_elemul_index_3d_kernel(
    Y,
    A,
    X,
    X_1,
    IDX,
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    stride_am,
    BEAMSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (idx[:, None] * stride_am + rn[None, :])
    X_1 = X_1 + rm
    X = X + rn

    if BEAMSIZE == 1:
        x1 = tl.load(X_1, mask=rm < M, other=0.0).to(tl.float32)
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A)
            x0 = tl.load(X)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        acc0 = acc0 * x1
    
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    Y = Y + rm
    tl.store(Y, acc0, mask=rm < M)

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BEAMSIZE"],
)

@triton.jit
def gather_transposed_gemv_index_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    stride_am,
    BEAMSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (idx[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    a = tl.load(A)
    if BEAMSIZE == 1:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)

def gather_gemv_elemul_indirect_index_2d(
        x: torch.Tensor, 
        x_1: torch.Tensor,
        Wup: torch.Tensor,  
        idx: torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute y = x_1 * (x @ Wup[idx, :].T)
    :param x: input tensor, (batch, N)
    :param x_1: wgate activation, (beam_width, batch, M)
    :param Wup: up weigth matrix, (Z, N)
    :param idx: live indices, (M,)
    :return: result tensor, (beam_width, batch, M)
    """
    _, N = Wup.shape
    beam_width,  _ = x.shape
    (M,) = idx.shape
    assert x.shape[1] == N
    assert beam_width == 1

    output = torch.empty(beam_width, M, device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa

    gather_gemv_elemul_index_3d_kernel[grid](
        output,  # data ptrs
        Wup,
        x,
        x_1,
        idx,
        M,  # shapes
        N,
        M // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        Wup.stride(0),  # strides
        beam_width,
    )

    return output

def gather_transposed_gemv_indirect_index_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor, (beam_width, batch, M)
    :param weight: weight matrix, (Z, N)
    :param idx: indices, (M,)
    :return: result tensor, (beam_width, batch, N)
    """
    _, N = weight.shape
    beam_width, _ = x.shape
    (M,) = idx.shape
    assert x.shape[1] == M
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()
    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"

    output = torch.empty(
        beam_width,
        N,
        device=x.device,
        dtype=torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa

    gather_transposed_gemv_index_atomicadd_kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        M,  # shapes
        N,
        M // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        weight.stride(0),  # strides
        beam_width,
    )

    return output.to(x.dtype)

def mistral_mlp_partial_sparse(x, x_1, Wup, Wdownt, idx):
    x = gather_gemv(x, Wup, idx) * x_1
    x = gather_transposed_gemv(x, Wdownt, idx)
    return x

def mistral_mlp_sparse_direct_index_2d(x, x_1, Wup, Wdownt, idx):
    x_2 = gather_gemv_elemul_indirect_index_2d(x, x_1, Wup, idx)
    return gather_transposed_gemv_indirect_index_2d(x_2, Wdownt, idx)