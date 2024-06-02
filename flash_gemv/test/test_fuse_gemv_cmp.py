import torch
import torch.nn as nn
import flash_gemv
import triton
import triton.language as tl
import numpy as np
import argparse

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
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X_1 = X_1 + rm
    X = X + rn
    x1 = tl.load(X_1, mask=rm < M, other=0.0).to(tl.float32)

    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        i_mask = idx[:, None] > 0
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=i_mask, other=0.0)
            x0 = tl.load(X)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        
    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    Y = Y + rm
    acc = acc0 * x1
    tl.store(Y, acc, mask=rm < M)

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
    beam_width, batch, _ = x.shape
    # assert x.shape == (batch, N)
    # assert x_1.shape == (batch, Z)
    assert batch == 1
    assert beam_width == 1
    x = x.contiguous()
    x_1 = x_1.contiguous()
    if wup.stride(1) > 1:
        wup = wup.contiguous()
    assert (
        x.dtype == wup.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {wup.dtype}"

    output = torch.empty(beam_width, batch, Z, device=x.device, dtype=x.dtype)

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
        batch,  # Can't use kwargs because auto-tuner requires args
    )

    return output

def test():
    input = torch.randn(1, 1, 4096).half().cuda()
    weight = torch.randn(4096, 14336).half().cuda()
    weight_t = weight.t().contiguous()
    threshold = 0.1
    idx = torch.zeros(14336, dtype=torch.int64).cuda()
    value = torch.zeros(1, 1, 14336).half().cuda()
    N = flash_gemv.fuse_gemv_cmp(input, weight_t, value, idx, threshold)
    torch.cuda.synchronize()
    print("finish")
    act_fn = nn.SiLU()
    output = torch.abs(act_fn(torch.matmul(input, weight)))
    (_,_,golden_idx) = torch.nonzero(output > threshold, as_tuple=True)
    print(golden_idx.shape)
    # print(value.shape)
    # print(value[0])
    print(N)
    idx = idx[:N]
    # Sort idx and value together

    idx_sorted, indices = torch.sort(idx[:N])
    value_sorted = value[0][0][indices]
    print(value_sorted)
    print(output[0][0][golden_idx])
    print(idx_sorted)
    print(golden_idx)

    # assert torch.allclose(value[0][0][idx_sorted], golden)

def test_flag():
    torch.manual_seed(0)
    input = torch.rand(1, 1, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    threshold = 0.2
    value = torch.zeros(1, 1, 14336).to(torch.float32).cuda()
    flag = torch.zeros(14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=True)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold).squeeze(0).squeeze(0)
    golden_indices = torch.nonzero(golden_flag, as_tuple=True)

    print(torch.nonzero(torch.logical_xor(flag, golden_flag)))
    print(indices[0].shape)
    print(golden_indices[0].shape)
    print(indices)
    print(golden_indices)
    np.savetxt('value.txt', value[0][0][indices].cpu().numpy())
    np.savetxt('golden_value.txt', golden_value[0][0][golden_indices].cpu().numpy())
    print(value[0][0][indices])
    print(golden_value[0][0][golden_indices])
    # print(value[0][0][indices].view(dtype=torch.int16))
    # print(golden_value[0][0][golden_indices].view(dtype=torch.int16))
    # value_bytes = value[0][0][indices]
    # value_bytes = value_bytes.view(dtype=torch.uint8)
    # for i in range(6):
    #     print(value_bytes[i].item().to_bytes(1, byteorder='little').hex())
    # golden_value_bytes = golden_value[0][0][golden_indices]
    # golden_value_bytes = golden_value_bytes.view(dtype=torch.uint8)
    # print("Golden")
    # for i in range(6):
    #     print(golden_value_bytes[i].item().to_bytes(1, byteorder='little').hex())

def test_flag_gemv():
    torch.manual_seed(0)
    input = torch.randn(1, 1, 4096).half().cuda()
    wgate = torch.randn(4096, 14336).half().cuda()
    wup = torch.randn(4096, 14336).half().cuda()
    # input = 0.1 * torch.ones(1, 1, 4096).half().cuda()
    # wgate = 0.1 * torch.ones(4096, 14336).half().cuda()
    # wup = 0.1 * torch.ones(4096, 14336).half().cuda()
    wgate_t = wgate.t().contiguous()
    wup_t = wup.t().contiguous()
    threshold = 0.1
    value = torch.zeros(1, 1, 14336).half().cuda()
    flag = torch.zeros(14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_gemv(input, wgate_t, wup_t, value, flag, threshold)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=True)
    act_fn = nn.SiLU()
    golden_inter = act_fn(torch.matmul(input, wgate))
    golden_inter = golden_inter.squeeze(0).squeeze(0)
    golden_flag = (torch.abs(golden_inter) > threshold)
    golden_flag_not = ~golden_flag
    golden_inter[golden_flag_not] = 0
    golden_value = torch.matmul(input, wup) * golden_inter
    golden_indices = torch.nonzero(golden_flag, as_tuple=True)
    print(indices[0].shape)
    print(golden_indices[0].shape)
    print(indices)
    print(golden_indices)
    print(value)
    print(golden_value)
    print(torch.nonzero(torch.abs(golden_value-value) > 0.01 * torch.abs(value)))

def test_flag_gemv_gemv():
    torch.manual_seed(0)
    input = torch.rand(1, 1, 4096).half().cuda() - 0.5
    wgate = torch.rand(4096, 14336).half().cuda() - 0.5
    wup = torch.rand(4096, 14336).half().cuda() - 0.5
    wdown = torch.rand(14336, 4096).half().cuda() - 0.5
    wgate_t = wgate.t().contiguous()
    wup_t = wup.t().contiguous()
    threshold = 0.1
    output = torch.zeros(1, 1, 4096).cuda()
    flag = torch.zeros(14336, dtype=torch.bool).cuda()
    # print("Start")
    flash_gemv.fuse_gemv_flag_gemv_gemv(input, wgate_t, wup_t, wdown, output, flag, threshold)
    torch.cuda.synchronize()
    output = output.half()
    indices = torch.nonzero(flag, as_tuple=True)
    act_fn = nn.SiLU()
    golden_inter = act_fn(torch.matmul(input, wgate))
    golden_inter = golden_inter.squeeze(0).squeeze(0)
    golden_flag = (torch.abs(golden_inter) > threshold)
    golden_flag_not = ~golden_flag
    golden_inter[golden_flag_not] = 0
    golden_value = torch.matmul(input, wup) * golden_inter
    golden_value = golden_value.squeeze(0).squeeze(0)
    golden_value[golden_flag_not] = 0
    golden_output = torch.matmul(golden_value, wdown)
    golden_indices = torch.nonzero(golden_flag, as_tuple=True)
    print(indices[0].shape)
    print(golden_indices[0].shape)
    print(indices)
    print(golden_indices)
    print(torch.sum(indices[0]))
    np.savetxt('indices.txt', indices[0].cpu().numpy())
    np.savetxt('golden_indices.txt', golden_indices[0].cpu().numpy())
    print(output)
    print(golden_output)
    np.savetxt('value.txt', output.squeeze(0).squeeze(0).cpu().numpy())
    np.savetxt('golden_value.txt', golden_output.cpu().numpy())

def test_gemv_gemv_gemv():
    torch.manual_seed(0)
    input = torch.rand(1, 1, 4096).half().cuda() - 0.5
    wgate = torch.rand(4096, 14336).half().cuda() - 0.5
    wup = torch.rand(4096, 14336).half().cuda() - 0.5
    wdown = torch.rand(14336, 4096).half().cuda() - 0.5
    wgate_t = wgate.t().contiguous()
    wup_t = wup.t().contiguous()
    threshold = 0.1
    output = torch.zeros(1, 1, 4096).cuda()
    # print("Start")
    flash_gemv.fuse_gemv_gemv_gemv(input, wgate_t, wup_t, wdown, output, threshold)
    torch.cuda.synchronize()
    # output = output.half()
    act_fn = nn.SiLU()
    golden_inter = act_fn(torch.matmul(input, wgate))
    golden_inter = golden_inter.squeeze(0).squeeze(0)
    golden_flag = (torch.abs(golden_inter) > threshold)
    golden_flag_not = ~golden_flag
    golden_inter[golden_flag_not] = 0
    golden_value = torch.matmul(input, wup) * golden_inter
    golden_value = golden_value.squeeze(0).squeeze(0)
    golden_value[golden_flag_not] = 0
    golden_output = torch.matmul(golden_value, wdown)
    print(output)
    print(golden_output)
    np.savetxt('value.txt', output.squeeze(0).squeeze(0).cpu().numpy())
    np.savetxt('golden_value.txt', golden_output.cpu().numpy())

def test_flag_batch():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    threshold = 0.2
    value = torch.zeros(1, B, 14336).to(torch.float16).cuda()
    flag = torch.zeros(1, B, 14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_batch(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=False)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold)
    golden_indices = torch.nonzero(golden_flag, as_tuple=False)
    print(indices.shape)
    print(golden_indices.shape)
    golden_value[torch.abs(golden_value) <= threshold] = 0
    np.savetxt('value.txt', value.flatten().cpu().numpy())
    np.savetxt('golden_value.txt', golden_value.flatten().cpu().numpy())

def test_flag_local():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    threshold = 0.2
    value = torch.zeros(1, B, 14336).to(torch.float16).cuda()
    flag = torch.zeros(1, B, 14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_local(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=False)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold)
    golden_indices = torch.nonzero(golden_flag, as_tuple=False)
    print(indices.shape)
    print(golden_indices.shape)
    golden_value[torch.abs(golden_value) <= threshold] = 0
    np.savetxt('value.txt', value.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_value.txt', golden_value.to(torch.float32).flatten().cpu().numpy())

def test_flag_gemv_local():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    wup = torch.rand(4096, 14336).half().cuda() - 0.5
    wup_t = wup.t().contiguous()
    threshold = 0.2
    value = torch.zeros(1, B, 14336).to(torch.float16).cuda()
    flag = torch.zeros(1, B, 14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_local(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    flash_gemv.fuse_flag_gemv_local(input, wup_t, value, flag)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=False)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold)
    golden_indices = torch.nonzero(golden_flag, as_tuple=False)
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    print(indices.shape)
    print(golden_indices.shape)
    np.savetxt('value_1.txt', value.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_up.txt', golden_up.to(torch.float32).flatten().cpu().numpy())

def test_flag_gemv_gemv_batch():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    wup = torch.rand(4096, 14336).half().cuda() - 0.5
    wup_t = wup.t().contiguous()
    wdown = torch.rand(14336, 4096).half().cuda() - 0.5
    threshold = 0.2
    value = torch.zeros(1, B, 14336).to(torch.float16).cuda()
    flag = torch.zeros(1, B, 14336, dtype=torch.bool).cuda()
    output = torch.zeros(1, B, 4096).cuda()
    flash_gemv.fuse_gemv_flag_local(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    flash_gemv.fuse_flag_gemv_local(input, wup_t, value, flag)
    torch.cuda.synchronize()
    flash_gemv.atomic_gemv(value, wdown, output, flag)
    torch.cuda.synchronize()
    output = output.half()
    indices = torch.nonzero(flag, as_tuple=False)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold)
    golden_indices = torch.nonzero(golden_flag, as_tuple=False)
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    golden_output = torch.matmul(golden_up, wdown)
    print(indices.shape)
    print(golden_indices.shape)
    np.savetxt('value_whole.txt', output.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_whole.txt', golden_output.to(torch.float32).flatten().cpu().numpy())

def test_flag_gemv_gemv_atomic():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    wup = torch.rand(4096, 14336).half().cuda() - 0.5
    wup_t = wup.t().contiguous()
    wdown = torch.rand(14336, 4096).half().cuda() - 0.5
    threshold = 0.3

    output = torch.zeros(1, B, 4096).cuda()
    flash_gemv.flag_gemv_gemv_atomic(input, weight_t, wup_t, wdown, output, threshold)
    torch.cuda.synchronize()
    output = output.half()
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    golden_output = torch.matmul(golden_up, wdown)
    # print the indices that |output - golden_output| > 0.01 * |output|
    nq_indices = torch.nonzero(torch.abs(golden_output-output) > 0.01 * torch.abs(output))
    print(nq_indices)
    print(nq_indices.shape)

    np.savetxt('value_whole.txt', output.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_whole.txt', golden_output.to(torch.float32).flatten().cpu().numpy())

def test_flag_gemv_gemv_local():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight = torch.rand(4096, 14336).half().cuda() - 0.5
    weight_t = weight.t().contiguous()
    wup = torch.rand(4096, 14336).half().cuda() - 0.5
    wup_t = wup.t().contiguous()
    wdown = torch.rand(14336, 4096).half().cuda() - 0.5
    wdown_t = wdown.t().contiguous()
    threshold = 0.3

    # output = torch.zeros(1, B, 4096).half().cuda()
    # flash_gemv.flag_gemv_gemv(input, weight_t, wup_t, wdown_t, output, threshold)
    output = flash_gemv.flag_gemv_gemv_inner(input, weight_t, wup_t, wdown_t, threshold)
    torch.cuda.synchronize()
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    golden_output = torch.matmul(golden_up, wdown)
    # print the indices that |output - golden_output| > 0.01 * |output|
    nq_indices = torch.nonzero(torch.abs(golden_output-output) > 0.01 * torch.abs(output))
    print(nq_indices)
    print(nq_indices.shape)

    np.savetxt('value_whole.txt', output.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_whole.txt', golden_output.to(torch.float32).flatten().cpu().numpy())

def test_flag_gemv_gemv_local_float():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).cuda() - 0.5
    weight = torch.rand(4096, 14336).cuda() - 0.5
    weight_t = weight.t().contiguous()
    wup = torch.rand(4096, 14336).cuda() - 0.5
    wup_t = wup.t().contiguous()
    wdown = torch.rand(14336, 4096).cuda() - 0.5
    wdown_t = wdown.t().contiguous()
    threshold = 0.3

    # output = torch.zeros(1, B, 4096).half().cuda()
    # flash_gemv.flag_gemv_gemv(input, weight_t, wup_t, wdown_t, output, threshold)
    output = flash_gemv.flag_gemv_gemv_inner_fp32(input, weight_t, wup_t, wdown_t, threshold)
    torch.cuda.synchronize()
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    golden_output = torch.matmul(golden_up, wdown)
    # print the indices that |output - golden_output| > 0.01 * |output|
    nq_indices = torch.nonzero(torch.abs(golden_output-output) > 0.01 * torch.abs(output))
    print(nq_indices)
    print(nq_indices.shape)

def test_flag_gemv_gemv_local_bf16():
    torch.manual_seed(0)
    B = 2
    input = torch.randn(1, B, 4096).to(torch.bfloat16).cuda() - 0.5
    weight = torch.rand(4096, 14336).to(torch.bfloat16).cuda() - 0.5
    weight_t = weight.t().contiguous()
    wup = torch.rand(4096, 14336).to(torch.bfloat16).cuda() - 0.5
    wup_t = wup.t().contiguous()
    wdown = torch.rand(14336, 4096).to(torch.bfloat16).cuda() - 0.5
    wdown_t = wdown.t().contiguous()
    threshold = 0.3

    # output = torch.zeros(1, B, 4096).half().cuda()
    # flash_gemv.flag_gemv_gemv(input, weight_t, wup_t, wdown_t, output, threshold)
    output = flash_gemv.flag_gemv_gemv_inner_bf16(input, weight_t, wup_t, wdown_t, threshold)
    torch.cuda.synchronize()
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    golden_output = torch.matmul(golden_up, wdown)
    # print the indices that |output - golden_output| > 0.01 * |output|
    nq_indices = torch.nonzero(torch.abs(golden_output-output) > 0.01 * torch.abs(output))
    print(nq_indices)
    print(nq_indices.shape)

    np.savetxt('value_whole.txt', output.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_whole.txt', golden_output.to(torch.float32).flatten().cpu().numpy())

def test_flag_gemv_gemv_triton():
    torch.manual_seed(0)
    B = 3
    input = torch.rand(B, 1, 4096).to(torch.bfloat16).cuda() - 0.5
    weight = torch.rand(4096, 14336).to(torch.bfloat16).cuda() - 0.5
    wup = torch.rand(4096, 14336).to(torch.bfloat16).cuda() - 0.5
    wup_t = wup.t().contiguous()
    wdown = torch.rand(14336, 4096).to(torch.bfloat16).cuda() - 0.5
    threshold = 0.3
    act_fn = nn.SiLU()

    output = flash_gemv.gemv_gemv_triton(input, act_fn(torch.matmul(input, weight)), wup_t, wdown, threshold)
    torch.cuda.synchronize()
    
    golden_value = act_fn(torch.matmul(input, weight))
    golden_up = torch.matmul(input, wup) * golden_value
    golden_up[torch.abs(golden_value) <= threshold] = 0
    golden_output = torch.matmul(golden_up, wdown)
    # print the indices that |output - golden_output| > 0.01 * |output|
    nq_indices = torch.nonzero(torch.abs(golden_output-output) > 0.01 * torch.abs(output))
    print(nq_indices)
    print(nq_indices.shape)

    np.savetxt('value_whole.txt', output.to(torch.float32).flatten().cpu().numpy())
    np.savetxt('golden_whole.txt', golden_output.to(torch.float32).flatten().cpu().numpy())

def bench_flag():
    torch.manual_seed(0)
    input = torch.randn(1, 1, 4096).half().cuda()
    weight = torch.randn(4096, 14336).half().cuda()
    weight_t = weight.t().contiguous()
    threshold = 0.2
    value = torch.zeros(1, 1, 14336).half().cuda()
    flag = torch.zeros(14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = torch.abs(golden_value) > threshold
    print(golden_flag.shape)

def bench_flag_gemv():
    torch.manual_seed(0)
    input = torch.randn(1, 1, 4096).half().cuda()
    wgate = torch.randn(4096, 14336).half().cuda()
    wgate_t = wgate.t().contiguous()
    wup = torch.randn(4096, 14336).half().cuda()
    wup_t = wup.t().contiguous()
    threshold = 0.1
    value = torch.zeros(1, 1, 14336).half().cuda()
    flag = torch.zeros(14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_gemv(input, wgate_t, wup_t, value, flag, threshold)
    torch.cuda.synchronize()
    value = torch.zeros(1, 1, 14336).half().cuda()
    flag = torch.zeros(14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag(input, wgate_t, value, flag, threshold)
    # gather_gemv_elemul_flag_3d(input, value, wup_t, flag)

def bench_flag_batch():
    torch.manual_seed(0)
    B = 16
    input = torch.randn(1, B, 4096).half().cuda()
    weight = torch.randn(4096, 14336).half().cuda()
    weight_t = weight.t().contiguous()
    threshold = 0.2
    value = torch.zeros(1, B, 14336).to(torch.float16).cuda()
    flag = torch.zeros(1, B, 14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_batch(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=False)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold)
    golden_indices = torch.nonzero(golden_flag, as_tuple=False)
    print(indices.shape)
    print(golden_indices.shape)

def bench_flag_local():
    torch.manual_seed(0)
    B = 1
    input = torch.randn(1, B, 4096).half().cuda()
    weight = torch.randn(4096, 14336).half().cuda()
    weight_t = weight.t().contiguous()
    threshold = 0.1
    value = torch.zeros(1, B, 14336).to(torch.float16).cuda()
    flag = torch.zeros(1, B, 14336, dtype=torch.bool).cuda()
    flash_gemv.fuse_gemv_flag_local(input, weight_t, value, flag, threshold)
    torch.cuda.synchronize()
    indices = torch.nonzero(flag, as_tuple=False)
    act_fn = nn.SiLU()
    golden_value = act_fn(torch.matmul(input, weight))
    golden_flag = (torch.abs(golden_value) > threshold)
    golden_indices = torch.nonzero(golden_flag, as_tuple=False)
    print(indices.shape)
    print(golden_indices.shape)

def bench_flag_gemv_gemv_atomic(B, threshold):
    torch.manual_seed(0)
    # B = 2
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight_t = torch.rand(14336, 4096).half().cuda() - 0.5
    wup_t = torch.rand(14336, 4096).half().cuda() - 0.5
    wdown = torch.rand(14336, 4096).half().cuda() - 0.5
    # threshold = 0.3

    output = torch.zeros(1, B, 4096).cuda()
    flash_gemv.flag_gemv_gemv_atomic(input, weight_t, wup_t, wdown, output, threshold)
    output = output.half()
    torch.cuda.synchronize()

def bench_flag_gemv_gemv_local(B, threshold):
    torch.manual_seed(0)
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight_t = torch.rand(14336, 4096).half().cuda() - 0.5
    wup_t = torch.rand(14336, 4096).half().cuda() - 0.5
    wdown_t = torch.rand(4096, 14336).half().cuda() - 0.5

    output = torch.zeros(1, B, 4096).half().cuda()
    flash_gemv.flag_gemv_gemv(input, weight_t, wup_t, wdown_t, output, threshold)
    torch.cuda.synchronize()

def bench_flag_gemv_gemv_local(B, threshold):
    torch.manual_seed(0)
    input = torch.randn(1, B, 4096).half().cuda() - 0.5
    weight_t = torch.rand(14336, 4096).half().cuda() - 0.5
    weight = weight_t.t().contiguous()
    wup_t = torch.rand(14336, 4096).half().cuda() - 0.5
    wup = wup_t.t().contiguous()
    wdown_t = torch.rand(4096, 14336).half().cuda() - 0.5

    output = torch.zeros(1, B, 4096).half().cuda()
    flash_gemv.flag_gemv_gemv(input, weight_t, wup_t, wdown_t, output, threshold)
    torch.cuda.synchronize()
    act_fn = nn.SiLU()
    torch.matmul(act_fn(torch.matmul(input, weight)) * (torch.matmul(input, wup)), weight)

def bench_flag_gemv_gemv_70b(B, threshold):
    torch.manual_seed(0)
    input = torch.randn(1, B, 8192).half().cuda() - 0.5
    weight_t = torch.rand(28672, 8192).half().cuda() - 0.5
    weight = weight_t.t().contiguous()
    wup_t = torch.rand(28672, 8192).half().cuda() - 0.5
    wup = wup_t.t().contiguous()
    wdown_t = torch.rand(8192, 28672).half().cuda() - 0.5
    wdown = wdown_t.t().contiguous()
    act_fn = nn.SiLU()
    golden = torch.matmul(act_fn(torch.matmul(input, weight)) * (torch.matmul(input, wup)), wdown)
    torch.cuda.synchronize()
    output = flash_gemv.flag_gemv_gemv_inner_bf16(input, weight_t, wup_t, wdown_t, threshold)
    torch.cuda.synchronize()
    act_fn = nn.SiLU()


if __name__ == '__main__':
    # test_flag()
    # bench_flag_gemv()
    # test_flag_gemv()
    # test_flag_gemv_gemv()
    # test_gemv_gemv_gemv()
    # test_flag_batch()
    # bench_flag_batch()
    # bench_flag()
    # test_flag_local()
    # bench_flag_local()
    # test_flag_gemv_local()
    # test_flag_gemv_gemv_batch()
    # test_flag_gemv_gemv_atomic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--ths", type=float, default=0.1)
    args = parser.parse_args()

    # bench_flag_gemv_gemv_atomic(args.B, args.ths)
    # test_flag_gemv_gemv_local()
    # bench_flag_gemv_gemv_local(args.B, args.ths)
    # test_flag_gemv_gemv_local_float()
    # test_flag_gemv_gemv_local_bf16()
    # bench_flag_gemv_gemv_local(args.B, args.ths)
    # bench_flag_gemv_gemv_70b(args.B, args.ths)
    test_flag_gemv_gemv_triton()