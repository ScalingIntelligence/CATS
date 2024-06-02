import argparse
import torch.nn as nn
import torch
from utils import benchmark
import flash_gemv

def mistral_direct_mlp_optimal(x, Wgate, Wup, Wdownt, threshold, act_fn):
    return torch.matmul((act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup)), Wdownt)

def mistral_direct_baseline(x, Wgate, Wup, Wdownt, threshold, act_fn):
    return flash_gemv.flag_gemv_gemv_dejavu(x, Wgate, Wup, Wdownt, threshold)

def mistral_direct_mlp(x, Wgate, Wup, Wdownt, threshold, act_fn):
    return flash_gemv.flag_gemv_gemv_fuse_dejavu(x, Wgate, Wup, Wdownt, threshold)

def mistral_direct_mlp_triton(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    return flash_gemv.gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold)
    # return mistral_mlp_sparse_direct_notranspose(x, x_1, Wup, Wdownt, torch.abs(x_1) > threshold)


def get_k_mistral_direct_mlp(x, Wgate, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return idx.shape[-1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ths", type=float, default=0.1)
    parser.add_argument("--cuths", type=float, default=0.1)
    parser.add_argument("--filename", type=str, default="mistral_direct.csv")

    args = parser.parse_args()

    torch.manual_seed(0)
    THS = args.cuths
    hidden_size = 4096
    intermediate_size = 14336
    B = 1
    hidden_act = "swish"
    svd_act = "swish"
    dtype = torch.float32
    input = torch.rand((B, hidden_size), dtype=dtype).cuda() - 0.5
 
    gate = torch.rand(hidden_size, intermediate_size, dtype=dtype).cuda() - 0.5
    up = torch.rand(hidden_size, intermediate_size, dtype=dtype).cuda() - 0.5
    down = torch.rand(intermediate_size, hidden_size, dtype=dtype).cuda() - 0.5
    gate_t = gate.t().contiguous()
    up_t = up.t().contiguous()
    down_t = down.t().contiguous()
    value = torch.zeros((B, intermediate_size), dtype=dtype).cuda()
    flag = torch.zeros((intermediate_size), dtype=torch.bool).cuda()
    
    k_fp32 = get_k_mistral_direct_mlp(input, gate, args.ths, nn.SiLU())
    # torch.cuda.synchronize()
    
    # mistral_direct_mlp(input, gate, up_t, down, args.ths, hidden_act)
    time_baseline = benchmark(mistral_direct_baseline, input, gate, up_t, down, args.ths, nn.SiLU())
    time_direct = benchmark(mistral_direct_mlp, input, gate, up_t, down, args.ths, nn.SiLU())
    input_3d = input.unsqueeze(1).contiguous()
    time_direct_triton = benchmark(mistral_direct_mlp_triton, input_3d, gate, up_t, down, args.ths, nn.SiLU())
    time_dense_mlp = benchmark(mistral_direct_mlp_optimal, input, gate, up, down, args.ths, nn.SiLU())

    gate = torch.rand((hidden_size, k_fp32), dtype=dtype).cuda() - 0.5
    up = torch.rand((hidden_size, k_fp32), dtype=dtype).cuda() - 0.5
    down = torch.rand((k_fp32, hidden_size), dtype=dtype).cuda() - 0.5
    time_direct_optimal = benchmark(mistral_direct_mlp_optimal, input, gate, up, down, args.ths, nn.SiLU())
    print(f"[{k_fp32}]Dense: {time_dense_mlp}, Baseline: {time_baseline}, Direct: {time_direct}, Triton: {time_direct_triton}, Optimal: {time_direct_optimal}")
    with open(args.filename, "a") as f:
        print(f"{1-k_fp32/intermediate_size},{k_fp32},{time_dense_mlp},{time_baseline},{time_direct},{time_direct_triton},{time_direct_optimal}", file=f)

