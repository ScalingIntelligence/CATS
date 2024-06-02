import torch
import torch.nn as nn
from scipy.stats.mstats import gmean

ACT2FN = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "swish": nn.SiLU(),
}


def benchmark(fn, input, Wgate, Wup, Wdown, idx, activation="swish", warmup=20, rep=80, quantiles=None, fast_flush=True):
    # https://github.com/nox-410/tvm.tl/blob/tl/python/tvm/tl/utils.py#L144    
    fn(input, Wgate, Wup, Wdown, idx, activation)
    torch.cuda.synchronize()

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn(input, Wgate, Wup, Wdown, idx, activation)
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn(input, Wgate, Wup, Wdown, idx, activation)
    
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        start_event[i].record()
        fn(input, Wgate, Wup, Wdown, idx, activation)
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    times_list = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return gmean(times_list)

def benchmark_pred(fn, input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K, warmup=20, rep=80, quantiles=None, fast_flush=True, return_mode="mean"):
    # https://github.com/nox-410/tvm.tl/blob/tl/python/tvm/tl/utils.py#L144    
    fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
    torch.cuda.synchronize()

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
    
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        start_event[i].record()
        fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()

class Config:
    def __init__(self, hidden_size, intermediate_size, rank, K, dtype, hidden_act='swish', svd_act='relu'):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank
        self.K = K
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.svd_act = svd_act

class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        l1 = self.gate_proj(x)
        print(l1.dtype)
        return self.down_proj(self.act_fn(l1) * self.up_proj(x))

class MistralSparseMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.rank = config.rank
        self.K = config.K
        self.u_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.sv_proj = nn.Linear(self.rank, self.intermediate_size, bias=True)
        self.svd_act_fn = ACT2FN[config.svd_act]

    def forward(self, x):
        probs = self.svd_act_fn(self.sv_proj(self.u_proj(x)))
        self.probs = probs[0]
        _, topk_idcs = torch.topk(self.probs, k=self.K, dim=-1)
        self.topk_idcs, _ = torch.sort(topk_idcs) # [0] to make i compatible with deja vu
        mask = torch.zeros(
            self.intermediate_size,
            dtype=probs.dtype,
            device=probs.device,
        ).scatter(-1, self.topk_idcs, 1)
        masked_up = (self.act_fn(self.gate_proj(x)) * self.up_proj(x) * mask)
        return self.down_proj(masked_up)
    
def degrad_tensor(x):
    x.grad = None
    x.requires_grad = False
    return x