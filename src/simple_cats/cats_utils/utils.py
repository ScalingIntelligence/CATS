import logging
import warnings
import torch

from ..kernels.sparse_mm import gemv_gemv_triton, gemv_triton


logger = logging.getLogger(__name__)

def get_threshold(
    bin_edges: torch.tensor, histogram_counts: torch.tensor, sparsity_level: float
):  # Only for L1 Regularization
    assert (
        len(bin_edges.shape) == len(histogram_counts.shape) == 1
    ), "bin_edges and histogram are expected to be 1-dimensional."
    histogram_counts /= histogram_counts.sum()
    threshold_idx = torch.searchsorted(histogram_counts.cumsum(0), sparsity_level, side="right")

    return bin_edges[threshold_idx]
def _get_submodules(model, key):  # Copied from peft package github repo
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def model_class_to_pkg_str(model_obj):
    model_class = model_obj.__class__
    return model_class.__name__
    # raise NotImplementedError(f"{model_obj.__class__} is not implemented ")


def custom_forward_for_swiglu(self, x):
    assert "MLP" in self.__class__.__name__, "{} is not an MLP".format(
        self.__class__.__name__
    )
    warnings.warn(
        "Using custom forward for MLP. Note that only batch size <= 4 is possible."
    )
    beam_width, seq_len, _ = x.shape
    if seq_len > 1:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    threshold = self.act_fn.threshold
    return gemv_gemv_triton(
        x,
        self.act_fn(self.gate_proj(x)),
        self.up_proj.weight,
        self.wdown_t,
        threshold,
    )


# Kernel Method 1 (simple 2-layer mlp)
def custom_forward_for_simple_mlp(self, x):
    assert "MLP" in self.__class__.__name__, "{} is not an MLP".format(
        self.__class__.__name__
    )
    warnings.warn(
        "Using custom forward for MLP. Note that only batch size <= 4 is possible."
    )
    beam_width, seq_len, _ = x.shape
    if seq_len > 1:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    threshold = self.threshold
    return gemv_triton(
        x,
        self.weight_t,
        threshold,
    )


# Kernel method 2 (LLama like mlp)
def replace_mlp_with_gemv_gemv(
    target_module,
    target_name: str,
):
    """
    Replace the forward function of the target module (llama-like MLP module) with a custom forward function that uses gemv_gemv_triton.
    """
    if "mlp" in target_name:
        logger.info(f"Replacing {target_name} forward function with Gemv.")
        target_module.wdown_t = target_module.down_proj.weight.t().contiguous()
        setattr(
            target_module,
            "forward",
            lambda x: custom_forward_for_swiglu(target_module, x),
        )


def replace_mlp_with_gemv(
    target_module,
    target_name: str,
):
    """
    Replace the forward function of the target module (2-layer MLP module) with a custom forward function that uses gemv_triton.
    """
    if "mlp" in target_name:
        logger.info(f"Replacing {target_name} forward function with Gemv.")
        target_module.weight_t = target_module.wrapped_module.weight.t().contiguous()
        setattr(
            target_module,
            "forward",
            lambda x: custom_forward_for_swiglu(target_module, x),
        )

