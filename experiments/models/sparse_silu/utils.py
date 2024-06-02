from transformers import TrainerCallback, Trainer
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from typing import Any, Dict, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral.modeling_mistral import MistralModel

from experiments.models.sparse_mistral.sparse_silu import *
from experiments.models.sparse_llama.sparse_silu import *
from utils.utils import is_running_deepspeed, is_mainprocess, ds_print, get_model_type, get_model_type_from_name
from utils.constants import MISTRAL


def get_mlp_class(model):
    model_type = get_model_type(model)
    return MistralSparseSiluMLP if model_type == MISTRAL else LlamaSparseSiluMLP


def get_decoder_class(model):
    model_type = get_model_type(model)
    return SparseMistralDecoderLayer if model_type == MISTRAL else LlamaSparseDecoderLayer


def get_model_class(model):
    model_type = get_model_type(model)
    return MistralModel if model_type == MISTRAL else LlamaModel


class SparseSiLU(nn.SiLU):
    def __init__(self, threshold):
        super(SparseSiLU, self).__init__()
        self.threshold = threshold
        self.m = nn.Threshold(self.threshold, 0)

    def set_new_threshold(self, threshold):
        self.threshold = threshold
        self.m = nn.Threshold(threshold, 0)

    def forward(self, x):
        act = super(SparseSiLU, self).forward(x)
        return self.m(act) - self.m(-act)


def get_sparse_config(
    config: PretrainedConfig,
    model_type: str = None,
    use_sparse_model=False,
    use_sparse_predictor=False,
    use_sparse_regularization=False,
    use_graceful_regularization=False,
    thresholds=None,
):
    if model_type == MISTRAL:
        new_config = SparseMistralConfig()
    else:
        new_config = SparseLlamaConfig()
    new_config.__dict__.update(config.__dict__)
    config = new_config
    config.use_sparse_model = use_sparse_model
    config.use_sparse_predictor = use_sparse_predictor
    config.use_sparse_regularization = use_sparse_regularization
    config.use_graceful_regularization = use_graceful_regularization
    config.thresholds = thresholds

    return config


def apply_sparse_silu_mlp(
    model,
    config,
    use_sparse_regularization: bool = False,
):
    SparseMLP = get_mlp_class(model)
    for layer in model.model.layers:
        original_mlp = layer.mlp
        new_mlp = SparseMLP(config, use_sparse_regularization=use_sparse_regularization)
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        layer.mlp = new_mlp


def apply_sparse_decoder_layer(
    model,
    config,
    init_svd: bool = True,
):
    Model = get_model_type(model)
    SparseMLP = get_mlp_class(model)
    DecoderLayer = get_decoder_class(model)

    assert isinstance(model.model, Model), "model.model must be a MistralModel."
    new_layers = []
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, SparseMLP):
            new_layers.append(
                DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    decoder_layer=layer,
                    init_svd=init_svd,
                )
            )
            print(f"{layer_idx}th mlp layer activation: {layer.mlp.sparse_act_fn}")
        else:
            new_layers.append(layer)
    model.model.layers = nn.ModuleList(new_layers)


def enable_sparse_predictor(
    model,
):
    DecoderLayer = get_decoder_class(model)
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer, DecoderLayer):
            layer.use_sparse_predictor = True


def disable_sparse_predictor(
    model,
):
    DecoderLayer = get_decoder_class(model)
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer, DecoderLayer):
            layer.use_sparse_predictor = False


def activate_stats(model, model_type: str = None, is_collect_histogram: bool = True):
    SparseMLP = get_mlp_class(model)
    for layer in model.model.layers:
        if isinstance(layer.mlp, SparseMLP):
            layer.mlp.activate_stats(is_collect_histogram=is_collect_histogram)


def deactivate_stats(
    model,
):
    SparseMLP = get_mlp_class(model)
    for layer in model.model.layers:
        if isinstance(layer.mlp, SparseMLP):
            layer.mlp.deactivate_stats()


def enable_sparse_silu(model):
    print("Enabling SparseSilu")
    SparseMLP = get_mlp_class(model)
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, SparseMLP):
            layer.mlp.kill_sparse_swish_outputs = True


def disable_sparse_silu(model):
    print("Disabling SparseSilu")
    SparseMLP = get_mlp_class(model)
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, SparseMLP):
            layer.mlp.kill_sparse_swish_outputs = False


def print_dead_neuron_stats(model):
    SparseMLP = get_mlp_class(model)
    total_sparsity = 0
    counts = 0
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, SparseMLP):
            dead_percentage = layer.mlp.dead_percentage * 100
            agg_sparsity = layer.mlp.agg_sparsity * 100
            print(f"layer {i} sparsity: {dead_percentage:.3f}%")
            print(f"layer {i} agg sparsity: {agg_sparsity:.3f}%")
            total_sparsity += dead_percentage
            counts += 1

    print(f"Total sparsity: {total_sparsity/counts: .3f}%")
    return total_sparsity / counts


def get_sparse_layers(model):
    SparseMLP = get_mlp_class(model)
    sparse_layers = [m.mlp for m in model.layers() if isinstance(m.mlp, SparseMLP)]
    return sparse_layers


def get_threshold(
    bin_edges: torch.tensor, histogram_counts: torch.tensor, sparsity_level: float
):  # Only for L1 Regularization
    assert (
        len(bin_edges.shape) == len(histogram_counts.shape) == 1
    ), "bin_edges and histogram are expected to be 1-dimensional."
    histogram_counts /= histogram_counts.sum()
    threshold_idx = torch.searchsorted(histogram_counts.cumsum(0), sparsity_level, side="right")

    return bin_edges[threshold_idx]


def set_regularization_threshold(model, threshold: float = 0.1):
    SparseMLP = get_mlp_class(model)
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, SparseMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            layer.mlp.regularization_threshold = threshold  # TODO: find better param


def set_sparse_threshold(model, sparsity_level: float, use_relu: bool = False):
    SparseMLP = get_mlp_class(model)
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, SparseMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            if use_relu:
                layer.mlp.sparse_act_fn = nn.ReLU()
                layer.mlp.use_relu = True
            else:
                layer.mlp.dead_threshold = get_threshold(
                    layer.mlp.histogram_bins,
                    layer.mlp.post_act_hist_counts,
                    sparsity_level,
                )
                layer.mlp.sparse_act_fn.set_new_threshold(layer.mlp.dead_threshold)
                layer.mlp.regularization_threshold = layer.mlp.dead_threshold * 1.2  # TODO: find better param


def plot_histogram(
    bin_edges,
    histogram_counts: torch.tensor,
    title: str = "Activation Distribution",
    fig_dir: str = "figures",
):
    plt.bar(bin_edges[:-1], histogram_counts, width=np.diff(bin_edges), edgecolor="black")
    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{title}.png")
    # plt.show()
    plt.clf()


def plot_activation_histogram(model, fig_dir: str = "figures"):
    SparseMLP = get_mlp_class(model)

    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, SparseMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            plot_title = f"Layer: {i} Pre-Activation Distribution"
            plot_histogram(layer.mlp.histogram_bins, layer.mlp.pre_act_hist_counts, plot_title)

            plot_title = f"Layer: {i} Post-Activation Absolute Distribution"
            plot_histogram(layer.mlp.histogram_bins, layer.mlp.post_act_hist_counts, plot_title)


def save_act_hist(model, filename="/scr/jay/models/mistral/pre_finetune/cola_act_hist.pt"):
    SparseMLP = get_mlp_class(model)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    act_dict = {}
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, SparseMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            act_dict[i] = (
                layer.mlp.histogram_bins,
                layer.mlp.pre_act_hist_counts,
                layer.mlp.post_act_hist_counts,
            )
    print("Saving activation histograms...\n\n\n")
    torch.save(act_dict, filename)


def load_act_hist(model, filename="/scr/jay/models/mistral/pre_finetune/cola_act_hist.pt"):
    assert os.path.exists(
        filename
    ), f"{filename} does not exist when loading pre/post-activation histogram of SparseMistralSiluMLP."
    SparseMLP = get_mlp_class(model)

    print("Loading activation histograms...\n\n\n")

    act_dict = torch.load(filename)
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, SparseMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            (
                layer.mlp.histogram_bins,
                layer.mlp.pre_act_hist_counts,
                layer.mlp.post_act_hist_counts,
            ) = act_dict[i]


def enable_last_k_modules(model, start_module_idx: int):
    assert 32 > start_module_idx >= 0
    new_modules = []
    new_idx = 0
    for idx in range(start_module_idx, len(model.model.original_layers)):
        module = model.model.original_layers[idx]
        module.layer_idx = new_idx
        module.self_attn.layer_idx = new_idx
        new_modules.append(module)
        new_idx += 1
        print(module.layer_idx)

    model.model.layers = nn.ModuleList(new_modules)


def enable_first_k_modules(model, end_module_idx: int):
    assert 32 > end_module_idx >= 0
    new_modules = []
    new_idx = 0
    for idx in range(0, end_module_idx + 1):
        module = model.model.original_layers[idx]
        module.layer_idx = new_idx
        module.self_attn.layer_idx = new_idx
        new_modules.append(module)
        new_idx += 1
        print(module.layer_idx)

    model.model.layers = nn.ModuleList(new_modules)
