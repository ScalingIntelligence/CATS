from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MistralMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import gc

from scipy.sparse.linalg import svds


class SparsePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = None

    def forward(self, x):
        return self.router(x)


def low_rank_approximation(
    linear_layer, act_func=nn.ReLU(), rank=64, init_svd: bool = True
):
    # Decompose the weight matrix of the layer
    # Solve "RuntimeError: "svd_cuda_gesvdj" not implemented for 'BFloat16'"
    weight_data = linear_layer.weight.T.data
    weight_data_type = weight_data.dtype
    weight_data = weight_data.float()

    # U, S, V = torch.svd(weight_data)
    if init_svd:
        U_approx, S_approx, V_approx = svds(
            np.array(weight_data),
            k=min(rank, weight_data.shape[1] - 1, weight_data.shape[0] - 1),
        )
        U_approx = torch.from_numpy(U_approx.copy())
        S_approx = torch.from_numpy(S_approx.copy()).unsqueeze(-1)
        V_approx = torch.from_numpy(V_approx.copy())
    else:
        sparse_predictor = SparsePredictor()
        sparse_predictor.router = nn.Sequential(
            nn.Linear(linear_layer.in_features, rank, bias=False),
            nn.Linear(rank, linear_layer.out_features, bias=True),
            act_func,
        )
        sparse_predictor = sparse_predictor.type(weight_data_type)
        return sparse_predictor

    # Take the top `rank` components
    # U_approx = U[:, :rank]
    # S_approx = torch.diag(S[:rank])
    # V_approx = V[:, :rank].t()

    # Create two new linear layers for low-rank approximation
    first_layer = nn.Linear(linear_layer.in_features, rank, bias=False)
    second_layer = nn.Linear(rank, linear_layer.out_features, bias=True)

    # Assign the low-rank matrices to the layers' weights
    first_layer.weight.data = U_approx.T.contiguous()
    second_layer.weight.data = (S_approx * V_approx).T.contiguous()

    # If the original linear layer had a bias, assign it to the second layer's bias
    if linear_layer.bias is not None:
        second_layer.bias.data = linear_layer.bias.data
    else:
        second_layer.bias.data *= 0

    sparse_predictor = SparsePredictor()
    sparse_predictor.router = nn.Sequential(first_layer, second_layer, act_func)
    sparse_predictor = sparse_predictor.type(weight_data_type)

    return sparse_predictor
