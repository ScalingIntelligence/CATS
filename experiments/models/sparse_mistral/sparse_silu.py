from transformers import TrainerCallback, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel
from datasets import Dataset
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from typing import Any, Dict, Union, Optional, Tuple
from torch.nn import MSELoss
from transformers.utils import is_flash_attn_2_available, logging
import inspect
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralAttention,
    MistralModel,
    MistralDecoderLayer,
    MistralConfig,
    MISTRAL_ATTENTION_CLASSES,
    MistralRMSNorm,
    MistralForCausalLM,
    MistralFlashAttention2,
)
from experiments.models.sparse_mistral.svd_router import (
    low_rank_approximation,
    SparsePredictor,
)
from utils.utils import (
    print_size_of_model,
    is_running_deepspeed,
    is_mainprocess,
    get_datetime,
    ds_print,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
logger = logging.get_logger(__name__)


class SparseSFTTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop("regularization_coefficient", 10)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", False)
        self.use_spm_loss = False
        self.freeze_original_weights = False
        self.regularization_type = kwargs.pop("regularization_type", "L1 positive activation")
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        self.sparse_decoder_layers = []
        super(SparseSFTTTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        self.sparse_layers = [m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)]

    def initialize_sparse_decoder_layers(self, model):
        self.sparse_decoder_layers = [m for m in model.modules() if isinstance(m, SparseMistralDecoderLayer)]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Override the huggingface's training_step function to add a regularization term.
        A regularization term is computed with intermediate values, which are freed after "backward()."
        You need to set `retain_graph=True` inside `backward` function to keep the values.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if not self.freeze_original_weights:
            if loss is not None:
                self.accelerator.backward(loss, retain_graph=False)

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            if regularization_loss is not None:
                self.accelerator.backward(regularization_loss, retain_graph=True)
            loss += regularization_loss

        if self.use_spm_loss:
            spm_loss = self.compute_spm_loss(model)
            if self.args.n_gpu > 1:
                spm_loss = spm_loss.mean()
            if spm_loss is not None:
                self.accelerator.backward(spm_loss, retain_graph=False)
            loss += spm_loss

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        if len(self.sparse_layers) == 0:
            self.initialize_sparse_silu_layers(model)
        num_layers = len(self.sparse_layers)

        for module in self.sparse_layers:
            if module.activation_norm is not None:
                loss += module.activation_norm

        loss /= num_layers
        loss *= self.regularization_coefficient

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss

    def compute_spm_loss(self, model):
        loss = 0
        if len(self.sparse_decoder_layers) == 0:
            self.initialize_sparse_decoder_layers(model)
        for module in self.sparse_decoder_layers:
            if module.distill_loss != None:
                loss += module.distill_loss
        if self.state.global_step % 20 == 0 and loss != 0:
            print("Sparse Predictor Distillation loss: ", loss.item())
        return loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     loss = super().compute_loss(model, inputs, return_outputs)
    #
    #     if is_sagemaker_mp_enabled():
    #         import smdistributed.modelparallel.torch as smp
    #         @smp.step()
    #         def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
    #             outputs = model(**inputs)
    #             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #             loss /= gradient_accumulation_steps
    #             model.backward(loss)
    #             return loss
    #
    #         loss_mb = smp_forward_backward(
    #             model, inputs, self.args.gradient_accumulation_steps
    #         )
    #         if self.use_sparse_regularization:
    #             return loss_mb.reduce_mean().detach().to(
    #                 self.args.device
    #             ) + self.regularization_coefficient * self.compute_regularization(model)
    #         else:
    #             return loss_mb.reduce_mean().detach().to(self)
    #
    #     if return_outputs:
    #         classification_loss, outputs = loss
    #     else:
    #         classification_loss = loss
    #
    #     loss = classification_loss
    #     if self.use_sparse_regularization:
    #         regularization_loss = self.compute_regularization(model)
    #         loss += self.regularization_coefficient * regularization_loss
    #
    #     return (loss, outputs) if return_outputs else loss


class SparseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop("regularization_coefficient", 10)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", False)
        self.use_spm_loss = False
        self.freeze_original_weights = False
        self.regularization_type = kwargs.pop("regularization_type", "L1 positive activation")
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        self.sparse_decoder_layers = []
        super(SparseTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        self.sparse_layers = [m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)]

    def initialize_sparse_decoder_layers(self, model):
        self.sparse_decoder_layers = [m for m in model.modules() if isinstance(m, SparseMistralDecoderLayer)]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Override the huggingface's training_step function to add a regularization term.
        A regularization term is computed with intermediate values, which are freed after "backward()."
        You need to set `retain_graph=True` inside `backward` function to keep the values.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if not self.freeze_original_weights:
            if loss is not None:
                self.accelerator.backward(loss, retain_graph=False)

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            if regularization_loss is not None:
                self.accelerator.backward(regularization_loss, retain_graph=True)
            loss += regularization_loss

        if self.use_spm_loss:
            spm_loss = self.compute_spm_loss(model)
            if self.args.n_gpu > 1:
                spm_loss = spm_loss.mean()
            if spm_loss is not None:
                self.accelerator.backward(spm_loss, retain_graph=False)
            loss += spm_loss

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        if len(self.sparse_layers) == 0:
            self.initialize_sparse_silu_layers(model)
        num_layers = len(self.sparse_layers)

        for module in self.sparse_layers:
            if module.activation_norm is not None:
                loss += module.activation_norm

        loss /= num_layers
        loss *= self.regularization_coefficient

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss

    def compute_spm_loss(self, model):
        loss = 0
        if len(self.sparse_decoder_layers) == 0:
            self.initialize_sparse_decoder_layers(model)
        for module in self.sparse_decoder_layers:
            if module.distill_loss != None:
                loss += module.distill_loss
        if self.state.global_step % 20 == 0 and loss != 0:
            print("Sparse Predictor Distillation loss: ", loss.item())
        return loss


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class SparseMistralFlashAttention(MistralFlashAttention2):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = 0
        self.pre_attn_sparsity = 0
        self.visit_counts = 0
        self.is_stats = False
        self.pre_attn_std = 0
        self.pre_attn_threshold = 0

        # Activation Histograms
        self.is_collect_histogram = False
        num_bins = 20000
        self.num_bins = num_bins
        self.hist_min = -2
        self.hist_max = 2
        self.histogram_bins = torch.linspace(self.hist_min, self.hist_max, num_bins - 2)
        self.histogram_bins = torch.cat([torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])])
        self.pre_mlp_std = 0
        self.pre_mlp_hist_counts = torch.zeros(num_bins - 1)
        self.pre_act_hist_counts = torch.zeros(num_bins - 1)
        self.post_act_hist_counts = torch.zeros(num_bins - 1)

    def activate_stats(self):
        self.is_stats = True
        self.visit_counts = 0
        # self.pre_attn_sparsity = 0
        self.pre_attn_std = 0

    def deactivate_stats(self):
        self.is_stats = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()
        mask = abs(hidden_states - hidden_states.mean()) < self.pre_attn_threshold
        hidden_states[mask] = 0
        self.counts += 1

        if self.is_stats:
            self.pre_attn_sparsity = (
                self.pre_attn_sparsity * self.visit_counts + (hidden_states == 0).float().mean()
            ) / (self.visit_counts + 1)
            self.pre_attn_std = (self.pre_attn_std * self.visit_counts + 0.5 * hidden_states.std()) / (
                self.visit_counts + 1
            )
            self.visit_counts += 1
            self.counts -= 1

        if self.counts == 10:
            print(f"Attention {self.layer_idx}: ", (hidden_states == 0).float().mean())
            print(
                mask.shape,
            )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class SparseMistralAttention(MistralAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = 0
        self.pre_attn_sparsity = 0
        self.visit_counts = 0
        self.is_stats = False
        self.pre_attn_std = 0
        self.pre_attn_threshold = 0

        # Activation Histograms
        self.is_collect_histogram = False
        num_bins = 20000
        self.num_bins = num_bins
        self.hist_min = -2
        self.hist_max = 2
        self.histogram_bins = torch.linspace(self.hist_min, self.hist_max, num_bins - 2)
        self.histogram_bins = torch.cat([torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])])
        self.pre_mlp_std = 0
        self.pre_attn_hist_counts = torch.zeros(num_bins - 1)
        self.post_qk_hist_counts = torch.zeros(num_bins - 1)

    def activate_stats(self):
        self.is_stats = True
        self.visit_counts = 0
        self.pre_attn_sparsity = 0
        self.pre_attn_std = 0

    def deactivate_stats(self):
        self.is_stats = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        mask = abs(hidden_states - hidden_states.mean()) < self.pre_attn_threshold
        hidden_states[mask] = 0

        if self.is_stats:
            self.pre_attn_hist_counts += torch.cat(
                (
                    (hidden_states < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        hidden_states.float(),
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (hidden_states > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()

        self.counts += 1
        if self.counts == 10:
            print(f"Attention {self.layer_idx}: {float((hidden_states == 0).float().mean()) * 100 : .3f}")
            self.counts += 1

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        if self.is_stats:
            self.post_qk_hist_counts += torch.cat(
                (
                    (attn_weights < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        attn_weights.float(),
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (attn_weights > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralSparseSiluMLP(MistralMLP):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.swish_outputs = None
        self.relu = nn.ReLU()
        self.resilu = nn.Sequential(nn.SiLU())

        self.kill_sparse_swish_outputs = False
        self.cut_pre_mlp = False
        self.dead_percentage = 0
        self.pre_mlp_sparsity = 0
        self.is_stats = False
        self.visit_counts = 0
        self.is_profile = False

        # Hyperparameters to tune
        self.dead_threshold = kwargs.pop("dead_threshold", 0)
        self.pre_mlp_threshold = kwargs.pop("pre_mlp_threshold", 0)
        self.pre_mlp_dead_threshold = kwargs.pop("pre_mlp_dead_threshold", 0)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", True)
        self.regularization_type = kwargs.pop("regularization_type", "L1 regularization")
        self.regularization_threshold = kwargs.pop("regularization_threshold", 0.5)
        self.use_relu = kwargs.pop("use_relu", False)
        self.use_resilu = kwargs.pop("use_resilu", False)
        self.activation_norm = None

        # Activation Histograms
        self.is_collect_histogram = False
        num_bins = 20000
        self.num_bins = num_bins
        self.hist_min = -2
        self.hist_max = 2
        self.histogram_bins = torch.linspace(self.hist_min, self.hist_max, num_bins - 2)
        self.histogram_bins = torch.cat([torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])])
        self.pre_mlp_std = 0
        self.pre_mlp_hist_counts = torch.zeros(num_bins - 1).to(self.gate_proj.weight.device)
        self.pre_act_hist_counts = torch.zeros(num_bins - 1).to(self.gate_proj.weight.device)
        self.post_act_hist_counts = torch.zeros(num_bins - 1).to(self.gate_proj.weight.device)
        self.t = 0
        self.count = 0
        self.agg_sparsity = 0

        # Sparse activation function
        self.sparse_act_fn = SparseSiLU(threshold=self.dead_threshold)

    def activate_stats(self, is_collect_histogram: bool = True):
        self.is_stats = True
        self.dead_percentage = 0
        self.visit_counts = 0
        self.is_collect_histogram = is_collect_histogram
        self.histogram_counts = torch.zeros(2000)  # .to(self.down_proj.weight.device)

    def deactivate_stats(self):
        self.is_stats = False

    def collect_stats(
        self,
        pre_mlp,
        pre_activation,
        post_activation,
    ):
        start_time = time.time()
        pre_mlp = pre_mlp.float()
        pre_activation = pre_activation.float()
        post_activation = torch.abs(post_activation.float())
        # self.histogram_bins=self.histogram_bins.to(pre_activation.device).type(pre_activation.dtype)
        # self.pre_mlp_hist_counts = torch.histogram(pre_mlp, bins=self.histogram_bins)[0]
        if torch.cuda.is_available():
            self.pre_mlp_hist_counts += torch.cat(
                (
                    (pre_mlp < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        pre_mlp,
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (pre_mlp > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()
            self.pre_act_hist_counts += torch.cat(
                (
                    (pre_activation < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        pre_activation,
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (pre_activation > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()
            if torch.cuda.is_available():
                self.post_act_hist_counts += torch.cat(
                    (
                        (post_activation < self.hist_min).sum().unsqueeze(0),
                        torch.histc(
                            post_activation,
                            bins=self.num_bins - 3,
                            min=self.hist_min,
                            max=self.hist_max,
                        ),
                        (pre_activation > self.hist_max).sum().unsqueeze(0),
                    )
                ).cpu()
        else:
            self.pre_mlp_hist_counts = torch.histogram(pre_mlp, bins=self.histogram_bins)[0]
            self.pre_act_hist_counts += torch.histogram(pre_activation, bins=self.histogram_bins)[0]
            self.post_act_hist_counts += torch.histogram(post_activation, bins=self.histogram_bins)[0]

        self.t += time.time() - start_time
        if self.visit_counts % 30 == 0:
            print(f"Time taken to collect stats: {self.t}s.")

    def forward(
        self,
        x,
        sp_mask: torch.tensor = None,
    ):
        """
        If kill_sparse_swish_outputs is set to False, this layer functions exactly like a normal MLP layer.
        """
        if sp_mask != None:  # When sparse mask is given
            return self.down_proj(
                self.sparse_act_fn(self.gate_proj(x) * sp_mask) * self.up_proj(x)
            )  # Todo: This doesn't accelerate runtime (instead slowing down)

        elif self.use_relu or self.use_resilu:
            if self.use_relu:
                post_act = self.relu(self.gate_proj(x))
            else:
                post_act = self.resilu(self.gate_proj(x))
            self.count += 1

            if self.is_stats:
                dead_neurons = post_act == 0
                dead_percentage = dead_neurons.float().mean()
                agg_sparsity = dead_neurons.all(dim=0).float().mean()

                self.dead_percentage = (self.dead_percentage * self.visit_counts + dead_percentage) / (
                    self.visit_counts + 1
                )
                self.agg_sparsity = (self.agg_sparsity * self.visit_counts + agg_sparsity) / (self.visit_counts + 1)
                self.visit_counts += 1

            return self.down_proj(post_act * self.up_proj(x))

        else:
            self.count += 1

            if self.cut_pre_mlp:
                if (
                    self.is_stats
                ):  # collect statistics for deciding threhold value to cut values of hidden vec before mlp
                    self.pre_mlp_std = (x.std() * 0.6 + self.visit_counts * self.pre_mlp_std) / (self.visit_counts + 1)
                    self.count -= 1
                x[abs(x) < self.pre_mlp_threshold] = 0

            pre_act = self.gate_proj(x)
            post_act = self.act_fn(pre_act)
            if self.kill_sparse_swish_outputs:
                dead_neurons = post_act.abs() <= self.dead_threshold
                # print("pre act sparsity: ", (pre_act==0).float().mean())

                dead_percentage = dead_neurons.float().mean()
                agg_sparsity = dead_neurons.all(dim=0).float().mean()

                if self.is_stats:
                    self.dead_percentage = (self.dead_percentage * self.visit_counts + dead_percentage) / (
                        self.visit_counts + 1
                    )
                    self.agg_sparsity = (self.agg_sparsity * self.visit_counts + agg_sparsity) / (self.visit_counts + 1)
                    self.pre_mlp_sparsity = (self.pre_mlp_sparsity * self.visit_counts + (x == 0).float().mean()) / (
                        self.visit_counts + 1
                    )

                    self.visit_counts += 1

                    self.a = dead_percentage

                    # print(self.agg_sparsity)

                    # Collect histogram stats
                    if self.is_collect_histogram and pre_act.eq(0).float().mean() < 0.99:  # Padded dataset
                        self.collect_stats(x, pre_act, post_act)

                post_act[dead_neurons] = 0

            out = self.down_proj(post_act * self.up_proj(x))
            if self.use_sparse_regularization:
                if self.regularization_type == "L1 regularization":
                    self.activation_norm = torch.abs(post_act)[post_act < self.regularization_threshold].mean()
                elif self.regularization_type == "L2 regularization":
                    self.activation_norm = torch.sqrt(
                        torch.square(post_act)[post_act < self.regularization_threshold]
                    ).mean()

            return out


class SparseMistralDecoderLayer(MistralDecoderLayer):
    def __init__(
        self,
        config: MistralConfig,
        layer_idx: int,
        decoder_layer: MistralDecoderLayer,
        init_svd: bool = True,
        *args,
        **kwargs,
    ):
        assert isinstance(
            decoder_layer.mlp, MistralSparseSiluMLP
        ), f"{type(decoder_layer.mlp)} should MistralSparseSiluMLP."

        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.init_svd = init_svd
        self.self_attn = decoder_layer.self_attn

        self.mlp = decoder_layer.mlp
        self.input_layernorm = decoder_layer.input_layernorm
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm

        # Sparse predictor for mlp (initialized with SVD decomposed matrix)
        self.low_rank = kwargs.pop("low_rank", 64)
        self.sparse_act_func = decoder_layer.mlp.sparse_act_fn

        print(f"Setting {layer_idx}th mlp layer's sparse predictor... svd init: {init_svd}")
        self.sp_mlp = low_rank_approximation(
            decoder_layer.mlp.gate_proj,
            act_func=self.sparse_act_func,
            init_svd=init_svd,
        )
        self.use_async = kwargs.pop("use_async", False)
        self.use_sparse_predictor = False
        self.distill_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        print("hidden_states shape: ", hidden_states.shape)
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        sp_mask = None

        if self.use_async:
            sp_mask = self.sp_mlp(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if not self.use_async:
            sp_mask = self.sp_mlp(hidden_states)

        # Compute distillation loss
        gating_output = self.mlp.sparse_act_fn(self.mlp.gate_proj(hidden_states))
        loss_func = MSELoss()
        self.distill_loss = loss_func(sp_mask, gating_output)

        # Convert sp mask into binary form
        sp_mask = sp_mask > 0

        if self.training:
            sp_mask = None
        # if not self.use_sparse_predictor:
        #     sp_mask = None

        hidden_states = self.mlp(hidden_states, sp_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class SparseMistralConfig(MistralConfig):
    model_type = "sparse_mistral"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SparseMistralforCausalLM(MistralForCausalLM):
    config_class = SparseMistralConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.use_sparse_model:
            self.apply_sparse_mlp()
            if config.thresholds is not None:
                for idx, m in enumerate(self.model.layers):
                    if isinstance(m.mlp, MistralSparseSiluMLP):
                        m.mlp.dead_threshold = config.thresholds[idx]
                        m.mlp.pre_mlp_threshold = getattr(config, "pre_mlp_thresholds", [0] * len(self.model.layers))[
                            idx
                        ]
                        m.mlp.sparse_act_fn.set_new_threshold(m.mlp.dead_threshold)
                        m.mlp.kill_sparse_swish_outputs = True
                        m.mlp.use_relu = getattr(config, "use_relu", False)
                        m.mlp.use_resilu = getattr(config, "use_resilu", False)
                    if isinstance(
                        m.self_attn,
                        (SparseMistralAttention, SparseMistralFlashAttention),
                    ):
                        m.self_attn.pre_attn_threshold = config.pre_attn_thresholds[idx]
        if config.use_sparse_predictor:
            self.apply_sparse_predictor(init_svd=config.init_svd)

    def apply_sparse_mlp(self):
        apply_mistral_sparse_silu_mlp(
            self,
            config=self.config,
            use_sparse_regularization=self.config.use_sparse_regularization,
            cut_pre_mlp=getattr(self.config, "cut_pre_mlp", False),
            cut_pre_attn=getattr(self.config, "cut_pre_attn", False),
        )

    def apply_sparse_predictor(self, init_svd: bool = True):
        apply_mistral_sparse_decoder_layer(self, config=self.config, init_svd=init_svd)


class GracefulRegularizationScheduler(TrainerCallback):
    def __init__(
        self,
        num_warmup_steps=40,
        is_enabled: bool = False,
        model_name: str = "mistral",
        test_dataset: Dataset = None,
        targeted_sparsity: float = 0.5,
        keep_regularization_with_kill: bool = False,
    ):
        """Scheduler for regularizing the model first before applying the dead threshold.

        :param num_warmup_steps: number of training steps required to reach the dead threshold, defaults to 40
        :param increment_ratio: by how much to increase the dead threshold.
            For example, 0.5 means "increase the threshold by 0.5 * desired threshold
        """
        self.num_warmup_steps = num_warmup_steps
        self.is_enabled = is_enabled
        self.model_name = model_name
        self.test_dataset = test_dataset
        self.targeted_sparsity = targeted_sparsity
        self.keep_regularization_with_kill = keep_regularization_with_kill
        self.act_hist_path = f"/matx/u/vxbrando/histograms/warm_up_reg_{targeted_sparsity}/act_hist.pt"
        if self.is_enabled:
            print("GracefulRegularizationScheduler is enabled.")
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_enabled:
            return

        model = kwargs["model"]
        if isinstance(model, PeftModel):
            base_model = model.get_base_model()
        else:
            base_model = model

        if state.global_step == 1:
            ds_print("Setting an initial reg threshold to 0.1")
            set_regularization_threshold(base_model, 0.1)

        # if state.global_step >= self.num_warmup_steps and state.global_step % 50 == 0:
        if state.global_step == self.num_warmup_steps:
            activate_stats(base_model)
            enable_sparse_silu(base_model)
            self.trainer.evaluate()
            save_act_hist(base_model, self.act_hist_path)
            set_sparse_threshold(base_model, self.targeted_sparsity, True)
            deactivate_stats(base_model)
            self.trainer.use_sparse_regularization = self.keep_regularization_with_kill
            # set_layer_specific_regularization(model.get_base_model())
            print_dead_neuron_stats(model.get_base_model())

        if state.global_step % 2000 == 0:
            if is_mainprocess():
                ds_print(
                    f"Saving to /scr/lukeai/{self.model_name}_{state.global_step}.pt",
                )
                torch.save(
                    model.state_dict(),
                    f"/scr/lukeai/{self.model_name}_{state.global_step}.pt",
                )


class GradualSparsificationScheduler(TrainerCallback):
    def __init__(
        self,
        num_warmup_steps=40,
        increment_ratio=0.5,
        is_enabled: bool = False,
        model_name: str = "mistral",
    ):
        """Scheduler for gradually increasing a dead threshold until it reaches the desired threshold.

        :param num_warmup_steps: number of training steps required to reach the dead threshold, defaults to 40
        :param increment_ratio: by how much to increase the dead threshold.
            For example, 0.5 means "increase the threshold by 0.5 * desired threshold
        """
        self.num_warmup_steps = num_warmup_steps
        self.increment_ratio = increment_ratio
        self.step_size = int(num_warmup_steps * increment_ratio)
        self.is_enabled = is_enabled
        self.model_name = model_name

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if not self.is_enabled:
            if state.global_step <= 10:
                for module in model.modules():
                    if isinstance(module, MistralSparseSiluMLP):
                        module.current_dead_threshold = module.dead_threshold
            return

        current_dead_threshold = 0
        desired_dead_threshold = 0

        if is_mainprocess():
            ds_print(state.global_step)

        if state.global_step % self.step_size == 2:
            for module in model.modules():
                if isinstance(module, MistralSparseSiluMLP):
                    desired_dead_threshold = copy.deepcopy(module.dead_threshold)
                    current_dead_threshold = module.current_dead_threshold
                    current_dead_threshold += self.increment_ratio * desired_dead_threshold
                    module.current_dead_threshold = min(desired_dead_threshold, current_dead_threshold)

            if is_running_deepspeed and is_mainprocess():
                ds_print(
                    state.global_step,
                    current_dead_threshold,
                    desired_dead_threshold,
                )

        if state.global_step % 2000 == 0:
            if is_running_deepspeed and is_mainprocess():
                ds_print(
                    f"Saving to /matx/u/lukeai/{self.model_name}_{state.global_step - 2}.pt",
                )
                torch.save(
                    model.state_dict(),
                    f"/matx/u/lukeai/{self.model_name}_{state.global_step - 2}.pt",
                )


def get_sparse_mistral_config(
    config: MistralConfig,
    use_sparse_model=False,
    use_sparse_predictor=False,
    use_sparse_regularization=False,
    thresholds=None,
    cut_pre_mlp=False,
    cut_pre_attn=False,
):
    new_config = SparseMistralConfig()
    new_config.__dict__.update(config.__dict__)
    config = new_config
    config.use_sparse_model = use_sparse_model
    config.use_sparse_predictor = use_sparse_predictor
    config.use_sparse_regularization = use_sparse_regularization
    config.thresholds = thresholds
    config.cut_pre_mlp = cut_pre_mlp
    config.cut_pre_attn = cut_pre_attn

    return config


def apply_mistral_sparse_silu_mlp(
    model,
    config,
    use_sparse_regularization: bool = False,
    use_flash_attn: bool = False,
    cut_pre_mlp: bool = False,
    cut_pre_attn: bool = False,
):
    for layer in model.model.layers:
        # counts += 1
        # if counts < 4:
        #     continue
        original_mlp = layer.mlp
        new_mlp = MistralSparseSiluMLP(config, use_sparse_regularization=use_sparse_regularization)
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        new_mlp.cut_pre_mlp = cut_pre_mlp
        layer.mlp = new_mlp
    if cut_pre_attn:
        for layer in model.model.layers:
            original_attention = layer.self_attn
            if use_flash_attn:
                new_attention = SparseMistralFlashAttention(
                    config=original_attention.config,
                    layer_idx=original_attention.layer_idx,
                )

            else:
                new_attention = SparseMistralAttention(
                    config=original_attention.config,
                    layer_idx=original_attention.layer_idx,
                )
            for attr in vars(original_attention):
                setattr(new_attention, attr, getattr(original_attention, attr))
                layer.self_attn = new_attention


def apply_mistral_sparse_attention(
    model,
    config,
):
    for layer in model.model.layers:
        layer.self_attention = layer.self_attention


def apply_mistral_sparse_decoder_layer(
    model,
    config,
    init_svd: bool = True,
):
    assert isinstance(model.model, MistralModel), "model.model must be a MistralModel."
    new_layers = []
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            new_layers.append(
                SparseMistralDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    decoder_layer=layer,
                    init_svd=init_svd,
                )
            )
            ds_print(f"{layer_idx}th mlp layer activation: {layer.mlp.sparse_act_fn}")
        else:
            new_layers.append(layer)
    model.model.layers = nn.ModuleList(new_layers)


def enable_sparse_predictor(
    model,
):
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer, MistralDecoderLayer):
            layer.use_sparse_predictor = True


def disable_sparse_predictor(
    model,
):
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer, MistralDecoderLayer):
            layer.use_sparse_predictor = False


def activate_stats(model, is_collect_histogram: bool = True):
    for layer in model.model.layers:
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.activate_stats(is_collect_histogram=is_collect_histogram)
        if isinstance(layer.self_attn, (SparseMistralAttention, SparseMistralFlashAttention)):
            layer.self_attn.activate_stats()


def deactivate_stats(model):
    for layer in model.model.layers:
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.deactivate_stats()
        if isinstance(layer.self_attn, (SparseMistralAttention, SparseMistralFlashAttention)):
            layer.self_attn.deactivate_stats()


def enable_sparse_silu(model):
    ds_print("Enabling SparseSilu")
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.kill_sparse_swish_outputs = True


def print_dead_neuron_stats(model):
    total_sparsity = 0
    counts = 0
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            dead_percentage = layer.mlp.dead_percentage * 100
            agg_sparsity = layer.mlp.agg_sparsity * 100
            pre_mlp_sparsity = layer.mlp.pre_mlp_sparsity * 100
            ds_print(f"layer {i} sparsity: {dead_percentage:.3f}%")
            ds_print(f"layer {i} agg sparsity: {agg_sparsity:.3f}%")
            ds_print(f"layer {i} pre_mlp_sparsity: {pre_mlp_sparsity:.3f}%")

            total_sparsity += dead_percentage
            counts += 1
        if isinstance(layer.self_attn, SparseMistralAttention) or isinstance(
            layer.self_attn, SparseMistralFlashAttention
        ):
            ds_print(f"Attention layer {i} sparsity: {layer.self_attn.pre_attn_sparsity * 100: .3f}%")

    ds_print(f"Total sparsity: {total_sparsity/counts: .3f}%")
    return total_sparsity / counts


def get_sparse_layers(model: MistralModel):
    sparse_layers = [m.mlp for m in model.layers() if isinstance(m.mlp, MistralSparseSiluMLP)]
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
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            layer.mlp.regularization_threshold = threshold  # TODO: find better param


def set_sparse_threshold(
    model,
    sparsity_level: float,
    use_relu: bool = False,
    use_resilu: bool = False,
    use_adaptive: bool = True,
):
    assert not (use_relu and use_resilu), "It's not allowed to use both relu and resilu"
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            if use_relu:
                layer.mlp.sparse_act_fn = nn.ReLU()
                layer.mlp.use_relu = True
                layer.mlp.use_resilu = False
            elif use_resilu:
                layer.mlp.sparse_act_fn = nn.Sequential(nn.ReLU(), nn.SiLU())
                layer.mlp.use_resilu = True
                layer.mlp.use_relu = False
            else:
                layer.mlp.dead_threshold = get_threshold(
                    layer.mlp.histogram_bins,
                    layer.mlp.post_act_hist_counts,
                    sparsity_level,
                )
                layer.mlp.sparse_act_fn.set_new_threshold(layer.mlp.dead_threshold)
                layer.mlp.regularization_threshold = layer.mlp.dead_threshold * 1.2  # TODO: find better param

            if layer.mlp.cut_pre_mlp:
                layer.mlp.pre_mlp_threshold = get_threshold(
                    layer.mlp.histogram_bins,
                    layer.mlp.pre_mlp_hist_counts,
                    sparsity_level,
                )
                ds_print(f"layer {i} pre-mlp threshold: {layer.mlp.pre_mlp_threshold}")

        if isinstance(layer.self_attn, (SparseMistralAttention, SparseMistralFlashAttention)):
            layer.self_attn.pre_attn_threshold = get_threshold(
                layer.self_attn.histogram_bins,
                layer.self_attn.pre_attn_hist_counts,
                sparsity_level,
            )
            ds_print(f"layer {i} pre-attn threshold: {layer.self_attn.pre_attn_threshold}")


def plot_histogram(
    bin_edges,
    histogram_counts: torch.tensor,
    title: str = "Activation Distribution",
    fig_dir: str = "figures",
    y_logscale: bool = False,
):
    plt.bar(bin_edges[:-1], histogram_counts, width=np.diff(bin_edges), edgecolor="black")
    if y_logscale:
        plt.yscale("log")
        # Find the indices of the histogram counts that are not zero
        non_zero_indices = np.nonzero(histogram_counts)
        if non_zero_indices[0] > 0:
            # Find the left boundary as the first non-zero bin edge
            first_non_zero_index = non_zero_indices[0]
            left = bin_edges[first_non_zero_index]  # This is your left boundary
            # Find the right boundary as the last non-zero bin edge
            last_non_zero_index = non_zero_indices[-1]
            right = bin_edges[last_non_zero_index + 1]  # This is your right boundary
        else:
            # Default to the first and last bin edge if all counts are zero
            left = bin_edges[0]
            right = bin_edges[-1]
        plt.xlim(left, right)
    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{title}.png")
    # plt.show()
    plt.clf()


def plot_activation_histogram(model, fig_dir: str = "figures"):
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.self_attn, SparseMistralAttention) and layer.self_attn.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            plot_title = f"Layer: {i} Pre-attention Distribution"
            # plot_histogram(
            #     layer.self_attn.histogram_bins,
            #     layer.self_attn.pre_attn_hist_counts,
            #     plot_title,
            # )

            plot_title = f"Layer: {i} Post QK_T Distribution"
            plot_histogram(
                layer.self_attn.histogram_bins,
                layer.self_attn.post_qk_hist_counts,
                plot_title,
                y_logscale=True,
            )


def save_act_hist(model, dirname="/scr/jay/models/mistral/pre_finetune/cola_act_hist"):
    os.makedirs(dirname, exist_ok=True)
    act_dict = {}
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            act_dict[i] = (
                layer.mlp.histogram_bins,
                layer.mlp.pre_act_hist_counts,
                layer.mlp.post_act_hist_counts,
                layer.mlp.pre_mlp_hist_counts,
            )
    ds_print("Saving activation histograms...\n\n\n")
    torch.save(act_dict, dirname + "/mlp_layers.pt")

    act_dict = {}
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.self_attn, SparseMistralAttention) and layer.self_attn.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            act_dict[i] = (
                layer.self_attn.histogram_bins,
                layer.self_attn.pre_attn_hist_counts,
                layer.self_attn.post_qk_hist_counts,
            )
    ds_print("Saving activation histograms...\n\n\n")
    torch.save(act_dict, dirname + "/attn_layers.pt")


def load_act_hist(model, dirname="/scr/jay/models/mistral/pre_finetune/cola_act_hist"):
    assert os.path.exists(
        dirname
    ), f"{dirname} does not exist when loading pre/post-activation histogram of SparseMistralSiluMLP."
    ds_print("Loading activation histograms...\n\n\n")

    act_dict = torch.load(dirname + "/mlp_layers.pt")
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            if len(act_dict[i]) == 4:
                (
                    layer.mlp.histogram_bins,
                    layer.mlp.pre_act_hist_counts,
                    layer.mlp.post_act_hist_counts,
                    layer.mlp.pre_mlp_hist_counts,
                ) = act_dict[i]
            else:
                (
                    layer.mlp.histogram_bins,
                    # layer.mlp.pre_mlp_hist_counts,
                    layer.mlp.pre_act_hist_counts,
                    layer.mlp.post_act_hist_counts,
                ) = act_dict[i]
    act_dict = torch.load(dirname + "/attn_layers.pt")
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.self_attn, SparseMistralAttention) and layer.self_attn.is_stats:
            (
                layer.self_attn.histogram_bins,
                layer.self_attn.pre_attn_hist_counts,
                layer.self_attn.post_qk_hist_counts,
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

    model.model.layers = nn.ModuleList(new_modules)
