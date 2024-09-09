import logging
import importlib
import json
import os
import warnings
from typing import List, Dict

import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    MistralConfig
    # GemmaForCausalLM,
)

from .cats_utils.constants import MISTRAL_7B
from .cats_utils.utils import *
from .cats_utils.utils import _get_submodules


logger = logging.getLogger(__name__)

class Cats(nn.Module):
    def __init__(
        self,
        wrapped_module: nn.Module,
        threshold: float = 0.0,
        hist_num_bins: int = 5000,
        hist_min: int = -5,
        hist_max: int = 5,
        pre_apply: bool = False,
    ):
        super(Cats, self).__init__()
        self.wrapped_module = wrapped_module
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.num_bins = hist_num_bins
        self.hist_min = hist_min
        self.hist_max = hist_max
        self.histogram_bins = torch.linspace(hist_min, hist_max, hist_num_bins - 2)
        self.histogram_bins = torch.cat(
            [torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])]
        )
        self.hist_counts = torch.zeros(hist_num_bins - 1)
        self.abs_hist_counts = torch.zeros(hist_num_bins - 1)
        self.num_zeros_act = 0
        self.num_elems_act = 0
        self.collect_stats = True
        self.print_sparsity = True
        self.is_kernel = False
        self.pre_apply = pre_apply

    def disable_collect_stats(self):
        self.collect_stats = False

    def enable_collect_stats(self):
        self.collect_stats = True

    def enable_kernel_injection(self):
        self.is_kernel = True

    def disable_kernel_injection(self):
        self.is_kernel = False

    def set_threshold(self, threshold: float):
        if isinstance(threshold, float):
            threshold = torch.tensor(threshold)
        self.threshold = nn.Parameter(threshold.clone().detach(), requires_grad=False)

    def _update_stats(self, x):
        if torch.cuda.is_available():
            pre_activation = x.float()
            post_activation = torch.abs(pre_activation)
            self.hist_counts += torch.cat(
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
            self.abs_hist_counts += torch.cat(
                (
                    (post_activation < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        post_activation,
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (post_activation > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()
        else:
            self.hist_counts += torch.histogram(x, bins=self.histogram_bins)[0]
            self.abs_hist_counts += torch.histogram(
                torch.abs(x), bins=self.histogram_bins
            )[0]

    def _update_sparsity(self, x):
        self.num_zeros_act += torch.sum(x == 0)
        self.num_elems_act += torch.numel(x)

    def forward(self, x):
        if self.pre_apply:
            if self.collect_stats:
                self._update_stats(x)
            if not self.is_kernel:
                x[abs(x) < self.threshold] = 0
            if self.print_sparsity:
                self._update_sparsity(x)
            x = self.wrapped_module(x)
        else:
            x = self.wrapped_module(x)
            if self.collect_stats:
                self._update_stats(x)
            if not self.is_kernel:
                x[abs(x) < self.threshold] = 0
            if self.print_sparsity:
                self._update_sparsity(x)

        return x


class CatsConfig(PretrainedConfig):
    model_type = "cats_model"

    def __init__(
        self,
        wrapped_model_config=MistralConfig(),
        wrapped_model_class_name: str = "MistralForCausalLM",
        pre_target_modules: List[str] = [],
        post_target_modules: List[str] = ["act_fn"],
        target_sparsity: float = 0.5,
        kernel_inject_targets: Dict[str, int] = {"mlp": 2},
        is_inject_kernel: bool = False,
        **kwargs,
    ):
        """
        Args:
            wrapped_model_config (PretrainedConfig): The configuration of the wrapped model.
            wrapped_model_class_name (str): The name of the wrapped model class.
            pre_target_modules (List[str]): The list of module names to replace with CATS (applied before wrapped module).
            post_target_modules (List[str]): The list of module names to replace with CATS (applied after wrapped module).
            target_sparsity (float): The target sparsity level.
            kernel_inject_targets (Dict[str, int]): The dictionary with module name as a key and with kernel method
            as a value.
            is_inject_kernel (bool): Whether to inject the kernel.
            **kwargs: Additional keyword arguments.
        """
        self.pre_target_modules = pre_target_modules
        self.post_target_modules = post_target_modules
        self.target_sparsity = target_sparsity
        self.kernel_inject_targets = kernel_inject_targets
        self.wrapped_model_class_name = wrapped_model_class_name
        self.is_inject_kernel = is_inject_kernel
        self.__dict__.update(wrapped_model_config.__dict__)
        super().__init__(**kwargs)


class CatsModelForCausalLM(PreTrainedModel, nn.Module):
    config_class = CatsConfig

    def __init__(
        self,
        config,
        pretrained_kwargs: dict = None,
        model: PreTrainedModel = None,
    ):
        super().__init__(config)
        transformers_module = importlib.import_module("transformers")
        self.wrapped_model_class = getattr(
            transformers_module, config.wrapped_model_class_name
        )
        if pretrained_kwargs is not None:
            self.wrapped_model = self.wrapped_model_class.from_pretrained(
                **pretrained_kwargs
            )
        else:
            if model is None:
                self.wrapped_model = self.wrapped_model_class(config)
            else:
                self.wrapped_model = model
        self.kernel_inject_targets = config.kernel_inject_targets
        self.inject_cats()

    def inject_kernel(self):
        if self.kernel_inject_targets is not None:
            for name, module in self.wrapped_model.named_modules():
                parent, target, target_name = _get_submodules(self.wrapped_model, name)
                if target_name in self.kernel_inject_targets.keys():
                    logger.info(f"injecting kernel to {name}")
                    if self.kernel_inject_targets[target_name] == 1:
                        replace_mlp_with_gemv(target, target_name)
                    elif self.kernel_inject_targets[target_name] == 2:
                        replace_mlp_with_gemv_gemv(target, target_name)
                    else:
                        raise NotImplementedError(
                            f"Kernel method {self.kernel_inject_targets[target_name]} is not implemented."
                        )
                if isinstance(target, Cats):
                    target.enable_kernel_injection()

    def inject_cats(self):
        for name, module in self.wrapped_model.named_modules():
            parent, target, target_name = _get_submodules(self.wrapped_model, name)
            if target_name in self.config.pre_target_modules:
                logger.info(
                    f"injecting cats into {target_name}, {type(target)} (pre-apply)"
                )

                # Replace target module with target module + CATS
                cats = Cats(wrapped_module=target, pre_apply=True)
                setattr(parent, target_name, cats)
            elif target_name in self.config.post_target_modules:
                logger.info(
                    f"injecting cats into {target_name}, {type(target)} (post-apply)"
                )

                cats = Cats(wrapped_module=target, pre_apply=False)
                setattr(parent, target_name, cats)

    def enable_collect_stats(self):
        logger.info("enable stats")
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.enable_collect_stats()
                module.print_sparsity = True

    def disable_collect_stats(self) -> None:
        logger.info("disable stats")

        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.disable_collect_stats()
                module.print_sparsity = False

    def init_stats(self) -> None:
        logger.info("Initialize stats")

        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.num_elems_act = 0
                module.num_zeros_act = 0

    def set_thresholds(self):
        logger.info("Setting threshold")
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                threshold = get_threshold(
                    module.histogram_bins,
                    module.abs_hist_counts,
                    self.config.target_sparsity,
                )
                module.set_threshold(threshold)

    def get_sparsity(self):
        sparsity_dict = {}
        idx = 0
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                sparsity_dict[name] = float(module.num_zeros_act / module.num_elems_act)
                idx += 1
        return sparsity_dict

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.wrapped_model, name)

    def __call__(self, *args, **kwargs):
        """Override the __call__ method to use the wrapped model's forward method."""
        return self.wrapped_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.wrapped_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.wrapped_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


def get_cats_model(
    model: PreTrainedModel,
    pre_target_modules: List[str] = [],
    post_target_modules: List[str] = ["act_fn"],
    target_sparsity: float = 0.5,
    sparsity_type: str = "Constant",
    kernel_inject_targets: Dict[str, int] = {"mlp": 2},
    is_inject_kernel: bool = False,
    is_share_params: bool = True,
) -> CatsModelForCausalLM:
    """
    Create a CATS model from an existing model.

    Args:
        model (PreTrainedModel): The base model to wrap with CATS.
        pre_target_modules (List[str]): Modules to apply CATS before.
        post_target_modules (List[str]): Modules to apply CATS after.
        target_sparsity (float): Target sparsity level.
        sparsity_type (str): Type of sparsity (currently unused).
        kernel_inject_targets (Dict[str, int]): Modules to inject kernels into.
        is_inject_kernel (bool): Whether to inject kernels.
        is_share_params (bool): Whether to share parameters with the base model.

    Returns:
        CatsModelForCausalLM: The CATS-wrapped model.
    """
    config = model.config
    wrapped_model_class_name = model_class_to_pkg_str(model)

    cats_config = CatsConfig(
        config,
        wrapped_model_class_name=wrapped_model_class_name,
        pre_target_modules=pre_target_modules,
        post_target_modules=post_target_modules,
        target_sparsity=target_sparsity,
        kernel_inject_targets=kernel_inject_targets,
        is_inject_kernel=is_inject_kernel,
    )

    pretrained_kwargs = {
        "pretrained_model_name_or_path": config.name_or_path,
        "torch_dtype": model.dtype,
        "attn_implementation": getattr(config, "_attn_implementation", None),
    }

    if is_share_params:
        logger.info("Creating CATS model with shared parameters")
        cats_model = CatsModelForCausalLM(cats_config, model=model)
    else:
        logger.info("Creating CATS model with separate parameters")
        cats_model = CatsModelForCausalLM(
            cats_config,
            pretrained_kwargs=pretrained_kwargs,
            model=None,
        )

    if is_inject_kernel:
        logger.info("Injecting kernels into CATS model")
        cats_model.inject_kernel()

    return cats_model
