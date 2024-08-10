import importlib
import json
import os
from typing import List

import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    AutoModelForCausalLM,
    MistralForCausalLM,
    LlamaForCausalLM,
    # GemmaForCausalLM,
)

from experiments.models.sparse_silu.ugly_utils import get_threshold
from utils.constants import MISTRAL_7B
from utils.utils import _get_submodules

def model_class_to_pkg_str(model_obj):
    if isinstance(model_obj, MistralForCausalLM):
        return "MistralForCausalLM"
    elif isinstance(model_obj, LlamaForCausalLM):
        return "LlamaForCausalLM"
    # elif isinstance(model_obj, GemmaForCausalLM):
    #     return "GemmaForCausalLM"
    else:
        raise NotImplementedError(f"{model_obj.__class__} is not implemented ")

class Cats(nn.Module):
    def __init__(
        self,
        wrapped_module: nn.Module,
        threshold: float = 0,
        hist_num_bins: int = 1000,
        hist_min: int = -1,
        hist_max: int = 1,
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

    def disable_collect_stats(self):
        self.collect_stats = False

    def enable_collect_stats(self):
        self.collect_stats = True

    def set_threshold(self, threshold: float):
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)

    def forward(self, x):
        x = self.wrapped_module(x)
        if self.collect_stats:
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
        x[abs(x) < self.threshold] = 0
        if self.print_sparsity:
            self.num_zeros_act += torch.sum(x == 0)
            self.num_elems_act += torch.numel(x)

        return x


# Function to load existing data from a JSON file
def load_data(file_path):
    try:
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the file does not exist


# Function to save the dictionary to a JSON file
def save_to_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


class CatsConfig(PretrainedConfig):
    model_type = "cats_model"

    def __init__(
        self,
        wrapped_model_config=AutoConfig.from_pretrained(MISTRAL_7B),
        wrapped_model_class_name: str = "MistralForCausalLM",
        target_modules: List[str] = ["act_fn"],
        target_sparsity: float = 0.5,
        **kwargs,
    ):
        self.target_modules = target_modules
        self.target_sparsity = target_sparsity
        self.wrapped_model_class_name = wrapped_model_class_name
        self.__dict__.update(wrapped_model_config.__dict__)
        super().__init__(**kwargs)


class CatsModel(PreTrainedModel, nn.Module):
    config_class = CatsConfig

    def __init__(
        self,
        config,
        pretrained_kwargs: dict = None,
    ):
        super().__init__(config)
        transformers_module = importlib.import_module("transformers")
        self.wrapped_model_class = getattr(
            transformers_module, config.wrapped_model_class_name
        )
        self.wrapped_model = self.wrapped_model_class(config)
        if pretrained_kwargs is not None:
            self.wrapped_model = self.wrapped_model_class.from_pretrained(
                **pretrained_kwargs
            )
        self.inject_cats()

    def inject_cats(self):
        for name, module in self.wrapped_model.named_modules():
            parent, target, target_name = _get_submodules(self.wrapped_model, name)
            if target_name in self.config.target_modules:
                print(f"{name} is replaced.")

                # Replace target module with target module + CATS
                cats = Cats(wrapped_module=target)
                setattr(parent, target_name, cats)

    def enable_collect_stats(self):
        print("enable stats")
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.enable_collect_stats()
                module.print_sparsity = True

    def disable_collect_stats(self) -> None:
        print("disable stats")

        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.disable_collect_stats()
                module.print_sparsity = False


    def set_thresholds(self):
        print("Setting threshold")
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
        for module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                sparsity_dict[idx] = float(module.num_zeros_act / module.num_elems_act)
                idx += 1
        return sparsity_dict

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.wrapped_model, name)

    def forward(self, *args, **kwargs):
        return self.wrapped_model.forward(*args, **kwargs)


def get_cats_model(model):
    config = model.config
    wrapped_model_class_name = model_class_to_pkg_str(model)

    cats_config = CatsConfig(config, wrapped_model_class_name)
    cats_model = CatsModel(cats_config, pretrained_kwargs=model.pretrained_kwargs)

    return cats_model

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MISTRAL_7B)
    get_cats_model()