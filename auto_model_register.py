from experiments.models.sparse_mistral.sparse_silu import (
    MistralSparseSiluMLP,
    SparseMistralforCausalLM,
    SparseMistralConfig,
    get_sparse_mistral_config,
)
from experiments.instruct_tuning import prepare_sparse_model
import os
from transformers import (
    MistralConfig,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    MistralForCausalLM,
)
from utils.constants import MISTRAL_7B

# AutoConfig.register("sparse_mistral", SparseMistralConfig)
# AutoModelForCausalLM.register(SparseMistralConfig, SparseMistralforCausalLM)

# config = get_sparse_mistral_config(MistralConfig.from_pretrained(MISTRAL_7B))
# model = SparseMistralforCausalLM.from_pretrained(MISTRAL_7B, config=config)
#
# path = "debugging_output/unit_tests"
# model.save_pretrained(path)
# model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,)
# print(model)

# path = "/scr/jay/ckpt/Mistral_Sparse_refined_web_70p"
# model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,)
# config = AutoConfig.from_pretrained(path, trust_remote_code=True,)
# print(config)
# print(model)
# for m in model.model.layers:
#     print(m.mlp.dead_threshold)
#
#

# prefix = "/scr/jay/ckpt/Mistral_Sparse_refined_web"


# for sparsity in ["50p", "70p", "90p"]:
#     path = prefix + f"_{sparsity}"
#     if os.path.exists(path):
#         config = AutoConfig.from_pretrained(path, trust_remote_code=True)
#         config.use_relu = False
#         config.save_pretrained(path)

# for sparsity in ["relu"]:
#     path = prefix + f"_{sparsity}"
#     if os.path.exists(path):
#         config = AutoConfig.from_pretrained(path, trust_remote_code=True)
#         config.use_relu = False
#         config.save_pretrained(path)

# path = "/scr/jay/ckpt/sparse_models/unit_test"
path = "/scr/jay/ckpt/2024-03-10/Mistral_Sparse_refined_web_50p_no_adapter"
# path = "thrunlab/Mistral_Sparse_refined_web_relu_2024-03-01"
# #
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
print(config)
print(model)


# Register for AutoConfig and AutoModelforCausalLM
# SparseMistralConfig.register_for_auto_class()
# SparseMistralforCausalLM.register_for_auto_class("AutoModelForCausalLM")

# config = SparseMistralConfig.from_pretrained(path)
# config.path = path
# print(config)
# model = AutoModelForCausalLM.from_pretrained(path, config=config, trust_remote_code=True)
# model.config = config
# print(model)
# print(model.config)

# model, tokenizer, config = prepare_sparse_model(False, use_lora=True)
# print(model)
# model = model.merge_and_unload()
# print(model)

# path = "/scr/jay/ckpt/2024-03-01/Mistral_Sparse_refined_web_relu"
# model.save_pretrained(path + "_auto")
# model = AutoModelForCausalLM.from_pretrained(path + "_auto", trust_remote_code=True)
# print(model)

#
# for m in model.model.layers:
#     m.mlp.dead_threshold = 100
# thresholds = [float(m.mlp.dead_threshold) for m in model.model.layers]
# model.config.thresholds = thresholds
# #
# print(config)
#
# # model.save_pretrained(path)
# # tokenizer.save_pretrained(path)
# # model.config.save_pretrained(path)
#
# a = tokenizer("hello world", return_tensors="pt")
# (model(**a))
