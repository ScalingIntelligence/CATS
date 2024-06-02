from experiments.models.sparse_mistral.sparse_silu import (
    MistralSparseSiluMLP,
    SparseMistralforCausalLM,
    SparseMistralConfig,
)
from experiments.instruct_tuning import (
    prepare_sparse_model,
    set_sparse_threshold,
    load_act_hist,
)
import os
from transformers import (
    MistralConfig,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    MistralForCausalLM,
)
from utils.constants import MISTRAL_7B

if __name__ == "__main__":
    act_hist_path = f"/scr/jay/exps/pre_finetune/Mistral_Sparse/refined_web_act_hist.pt"
    relu_model, relu_tokenizer, relu_config = prepare_sparse_model(
        use_flash_attn=True, base_model_name=MISTRAL_7B, use_relu=True, use_lora=False
    )
    sparse_model, sparse_tokenizer, sparse_config = prepare_sparse_model(
        use_flash_attn=True, base_model_name=MISTRAL_7B, use_relu=False, use_lora=False
    )
    load_act_hist(model=sparse_model, filename=act_hist_path)
    set_sparse_threshold(sparse_model, 0.5)

    thresholds = [float(m.mlp.dead_threshold) for m in sparse_model.model.layers]
    sparse_model.config.thresholds = thresholds

    relu_path = "/scr/jay/exps/relu_mistral"
    sparse_path = "/scr/jay/exps/sparse_mistral"

    relu_model.save_pretrained(relu_path)
    relu_tokenizer.save_pretrained(relu_path)

    sparse_model.save_pretrained(sparse_path)
    sparse_tokenizer.save_pretrained(sparse_path)

    relu_model = AutoModelForCausalLM.from_pretrained(relu_path, trust_remote_code=True)
    for m in relu_model.model.layers:
        print(m.mlp.use_relu)

    sparse_model = AutoModelForCausalLM.from_pretrained(sparse_path, trust_remote_code=True)
    for m in sparse_config.model.layers:
        print(m.mlp.use_relu)
        print(m.mlp.dead_threshold)


# AutoConfig.register("sparse_mistral", SparseMistralConfig)
# AutoModelForCausalLM.register(SparseMistralConfig, SparseMistralforCausalLM)

# path = "/scr/jay/ckpt/Mistral_Sparse_refined_web_70p"
# model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,)
# config = AutoConfig.from_pretrained(path, trust_remote_code=True,)
# print(config)
# print(model)
# for m in model.model.layers:
#     print(m.mlp.dead_threshold)
