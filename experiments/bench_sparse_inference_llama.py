from transformers import (
    AutoTokenizer,
    LlamaConfig,
)
from torch.utils.data import DataLoader
from utils.constants import LLAMA, REFINED_WEB
from experiments.models.sparse_silu.ugly_utils import *
from experiments.data.get_dataset import get_dataset
from tqdm import tqdm
import argparse
import os


def prepare_sparse_model(base_model_name):
    save_dir = os.getenv("CATS_RESPATH", "results")
    save_path = os.path.join(save_dir, "throughput.csv")

    with open(save_path, "a") as f:
        print(
            "Llama2_sparse",
            file=f,
        )
    BaseConfig = LlamaConfig
    SparseConfig = SparseLlamaConfig
    SparseCausalLM = SparseLlamaForCausalLM
    SparseConfig.register_for_auto_class()
    SparseCausalLM.register_for_auto_class("AutoModelForCausalLM")

    config = BaseConfig.from_pretrained(base_model_name)
    config = get_sparse_config(config)
    config.use_cache = True
    model = SparseCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.config_class = SparseConfig
    apply_sparse_silu_mlp(model, model.config, use_sparse_regularization=False)
    enable_sparse_silu(model)
    model.config.use_sparse_predictor = False
    model.config.use_relu = False
    model.config.use_sparse_model = True
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def benchmark_decode(B, bm, method, gen_len, act_hist_path, base_model_name):
    device = torch.device("cuda")
    dataset_type = REFINED_WEB
    model_type = LLAMA
    model, tokenizer = prepare_sparse_model(base_model_name)
    activate_stats(model)
    load_act_hist(model, act_hist_path)
    set_sparse_threshold(model, 0.5)
    deactivate_stats(model)
    dataset = get_dataset(dataset_type, tokenizer, model_type, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()

    dataloader = DataLoader(test_dataset, batch_size=B, collate_fn=data_collator)

    for m in model.model.layers:
        m.mlp.is_profile = True
        # m.mlp.use_flash_gemv = True
        if method == 0:
            m.mlp.sp_method = 0
            print(m.mlp.down_proj.weight.dtype)
        elif method == 1:
            m.mlp.sp_method = 1
        elif method == 2:
            m.mlp.sp_method = 2
            m.mlp.wdown_t = m.mlp.down_proj.weight.t().contiguous()
            print(m.mlp.down_proj.weight.dtype)
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    count = 0
    max_count = 50
    with torch.no_grad():
        for batch in tqdm(dataloader, total=max_count):
            count += 1
            if count > max_count:
                break
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + gen_len
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=max_lengths,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.7,
                    num_beams=bm,
                )
            except RuntimeError as _:
                print("Time interval is too short!")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--bm", type=int, default=1)
    parser.add_argument("--gen", type=int, default=501)
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--weights_dir", type=str, default="None")
    parser.add_argument("--results_dir", type=str, default="None")

    args = parser.parse_args()
    if args.weights_dir == "None":
        print("Please specify weights_dir")
        raise ValueError
    if args.results_dir == "None":
        print("Please specify results_dir")
        raise ValueError

    act_hist_path = os.path.join(
        args.root_path,
        args.results_dir,
        "general_finetuning",
        "sparse_llama_7b_hf2",
        "refined_web_activation_histogram.pt",
    )
    base_path = os.path.join(
        args.root_path, args.weights_dir, "general_finetuning", "sparse_llama_7b_hf2_refined_web_50p_no_adapter_3steps"
    )
    benchmark_decode(args.B, args.bm, args.method, args.gen, act_hist_path, base_path)
