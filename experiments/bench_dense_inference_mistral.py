import argparse
import torch
import os
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm
from data.get_dataset import get_dataset
from utils.constants import MISTRAL_7B, REFINED_WEB


def benchmark_decode(B, bm, gen_len):
    save_dir = os.getenv("CATS_RESPATH", "results")
    save_path = os.path.join(save_dir, "throughput.csv")

    with open(save_path, "a") as f:
        print(
            "Mistral_base",
            file=f,
        )

    path = "mistralai/Mistral-7B-v0.1"
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset_type = REFINED_WEB

    dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()

    dataloader = DataLoader(test_dataset, batch_size=B, collate_fn=data_collator)
    model.eval()
    for m in model.model.layers:
        print(m.mlp.down_proj.weight.dtype)
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
    args = parser.parse_args()
    # benchmark_generation()
    benchmark_decode(args.B, args.bm, args.gen)
