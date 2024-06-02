import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaConfig,
)
from torch.utils.data import DataLoader
from utils.constants import LLAMA, REFINED_WEB
from experiments.data.get_dataset import get_dataset


def prepare_dense_model():
    base_model_name = "meta-llama/Llama-2-7b-hf"
    BaseConfig = LlamaConfig
    BaseCausalLM = LlamaForCausalLM
    config = BaseConfig.from_pretrained(base_model_name)
    model = BaseCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def inference():
    device = torch.device("cuda")
    dataset_type = REFINED_WEB
    model_type = LLAMA
    model, tokenizer = prepare_dense_model()
    dataset = get_dataset(dataset_type, tokenizer, model_type, max_seq_length=1024)
    _, _, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)
    counts = 0
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + 500
            print(max_lengths)
            outputs = model.generate(
                input_ids=input_ids, max_length=max_lengths, pad_token_id=tokenizer.eos_token_id, use_cache=True
            )
            # Decode generated tokens to texts
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            counts += 1
            if counts <= 1:
                for idx in range(len(input_ids)):
                    ids = input_ids[idx]
                    # Decode the input_ids to a string
                    tokens_str = tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"Tokens before generation: \n {tokens_str}")
                    generated_text = generated_texts[idx]
                    print(f"Generated: \n {generated_text}\n")


if __name__ == "__main__":
    inference()
