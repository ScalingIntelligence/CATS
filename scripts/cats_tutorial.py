import torch
import os
import time
import random
import csv
import math
import bitsandbytes as bnb
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Subset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils.other import fsdp_auto_wrap_policy
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR


from simple_cats.data.get_dataset import get_dataset
from simple_cats.cats_model import (
    CatsModelForCausalLM,
    CatsConfig,
    get_cats_model,
)
from utils.constants import MISTRAL_7B, REFINED_WEB, LLAMA_7B


def print_gpu_memory():
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
    else:
        print("CUDA is not available")


def run_evaluation(
    model, eval_dataloader, total, max_lengths, tokenizer, with_kernel_injection=False
):
    start_time = time.time()
    print(
        f"Running {'with' if with_kernel_injection else 'without'} kernel injection..."
    )

    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=total):
            if step > total:
                break
            input_ids = batch["input_ids"].to(model.device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                min_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )

    elapsed_time = time.time() - start_time
    print(
        f"{'With' if with_kernel_injection else 'Without'} kernel injection: {elapsed_time}"
    )
    return elapsed_time


def record_measurements(file_name, data):
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(data.keys())
        writer.writerow(data.values())


def calculate_perplexity(model, eval_dataloader, accelerator):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(
        eval_dataloader,
        desc="Calculating Perplexity",
        disable=not accelerator.is_local_main_process,
    ):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss.detach().float() * batch["input_ids"].numel()
        total_tokens += batch["input_ids"].numel()

    # Gather total loss and tokens across all processes
    total_loss = accelerator.gather(
        torch.tensor(total_loss, device=accelerator.device)
    ).sum()
    total_tokens = accelerator.gather(
        torch.tensor(total_tokens, device=accelerator.device)
    ).sum()

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(avg_loss)

    return perplexity.item()


def train_model(
    model,
    train_dataloader,
    eval_dataloader,
    num_epochs,
    max_steps,
    learning_rate,
    accelerator,
    gradient_accumulation_steps=1,
):
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)

    total_steps = min(
        len(train_dataloader) * num_epochs // gradient_accumulation_steps, max_steps
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=5000)

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    steps = 0
    log_steps = 25
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        ):
            outputs = model(**batch)
            loss = outputs.loss
            # Scale the loss by the number of gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            total_loss += loss.detach().float() * gradient_accumulation_steps

            # Only perform optimization step after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1

                if steps >= max_steps:
                    break

                if steps % log_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    accelerator.print(
                        f"Step {steps} Avg Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    model.eval()
                    perplexity = calculate_perplexity(
                        model, eval_dataloader, accelerator
                    )
                    accelerator.print(f"Perplexity: {perplexity:.2f}")
                    model.train()

        if steps >= max_steps:
            break

        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    accelerator.print(f"Training stopped after {steps} steps")
    return accelerator.unwrap_model(model), optimizer


def run_experiment(config):
    accelerator = Accelerator()

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_dir"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Create CATS model if specified
    if config["use_cats"]:
        model = get_cats_model(
            base_model,
            target_sparsity=config["target_sparsity"],
            is_share_params=False,
            post_target_modules=config["post_target_modules"],
            pre_target_modules=config["pre_target_modules"],
            kernel_inject_targets=config["kernel_inject_targets"],
        )
        model.to(torch.bfloat16)
    else:
        model = base_model

    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
    tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if specified
    if config["use_lora"]:
        lora_target_modules = set([])

        # Lora is not allowed to inject low rank adapters to CATS class
        for module in config["lora_target_modules"]:
            if module in config["post_target_modules"] + config["pre_target_modules"]:
                lora_target_modules.add(f"{module}.wrapped_module")
            else:
                lora_target_modules.add(module)

        print("lora target modules: ", lora_target_modules)
        lora_config = LoraConfig(
            r=256,
            lora_alpha=512,
            target_modules=list(lora_target_modules),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare dataset
    dataset = get_dataset(
        REFINED_WEB,
        tokenizer,
        model_type="Cats_Mistral" if config["use_cats"] else "Mistral",
        max_seq_length=1024,
    )
    data_collator = dataset.get_data_collator()
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    )

    unwrapped_model = model.module if hasattr(model, "module") else model

    # Collect stats and set thresholds for CATS model
    if config["use_cats"]:
        unwrapped_model.enable_collect_stats()
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=10):
            if step > 10:
                break
            _ = model(batch["input_ids"])
        unwrapped_model.set_thresholds()
        unwrapped_model.init_stats()
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=10):
            if step > 10:
                break
            _ = model(batch["input_ids"])

        sparsities = unwrapped_model.get_sparsity()
        average_sparsity = 0
        for name, sparsity in sparsities.items():
            if config["verbose"]:
                print(f"{name} activation sparsity: {sparsity}")
            average_sparsity += sparsity
        print(f"Sparsity: {average_sparsity/len(sparsities)}")

        accelerator.wait_for_everyone()

    # Calculate initial perplexity
    initial_perplexity = calculate_perplexity(model, test_dataloader, accelerator)
    if accelerator.is_main_process:
        print(f"Initial model perplexity: {initial_perplexity}")

    # Train the model
    if config["train_model"]:
        unwrapped_model.disable_collect_stats()
        model, optimizer = train_model(
            model,
            train_dataloader,
            eval_dataloader,
            config["num_epochs"],
            config["max_steps"],
            config["learning_rate"],
            accelerator,
        )

        # Calculate final perplexity
        final_perplexity = calculate_perplexity(model, test_dataloader, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Final model perplexity: {final_perplexity}")
            record_measurements(
                "perplexity_measurements.csv",
                {
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Model Type": "CATS" if config["use_cats"] else "Base",
                    "Target Sparsity": config["target_sparsity"]
                    if config["use_cats"]
                    else "N/A",
                    "Initial Perplexity": initial_perplexity,
                    "Final Perplexity": final_perplexity,
                    "CATS Pre Target Modules": ",".join(config["pre_target_modules"])
                    if config["use_cats"]
                    else "N/A",
                    "CATS Post Target Modules": ",".join(config["post_target_modules"])
                    if config["use_cats"]
                    else "N/A",
                },
            )
    else:
        final_perplexity = initial_perplexity

        # Record perplexity measurements
        # Record perplexity measurements
        record_measurements(
            "perplexity_measurements.csv",
            {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Model Type": "CATS" if config["use_cats"] else "Base",
                "Target Sparsity": config["target_sparsity"]
                if config["use_cats"]
                else "N/A",
                "Initial Perplexity": initial_perplexity,
                "Final Perplexity": final_perplexity,
                "CATS Pre Target Modules": ",".join(config["pre_target_modules"])
                if config["use_cats"]
                else "N/A",
                "CATS Post Target Modules": ",".join(config["post_target_modules"])
                if config["use_cats"]
                else "N/A",
            },
        )

    # Model generation for latency measurement
    if config["run_generation"]:
        model.eval()
        if config["use_cats"]:
            unwrapped_model.disable_collect_stats()
            accelerator.wait_for_everyone()
        if config["run_generation_wo_kernel"]:
            # Run without kernel injection
            elapsed_time_without = run_evaluation(
                model,
                eval_dataloader,
                config["total_eval_steps"],
                config["max_lengths"],
                tokenizer,
                with_kernel_injection=False,
            )
            if accelerator.is_main_process:
                record_measurements(
                    "timing_measurements.csv",
                    {
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Model Type": "CATS" if config["use_cats"] else "Base",
                        "With Kernel Injection": False,
                        "Elapsed Time": elapsed_time_without,
                        "Target Sparsity": config["target_sparsity"]
                        if config["use_cats"]
                        else "N/A",
                    },
                )

        # Run with kernel injection for CATS model
        if config["use_cats"] and config["use_kernel_injection"]:
            print("Compiling for kernel injection...")
            unwrapped_model.inject_kernel()
            accelerator.wait_for_everyone()
            with torch.no_grad():
                for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
                    if step > 1:
                        break
                    input_ids = batch["input_ids"].to(model.device)
                    _ = model.generate(
                        input_ids=input_ids,
                        max_length=config["max_lengths"],
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=1,
                    )

            elapsed_time_with = run_evaluation(
                model,
                eval_dataloader,
                config["total_eval_steps"],
                config["max_lengths"],
                tokenizer,
                with_kernel_injection=True,
            )
            if accelerator.is_main_process:
                record_measurements(
                    "timing_measurements.csv",
                    {
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Model Type": "CATS",
                        "With Kernel Injection": True,
                        "Elapsed Time": elapsed_time_with,
                        "Target Sparsity": config["target_sparsity"],
                    },
                )

    # Save model
    if config["save_model"]:
        save_dir = os.path.join(
            config["save_dir"],
            f"{config['model_name']}_{int(config['target_sparsity']*100)}p_sparsity",
        )
        os.makedirs(save_dir, exist_ok=True)
        CatsConfig.register_for_auto_class()
        CatsModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Model saved to {save_dir}")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    default_config = {
        "model_dir": MISTRAL_7B,
        "use_cats": True,
        "target_sparsity": 0.5,
        "pre_target_modules": [],
        "post_target_modules": ["act_fn"],
        "lora_target_modules": [
            "q_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            # "down_proj",
        ],
        "kernel_inject_targets": {"mlp": 2},
        "use_lora": True,
        "batch_size": 4,
        "num_epochs": 1,
        "max_steps": 1000,
        "learning_rate": 5e-5,
        "total_eval_steps": 5,
        "max_lengths": 1500,
        "train_model": True,
        "run_generation_wo_kernel": False,
        "use_kernel_injection": True,
        "save_model": True,
        "save_dir": "cats/mistral_cats_zero_shot",
        "model_name": "CATS",
        "run_generation": False,  # New flag to control generation for latency measurement
        "pre_apply": False,
        "verbose": True,
    }

    # To run experiment with base model only
    # base_config = default_config.copy()
    # base_config["use_cats"] = False
    # base_config["use_lora"] = False
    # base_config["model_name"] = "Base_Mistral"
    # run_experiment(base_config)

    # To run experiment with different target modules

    # To run experiments with different configurations:
    target_sparsities = [0.5, 0.7, 0.85, 0.90]
    # for sparsity in target_sparsities:
    #     config = default_config.copy()
    #     config["target_sparsity"] = sparsity
    #
    #     config = default_config.copy()
    #     config["target_modules"] = [
    #         "q_proj",
    #     ]
    #     run_experiment(config)
    #
    # for sparsity in target_sparsities:
    #     config = default_config.copy()
    #     config["target_sparsity"] = sparsity
    #
    #     config = default_config.copy()
    #     config["target_modules"] = ["post_attention_layernorm"]
    #     run_experiment(config)

    for sparsity in target_sparsities:
        config = default_config.copy()
        config["target_sparsity"] = sparsity
        config["post_target_modules"] = ["act_fn"]
        config["pre_target_modules"] = ["q_proj", "o_proj", "gate_proj"]
        config["kernel_inject_targets"] = {
            "mlp": 2,
            "q_proj": 1,
            "o_proj": 1,
            "gate_proj": 1,
        }
        config["run_generation"] = True
        config["train_model"] = False
        run_experiment(config)

        # config["target_modules"] = [
        #     "post_attention_layernorm",
        #     "q_proj",
        #     "act_fn",
        #     "v_proj",
        # ]
        # run_experiment(config)

        # config["post_target_modules"] = [
        #     "post_attention_layernorm",
        #     "act_fn",
        # ]
        # run_experiment(config)
        #
        # config["post_target_modules"] = [
        #     "act_fn",
        # ]
        # run_experiment(config)
        #
        # config["post_target_modules"] = [
        #     "q_proj",
        # ]
        # run_experiment(config)
