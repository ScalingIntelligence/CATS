from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    MistralForCausalLM,
    MistralConfig,
)
from transformers.integrations.deepspeed import deepspeed_init
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, load_metric
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from evaluate import load
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)


import torch
import deepspeed
import torch.nn as nn
import os
import gc
import wandb
import numpy as np
import sys
import argparse
import time
import warnings
import json

from experiments.models.sparse_mistral.sparse_silu import (
    SparseSFTTTrainer,
    SparseMistralforCausalLM,
    SparseMistralConfig,
    apply_mistral_sparse_silu_mlp,
    apply_mistral_sparse_decoder_layer,
    activate_stats,
    enable_sparse_silu,
    print_dead_neuron_stats,
    set_sparse_threshold,
    plot_activation_histogram,
    deactivate_stats,
    load_act_hist,
    save_act_hist,
    enable_last_k_modules,
    enable_first_k_modules,
    enable_sparse_predictor,
    disable_sparse_predictor,
    get_sparse_mistral_config,
)
from experiments.models.sparse_mistral.svd_router import SparsePredictor
from utils.mistral_utils import compress_mistral
from utils.utils import (
    print_size_of_model,
    is_running_deepspeed,
    is_mainprocess,
    get_datetime,
)
from utils.parse_args import parse_args
from utils.constants import (
    MISTRAL_YES_ID,
    MISTRAL_NO_ID,
    MISTRAL_7B,
    OPENWEBTEXT,
    COLA,
    BOOLQ,
    QNLI,
    SST2,
    WIC,
)
from experiments.format_dataset_for_instruction_tuning import (
    get_formatting_func,
    get_dataset_for_instruction_tuning,
)


def load_glue(data_type: str, seed: int = 0):
    # Load = dataset
    if data_type == "cola":
        dataset = load_dataset("glue", data_type)
    elif data_type == "boolq":
        dataset = load_dataset("super_glue", data_type)
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")

    # Split the training dataset into 85% train and 15% new validation
    train_dataset = dataset["train"].train_test_split(test_size=0.15, seed=seed)

    # Set the new splits
    dataset["train"] = train_dataset["train"]
    dataset["validation"] = train_dataset["test"]
    dataset["test"] = dataset["validation"]  # Set original validation
    return dataset


def create_cola_prompt(sample):
    bos_token = "<s>"
    instruction = "Is the following 'input' sentence is grammatically correct? Respond in 'yes' or 'no'."
    input = sample["sentence"]
    response = "yes" if sample["label"] == 1 else "no"
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += f"[INST]### Instruction: {instruction}"
    full_prompt += f"### Input: {input}[/INST]"
    full_prompt += f"### Response: {response}"
    full_prompt += eos_token

    return full_prompt


def create_boolq_prompt(sample):
    bos_token = "<s>"
    instruction = (
        "Based on the following 'passage', determine if the 'question' is true or false. Respond in 'yes' or 'no'."
    )
    passage = sample["passage"]
    question = sample["question"]
    response = "yes" if sample["label"] == 1 else "no"
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += f"[INST]### Instruction: {instruction}"
    full_prompt += f"### Passage: {passage}"
    full_prompt += f"### Question: {question}[/INST]"
    full_prompt += f"### Response: {response}"
    full_prompt += eos_token

    return full_prompt


def formatting_func_cola(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Is the following 'input' sentence is grammatically correct? Respond in 'yes' or 'no'."
    eos_token = "</s>"
    for idx in range(len(samples["sentence"])):
        input = samples["sentence"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Input: {input}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)
    return formatted_prompts


def formatting_func_boolq(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Based on the following 'passage', is the 'question' true or false? Respond in 'yes' or 'no'."
    eos_token = "</s>"

    for idx in range(len(samples["question"])):
        passage = samples["passage"][idx]
        question = samples["question"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Passage: {passage}"
        full_prompt += f"### Question: {question}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)

    return formatted_prompts


def find_left_of_a(np_array, a):
    if not isinstance(np_array, np.ndarray):
        raise ValueError("Input must be a 2D NumPy array.")

    result = []
    for row in np_array:
        if row.size <= 1:
            raise ValueError("Sublist length must be greater than 1.")
        try:
            index = np.where(row == a)[0][0]  # Find the first occurrence of 'a'
            if index > 0:
                result.append(row[index - 1])
        except IndexError:
            # This occurs if 'a' is not found in the row
            print("ERROR: No EOS Token is found in the prompt when evaluating.")
            pass
    return np.array(result)


# https://github.com/huggingface/trl/issues/862
def extract_response(logits, labels):
    # eos_token_id = tokenizer.eos_token_id
    # logits = find_left_of_a(logits, eos_token_id)
    # labels = find_left_of_a(labels, eos_token_id)
    new_logits = []
    new_labels = []
    for row_idx in range(len(labels)):
        new_logits_row = []
        new_labels_row = []
        logits_row = logits[row_idx]
        labels_row = labels[row_idx]

        for idx in range(len(labels[0])):
            if labels_row[idx] not in [
                -100,
                2,
                # tokenizer.eos_token_id,
                # tokenizer.pad_token_id,
            ]:
                new_logits_row.append(logits_row[idx - 1])
                new_labels_row.append(labels_row[idx])
        if len(new_logits_row) > 0:
            new_logits.append(new_logits_row)
            new_labels.append(new_labels_row)

    return np.array(new_logits, dtype=int), np.array(new_labels, dtype=int)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    # Extract values at these positions
    values_at_yes = logits[..., MISTRAL_YES_ID]
    values_at_no = logits[..., MISTRAL_NO_ID]

    # Compare and get the index where the value is larger
    # If the value at c1 is greater, return c1; otherwise, return c2
    max_value_indices = torch.where(values_at_yes > values_at_no, MISTRAL_YES_ID, MISTRAL_NO_ID)

    return max_value_indices


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    preds, labels = extract_response(preds, labels)

    accuracy_metric = load("accuracy")
    results = accuracy_metric.compute(predictions=preds, references=labels)

    return results


def train(exp_config, use_wandb: bool = True, use_sweep: bool = False):
    print(f"Config: {exp_config}")
    if use_wandb:
        if is_running_deepspeed():
            if int(os.environ["LOCAL_RANK"]) == 0:
                wandb.init()
            if use_sweep:
                exp_config = wandb.config
            else:
                wandb.config = exp_config
            time.sleep(10)
        else:
            wandb.init()
            if use_sweep:
                exp_config = wandb.config
            else:
                wandb.config = exp_config

    model_name = exp_config.model_name
    num_epochs = exp_config.num_epochs
    train_batch_size = exp_config.train_batch_size
    test_batch_size = exp_config.test_batch_size
    gradient_checkpointing = exp_config.gradient_checkpointing
    push_to_hub = exp_config.push_to_hub
    is_debugging = exp_config.is_debugging
    dataset_type = exp_config.dataset_type
    is_plot = exp_config.is_plot
    seed = exp_config.seed

    # If not using sparse Mistral model, all flags related to sparse model should be set as zero or false.
    if not exp_config.use_sparse_model:
        exp_config.use_sparse_regularization = (
            exp_config.print_sparsity
        ) = exp_config.print_act_stats = exp_config.set_sparsity_aware_threshold = False
        exp_config.targeted_sparsity = 0
        exp_config.sparse_model_dir = None

    if exp_config.use_relu:
        exp_config.targeted_sparsity = "ReLU"

    run_name = f"{model_name}_{dataset_type}_{exp_config.targeted_sparsity}"
    folder_name = "task_finetuning"
    if is_debugging:
        folder_name = "debugging_" + folder_name

    # Load dataset
    dataset = get_dataset_for_instruction_tuning(dataset_type, seed=seed)
    formatting_func = get_formatting_func(dataset_type)
    max_seq_length = 2048

    checkpoint_dir = os.path.join(exp_config.checkpoint_dir, folder_name, run_name)
    results_dir = os.path.join(exp_config.results_dir, folder_name, run_name)
    act_hist_path = os.path.join(exp_config.results_dir, folder_name, model_name, f"{dataset_type}.pt")

    if is_debugging:
        dataset["train"] = Dataset.from_dict(dataset["train"][:300])
        dataset["validation"] = Dataset.from_dict(dataset["validation"][:300])
        dataset["test"] = Dataset.from_dict(dataset["test"][:])

    # Load model
    model, tokenizer, config = prepare_sparse_model(
        is_debugging=is_debugging,
        use_sparse_model=exp_config.use_sparse_model,
        use_sparse_regularization=exp_config.use_sparse_regularization,
        use_sparse_predictor=exp_config.use_spm,
        sparse_model_dir=exp_config.sparse_model_dir,
        use_lora=exp_config.use_lora,
        use_flash_attn=exp_config.use_flash_attn,
        base_model_name="mistralai/Mistral-7B-v0.1",
    )
    print(estimate_zero2_model_states_mem_needs_all_live(model))

    def tokenize(element):
        outputs = tokenizer(
            formatting_func(element),
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # Roundabout way to avoid error
    test_dataset = dataset["test"].map(
        tokenize,
        batched=True,
        remove_columns=dataset["test"].column_names,
        num_proc=None,
        batch_size=test_batch_size,
    )

    # Use only 1000 samples of training dataset to collect statistics
    if exp_config.set_sparsity_aware_threshold:
        sampled_train_dataset = Dataset.from_dict(dataset["train"][:1000])
        sampled_train_dataset = sampled_train_dataset.map(
            tokenize,
            batched=True,
            remove_columns=sampled_train_dataset.column_names,
            num_proc=None,
            batch_size=train_batch_size,
        )

    if use_wandb:
        if is_debugging:
            run_name += "_debugging"
        run_name += "_" + get_datetime(True)
        wandb.name = run_name

        if is_running_deepspeed():
            if is_mainprocess():
                wandb_run = wandb.init(
                    project=f"{model_name}_{dataset_type}",
                    name=run_name,
                    reinit=True,
                    config=exp_config,
                )
        else:
            wandb_run = wandb.init(
                project=f"{model_name}_{dataset_type}",
                name=run_name,
                reinit=True,
                config=exp_config,
            )

    response_template = "Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer_config = TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy="steps",
        eval_steps=50,  # early stopping counts only when eval step and save step match
        max_steps=10 if is_debugging else -1,
        # save_steps=50,
        logging_steps=5,
        save_strategy="steps",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir=results_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=False,
        # report_to="wandb" if use_wandb else None,
        gradient_accumulation_steps=exp_config.gradient_accumulation_steps,
        deepspeed=exp_config.ds_config_path if not is_debugging else None,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True if not is_debugging else False,
        ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{model_name}",
        push_to_hub=push_to_hub,
        seed=seed,
        data_seed=seed,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.002)
    trainer = SparseSFTTTrainer(
        model=model,
        # peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=formatting_func,
        args=trainer_config,
        train_dataset=Dataset.from_dict(dataset["train"][:]),
        eval_dataset=Dataset.from_dict(dataset["validation"][:]),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping],
        use_sparse_regularization=exp_config.use_sparse_regularization,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # See https://github.com/huggingface/transformers/issues/23170

    if exp_config.use_lora:
        base_model = model.get_base_model()
    else:
        base_model = model

    # Collect statistics and find the threshold for given sparsity
    if exp_config.set_sparsity_aware_threshold:
        activate_stats(base_model)
        if os.path.exists(act_hist_path):
            load_act_hist(base_model, act_hist_path)
        else:
            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = (
                False  # A way to go around the value error when using ds stage 2 for evaluation
            )
            trainer.evaluate(sampled_train_dataset)
            trainer.is_deepspeed_enabled = is_deepspeed_enabled
            save_act_hist(base_model, act_hist_path)
        if exp_config.use_relu:
            set_sparse_threshold(base_model, 0, True)
        else:
            set_sparse_threshold(base_model, exp_config.targeted_sparsity)

    # Plot activation histograms
    if is_plot:
        if is_running_deepspeed():
            if is_mainprocess():
                plot_activation_histogram(base_model)
        else:
            plot_activation_histogram(base_model)

    # Train Model after deactivating collecting statistics
    deactivate_stats(base_model)

    # Only use distillation loss if training sparse predictor
    if exp_config.use_spm:
        if exp_config.use_sparse_model:
            trainer.freeze_original_weights = False
            trainer.initialize_sparse_decoder_layers(base_model)
            trainer.use_spm_loss = True
            disable_sparse_predictor(base_model)

            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = (
                False  # A way to go around the value error when using ds stage 2 for evaluation
            )
            print(trainer.evaluate(test_dataset))
            trainer.is_deepspeed_enabled = is_deepspeed_enabled

            enable_sparse_predictor(base_model)
        else:
            warnings.warn("use_spm arg is ignored as use_spare_model arg is not activated.")
    trainer.train()

    # Evaluate on test dataset
    if exp_config.use_lora:
        model = model.merge_and_unload()

    # Enable sparse predictor in inference stage
    if exp_config.use_spm:
        if exp_config.use_sparse_model:
            enable_sparse_predictor(model)
        else:
            warnings.warn("use_spm arg is ignored as use_spare_model arg is not activated.")

    if exp_config.print_sparsity:
        activate_stats(model, exp_config.print_act_stats)
    eval_result = trainer.evaluate(test_dataset)
    print("===Test Result==")
    print(eval_result)
    log_dict = {
        "test_accuracy": float(eval_result["eval_accuracy"]),
    }

    if exp_config.print_sparsity:
        total_sparsity = print_dead_neuron_stats(model)
        log_dict.update({"total_sparsity": float(total_sparsity)})

    if exp_config.print_act_stats:
        save_act_hist(model, act_hist_path)
        plot_activation_histogram(model)

    if exp_config.model_save:
        # Save thresholds
        if exp_config.use_sparse_model:
            thresholds = [float(m.mlp.dead_threshold) for m in model.model.layers]
            model.config.thresholds = thresholds
        model.config.save_pretrained(checkpoint_dir)
        model.save_pretrained(checkpoint_dir)

    if push_to_hub:
        trainer.model = model
        trainer.push_to_hub()

    if use_wandb:
        if is_running_deepspeed():
            if is_mainprocess():
                wandb.log(log_dict)
        else:
            wandb.log(log_dict)

    # Also save log dictionary in json file
    log_dict_filename = os.path.join(results_dir, "log.json")
    os.makedirs(
        os.path.dirname(log_dict_filename),
        exist_ok=True,
    )
    try:
        with open(log_dict_filename, "w") as f:
            json.dump(log_dict, f, indent=4)
        print("Metrics successfully saved.")
    except Exception as e:
        print("Failed to dump metrics:", e)

    # Empty out gpu memory
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    if use_wandb:
        wandb.finish()


def prepare_sparse_model(
    is_debugging=False,
    use_sparse_model: bool = True,
    use_sparse_regularization: bool = False,
    use_sparse_predictor: bool = False,
    use_graceful_regularization: bool = False,
    sparse_model_dir: str = None,
    use_lora: bool = True,
    use_flash_attn: bool = False,
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    gradient_checkpointing: bool = False,
    use_relu: bool = False,
    cutoff_large: bool = False,
):
    # Register for AutoConfig and AutoModelforCausalLM
    SparseMistralConfig.register_for_auto_class()
    SparseMistralforCausalLM.register_for_auto_class("AutoModelForCausalLM")

    if use_flash_attn:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = None
    if sparse_model_dir is None:
        if is_debugging:
            config = MistralConfig(
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=4,
                # num_attention_heads=4,
            )
            if use_sparse_model:
                config = get_sparse_mistral_config(config)
                model = SparseMistralforCausalLM(config=config)
            else:
                model = MistralForCausalLM(config=config)
        else:
            config = MistralConfig.from_pretrained(base_model_name)
            config = get_sparse_mistral_config(config)
            config.use_cache = True
            model = SparseMistralforCausalLM.from_pretrained(
                base_model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            model.config_class = SparseMistralConfig

        if use_sparse_model:
            model.config.use_sparse_model = use_sparse_model
            apply_mistral_sparse_silu_mlp(model, model.config, use_sparse_regularization=use_sparse_regularization)
            enable_sparse_silu(model)
        if use_sparse_predictor:
            model.config.use_sparse_predictor = use_sparse_predictor
            apply_mistral_sparse_decoder_layer(model, model.config, init_svd=not is_debugging)
        if use_sparse_regularization:
            model.config.us_sparse_regularization = use_sparse_regularization
        if use_graceful_regularization:
            model.config.use_graceful_regularization = use_graceful_regularization
    else:
        config = SparseMistralConfig.from_pretrained(sparse_model_dir)
        print(config)
        model = SparseMistralforCausalLM.from_pretrained(
            sparse_model_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        if use_sparse_predictor:
            svd_model_dir = sparse_model_dir + "_svd"
            init_svd = not os.path.exists(svd_model_dir)
            model.config.use_sparse_predictor = use_sparse_predictor

            if init_svd:
                apply_mistral_sparse_decoder_layer(
                    model,
                    model.config,
                    init_svd=False,
                )
                model.config.init_svd = False
                model.save_pretrained(svd_model_dir)
            else:
                config = SparseMistralConfig.from_pretrained(svd_model_dir)
                model = SparseMistralforCausalLM.from_pretrained(
                    svd_model_dir,
                    config=config,
                )
    model.config.use_relu = use_relu
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # See https://github.com/huggingface/transformers/issues/23170
    if use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())

    # Register for AutoConfig and AutoModelforCausalLM
    # SparseMistralConfig.register_for_auto_class()
    # SparseMistralforCausalLM.register_for_auto_class("AutoModelForCausalLM")
    # model.config.register_for_auto_class()
    # model.register_for_auto_class("AutoModelForCausalLM")

    return model, tokenizer, model.config


class BooleanOptionalAction(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        super(BooleanOptionalAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values or self.const)


if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    args = parse_args()
    print(args)
    train(args, use_wandb=args.use_wandb)
    # torch.cuda.memory._dump_snapshot("snapshot_new.pickle")
