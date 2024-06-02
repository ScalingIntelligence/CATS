from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    MistralForCausalLM,
    MistralConfig,
    LlamaForCausalLM,
    LlamaConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
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

from experiments.models.sparse_silu.ugly_utils import *

# from experiments.models.sparse_silu.callbacks import GracefulRegularizationScheduler
# from trainer import SparseTrainer
from utils.utils import get_model_type_from_name
from utils.constants import (
    REFINED_WEB,
)
from utils.utils import ds_print, is_mainprocess, is_running_deepspeed, get_datetime
from utils.parse_args import parse_args
from experiments.data.get_dataset import get_dataset


def get_run_name(exp_config):
    model_name = exp_config.model_name
    dataset_type = exp_config.dataset_type

    run_name = f"{model_name}_{dataset_type}"

    ds_print("USE RELU: ", exp_config.use_relu)
    ds_print("TARGETED SPARSITY: ", exp_config.targeted_sparsity)

    if exp_config.use_relu:
        run_name += "_relu"
    elif exp_config.targeted_sparsity:
        run_name += f"_{int(exp_config.targeted_sparsity * 100)}p"

    if exp_config.use_graceful_regularization:
        run_name += "_graceful_reg"

    return run_name


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
    model_type = get_model_type_from_name(base_model_name)
    BaseConfig = MistralConfig if model_type == MISTRAL else LlamaConfig
    BaseCausalLM = MistralForCausalLM if model_type == MISTRAL else LlamaForCausalLM
    SparseConfig = SparseMistralConfig if model_type == MISTRAL else SparseLlamaConfig
    SparseCausalLM = SparseMistralforCausalLM if model_type == MISTRAL else SparseLlamaForCausalLM
    print(SparseConfig)
    SparseConfig.register_for_auto_class()
    SparseCausalLM.register_for_auto_class("AutoModelForCausalLM")

    if use_flash_attn:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = None
    if sparse_model_dir is None:
        if is_debugging:
            config = BaseConfig(
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=4,
            )
            if use_sparse_model:
                config = get_sparse_config(config, model_type)
                model = SparseCausalLM(config=config)
            else:
                model = BaseCausalLM(config=config)
        else:
            config = BaseConfig.from_pretrained(base_model_name)
            config = get_sparse_config(config, model_type=model_type)
            config.use_cache = True
            model = SparseCausalLM.from_pretrained(
                base_model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            model.config_class = SparseConfig

        if use_sparse_model:
            apply_sparse_silu_mlp(model, model.config, use_sparse_regularization=use_sparse_regularization)
            enable_sparse_silu(model)
        if use_sparse_predictor:
            apply_sparse_decoder_layer(model, model.config, init_svd=not is_debugging)

        model.config.use_sparse_predictor = use_sparse_predictor
        model.config.use_sparse_model = use_sparse_model
        model.config.us_sparse_regularization = use_sparse_regularization
        model.config.use_graceful_regularization = use_graceful_regularization
    else:
        config = SparseConfig.from_pretrained(sparse_model_dir)
        print(config)
        model = SparseCausalLM.from_pretrained(
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
                apply_sparse_decoder_layer(
                    model,
                    model.config,
                    init_svd=False,
                )
                model.config.init_svd = False
                model.save_pretrained(svd_model_dir)
            else:
                config = SparseConfig.from_pretrained(svd_model_dir)
                model = SparseCausalLM.from_pretrained(
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
        ds_print(model.print_trainable_parameters())

    # Register for AutoConfig and AutoModelforCausalLM
    # SparseMistralConfig.register_for_auto_class()
    # SparseMistralforCausalLM.register_for_auto_class("AutoModelForCausalLM")
    # model.config.register_for_auto_class()
    # model.register_for_auto_class("AutoModelForCausalLM")

    return model, tokenizer, model.config


def train(exp_config, use_wandb: bool = True, use_sweep: bool = False):
    ds_print(f"saving to model_directory{exp_config.process_index}.txt")
    ds_print(f"Config: {exp_config}")
    if use_wandb and is_mainprocess():
        wandb.init()
        if use_sweep:
            exp_config = wandb.config
        else:
            wandb.config = exp_config
        if is_running_deepspeed():
            time.sleep(1)

    model_name = exp_config.model_name
    model_type = get_model_type_from_name(model_name)
    gradient_checkpointing = exp_config.gradient_checkpointing
    push_to_hub = exp_config.push_to_hub
    is_debugging = exp_config.is_debugging
    dataset_type = exp_config.dataset_type
    is_plot = exp_config.is_plot
    is_first_training = exp_config.is_first_training

    # Graceful Regularization
    num_warmup_steps = exp_config.num_warmup_steps
    keep_regularization_with_kill = exp_config.keep_regularization_with_kill
    use_graceful_regularization = exp_config.use_graceful_regularization

    use_sparse_regularization = exp_config.use_sparse_regularization
    if use_graceful_regularization and not is_first_training:
        print("No sparse reg!")
        use_sparse_regularization = False

    print("FIRST TRAINING? : ", is_first_training)

    run_name = get_run_name(exp_config)
    folder_name = "general_finetuning"
    if is_debugging:
        folder_name = "debugging_" + folder_name
    checkpoint_dir = os.path.join(exp_config.checkpoint_dir, folder_name, run_name)
    results_dir = os.path.join(exp_config.results_dir, folder_name, exp_config.model_name, run_name)
    fig_dir = os.path.join(results_dir, "figures")
    bin_edge_dir = os.path.join(results_dir, "bin_edges")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(bin_edge_dir, exist_ok=True)
    act_hist_path = os.path.join(
        exp_config.results_dir, folder_name, exp_config.model_name, f"{dataset_type}_activation_histogram.pt"
    )

    # If not using sparse Mistral model, all flags related to sparse model should be set as zero or false.
    if not exp_config.use_sparse_model:
        exp_config.use_sparse_regularization = (
            exp_config.print_sparsity
        ) = exp_config.print_act_stats = exp_config.set_sparsity_aware_threshold = False
        exp_config.targeted_sparsity = 0

    if exp_config.use_relu:
        exp_config.targeted_sparsity = "ReLU"

    # Load model
    model, tokenizer, config = prepare_sparse_model(
        is_debugging,
        exp_config.use_sparse_model,
        use_sparse_regularization,
        exp_config.use_spm,
        exp_config.use_graceful_regularization,
        exp_config.sparse_model_dir,
        exp_config.use_lora,
        exp_config.use_flash_attn,
        base_model_name=exp_config.base_model_repo_id,
        gradient_checkpointing=gradient_checkpointing,
        use_relu=exp_config.use_relu,
    )

    # Load dataset
    dataset = get_dataset(dataset_type, tokenizer, model_type, max_seq_length=exp_config.max_seq_length)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()

    # Use only 500 samples of training dataset to collect statistics
    sampled_train_dataset = train_dataset.select(range(100))

    if use_wandb:
        if is_debugging:
            run_name += "_debugging"
        run_name += "_" + get_datetime(True)
        wandb.name = run_name

        if is_mainprocess():
            wandb.init(
                project=f"{model_name}_{dataset_type}",
                name=run_name,
                reinit=True,
                config=exp_config,
            )

    trainer_config = TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy="steps",
        eval_steps=25,  # early stopping counts only when eval step and save step match
        max_steps=10 if is_debugging else exp_config.max_steps,
        save_steps=min(250, exp_config.max_steps),
        logging_steps=5,
        save_strategy="steps",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=exp_config.num_epochs,
        logging_dir=results_dir,
        save_total_limit=1,
        greater_is_better=False,
        gradient_accumulation_steps=exp_config.gradient_accumulation_steps,
        deepspeed=exp_config.ds_config_path if not is_debugging else None,
        per_device_train_batch_size=exp_config.train_batch_size,
        per_device_eval_batch_size=exp_config.test_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True if not is_debugging else False,
        hub_model_id=f"thrunlab/{run_name}",
        push_to_hub=push_to_hub,
        seed=exp_config.seed,
        data_seed=exp_config.seed,
    )

    graceful_regularization_scheduler = GracefulRegularizationScheduler(
        num_warmup_steps=num_warmup_steps,
        is_enabled=use_graceful_regularization and is_first_training,
        model_name=model_name,
        test_dataset=test_dataset,
        targeted_sparsity=exp_config.targeted_sparsity,
        keep_regularization_with_kill=keep_regularization_with_kill,
    )

    # See https://discuss.huggingface.co/t/using-iterabledataset-with-trainer-iterabledataset-has-no-len/15790/2
    trainer = SparseTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_config,
        train_dataset=train_dataset.with_format("torch"),  # train_dataset.with_format("torch"),
        eval_dataset=val_dataset.with_format("torch"),
        callbacks=[graceful_regularization_scheduler],
        use_sparse_regularization=use_sparse_regularization,
    )
    graceful_regularization_scheduler.set_trainer(trainer)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        config.use_cache = False
        model.config.use_cache = False
        # model.enable_input_require_grads()  # See https://github.com/huggingface/transformers/issues/23170

    if exp_config.use_lora:
        base_model = model.get_base_model()
    else:
        base_model = model

    # Collect statistics and find the threshold for given sparsity
    if not use_graceful_regularization and exp_config.set_sparsity_aware_threshold:
        activate_stats(base_model)
        if os.path.exists(act_hist_path):
            load_act_hist(base_model, act_hist_path)
        elif not exp_config.use_relu:
            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = (
                False  # A way to go around the value error when using ds stage 2 for evaluation
            )
            trainer.evaluate(sampled_train_dataset)
            trainer.is_deepspeed_enabled = is_deepspeed_enabled
            save_act_hist(base_model, act_hist_path)

        print("Setting sparse threshold")
        if exp_config.use_relu:
            set_sparse_threshold(base_model, 0, True)
        else:
            set_sparse_threshold(base_model, exp_config.targeted_sparsity)

    if not use_graceful_regularization and exp_config.print_sparsity:
        activate_stats(base_model, exp_config.print_act_stats)

        is_deepspeed_enabled = trainer.is_deepspeed_enabled
        trainer.is_deepspeed_enabled = False
        eval_result = trainer.evaluate(sampled_train_dataset)
        trainer.is_deepspeed_enabled = is_deepspeed_enabled

        ds_print("===Pre-training test Result==")
        ds_print(eval_result)
        total_sparsity, _ = print_dead_neuron_stats(base_model)
        ds_print(f"pre-training sparsity: {total_sparsity}")

    # Plot activation histograms
    if is_plot:
        if is_running_deepspeed():
            if is_mainprocess():
                plot_activation_histogram(base_model, fig_dir, bin_edge_dir)
        else:
            plot_activation_histogram(base_model, fig_dir, bin_edge_dir)

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

    if exp_config.do_training:
        trainer.train(resume_from_checkpoint=not is_first_training)

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
    log_dict = {}

    log_dict.update({"perplexity": eval_result["eval_loss"]})

    if exp_config.print_sparsity:
        total_sparsity, sparsity_list = print_dead_neuron_stats(model)
        log_dict.update({"total_sparsity": total_sparsity})
        log_dict.update({"sparsity_list": sparsity_list})

    if exp_config.print_act_stats:
        save_act_hist(model, act_hist_path)
        plot_activation_histogram(model, fig_dir, bin_edge_dir)

    if exp_config.model_save:
        # Save thresholds
        if exp_config.use_sparse_model:
            thresholds = [float(m.mlp.dead_threshold) for m in model.model.layers]
            model.config.thresholds = thresholds
        if exp_config.use_relu:
            no_adapter_checkpoint_dir = checkpoint_dir + f"_no_adapter"
        else:
            no_adapter_checkpoint_dir = checkpoint_dir + f"_no_adapter_{exp_config.max_steps}steps"
        with open(f"model_directory{exp_config.process_index}.txt", "w") as f:
            f.write(no_adapter_checkpoint_dir)
        model.save_pretrained(no_adapter_checkpoint_dir)
        tokenizer.save_pretrained(no_adapter_checkpoint_dir)

    if push_to_hub:
        trainer.model = model
        trainer.deepspeed = model
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


if __name__ == "__main__":
    args = parse_args()
    args.dataset_type = REFINED_WEB
    train(args, use_wandb=args.use_wandb)
