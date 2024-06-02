#!/bin/bash

echo "Plotting the MLP activation histograms for Llama"
deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
  --use_sparse_model --targeted_sparsity 0.5 \
  --set_sparsity_aware_threshold --print_sparsity --is_plot \
  --max_steps 1 \
  --train_batch_size 1 --test_batch_size 1 --use_flash_attn \
  --ds_config_path ds_config.json --max_seq_length 1024  \
  --checkpoint_dir $1 --results_dir $2 --is_first_training 1 \
  --gradient_checkpointing \
  --model_name "sparse_llama_7b_hf2" \
  --base_model_repo_id "meta-llama/Llama-2-7b-hf"

echo "Plotting the MLP activation histograms for Mistral"
deepspeed --master_port 12330 experiments/pretrain_sparse_model.py \
  --use_sparse_model --targeted_sparsity 0.5 \
  --set_sparsity_aware_threshold --print_sparsity --is_plot \
  --max_steps 1 \
  --train_batch_size 1 --test_batch_size 1 --use_flash_attn \
  --ds_config_path ds_config.json --max_seq_length 1024  \
  --checkpoint_dir $1 --results_dir $2 --is_first_training 1 \
  --gradient_checkpointing \
  --model_name "sparse_mistral_7b" \
  --base_model_repo_id "mistralai/Mistral-7B-v0.1"

