#!/bin/bash

# Initial steps and increment
initial_steps=100
step_increment=250
max_iterations=5
is_first_training=1

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
      --set_sparsity_aware_threshold \
      --use_sparse_model --print_sparsity \
      --use_wandb --max_steps $current_steps --model_save \
      --train_batch_size 1 --test_batch_size 2 --use_flash_attn \
      --gradient_accumulation_steps 4 --ds_config_path ds_config.json \
      --max_seq_length 1024  \
      --checkpoint_dir $1 --results_dir $2 \
      --gradient_checkpointing \
      --is_first_training $is_first_training \
      --model_name "relu_mistral_7b" \
      --base_model_repo_id mistralai/Mistral-7B-v0.1 \
      --use_relu --process_index 2

    is_first_training=0
    model_directory=$(cat model_directory2.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 16 \
        --log_samples \
        --output_path ${2}/evaluations/relu_mistral_${current_steps}steps
done


