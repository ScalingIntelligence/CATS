#!/bin/bash

initial_steps=100
step_increment=250
max_iterations=5
targeted_sparsity_list=(0.5 0.7 0.9)
model_name="sparse_mistral_7b"
base_model_repo_id="mistralai/Mistral-7B-v0.1"

for targeted_sparsity in "${targeted_sparsity_list[@]}"; do
  sparsity_percentage=$(printf "%.0f" $(echo "$targeted_sparsity * 100" | bc))
  echo "Targeted Sparsity Percentage: $sparsity_percentage"
  is_first_training=1
  for ((i=1; i<=max_iterations; i++)); do
      current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

      echo "Training for $current_steps steps..."
      deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
        --use_sparse_model --targeted_sparsity $targeted_sparsity \
        --set_sparsity_aware_threshold --print_sparsity \
        --use_wandb --max_steps $current_steps --model_save \
        --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 8 \
        --ds_config_path ds_config.json --max_seq_length 1024  \
        --checkpoint_dir $1 --results_dir $2 --is_first_training $is_first_training \
        --gradient_checkpointing \
        --model_name $model_name \
        --base_model_repo_id $base_model_repo_id \
        --process_index 1

      is_first_training=0
      model_directory=$(cat model_directory1.txt)
      echo "model directory: $model_directory"

      echo "Evaluating after $current_steps steps..."
      accelerate launch -m --main_process_port 12329 lm_eval \
          --model hf \
          --model_args pretrained=$model_directory,trust_remote_code=True \
          --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
          --batch_size 32 \
          --log_samples \
          --output_path ${2}/evaluations/mistral_sparse_${sparsity_percentage}p_${current_steps}steps2
  done
done
