#!/bin/bash

##########################################
#             7B-Based models            #
##########################################
# Zero-shot accuracy wo fine-tuning
# Original Llama
accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 8 \
        --log_samples \
        --output_path ${2}/evaluations/llama_7b_hf

# Llama 50%, 70%, and 90%
initial_steps=0
step_increment=1
max_iterations=1
is_first_training=1
targeted_sparsity_list=(0.5 0.7 0.9)
model_name="sparse_llama_7b_hf"
base_model_repo_id="meta-llama/Llama-2-7b-hf"

for targeted_sparsity in "${targeted_sparsity_list[@]}"; do
  sparsity_percentage=$(printf "%.0f" $(echo "$targeted_sparsity * 100" | bc))
  echo "Targeted Sparsity Percentage: $sparsity_percentage"
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

      model_directory=$(cat model_directory1.txt)
      echo "model directory: $model_directory"

      echo "Evaluating after $current_steps steps..."
      accelerate launch -m --main_process_port 12329 lm_eval \
          --model hf \
          --model_args pretrained=$model_directory,trust_remote_code=True \
          --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
          --batch_size 32 \
          --log_samples \
          --output_path ${2}/evaluations/llama_sparse_${sparsity_percentage}p_${current_steps}steps2
  done
done

# Llama2 ReLU
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
      --model_name "relu_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf" \
      --use_relu --process_index 2

    model_directory=$(cat model_directory2.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 16 \
        --log_samples \
        --output_path ${2}/evaluations/relu_llama_${current_steps}steps
done

# Original Mistral
accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=mistralai/Mistral-7B-v0.1,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 8 \
        --log_samples \
        --output_path output/mistral_base

# Mistral 50%, 70%, and 90%
model_name="sparse_mistral_7b"
base_model_repo_id="mistralai/Mistral-7B-v0.1"

for targeted_sparsity in "${targeted_sparsity_list[@]}"; do
  sparsity_percentage=$(printf "%.0f" $(echo "$targeted_sparsity * 100" | bc))
  echo "Targeted Sparsity Percentage: $sparsity_percentage"
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

# Mistral ReLU
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

##########################################
#            13B-Based models            #
##########################################

# Zero-shot accuracy wo fine-tuning
# Original Llama
accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=meta-llama/Llama-2-13b-hf,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 8 \
        --log_samples \
        --output_path ${2}/evaluations/llama_13b_hf

# Llama 50%, 70%, and 90%
initial_steps=0
step_increment=1
max_iterations=1
is_first_training=1
targeted_sparsity_list=(0.5 0.7 0.9)
model_name="sparse_llama_13b_hf"
base_model_repo_id="meta-llama/Llama-2-13b-hf"

for targeted_sparsity in "${targeted_sparsity_list[@]}"; do
  sparsity_percentage=$(printf "%.0f" $(echo "$targeted_sparsity * 100" | bc))
  echo "Targeted Sparsity Percentage: $sparsity_percentage"
  for ((i=1; i<=max_iterations; i++)); do
      current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

      echo "Training for $current_steps steps..."
      deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
        --use_sparse_model --targeted_sparsity $targeted_sparsity \
        --set_sparsity_aware_threshold --print_sparsity \
        --use_wandb --max_steps $current_steps --model_save \
        --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 8 \
        --ds_config_path ds_config.json --max_seq_length 1024  \
        --checkpoint_dir $1 --results_dir $2 --is_first_training $is_first_training \
        --gradient_checkpointing \
        --model_name $model_name \
        --base_model_repo_id $base_model_repo_id \
        --process_index 1

      model_directory=$(cat model_directory1.txt)
      echo "model directory: $model_directory"

      echo "Evaluating after $current_steps steps..."
      accelerate launch -m --main_process_port 12329 lm_eval \
          --model hf \
          --model_args pretrained=$model_directory,trust_remote_code=True \
          --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
          --batch_size 2 \
          --log_samples \
          --output_path ${2}/evaluations/llama_sparse_${sparsity_percentage}p_${current_steps}steps1
  done
done

# Llama2 ReLU
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
      --model_name "relu_llama_13b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-13b-hf" \
      --use_relu --process_index 2

    model_directory=$(cat model_directory2.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 2 \
        --log_samples \
        --output_path ${2}/evaluations/relu_llama_${current_steps}steps
done
