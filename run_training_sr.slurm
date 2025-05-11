#!/bin/bash -l

#SBATCH --job-name=juddges_sft
#SBATCH --output=logs/%j-%x.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --gres=storage:lustre

srun python embedings_prediction_mean.py
        --data_dir data/lmsys_vicuna-13b_preview0_100K
        --output_dir results/embeddings_predictor_mean
        --do_train
        --do_eval
        --use_wandb
        --wandb_project output-length-prediction
        --layer_indices "[10, 11, 12]"
        --aggregation mean
        --batch_size 10
        --learning_rate 0.01
        --num_epochs 30
        --seed 42