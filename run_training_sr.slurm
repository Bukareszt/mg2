#!/bin/bash -l

#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=7:00:00
#SBATCH --account=plghallugrant-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --gres=gpu:1
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