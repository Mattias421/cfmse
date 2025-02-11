#!/bin/bash
#SBATCH --partition=gpu-h100,gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/slurm/%x-%a.out
#SBATCH --array=0-4

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCCcore/12.3.0

source activate cfmse

# Define the values for c and k
c_values=(0.1 0.15 0.2 0.25 0.3)

# Get the c and k values
c=${c_values[$SLURM_ARRAY_TASK_ID]}

export WANDB_NAME=sbve_c=${c}_stat_sigma_xt
python train.py --base_dir $DATA/VB+DMD --backbone ncsnpp_v2 --sde sbve --loss_type data_prediction --variance_type stationary --max_epochs 300 --wandb_name $WANDB_NAME --log_dir ./logs/sbve_stat_sigma_xt --c $c
