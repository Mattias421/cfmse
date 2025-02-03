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
#SBATCH --array=1-16

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCCcore/12.3.0

source activate cfmse

# Define the values for c and k
c_values=(0.01 0.13 0.375 0.5)
k_values=(0.1 0.99 1.99 2.99)

# Calculate the index for c and k based on SLURM_ARRAY_TASK_ID
c_index=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
k_index=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))

# Get the c and k values
c=${c_values[$c_index]}
k=${k_values[$k_index]}

export WANDB_NAME=sbve_c=${c}_k=${k}
python train.py --base_dir $DATA/VB+DMD --backbone ncsnpp_v2 --sde sbve --loss_type data_prediction --wandb_name $WANDB_NAME --log_dir ./logs/sbve_ck_sweep --c $c --k $k
