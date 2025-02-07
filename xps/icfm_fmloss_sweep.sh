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
#SBATCH --array=1-12

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCCcore/12.3.0

source activate cfmse

sigma_values=(0.1 0.15 0.2 0.3)
l1_weights=(0.001 0.0005 0)

i=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
j=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))

sigma="${sigma_values[$i]}"
l1_weight="${l1_weights[$j]}"

export WANDB_NAME=icfm_sigma=${sigma}_l1_weight=${l1_weight}
python train.py --base_dir $DATA/VB+DMD --backbone ncsnpp_v2 --sde icfm --loss_type flow_matching --wandb_name $WANDB_NAME --log_dir ./logs/icfm_fm_sweep/${WANDB_NAME} --sigma $sigma --l1_weight $l1_weight
