#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/slurm/%x-%a.out
#SBATCH --array=1-5

# module load Anaconda3/2022.05
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# module load GCCcore/12.3.0

module load Anaconda3/2019.07
module load cuDNN/8.0.4.30-CUDA-11.1.1


source activate cfmse

# Define the values for c and k
sigma_values=(0.01 0.1 0.15 0.2 0.3)

sigma=${sigma_values[$((SLURM_ARRAY_TASK_ID - 1))]}

export WANDB_NAME=icfm_sigma=${sigma}
python train.py --base_dir $DATA/VB+DMD --backbone ncsnpp_v2 --sde icfm --loss_type data_prediction --wandb_name $WANDB_NAME --log_dir ./logs/icfm_sigma_sweep --sigma $sigma
