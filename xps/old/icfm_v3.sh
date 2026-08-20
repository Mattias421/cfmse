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
#SBATCH --output=logs/slurm/%x.out

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCCcore/12.3.0

source activate cfmse

ckpt_metric=pesq

export WANDB_NOTES="copy sb loss but replace x and x_hat with vt and ut to enable best icfm model for se, use data_prediction loss then reformulate x_1 - x_0 as F(x_t) - y"
export WANDB_NAME=icfm_v3
python train.py --base_dir $DATA/VB+DMD --max_epochs 300 --backbone ncsnpp_v2 --sde icfm --loss_type data_prediction --wandb_name $WANDB_NAME --log_dir ./logs/${WANDB_NAME}
