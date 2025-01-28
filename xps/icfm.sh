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
#SBATCH --output=slurm/%x.out

module load Anaconda3/2022.05
module load SoX

source activate cfmse

ckpt_metric=pesq

WANDB_NAME=icfm_default
python train.py --base_dir $DATA/VB+DMD/ --backbone ncsnpp_v2 --sde icfm --loss_type flow_matching 

python enhancement.py --test_dir $DATA/VB+DMD/test/noisy --enhanced_dir logs/${WANDB_NAME}/enhanced_${ckpt_metric} --ckpt logs/${WANDB_NAME}/epoch=*-${ckpt_metric}=*.ckpt

python calc_metrics.py --clean_dir $DATA/VB+DMD/test/clean --noisy_dir $DATA/VB+DMD/test/noisy --enhanced_dir logs/${WANDB_NAME}/enhanced_${ckpt_metric}
