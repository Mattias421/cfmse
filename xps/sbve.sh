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

export WANDB_NAME=sbve_no_pesq
python train.py --base_dir $DATA/VB+DMD/ --backbone ncsnpp_v2 --sde sbve --loss_type data_prediction --max_epochs 300

python enhancement.py --test_dir $DATA/VB+DMD/test/noisy --enhanced_dir logs/${WANDB_NAME}/enhanced_${ckpt_metric} --ckpt logs/${WANDB_NAME}/epoch=*-${ckpt_metric}=*.ckpt

python calc_metrics.py --clean_dir $DATA/VB+DMD/test/clean --noisy_dir $DATA/VB+DMD/test/noisy --enhanced_dir logs/${WANDB_NAME}/enhanced_${ckpt_metric}
