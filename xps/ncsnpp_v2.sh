#!/bin/bash
#SBATCH --partition=gpu-h100,gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/slurm/%x.log

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCCcore/12.3.0

source activate cfmse

python train.py --base_dir $DATA/VB+DMD/ --backbone ncsnpp_v2
