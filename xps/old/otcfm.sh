#!/bin/bash
#SBATCH --partition=gpu-h100,gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm/%x-%a.out
#SBATCH --array=0-1

module load Anaconda3/2022.05
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load GCCcore/11.3.0

source activate cfmse
export WANDB_TAGS=("ot_minibatch alpha xp")

if [ $SLURM_ARRAY_TASK_ID == 0]; then

  export WANDB_NAME=unpaired
  python train.py --base_dir $DATA/VB+DMD --max_epochs 300 --backbone ncsnpp_v2 --sde icfm --loss_type flow_matching --wandb_name $WANDB_NAME --log_dir ./logs/unpaired/${WANDB_NAME} 
else
  export WANDB_NAME=unpaired_ot
  python train.py --base_dir $DATA/VB+DMD --max_epochs 300 --backbone ncsnpp_v2 --sde icfm --loss_type flow_matching --wandb_name $WANDB_NAME --log_dir ./logs/unpaired/${WANDB_NAME} --unpaired --ot_minibatch
fi
