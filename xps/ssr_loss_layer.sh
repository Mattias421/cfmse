#!/bin/bash
#SBATCH --partition=gpu-h100,gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm/%x-%a.out
#SBATCH --array=1-12

module load Anaconda3/2022.05
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load GCCcore/11.3.0

source activate cfmse

layer=$SLURM_ARRAY_TASK_ID
model=wavlm

export WANDB_NAME=${model}_l${layer}
export WANDB_TAGS=("${model}_loss_layer")
python train.py --base_dir $DATA/VB+DMD --max_epochs 300 --backbone ncsnpp_v2 --sde icfm --loss_type flow_matching --wandb_name $WANDB_NAME --log_dir ./logs/${model}_loss_layer/${WANDB_NAME} --sigma $sigma --l1_weight $l1_weight --ssr_weight 0.001 --ssr_layer $layer --ssr_model $model
