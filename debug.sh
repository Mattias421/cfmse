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
#SBATCH --array=1


module load Anaconda3/2019.07
module load CUDA/12.4.0 
module load GCCcore/10.3.0
source activate cfmse
pwd
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

layer=$SLURM_ARRAY_TASK_ID
# xp_id=$(ls -I *=* logs/sbve_stat_sigma_xt/ | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)
# root_dir=$EXP/cfmse/logs/sbve_stat_sigma_xt/${xp_id}
root_dir=$EXP/cfmse/logs/sbve_stat_sigma_xt_k1
xp_id=$(ls -t $root_dir --ignore="*=*" | head -$SLURM_ARRAY_TASK_ID | tail -1)
root_dir=$root_dir/$xp_id

echo "boutta debug"
python debug.py


# stanage
##!/bin/bash
##SBATCH --partition=gpu-h100,gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:1
##SBATCH --time=50:00:00
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=32G
##SBATCH --output=logs/slurm/%x.log
##SBATCH --array=1-16
#
#module load Anaconda3/2022.05
#module load cuDNN/8.9.2.26-CUDA-12.1.1
#module load GCCcore/12.3.0
