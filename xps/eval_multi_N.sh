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
#SBATCH --array=1,5,10,20,30,40,50


module load Anaconda3/2019.07
module load CUDA/12.4.0 
module load GCCcore/10.3.0
source activate cfmse

root_dir=$EXP/cfmse/logs/icfm_fm_sweep/icfm_sigma=0.1_l1_weight=0.001/6foffv3i
N=$SLURM_ARRAY_TASK_ID

python enhancement.py --test_dir $DATA/VB+DMD/test/noisy --enhanced_dir ${root_dir}/enhanced_${N} --ckpt ${root_dir}/epoch=*pesq*.ckpt --N $N

python calc_metrics.py --clean_dir $DATA/VB+DMD/test/clean --noisy_dir $DATA/VB+DMD/test/noisy --enhanced_dir ${root_dir}/enhanced_${N}

# WhiSQA calculation
cd $EXP/WhiSQA/
python get_score_batch.py --model_type multi --output_csv
${root_dir}/enhanced_${N}/_results_whisqa.csv --output_txt
${root_dir}/enhanced_${N}/_avg_results_whisqa.txt ${root_dir}/enhanced_${N}

# DNSMOS calculation
cd $EXP/DNS-Challenge/DNSMOS/
python dnsmos_local.py -t ${root_dir}/enhanced_${N} -o
${root_dir}/enhanced_${N}/_results_dnsmos.csv
awk -F',' '{sum+=$12; ++n} END { print "DNSMOS: " sum/(n-1) }'
< ${root_dir}/enhanced_${N}/_results_dnsmos.csv > ${root_dir}/enhanced_${N}/_results_avg_dnsmos.txt

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
