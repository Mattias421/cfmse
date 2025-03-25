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
#SBATCH --array=0-1


module load Anaconda3/2019.07
module load CUDA/12.4.0 
module load GCCcore/10.3.0
source activate cfmse

#root_dir=$EXP/cfmse/logs/icfm_sigma_sweep/46pucckl
#root_dir=$EXP/cfmse/logs/icfm_fm_sweep/icfm_sigma=0.1_l1_weight=0.001/6foffv3i
#root_dir=$EXP/cfmse/logs/sbve_no_pesq
#root_dir=$EXP/cfmse/logs/sbve_stat_sigma_xt/7arzavsi
#root_dir=$EXP/cfmse/logs/sbve_stat_sigma_xt_k1/j39kz2aj


#paths=(icfm_sigma_sweep/46pucckl icfm_fm_sweep/icfm_sigma=0.1_l1_weight=0.001/6foffv3i sbve_no_pesq sbve_stat_sigma_xt/7arzavsi sbve_stat_sigma_xt_k1/j39kz2aj)
paths=(clean noisy)
root_dir=$EXP/cfmse/logs/${paths[$SLURM_ARRAY_TASK_ID]}

echo "Evaluating ${root_dir}"

echo "Enhancing"
#python enhancement.py --test_dir $DATA/VB+DMD/test/noisy --enhanced_dir ${root_dir}/enhanced_dp_v2 --ckpt ${root_dir}/epoch=*pesq*.ckpt --sampler_type dp

echo "Calculating metrics"
python calc_metrics.py --clean_dir $DATA/VB+DMD/test/clean --noisy_dir $DATA/VB+DMD/test/noisy --enhanced_dir $DATA/VB+DMD/test/${paths[$SLURM_ARRAY_TASK_ID]}

echo "Computing WhiSQA"
#cd $EXP/WhiSQA/
#python get_score_batch.py --model_type multi --output_csv ${root_dir}/enhanced_dp_v2/_results_whisqa.csv --output_txt ${root_dir}/enhanced_dp_v2/_avg_results_whisqa.txt ${root_dir}/enhanced_dp_v2

echo "Computing DNSMOS"
#cd $EXP/DNS-Challenge/DNSMOS/
#python dnsmos_local.py -t ${root_dir}/enhanced_dp_v2 -o ${root_dir}/enhanced_dp_v2/_results_dnsmos.csv
#awk -F',' '{sum+=$12; ++n} END { print "DNSMOS: " sum/(n-1) }' < ${root_dir}/enhanced_dp_v2/_results_dnsmos.csv > ${root_dir}/enhanced_dp_v2/_results_avg_dnsmos.txt

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
