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
#SBATCH --array=0-15

module load Anaconda3/2022.05
module load GCCcore/12.3.0
module load cuDNN/8.9.2.26-CUDA-12.1.1


source activate cfmse

array=("0n52exz7" "0p27oa57" "2hqvja7z" "2t2eudjv" "594lx3dt" "c6rvx7ck" "d4hg8fag" "exli5xss" "hcrt9hsh" "ieq9g228" "ljgevzv7" "pvnb0p90" "pxd0h4rx" "u7qjqp73" "xi9txau8" "y6md6luk")
# array=("ieq9g228")
# array=("0n52exz7")

xp_code="${array[$SLURM_ARRAY_TASK_ID]}"

root_dir=$EXP/cfmse/logs/sbve_ck_sweep/${xp_code}

echo "evaluating ${root_dir}"

# python enhancement.py --test_dir $DATA/VB+DMD/test/noisy --enhanced_dir ${root_dir}/enhanced --ckpt ${root_dir}/epoch=*pesq*.ckpt
# python calc_metrics.py --clean_dir $DATA/VB+DMD/test/clean --noisy_dir $DATA/VB+DMD/test/noisy --enhanced_dir ${root_dir}/enhanced

# WhiSQA calculation
# cd $EXP/WhiSQA/
# python get_score_batch.py --model_type multi --output_csv ${root_dir}/enhanced/_results_whisqa.csv --output_txt ${root_dir}/enhanced/_avg_results_whisqa.txt ${root_dir}/enhanced

# DNSMOS calculation
cd $EXP/DNS-Challenge/DNSMOS/
python dnsmos_local.py -t ${root_dir}/enhanced -o ${root_dir}/enhanced/_results_pdnsmos.csv -p
awk -F',' '{sum+=$12; ++n} END { print "DNSMOS: " sum/(n-1) }' < ${root_dir}/enhanced/_results_pdnsmos.csv > ${root_dir}/enhanced/_avg_results_pdnsmos.txt
