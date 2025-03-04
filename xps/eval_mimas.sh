export OMP_NUM_THREADS=8
root_dir=$EXP/cfmse/logs/nn_path_stable
root_dir=$EXP/cfmse/logs/2k381hhl

echo "evaluating ${root_dir}"

python enhancement.py --test_dir $DATA/VB+DMD/valid/noisy --enhanced_dir ${root_dir}/enhanced --ckpt ${root_dir}/epoch=*pesq*.ckpt

python calc_metrics.py --clean_dir $DATA/VB+DMD/test/clean --noisy_dir $DATA/VB+DMD/test/noisy --enhanced_dir ${root_dir}/enhanced

# WhiSQA calculation
cd $EXP/WhiSQA/
python get_score_batch.py --model_type multi --output_csv ${root_dir}/enhanced/_results_whisqa.csv --output_txt ${root_dir}/enhanced/_avg_results_whisqa.txt ${root_dir}/enhanced

# DNSMOS calculation
cd $EXP/DNS-Challenge/DNSMOS/
python dnsmos_local.py -t ${root_dir}/enhanced -o ${root_dir}/enhanced/_results_dnsmos.csv
awk -F',' '{sum+=$12; ++n} END { print "DNSMOS: " sum/(n-1) }' < ${root_dir}/enhanced/_results_dnsmos.csv > ${root_dir}/enhanced/_results_avg_dnsmos.txt
