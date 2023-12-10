#!/bin/bash -x
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=128G
###SBATCH --cpus-per-task=4
#SBATCH -t 1-00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=CCLM_30k_captions_rationales
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyushkh@andrew.cmu.edu
#SBATCH --partition=babel-shared-long


source /data/tir/projects/tir6/general/piyushkh/conda/bin/activate cclm

variant=$1

echo "CCLM Variant Selected: $variant"
output_dir="CCLM_output_$variant"

echo "Output Directory: $output_dir"

if [ ! -d "$output_dir" ]; then
  mkdir "$output_dir"
  echo "Folder '$output_dir' created successfully."
else
  echo "Folder '$output_dir' already exists."
fi

# python3 run.py --dist "f4" --task gqa --output_dir CCLM_output3_10kcaptions/ --checkpoint cclm_3m_epoch_29.th --bs 128 --seed 42 > CCLM_output3_10kcaptions/train.log
python3 run.py --dist "f4" --task gqa --output_dir $output_dir/ --checkpoint cclm_3m_epoch_29.th --bs 128 --seed 42 --variant $variant > $output_dir/train.log 