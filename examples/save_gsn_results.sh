#!/bin/bash
#SBATCH --job-name=gsn-%j
#SBATCH --time=02:00:00
#SBACTH --ntasks=1
#SBATCH --output="out/gsn-%j.out"
#SBATCH --mem=42G
#SBATCH -p evlab

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate gsn

python save_gsn_results.py --uid "${1}" --hemi "${2}" --parc "${3}" --parc_col  "${4}" --permute "${5}"