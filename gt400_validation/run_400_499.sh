#!/bin/bash
#SBATCH --job-name=gt400_methods
#SBATCH --output=/home/ab9884/lhs1/gt400/gt400_%j.out
#SBATCH --error=/home/ab9884/lhs1/gt400/gt400_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute

source /etc/profile
module load python/3.9.0

# >>> EDIT THESE PATHS (REPO_DIR, CT_DIR, COMBINED_MODEL, COMBINED_RANGE) <<<
export REPO_DIR=/home/ab9884/lhs1/rf_ct_pipeline        # your checkout of the repo
export CORE_END=490                                      # scored core = slices 400..489 (window capped at 499; slice 500 never loaded)
export CT_DIR=/home/ab9884/lhs1/ct_data                  # folder holding slice_000400.tif etc. (plain naming; content-verified identical to the "Copy of..." set used for OTSU_LOWER=0.475 validation)
export COMBINED_MODEL=/home/ab9884/lhs1/benchmark/combined_model.joblib
export COMBINED_RANGE=/home/ab9884/lhs1/benchmark/combined_intensity_range.joblib
export OTSU_LOWER=0.475                                  # corrected A/B/E threshold used for the updated tables
export ABE_ALSO_ORIGINAL=1                                # also save the superseded min-max A/B/E (documents the failure)
export OUT_DIR=/home/ab9884/lhs1/gt400/out

mkdir -p /home/ab9884/lhs1/gt400 $OUT_DIR
cd /home/ab9884/lhs1/gt400
python3 -u run_400_499.py
