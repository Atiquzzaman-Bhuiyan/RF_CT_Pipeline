#!/bin/bash
#SBATCH --job-name=method_timing_benchmark
#SBATCH --output=/home/ab9884/lhs1/benchmark/benchmark_%j.out
#SBATCH --error=/home/ab9884/lhs1/benchmark/benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute

echo "=========================================="
echo "SLURM benchmark job environment"
echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo ""
echo "--- CPU ---"
lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\):|MHz"
echo ""
echo "--- Memory ---"
free -h
echo ""
echo "=========================================="
echo ""

source /etc/profile
module load python/3.9.0
echo "Python: $(python3 --version)  ($(which python3))"
echo ""

cd /home/ab9884/lhs1/benchmark
python3 -u benchmark_methods.py

echo ""
echo "Job finished: $(date)"
