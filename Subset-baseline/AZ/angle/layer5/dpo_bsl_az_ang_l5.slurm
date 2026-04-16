#!/bin/bash
#SBATCH --job-name=dpo_bsl_az_ang_l5
#SBATCH --account=jakar_general
#SBATCH --qos=jakar_medium_general
#SBATCH --partition=medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dpo_bsl_az_ang_l5_%j.out
#SBATCH --error=logs/dpo_bsl_az_ang_l5_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=srnowmi@miners.utep.edu

# ---- environment ----
source ~/work/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ~/work/Nowmi/Sok/quid_env

python --version
which python

# ---- confirm resources ----
echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Start time  : $(date)"
echo "========================================"

set -e
mkdir -p ~/work/Nowmi/Sok/Baseline/AZ/angle/layer5/logs
echo "Changing directory..."
cd ~/work/Nowmi/Sok/Baseline/AZ/angle/layer5
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
python depo_baseline_az_ang_l5.py
