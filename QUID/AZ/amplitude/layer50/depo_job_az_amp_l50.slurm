#!/bin/bash
#SBATCH --job-name=depo_quid_az_amp_l50
#SBATCH --account=jakar_general
#SBATCH --qos=jakar_medium_general
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=288:00:00
#SBATCH --output=depo_logs/depo_quid_az_amp_l50_%j.out
#SBATCH --error=depo_logs/depo_quid_az_amp_l50_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=srnowmi@miners.utep.edu

# ---- environment ----
source /work/srnowmi/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /work/srnowmi/envs/quid_env

/work/srnowmi/envs/quid_env/bin/python --version
which python
python --version

# ---- confirm resources ----
echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Start time  : $(date)"
echo "========================================"

set -e
mkdir -p /work/srnowmi/QUID_full/AZ/amplitude/layer50/depo_logs
echo "Changing directory..."
cd /work/srnowmi/QUID_full/AZ/amplitude/layer50
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/work/srnowmi/envs/quid_env/bin/python depo_quid_az_amplitude_l50.py
