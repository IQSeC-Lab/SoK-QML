#!/bin/bash
#SBATCH --job-name=depo_quid_az_ang_l2
#SBATCH --account=jakar_general
#SBATCH --qos=jakar_medium_general
#SBATCH --partition=medium
#SBATCH --cpus-per-task=40            # max CPUs; qiskit.aer parallelises over cores
#SBATCH --mem=64G                     # inference only — no J matrix, no proto cache
#SBATCH --time=288:00:00              # ~30-60 min expected, 4hr for safety
#SBATCH --output=depo_logs/depo_quid_az_ang_l2_%j.out
#SBATCH --error=depo_logs/depo_quid_az_ang_l2_%j.err
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
mkdir -p /work/srnowmi/QUID_full/AZ/angle/layer2/depo_logs
echo "Changing directory..."
cd /work/srnowmi/QUID_full/AZ/angle/layer2
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/work/srnowmi/envs/quid_env/bin/python depo_quid_az_angle_l2.py
