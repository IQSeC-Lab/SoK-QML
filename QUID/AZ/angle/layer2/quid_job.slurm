#!/bin/bash
#SBATCH --job-name=quid_az_l2
#SBATCH --account=punakha_partner_sdriver
#SBATCH --qos=punakha_dgx2_general
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2                  # max allowed by punakha_dgx2_general QoS
#SBATCH --cpus-per-task=32            # joblib parallelism for rho computation
#SBATCH --mem=256G                     # proto cache (~96 MB) + J matrices + data
#SBATCH --time=96:00:00               # ~12h rho compute + ~5h training across 2 GPUs
#SBATCH --output=logs/quid_az_l2_%j.out
#SBATCH --error=logs/quid_az_l2_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=srnowmi@miners.utep.edu





# ---- environment ----
/work/srnowmi/envs/quid_env/bin/python --version
which python
python --version

# ---- confirm resources ----
echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Start time  : $(date)"
echo "========================================"

nvidia-smi

set -e
mkdir -p logs
echo "Changing directory..."
cd /work/srnowmi/QUID_full/AZ/angle/layer2
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/work/srnowmi/envs/quid_env/bin/python3.11 NN_quid_az_angle_l2.py
