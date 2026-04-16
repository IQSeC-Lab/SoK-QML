#!/bin/bash
#SBATCH --job-name=quid_az_amp_l2
#SBATCH --account=punakha_partner_sdriver
#SBATCH --qos=punakha_dgx2_general
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2                   # 2 GPUs — parallel workers round-robin
#SBATCH --cpus-per-task=47            # max CPUs; joblib uses all for rho cache build
#SBATCH --mem=256G                     # proto cache (~96 MB) + J matrices + data
#SBATCH --time=240:00:00              # 10 days max
#SBATCH --output=logs/quid_az_amp_l2_%j.out
#SBATCH --error=logs/quid_az_amp_l2_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=srnowmi@miners.utep.edu

# ---- environment ----
source /home/msrahman3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/msrahman3/work/Nowmi/Sok/quid_env

/home/msrahman3/work/Nowmi/Sok/quid_env/bin/python --version
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
mkdir -p /home/msrahman3/work/Nowmi/Sok/QUID/AZ/amplitude/layer2/logs
echo "Changing directory..."
cd /home/msrahman3/work/Nowmi/Sok/QUID/AZ/amplitude/layer2
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/home/msrahman3/work/Nowmi/Sok/quid_env/bin/python NN_quid_az_amplitude_l2.py
