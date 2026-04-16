#!/bin/bash
#SBATCH --job-name=quid_mnist_amp_l10
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_dgx_general
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1                  # max allowed by punakha_dgx_general QoS
#SBATCH --cpus-per-task=28            # joblib parallelism for rho computation
#SBATCH --mem=128G                    # proto cache + J matrices + data
#SBATCH --time=96:00:00               # rho compute + sequential training (1 GPU)
#SBATCH --output=logs/quid_mnist_amp_l10_%j.out
#SBATCH --error=logs/quid_mnist_amp_l10_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=clalarcon@miners.utep.edu


# ---- environment ----
source /home/clalarcon/work/miniconda3/etc/profile.d/conda.sh
conda activate /home/clalarcon/work/Nowmi/Sok/quid_env
/home/clalarcon/work/Nowmi/Sok/quid_env/bin/python --version
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
cd /home/clalarcon/work/Nowmi/Sok/QUID/MNIST/amplitude/layer10
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/home/clalarcon/work/Nowmi/Sok/quid_env/bin/python NN_quid_mnist_amplitude_l10.py
