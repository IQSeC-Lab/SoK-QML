#!/bin/bash
#SBATCH --job-name=quid_mnist_angle_l10
#SBATCH --partition=dgx
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_dgx_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=27
#SBATCH --mem=64G
#SBATCH --time=280:00:00
#SBATCH --output=logs/quid_mnist_angle_l10_%j.out
#SBATCH --error=logs/quid_mnist_angle_l10_%j.err

set -e

echo "================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Start    : $(date)"
echo "CWD      : $SLURM_SUBMIT_DIR"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================"

# Activate conda env
source /home/vcadena1/work/miniconda3/etc/profile.d/conda.sh
conda activate /home/vcadena1/work/Nowmi/Sok/quid_env

# Run from script directory
cd /home/vcadena1/work/Nowmi/Sok/QUID/MNIST/Angle/layer10
mkdir -p logs

nvidia-smi || true

python NN_quid_mnist_angle_l10.py

echo "Finished : $(date)"
