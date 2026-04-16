#!/bin/bash
#SBATCH --job-name=qtrojan_az23_l10
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_dgx_general
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
#SBATCH --output=/work/clalarcon/Nowmi/Sok/Qtrojan/AZ/angle/layer10/logs/qtrojan_az23_l10_%j.out
#SBATCH --error=/work/clalarcon/Nowmi/Sok/Qtrojan/AZ/angle/layer10/logs/qtrojan_az23_l10_%j.err
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
echo "Changing directory..."
cd /work/clalarcon/Nowmi/Sok/Qtrojan/AZ/angle/layer10
mkdir -p logs
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/home/clalarcon/work/Nowmi/Sok/quid_env/bin/python /work/clalarcon/Nowmi/Sok/Qtrojan/AZ/angle/layer10/qtrojan_az23_angle_l10.py
