#!/bin/bash
#SBATCH --job-name=qtrojan_ang_l2
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_hgx_general
#SBATCH --partition=hgx
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
#SBATCH --output=/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer2/logs/qtrojan_ang_l2_%j.out
#SBATCH --error=/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer2/logs/qtrojan_ang_l2_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vcadena1@miners.utep.edu


# ---- environment ----
source /home/vcadena1/work/miniconda3/etc/profile.d/conda.sh
conda activate /home/vcadena1/work/Nowmi/Sok/quid_env
/home/vcadena1/work/Nowmi/Sok/quid_env/bin/python --version
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
cd /work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer2
mkdir -p logs
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/home/vcadena1/work/Nowmi/Sok/quid_env/bin/python /work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer2/qtrojan_mnist_angle_l2.py
