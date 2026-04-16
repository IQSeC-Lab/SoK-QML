#!/bin/bash
#SBATCH --job-name=depo_qtr_mn_l10
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_hgx_general
#SBATCH --partition=hgx
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
#SBATCH --output=/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer10/logs/depo_qtrojan_mn_l10_%j.out
#SBATCH --error=/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer10/logs/depo_qtrojan_mn_l10_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vcadena1@miners.utep.edu

# ---- create depo_logs and copy logs there at end ----
WORK_DIR=/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/angle/layer10
mkdir -p ${WORK_DIR}/depo_logs

LOG_OUT=${WORK_DIR}/logs/depo_qtrojan_mn_l10_${SLURM_JOB_ID}.out
LOG_ERR=${WORK_DIR}/logs/depo_qtrojan_mn_l10_${SLURM_JOB_ID}.err
trap "cp ${LOG_OUT} ${WORK_DIR}/depo_logs/ 2>/dev/null; \
      cp ${LOG_ERR} ${WORK_DIR}/depo_logs/ 2>/dev/null" EXIT

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
cd ${WORK_DIR}
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/home/vcadena1/work/Nowmi/Sok/quid_env/bin/python ${WORK_DIR}/depo_qtrojan_mnist_angle_vca_l10.py
