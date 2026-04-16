#!/bin/bash
#SBATCH --job-name=depo_qtr_az_l5
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
#SBATCH --output=/home/clalarcon/work/Nowmi/Sok/Qtrojan/AZ/angle/layer5/logs/depo_qtrojan_az_l5_%j.out
#SBATCH --error=/home/clalarcon/work/Nowmi/Sok/Qtrojan/AZ/angle/layer5/logs/depo_qtrojan_az_l5_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=clalarcon@miners.utep.edu

# ---- create depo_logs and symlink this job's logs into it ----
WORK_DIR=/home/clalarcon/work/Nowmi/Sok/Qtrojan/AZ/angle/layer5
mkdir -p ${WORK_DIR}/depo_logs

# copy logs to depo_logs at the end via trap
LOG_OUT=${WORK_DIR}/logs/depo_qtrojan_az_l5_${SLURM_JOB_ID}.out
LOG_ERR=${WORK_DIR}/logs/depo_qtrojan_az_l5_${SLURM_JOB_ID}.err
trap "cp ${LOG_OUT} ${WORK_DIR}/depo_logs/ 2>/dev/null; \
      cp ${LOG_ERR} ${WORK_DIR}/depo_logs/ 2>/dev/null" EXIT

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
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Start time  : $(date)"
echo "========================================"

set -e
echo "Changing directory..."
cd ${WORK_DIR}
echo "Now in: $(pwd)"
echo "Files: $(ls)"
echo "Running python script..."
/home/clalarcon/work/Nowmi/Sok/quid_env/bin/python ${WORK_DIR}/depo_qtrojan_az23_angle_l5.py
