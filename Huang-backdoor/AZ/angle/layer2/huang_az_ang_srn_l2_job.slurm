#!/bin/bash
#SBATCH --job-name=hng_az_ang_l2
#SBATCH --output=logs/huang_az_ang_l2_%j.out
#SBATCH --error=logs/huang_az_ang_l2_%j.err
#SBATCH --partition=hgx
#SBATCH --account=punakha_partner_sdriver
#SBATCH --qos=punakha_hgx2_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --time=12-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@utep.edu

# ============================================================
echo "============================================================"
echo " Job        : $SLURM_JOB_NAME"
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $SLURMD_NODENAME"
echo " CPUs       : $SLURM_CPUS_PER_TASK"
echo " Start      : $(date)"
echo "============================================================"

WORKDIR=/home/srnowmi/work/huang-backdoor/AZ/angle/layer2
cd $WORKDIR

mkdir -p logs
mkdir -p runs

source /work/srnowmi/miniconda3/etc/profile.d/conda.sh
conda activate /work/srnowmi/envs/quid_env

echo ""
echo "[ENV] Python     : $(which python)"
echo "[ENV] Python ver : $(python --version)"
echo "[ENV] Working dir: $(pwd)"
echo "[ENV] CUDA avail : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[ENV] GPU        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo "[ENV] PennyLane  : $(python -c 'import pennylane; print(pennylane.__version__)')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

AZ_DIR=/home/srnowmi/work/huang-backdoor/AZ/AZ_23
echo "[Data] Checking AZ_23 files in $AZ_DIR ..."
for f in AZ-Class-Task_23_families_train.npz AZ-Class-Task_23_families_test.npz; do
    if [ -f "$AZ_DIR/$f" ]; then
        echo "       FOUND    $f"
    else
        echo "       MISSING  $f  <-- fix before submitting"
    fi
done
echo ""

echo "[RUN] Starting at $(date)"
echo ""

python huang_bd_az_ang_l2.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End time  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
