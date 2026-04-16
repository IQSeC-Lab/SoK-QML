#!/bin/bash
#SBATCH --job-name=hng_az_amp_l2
#SBATCH --output=logs/huang_az_amp_l2_%j.out
#SBATCH --error=logs/huang_az_amp_l2_%j.err
#SBATCH --partition=hgx
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_hgx_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --time=12-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@utep.edu

echo "============================================================"
echo " Job        : $SLURM_JOB_NAME"
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $SLURMD_NODENAME"
echo " CPUs       : $SLURM_CPUS_PER_TASK"
echo " Start      : $(date)"
echo "============================================================"

WORKDIR=/home/vcadena1/work/Nowmi/Sok/huang-backdoor/AZ/amplitude/layer2
cd $WORKDIR
mkdir -p logs runs

source /home/vcadena1/work/miniconda3/etc/profile.d/conda.sh
conda activate /home/vcadena1/work/Nowmi/Sok/quid_env

echo ""
echo "[ENV] Python     : $(which python)"
echo "[ENV] Python ver : $(python --version)"
echo "[ENV] Working dir: $(pwd)"
echo "[ENV] CUDA avail : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[ENV] PennyLane  : $(python -c 'import pennylane; print(pennylane.__version__)')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[Data] Checking AZ_23 files..."
for f in AZ-Class-Task_23_families_train.npz AZ-Class-Task_23_families_test.npz; do
    if [ -f "/home/vcadena1/work/Nowmi/Sok/huang-backdoor/AZ/AZ_23/$f" ]; then
        echo "       FOUND    $f"
    else
        echo "       MISSING  $f  <-- fix before submitting"
    fi
done
echo ""

echo "[RUN] Starting at $(date)"
python huang_bd_az_amp_l2.py
EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End time  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
