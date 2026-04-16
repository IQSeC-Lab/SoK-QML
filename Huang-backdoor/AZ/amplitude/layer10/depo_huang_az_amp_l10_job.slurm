#!/bin/bash
#SBATCH --job-name=dpo_hz_az_amp_l10
#SBATCH --output=logs/depo_huang_az_amp_l10_%j.out
#SBATCH --error=logs/depo_huang_az_amp_l10_%j.err
#SBATCH --partition=medium
#SBATCH --account=jakar_general
#SBATCH --qos=jakar_medium_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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

WORKDIR=/home/mkamol/work/Nowmi/Sok/huang-backdoor/AZ/amplitude/layer10
cd $WORKDIR

mkdir -p logs
mkdir -p depo_runs
mkdir -p depo_huang_az_amp_l10

source /home/mkamol/work/miniconda3/etc/profile.d/conda.sh
conda activate quid_env

echo ""
echo "[ENV] Python     : $(which python)"
echo "[ENV] Python ver : $(python --version)"
echo "[ENV] Working dir: $(pwd)"
echo "[ENV] PennyLane  : $(python -c 'import pennylane; print(pennylane.__version__)')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[Models] Checking saved model files ..."
for f in huang_az_amp_l10_best_ratio01.pt           huang_az_amp_l10_best_ratio03.pt           huang_az_amp_l10_best_ratio05.pt           huang_az_amp_l10_trigger.pt; do
    if [ -f "$f" ]; then
        echo "       FOUND    $f"
    else
        echo "       MISSING  $f  <-- fix before submitting"
    fi
done
echo ""

echo "[RUN] Starting at $(date)"
echo ""

python depo_huang_az_amp_l10.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End time  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
