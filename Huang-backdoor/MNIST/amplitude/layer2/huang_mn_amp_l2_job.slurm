#!/bin/bash
#SBATCH --job-name=huang_bd_ampl_l2
#SBATCH --output=logs/huang_%j.out
#SBATCH --error=logs/huang_%j.err
#SBATCH --partition=dgx
#SBATCH --account=punakha_general
#SBATCH --qos=punakha_dgx_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
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

# ── Working directory ─────────────────────────────────────────
WORKDIR=/home/mkamol/work/Nowmi/Sok/huang-backdoor/MNIST/amplitude/layer2
cd $WORKDIR

mkdir -p logs
mkdir -p runs

# ── Activate quid_env ─────────────────────────────────────────
source /home/mkamol/work/miniconda3/etc/profile.d/conda.sh
conda activate quid_env

# ── Confirm setup ─────────────────────────────────────────────
echo ""
echo "[ENV] Python     : $(which python)"
echo "[ENV] Python ver : $(python --version)"
echo "[ENV] Working dir: $(pwd)"
echo "[ENV] CUDA avail : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[ENV] GPU info   : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo "[ENV] PennyLane  : $(python -c 'import pennylane; print(pennylane.__version__)')"
echo ""

# ── Thread settings ───────────────────────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── MNIST data path check ─────────────────────────────────────
MNIST_DIR=/home/mkamol/work/Nowmi/Sok/huang-backdoor/MNIST/MNIST
echo "[Data] Checking MNIST files in $MNIST_DIR ..."
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
          t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
    if [ -f "$MNIST_DIR/$f" ]; then
        echo "       FOUND    $f"
    else
        echo "       MISSING  $f  <-- fix before submitting"
    fi
done
echo ""

# ── Run ───────────────────────────────────────────────────────
echo "[RUN] Starting at $(date)"
echo ""

python huang_bd_mn_amp_l2_phase2.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End time  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
