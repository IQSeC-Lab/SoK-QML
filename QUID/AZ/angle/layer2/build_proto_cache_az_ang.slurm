#!/bin/bash
#SBATCH --job-name=proto_cache_az_ang
#SBATCH --account=jakar_general
#SBATCH --qos=jakar_medium_general
#SBATCH --partition=medium
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=288:00:00
#SBATCH --output=depo_logs/proto_cache_az_ang_%j.out
#SBATCH --error=depo_logs/proto_cache_az_ang_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=srnowmi@miners.utep.edu

# ---- environment ----
source /work/srnowmi/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /work/srnowmi/envs/quid_env

/work/srnowmi/envs/quid_env/bin/python --version
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
mkdir -p /work/srnowmi/QUID_full/AZ/angle/layer2/depo_logs
echo "Changing directory..."
cd /work/srnowmi/QUID_full/AZ/angle/layer2
echo "Now in: $(pwd)"
echo "Building angle proto cache..."
/work/srnowmi/envs/quid_env/bin/python - << 'PYEOF'
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import time

N_QUBITS  = 9
N_CLASSES = 23

def _compute_one_rho(x9):
    dev = qml.device('default.qubit', wires=N_QUBITS)
    @qml.qnode(dev, interface=None, diff_method=None)
    def rho_qnode(x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS))
        return qml.density_matrix(wires=list(range(N_QUBITS)))
    return np.array(rho_qnode(pnp.array(x9, dtype=float)))

tr  = np.load('./AZ_23/AZ-Class-Task_23_families_train.npz')
Xtr = tr['X_train'].astype('float32')
ytr = tr['Y_train'].astype('int64')

scaler = MinMaxScaler()
Xtr    = scaler.fit_transform(Xtr)
pca    = PCA(n_components=N_QUBITS)
Xtr    = pca.fit_transform(Xtr).astype('float32')
print(f'[Data] Train: {Xtr.shape}')

dim    = 2 ** N_QUBITS
protos = np.zeros((N_CLASSES, dim, dim), dtype=np.complex128)
t_all  = time.time()

for c in range(N_CLASSES):
    idx_c = np.where(ytr == c)[0]
    print(f'  Class {c:>2}: averaging {len(idx_c):>6} density matrices ...')
    tc   = time.time()
    rhos = Parallel(n_jobs=-1)(delayed(_compute_one_rho)(Xtr[i]) for i in idx_c)
    protos[c] = np.stack(rhos, axis=0).mean(axis=0)
    print(f'  Class {c:>2}: done in {time.time()-tc:.1f}s')

np.save('rho_proto_cache.npy', protos)
print(f'[ProtoCache] All done in {time.time()-t_all:.1f}s')
print(f'[ProtoCache] Saved rho_proto_cache.npy  ({protos.nbytes/1e6:.1f} MB)')
PYEOF
echo "Proto cache build complete."
echo "End time: $(date)"
