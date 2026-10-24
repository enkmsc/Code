#!/bin/bash
#SBATCH --job-name=mace_committee
#SBATCH --partition=GPU4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=committee-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source /data01/tian_02/mace-mliap/bin/activate
module load /data00/software/nvidia/hpc_sdk_253/modulefiles/nvhpc/25.3
export LD_LIBRARY_PATH=/data01/tian_02/python311/lib:${CUDA_HOME}/lib64:${HPCSDK}/Linux_x86_64/25.3/math_libs/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
# Only load your own trusted MACE model files.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u evaluate_committee.py

