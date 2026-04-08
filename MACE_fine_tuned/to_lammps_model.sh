#!/bin/bash

#SBATCH --job-name=test      # Job name
#SBATCH --partition=GPU4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00           # Walltime

source /data01/tian_02/mace-mliap/bin/activate
module purge
module load /data00/software/nvidia/hpc_sdk_253/modulefiles/nvhpc/25.3
export HPCSDK=/data00/software/nvidia/hpc_sdk_253
export CUDA_HOME=$HPCSDK/Linux_x86_64/25.3/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/data01/tian_02/python312/lib:$CUDA_HOME/lib64:$HPCSDK/Linux_x86_64/25.3/math_libs/lib64:${LD_LIBRARY_PATH:-}

MACE_MODEL="/data01/tian_02/Final/Test/GPU-MACE-IAP/Pre-trained/single/test/2023-12-03-mace-128-L1_epoch-199.pt"
MACE_CLI=$(python -c "import mace.cli, os; print(os.path.dirname(mace.cli.__file__))")
LAMMPS_SCRIPT=$MACE_CLI/create_lammps_model.py

python $LAMMPS_SCRIPT $MACE_MODEL --format=mliap
