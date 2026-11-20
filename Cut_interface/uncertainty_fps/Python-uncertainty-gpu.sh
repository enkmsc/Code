#!/bin/bash

#SBATCH --account=MST113461
#SBATCH --job-name=md_0
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

set -e
module purge
module load gcc/12.5.0
module load cuda/12.6
module load openmpi/4.1.6

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_ROOT=$CUDA_HOME
export OPENSSL_HOME=$HOME/openssl-3.5
export LD_LIBRARY_PATH=$HOME/python312/lib:$OPENSSL_HOME/lib:${LD_LIBRARY_PATH:-}
source $HOME/mace-mliap/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1

start_time=$(date +%s)
printf "start: $(date)\n" > record_date

src_path="$HOME/Cut_interface/uncertainty_fps/src/uncertainty.py"
md_path="$HOME/Cut_interface/md_sample/md_0/Dump"
model_path="$HOME/Cut_interface/fine_tune/bootstrap"

python $src_path \
    --config "$md_path/LCO_LLZO_*.data" \
    --model "$model_path/*/MACETest_stagetwo.model" \
    --output "$(pwd)/uncertainty.dat" \
    --device cuda \
    --default_dtype float32 \
    --reg 0.2 \
    --lower 0.15 \
    --upper 0.40 > "${SLURM_JOB_NAME}-out"

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date
