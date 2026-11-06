#!/bin/bash
#SBATCH --job-name=LCO_LLZO
#SBATCH --partition=GPU4090
#SBATCH -N 10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --exclude=gpu07


source /data01/tian_02/mace-mliap/bin/activate
module load /data00/software/nvidia/hpc_sdk_253/modulefiles/nvhpc/25.3
export HPCSDK=/data00/software/nvidia/hpc_sdk_253
export CUDA_HOME=$HPCSDK/Linux_x86_64/25.3/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/data01/tian_02/python311/lib:$CUDA_HOME/lib64:$HPCSDK/Linux_x86_64/25.3/math_libs/lib64:${LD_LIBRARY_PATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONWARNINGS="ignore::FutureWarning"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MACE_TIME=true
export MACE_PROFILE=true

EXE=/data01/tian_02/lammps-mace/build-mliap/lmp
INPUT=$(pwd)/in.nvt

start_time=$(date +%s)

mpirun -np $SLURM_NTASKS \
  --bind-to core --map-by ppr:1:node \
  --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0 --mca coll ^hcoll,ucc \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x CUDA_VISIBLE_DEVICES \
  $EXE -sf kk -k on g 1 -pk kokkos newton on neigh half -in $INPUT


end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date
