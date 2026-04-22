#!/bin/bash

#SBATCH --job-name=test      # Job name
#SBATCH --partition=GPU4090  # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=16     # Number of CPU cores per task
#SBATCH --time=72:00:00           # Walltime

start_time=$(date +%s)
printf "start: $(date)\n" > record_date
export LD_LIBRARY_PATH=/data01/tian_02/python312/lib:${LD_LIBRARY_PATH:-}
source /data01/tian_02/mace-md/bin/activate

python md.py > "$SLURM_JOB_NAME"-out

deactivate

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date
