#!/bin/bash

#SBATCH --job-name=fine_tuning      # Job name
#SBATCH --partition=GPU5090  # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=24      # Number of CPU cores per task
#SBATCH --time=120:00:00           # Walltime

start_time=$(date +%s)
printf "start: $(date)\n" > record_date
source /data00/software/python-envs/CHGNet-MD/bin/activate

#python fine_tuning.py > "$SLURM_JOB_NAME"-out
python MAE_E.py > plot-out
python MAE_F.py > plot-out

deactivate

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date

