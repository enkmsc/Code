!/bin/bash

#SBATCH --job-name=ddec      # Job name
#SBATCH --partition=CPU  # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=56       # Number of tasks per node
#SBATCH --cpus-per-task=1       #CPU core per task
#SBATCH --time=72:00:00           # Walltime
#SBATCH --mem=64G # memory per cpu core

base_path=$(pwd)

cd $base_path
cp $base_path/POTCAR $base_path/POTCAR_calculate
./ddec-potcar.sh

start=$(date +%s)
printf "start: $(date)\n" > ddec_record_date

Chargemol_09_26_2017_linux_serial

end=$(date +%s)
printf "end:   $(date)\n" >> ddec_record_date
echo "elapsed: $((end - start)) seconds" >> ddec_record_date                                                            
