#!/bin/bash

#SBATCH --job-name=fine_tuning      # Job name
#SBATCH --partition=GPU4090  # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=16       # Number of CPU cores per task
#SBATCH --time=72:00:00           # Walltime

start_time=$(date +%s)
printf "start: $(date)\n" > record_date
source /data01/tian_02/mace_env/bin/activate
module load /data00/software/nvidia/hpc_sdk_253/modulefiles/nvhpc/25.3

HOST_PATH="/data01/tian_02/MACE/fine_tune_LATP"
TRAIN_FILE="${HOST_PATH}/train.xyz"
TEST_FILE="${HOST_PATH}/test.xyz"
VALID_FILE="${HOST_PATH}/valid.xyz"
MACE_PATH="/data01/tian_02/mace_env/lib/python3.10/site-packages/mace/cli"
MACE_TRAIN_SCRIPT="${MACE_PATH}/run_train.py"
MACE_EVAL_SCRIPT="${MACE_PATH}/eval_configs.py"
LAMMPS_SCRIPT="${MACE_PATH}/create_lammps_model.py"
MACE_MODEL="/data01/tian_02/MACE/fine_tune_LATP/2023-12-10-mace-128-L0_energy_epoch-249.model"

cd "$HOST_PATH" 

python "$MACE_TRAIN_SCRIPT" \
    --name="MACETest" \
    --model="MACE" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --test_file="$TEST_FILE" \
    --foundation_model="$MACE_MODEL" \
    --E0s="foundation" \
    --energy_key="vasp_energy" \
    --forces_key="vasp_forces" \
    --stress_key="vasp_stress" \
    --loss="stress" \
    --energy_weight=20 \
    --forces_weight=1000 \
    --stress_weight=50 \
    --num_interactions=2 \
    --num_cutoff_basis=5 \
    --max_ell=3 \
    --lr=0.0001 \
    --correlation=3 \
    --r_max=6.0 \
    --batch_size=4 \
    --valid_batch_size=4 \
    --eval_interval=1 \
    --max_num_epochs=500 \
    --swa \
    --ema \
    --default_dtype="float64" \
    --device=cuda \
    --seed=42 \
    --save_cpu \
    --restart_latest

#python $LAMMPS_SCRIPT ./MACETest_stagetwo.model --format=mliap

python "$MACE_EVAL_SCRIPT" \
    --configs="$TEST_FILE" \
    --model="./MACETest_stagetwo_compiled.model" \
    --default_dtype="float64" \
    --output="./eval_test.xyz" > eval.out

python "$MACE_EVAL_SCRIPT" \
    --configs="$TRAIN_FILE" \
    --model="./MACETest_stagetwo_compiled.model" \
    --default_dtype="float64" \
    --output="./eval_train.xyz" >> eval.out

deactivate

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date

