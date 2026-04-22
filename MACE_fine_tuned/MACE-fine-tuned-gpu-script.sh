#!/bin/bash

#SBATCH --job-name=MATERIAL_SEED# Job name
#SBATCH --partition=GPU4090  # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=16      # Number of CPU cores per task
#SBATCH --time=120:00:00           # Walltime

start_time=$(date +%s)
printf "start: $(date)\n" > record_date

source /data01/tian_02/mace-mliap/bin/activate
module load /data00/software/nvidia/hpc_sdk_253/modulefiles/nvhpc/25.3
export LD_LIBRARY_PATH=/data01/tian_02/python311/lib:$CUDA_HOME/lib64:$HPCSDK/Linux_x86_64/25.3/math_libs/lib64:${LD_LIBRARY_PATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

base_path="BASEPATH"
cd $base_path

python mace_dataset.py > "$SLURM_JOB_NAME"-out

TRAIN=train.xyz
VALID=valid.xyz
TEST=test.xyz
MACE_MODEL="MODELPATH"
MACE_CLI=$(python -c "import mace.cli, os; print(os.path.dirname(mace.cli.__file__))")
TRAIN_SCRIPT=$MACE_CLI/run_train.py
EVAL_SCRIPT=$MACE_CLI/eval_configs.py
LAMMPS_SCRIPT=$MACE_CLI/create_lammps_model.py

python "$TRAIN_SCRIPT" \
    --name="MACETest" \
    --model="MACE" \
    --train_file="$TRAIN" \
    --valid_file="$VALID" \
    --test_file="$TEST" \
    --foundation_model="$MACE_MODEL" \
    --E0s="foundation" \
    --loss="stress" \
    --energy_key="vasp_energy" \
    --forces_key="vasp_forces" \
    --stress_key="vasp_stress" \
    --energy_weight=10 \
    --forces_weight=200 \
    --stress_weight=30 \
    --num_interactions=2 \
    --num_cutoff_basis=5 \
    --max_ell=3 \
    --lr=0.0001 \
    --correlation=3 \
    --r_max=6.0 \
    --batch_size=4 \
    --valid_batch_size=4 \
    --eval_interval=1 \
    --max_num_epochs=100 \
    --swa \
    --ema \
    --default_dtype="float64" \
    --device=cuda \
    --seed=SEED \
    --save_cpu \
    --restart_latest

python $LAMMPS_SCRIPT ./MACETest_stagetwo.model --format=mliap

python "$EVAL_SCRIPT" \
    --configs="$TEST" \
    --model="./MACETest_stagetwo.model" \
    --default_dtype="float64" \
    --compute_stress \
    --output="./eval_test.xyz" >> "$SLURM_JOB_NAME"-out

python "$EVAL_SCRIPT" \
    --configs="$TRAIN" \
    --model="./MACETest_stagetwo.model" \
    --default_dtype="float64" \
    --compute_stress \
    --output="./eval_train.xyz" >> "$SLURM_JOB_NAME"-out

deactivate

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date
