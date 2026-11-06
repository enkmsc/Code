#!/bin/bash
#SBATCH --job-name=LCO_LLZO_24681    # Job name
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
export XDG_CACHE_HOME=/data01/tian_02/.cache

base_path="/data01/tian_02/Interface/LCO_LLZO/fine_tune/LCO_LLZO/LCO_LLZO_400_seed24681"
cd $base_path

python mace_dataset.py > "$SLURM_JOB_NAME"-out

TRAIN=train.xyz
VALID=valid.xyz
TEST=test.xyz
MACE_MODEL="/data01/tian_02/.cache/mace/2023-12-10-mace-128-L0_energy_epoch-249.model"
MACE_CACHE="/data01/tian_02/.cache/mace"
MACE_CLI=$(python -c 'import mace.cli, os; print(os.path.dirname(mace.cli.__file__))')
TRAIN_SCRIPT="$MACE_CLI/run_train.py"
EVAL_SCRIPT="$MACE_CLI/eval_configs.py"
LAMMPS_SCRIPT="$MACE_CLI/create_lammps_model.py"

python "$TRAIN_SCRIPT" \
    --name="MACETest" \
    --model="MACE" \
    --train_file="$TRAIN" \
    --valid_file="$VALID" \
    --test_file="$TEST" \
    --foundation_model="$MACE_MODEL" \
    --multiheads_finetuning=True \
    --pt_train_file="mp" \
    --atomic_numbers="[3,8,17,28,49]" \
    --num_samples_pt=10000 \
    --filter_type_pt="combinations" \
    --subselect_pt="fps" \
    --weight_pt_head=1.0 \
    --E0s="foundation" \
    --energy_key="vasp_energy" \
    --forces_key="vasp_forces" \
    --stress_key="vasp_stress" \
    --compute_stress=True \
    --energy_weight=10 \
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
    --max_num_epochs=400 \
    --ema \
    --swa \
    --swa_lr=0.0001 \
    --default_dtype="float64" \
    --device=cuda \
    --seed=24681 \
    --save_cpu \
    --restart_latest >> "$SLURM_JOB_NAME"-out

python $LAMMPS_SCRIPT ./MACETest_stagetwo.model --format=mliap --head=Default

python "$EVAL_SCRIPT" \
    --configs="$TEST" \
    --model="./MACETest_stagetwo.model" \
    --device=cuda \
    --default_dtype="float64" \
    --batch_size=4 \
    --head=Default \
    --compute_stress \
    --output="./eval_test.xyz" >> "$SLURM_JOB_NAME"-out

python "$EVAL_SCRIPT" \
    --configs="$TRAIN" \
    --model="./MACETest_stagetwo.model" \
    --device=cuda \
    --default_dtype="float64" \
    --batch_size=4 \
    --head=Default \
    --compute_stress \
    --output="./eval_train.xyz" >> "$SLURM_JOB_NAME"-out

deactivate
end_time=$(date +%s)
duration=$((end_time - start_time))
printf "end: $(date)\n" >> record_date
printf "duration: $duration (s)\n" >> record_date
