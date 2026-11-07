#!/bin/bash
set -e

source /data01/tian_02/mace-mliap/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

base_path=$(cd "$(dirname "$0")" && pwd)
python "$base_path/uncertainty.py" --traj "$1" --outdir "$base_path/results" --max-frames 1 "${@:2}"
