# Standalone MACE committee force-uncertainty test

Independent of `Cut_interface/`; existing MD files and trained models are read-only.

## Input

- Four `MACETest_stagetwo.model` files in `LCO_LLZO_400_seed{13,1357,24681,97531}` under `--models-root`.
- An ASE-readable `.xyz` file, or LAMMPS text `.lammpstrj` / `.dump`.
- LAMMPS type order defaults to `Li La Zr Co O`, matching `Cut_interface/md_sample/md_0/PARM.FILE`.
- MACE 0.3.15 environment containing ASE, NumPy, PyTorch and MACE.

## First test (small structure recommended)

```bash
cd /data01/tian_02/Code/mace_uncertainty_test
bash run.sh /data01/tian_02/Interface/LCO_LLZO/fine_tune/LCO_LLZO/LCO_LLZO_400_seed13/test.xyz
```

The script processes **one frame** by default. To evaluate 5 frames:

```bash
bash run.sh /path/to/trajectory.lammpstrj --max-frames 5 --stride 100 --top-k 3
```

To use models from another path, add `--models-root /path/to/parent_of_four_seed_dirs`.

## Outputs

- `results/uncertainty.csv`: frame number, atom count, maximum, 95th percentile, mean atom uncertainty, and maximum-uncertainty atom index (1-based, sorted by LAMMPS ID for dump files).
- `results/selected_frames.csv`: ranking of the highest-uncertainty frames.
- `results/selected.xyz`: the highest-ranked **whole frames** (coordinates/cell only, with per-atom `force_uncertainty` array); no DFT reference labels are implied.

Per-atom uncertainty definition (population-RMS of model force-vector disagreement):

`u_i = sqrt(mean_m(||F_mi - mean_m(F_mi)||^2))`, in eV/Å.

**This is not the MTP extrapolation grade.** It is not a calibrated error threshold. Optional `--threshold` is a force-disagreement threshold in eV/Å. `--top-k` ranks frames; no FPS or DFT-cell cutting is performed in this independent test.

**Memory warning:** `MACECalculator` loads all 4 models, and a single 26,688-atom interface frame may exceed a GPU's memory. Test first using `test.xyz` (small cell); `--max-frames` only limits frame count, not atom count or per-frame GPU memory. Do not crop the full interface without carefully handling local environments and long-range effects.

## Formula test

```bash
python test_uncertainty.py
```
