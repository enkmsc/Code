import argparse
import os
import glob
import shutil

import ase.io
import numpy as np
from mace.calculators.mace import MACECalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="path to LAMMPS Data files", required=True)
    parser.add_argument("--model", help="path to MACE Committee models", required=True)
    parser.add_argument("--output", help="output DAT file", required=True)
    parser.add_argument("--lower", type=float, default=0.15)
    parser.add_argument("--upper", type=float, default=0.40)
    parser.add_argument("--reg", type=float, default=0.2)
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cuda"
    )
    parser.add_argument(
        "--default_dtype", type=str,
        choices=["float32", "float64"], default="float32"
    )
    return parser.parse_args()


def read_data(fname, show_masses=False):
    atoms = ase.io.read(fname, format="lammps-data", atom_style="atomic")

    if show_masses:
        types = atoms.arrays["type"]
        masses = atoms.get_masses()

        for t in np.unique(types):
            i = np.where(types == t)[0][0]
            print(f"Type {t}: {atoms[i].symbol}, Mass = {masses[i]:.5f}")

    return atoms


def calc_uncertainty(atoms, reg=0.2):
    force = atoms.get_forces()
    force_var = np.var(atoms.calc.results["forces_comm"], axis=0)

    ferr = np.sqrt(np.sum(force_var, axis=1))
    ferr_rel = ferr / (np.linalg.norm(force, axis=1) + reg)

    return force_var, ferr, ferr_rel


def check_threshold(ferr, lower=0.15, upper=0.40):
    ferr_max = np.max(ferr)
    selected = lower < ferr_max < upper

    return ferr_max, selected


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    mace_fname = args.model
    atoms_fname = args.config

    mace_calc = MACECalculator(
        model_paths=mace_fname,
        device=args.device,
        default_dtype=args.default_dtype,
        compute_stress=False,
    )

    data_files = sorted(
        glob.glob(atoms_fname),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].rsplit("_", 1)[1])
    )

    if not data_files:
        raise FileNotFoundError(atoms_fname)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    selected_dir = os.path.join(output_dir, "selected_data")
    os.makedirs(selected_dir, exist_ok=True)

    nselected = 0

    with open(args.output, "w") as f:
        f.write("# frame_index step filename ferr_max ferr_rel_max selected\n")

        for frame_index, fname in enumerate(data_files):
            step = int(os.path.splitext(os.path.basename(fname))[0].rsplit("_", 1)[1])

            atoms = read_data(fname, show_masses=(frame_index == 0))
            atoms.calc = mace_calc

            force_var, ferr, ferr_rel = calc_uncertainty(atoms, reg=args.reg)
            ferr_max, selected = check_threshold(ferr, lower=args.lower, upper=args.upper)

            idx = np.argmax(ferr)

            print(f"Max Ferr Atom: {idx}")
            print(f"Element: {atoms[idx].symbol}")
            print(f"Position: {atoms.positions[idx]}")
            print(f"Ferr: {ferr[idx]:.6f} eV/A")
            print(f"Committee Forces:\n{atoms.calc.results['forces_comm'][:, idx, :]}")
            print(f"Ferr Percentiles: {np.percentile(ferr, [50, 90, 99, 99.9, 100])}")
            print(f"Atoms above 0.40 eV/A: {np.sum(ferr > 0.40)}")
            
            f.write(
                f"{frame_index:6d} {step:8d} {os.path.basename(fname):25s} "
                f"{ferr_max:.6f} {np.max(ferr_rel):.6f} {int(selected)}\n"
            )
            f.flush()

            if selected:
                shutil.copy2(fname, os.path.join(selected_dir, os.path.basename(fname)))
                nselected += 1

            print(
                f"Frame {frame_index}, Step {step}, "
                f"Ferr = {ferr_max:.3f} eV/A, "
                f"Ferr_rel = {np.max(ferr_rel):.3f}, "
                f"Selected = {selected}",
                flush=True,
            )

    print(f"Total frames: {len(data_files)}", flush=True)
    print(f"Selected frames: {nselected}", flush=True)


if __name__ == "__main__":
    main()

