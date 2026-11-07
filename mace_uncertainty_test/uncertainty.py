import os
import csv
import heapq
import argparse
import numpy as np


def force_disagreement(forces):
    """(n_models, n_atoms, 3) -> per-atom vector RMS disagreement in eV/Angstrom."""
    return np.sqrt(np.mean(np.sum((forces - forces.mean(axis=0)) ** 2, axis=2), axis=0))


def main():
    from ase import Atoms
    from ase.io import iread, write
    from mace.calculators import MACECalculator

    parser = argparse.ArgumentParser(description="Four-model MACE force-uncertainty test")
    parser.add_argument("--traj", required=True, help="ASE-readable trajectory, e.g. .xyz or .lammpstrj")
    parser.add_argument("--models-root", default="/data01/tian_02/Interface/LCO_LLZO/fine_tune/LCO_LLZO")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth input frame")
    parser.add_argument("--max-frames", type=int, default=1, help="Maximum frames to evaluate; 0 means all")
    parser.add_argument("--top-k", type=int, default=5, help="Keep K frames with highest max uncertainty")
    parser.add_argument("--threshold", type=float, default=None, help="Optional min max-force-uncertainty (eV/Angstrom)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--head", default="Default")
    parser.add_argument("--specorder", nargs="+", default=["Li", "La", "Zr", "Co", "O"], help="LAMMPS atom-type order")
    args = parser.parse_args()

    seeds = [13, 1357, 24681, 97531]
    models = [os.path.join(args.models_root, f"LCO_LLZO_400_seed{seed}", "MACETest_stagetwo.model") for seed in seeds]
    missing = [path for path in models if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("MACE models missing:\n" + "\n".join(missing))
    os.makedirs(args.outdir, exist_ok=True)

    calc = MACECalculator(model_paths=models, device=args.device, default_dtype=args.dtype, head=args.head)
    options = {"format": "lammps-dump-text", "specorder": args.specorder} if args.traj.endswith((".lammpstrj", ".dump")) else {}
    frames = iread(args.traj, index=":", **options)
    top = []
    n = 0
    with open(os.path.join(args.outdir, "uncertainty.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "n_atoms", "max_eV_A", "p95_eV_A", "mean_eV_A", "max_atom_id"])
        for i, atoms in enumerate(frames):
            if i % args.stride:
                continue
            atoms.calc = calc
            atoms.get_forces()
            forces = np.asarray(calc.results["forces_comm"])
            u = force_disagreement(forces)
            u_max = float(u.max())
            writer.writerow([i, len(atoms), u_max, float(np.percentile(u, 95)), float(u.mean()), int(u.argmax()) + 1])
            f.flush()
            print(f"frame={i} atoms={len(atoms)} max={u_max:.6f} p95={np.percentile(u, 95):.6f} eV/A", flush=True)
            if args.top_k > 0 and (args.threshold is None or u_max >= args.threshold):
                structure = Atoms(symbols=atoms.get_chemical_symbols(), positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
                structure.info.update({"frame": i, "uncertainty_max": u_max})
                structure.new_array("force_uncertainty", u)
                entry = (u_max, i, structure)
                if len(top) < args.top_k:
                    heapq.heappush(top, entry)
                elif entry[:2] > top[0][:2]:
                    heapq.heapreplace(top, entry)
            n += 1
            if args.max_frames > 0 and n >= args.max_frames:
                break

    top.sort(reverse=True)
    if top:
        write(os.path.join(args.outdir, "selected.xyz"), [entry[2] for entry in top], format="extxyz")
    with open(os.path.join(args.outdir, "selected_frames.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "frame", "max_eV_A"])
        for rank, (score, frame, _) in enumerate(top, 1):
            writer.writerow([rank, frame, score])
    print(f"Completed {n} frames; selected {len(top)} frames; results in {args.outdir}")


if __name__ == "__main__":
    main()
