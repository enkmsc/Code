#!/usr/bin/env python3
"""Offline full-frame MACE committee evaluation, one model on the device at a time."""
import csv
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import find_mic
from ase.io import iread, read, write
from ase.neighborlist import neighbor_list


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_frames(path, type_map):
    """Stream metal-unit custom dumps; preserve IDs, timestep, origin and each cell.

    Supports orthogonal/restricted triclinic cells and x/xu/xs/xsu coordinates.
    General triclinic (abc origin) and non-metal units must be converted first.
    """
    with open(path) as stream:
        def line():
            value = stream.readline()
            if not value:
                raise ValueError("Truncated LAMMPS dump")
            return value.strip()

        while True:
            header = stream.readline()
            if not header:
                return
            if not header.strip():
                continue
            if header.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Expected ITEM: TIMESTEP, got {header.strip()!r}")
            step = int(line())
            if line() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing NUMBER OF ATOMS header")
            n = int(line())
            bounds_header = line().split()
            if bounds_header[:3] != ["ITEM:", "BOX", "BOUNDS"] or "abc" in bounds_header:
                raise ValueError("Only orthogonal/restricted-triclinic BOX BOUNDS are supported")
            tilted = bounds_header[3:6] == ["xy", "xz", "yz"]
            flags = bounds_header[6:] if tilted else bounds_header[3:]
            if len(flags) != 3 or any(len(flag) != 2 or any(c not in "pfsm" for c in flag) for flag in flags):
                raise ValueError("BOX BOUNDS must explicitly contain three boundary flags")
            bounds = np.array([[float(x) for x in line().split()] for _ in range(3)])
            if bounds.shape != (3, 3 if tilted else 2):
                raise ValueError("Invalid BOX BOUNDS rows")
            lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
            xy, xz, yz = bounds[:, 2] if tilted else (0.0, 0.0, 0.0)
            lo[0] -= min(0.0, xy, xz, xy + xz)
            hi[0] -= max(0.0, xy, xz, xy + xz)
            lo[1] -= min(0.0, yz)
            hi[1] -= max(0.0, yz)
            lengths = hi - lo
            if np.any(lengths <= 0):
                raise ValueError("Nonpositive box length")
            cell = np.array([[lengths[0], 0, 0], [xy, lengths[1], 0], [xz, yz, lengths[2]]])
            atom_header = line().split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Missing ATOMS header")
            cols = atom_header[2:]
            if "id" not in cols or "type" not in cols:
                raise ValueError("Dump must contain id and type columns")
            rows = [line().split() for _ in range(n)]
            if n == 0 or any(len(row) != len(cols) for row in rows):
                raise ValueError("Empty frame or malformed atom rows")
            ids = np.array([int(row[cols.index("id")]) for row in rows], dtype=np.int64)
            types = np.array([int(row[cols.index("type")]) for row in rows], dtype=int)
            if len(np.unique(ids)) != n:
                raise ValueError("Duplicate atom IDs in dump frame")
            missing = set(types) - set(type_map)
            if missing:
                raise ValueError(f"Missing TYPE_MAP entries: {sorted(missing)}")
            coordinates = next((keys for keys in [("xu", "yu", "zu"), ("xsu", "ysu", "zsu"), ("x", "y", "z"), ("xs", "ys", "zs")] if all(k in cols for k in keys)), None)
            if coordinates is None:
                raise ValueError("Dump needs x/y/z, xu/yu/zu, xs/ys/zs or xsu/ysu/zsu")
            positions = np.array([[float(row[cols.index(key)]) for key in coordinates] for row in rows])
            if coordinates[0] in ("xs", "xsu"):
                positions = positions @ cell + lo
            order = np.argsort(ids, kind="stable")
            atoms = Atoms(symbols=[type_map[t] for t in types[order]], positions=positions[order], cell=cell, pbc=[f == "pp" for f in flags])
            atoms.new_array("atom_id", ids[order])
            atoms.info.update(timestep=step, lammps_origin=lo.tolist(), atom_id_source="lammps_id", coordinate_style=" ".join(coordinates))
            yield atoms


def input_frames(cfg, path):
    if cfg.INPUT_FORMAT == "lammps-dump-text":
        yield from dump_frames(path, cfg.TYPE_MAP)
    elif cfg.INPUT_FORMAT == "lammps-data":
        atoms = read(path, format="lammps-data", atom_style=cfg.ATOM_STYLE, Z_of_type={k: atomic_numbers[v] for k, v in cfg.TYPE_MAP.items()}, units="metal", sort_by_id=True)
        yield atoms
    else:
        yield from iread(path, index=":", format=cfg.INPUT_FORMAT)


def ensure_ids(atoms):
    key = next((key for key in ("atom_id", "id") if key in atoms.arrays), None)
    raw = atoms.arrays[key] if key else np.arange(1, len(atoms) + 1)
    ids = np.asarray(raw, dtype=np.int64)
    if ids.shape != (len(atoms),) or not np.all(ids == raw) or len(np.unique(ids)) != len(atoms):
        raise ValueError("Atom IDs must be unique integers")
    atoms.set_array("atom_id", ids)
    atoms.info.setdefault("atom_id_source", "input_id" if key else "generated_row_1based")


def update_moments(mean, m2, forces, count):
    """Welford updates; population variance uses final m2 / number_of_models."""
    delta = forces - mean
    mean = mean + delta / count
    return mean, m2 + delta * (forces - mean)


def local_scores(atoms, force_std, radius):
    """Arithmetic mean over center plus unique neighbors within radius, with PBC."""
    if radius == 0:
        return force_std.copy()
    i, j = neighbor_list("ij", atoms, radius, self_interaction=False)
    pairs = np.unique(np.column_stack((i[i != j], j[i != j])), axis=0)
    sums, counts = force_std.copy(), np.ones(len(atoms))
    if len(pairs):
        np.add.at(sums, pairs[:, 0], force_std[pairs[:, 1]])
        np.add.at(counts, pairs[:, 0], 1)
    return sums / counts


def select_centers(atoms, scores, top_k, min_distance, elements=None, min_score=None):
    eligible = np.ones(len(atoms), dtype=bool)
    if elements is not None:
        eligible &= np.isin(atoms.get_chemical_symbols(), elements)
    if min_score is not None:
        eligible &= scores >= min_score
    selected = []
    order = np.lexsort((atoms.arrays["atom_id"], -scores))
    for index in order:
        if not eligible[index]:
            continue
        if selected:
            _, distances = find_mic(atoms.positions[selected] - atoms.positions[index], atoms.cell, atoms.pbc)
            if np.any(distances < min_distance):
                continue
        selected.append(int(index))
        if len(selected) == top_k:
            break
    return selected


def make_calculator(model_path, cfg):
    from mace.calculators import MACECalculator
    kwargs = {} if cfg.HEAD is None else {"head": cfg.HEAD}
    return MACECalculator(model_paths=str(model_path), device=cfg.DEVICE, default_dtype=cfg.DEFAULT_DTYPE, **kwargs)


def gpu_stats(cfg, reset=False):
    if not str(cfg.DEVICE).startswith("cuda"):
        return None
    import torch
    if reset:
        torch.cuda.reset_peak_memory_stats(cfg.DEVICE)
    torch.cuda.synchronize(cfg.DEVICE)
    return torch.cuda.max_memory_allocated(cfg.DEVICE) / 1024**2


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg, base_dir=None, calculator_factory=make_calculator):
    base = Path(base_dir or Path(__file__).parent).resolve()
    def resolve(value):
        return (base / value).resolve()

    models = [resolve(p) for p in cfg.MODEL_PATHS]
    if len(models) < 2:
        raise ValueError("At least two independently trained MACE models are required")
    fingerprints = [sha256(path) for path in models]
    if len(set(fingerprints)) != len(models):
        raise ValueError("Identical model files do not form an independent committee")
    if cfg.START_FRAME < 0 or cfg.FRAME_STRIDE < 1 or (cfg.MAX_FRAMES is not None and cfg.MAX_FRAMES < 1):
        raise ValueError("Invalid frame selection")
    if cfg.STOP_FRAME is not None and cfg.STOP_FRAME <= cfg.START_FRAME:
        raise ValueError("STOP_FRAME must exceed START_FRAME")
    if cfg.TOP_CENTERS_PER_FRAME < 1 or cfg.MIN_CENTER_DISTANCE < 0 or cfg.LOCAL_RADIUS < 0:
        raise ValueError("Invalid candidate selection settings")
    if cfg.RANK_BY not in ("local_mean", "force_std"):
        raise ValueError("RANK_BY must be local_mean or force_std")
    source = resolve(cfg.INPUT_FILE)
    if not source.is_file():
        raise FileNotFoundError(source)
    out = resolve(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=False)  # Never mix new results with an earlier run.
    cache = out / "frames"
    cache.mkdir()
    settings = {k: v for k, v in vars(cfg).items() if k.isupper()}
    versions = {}
    for package in ("mace-torch", "torch", "ase", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    metadata = {"status": "incomplete", "settings": settings, "versions": versions, "source": str(source), "source_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns, "models": [{"path": str(p), "sha256": h} for p, h in zip(models, fingerprints)], "score_units": "eV/Angstrom", "variance_ddof": 0}
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    started = time.perf_counter()
    paths = []
    frames = input_frames(cfg, source)
    try:
        for index, atoms in enumerate(frames):
            if cfg.STOP_FRAME is not None and index >= cfg.STOP_FRAME:
                break
            if index < cfg.START_FRAME or (index - cfg.START_FRAME) % cfg.FRAME_STRIDE:
                continue
            if len(atoms) == 0 or not np.all(np.isfinite(atoms.positions)) or not np.all(np.isfinite(atoms.cell)):
                raise ValueError(f"Invalid structure at frame {index}")
            ensure_ids(atoms)
            atoms.calc = None
            atoms.set_constraint()
            atoms.info["source_frame"] = index
            path = cache / f"frame_{index:08d}.extxyz"
            write(path, atoms, format="extxyz", write_results=False)
            paths.append(path)
            if cfg.MAX_FRAMES is not None and len(paths) >= cfg.MAX_FRAMES:
                break
    finally:
        frames.close()
    if not paths:
        raise ValueError("No frames selected")
    print(f"Selected {len(paths)} frames; evaluating {len(models)} models sequentially.", flush=True)
    for count, model in enumerate(models, start=1):
        load_start = time.perf_counter()
        calc = calculator_factory(model, cfg)
        gpu_stats(cfg)
        load_seconds = time.perf_counter() - load_start
        timings = []
        for path in paths:
            atoms = read(path)
            if hasattr(calc, "z_table"):
                unsupported = set(atoms.numbers) - set(calc.z_table.zs)
                if unsupported:
                    raise ValueError(f"{model.name} does not support atomic numbers {sorted(unsupported)}")
            gpu_stats(cfg, reset=True)
            tic = time.perf_counter()
            forces = np.asarray(calc.get_forces(atoms), dtype=np.float64)
            energy = float(calc.get_potential_energy(atoms))
            peak = gpu_stats(cfg)
            elapsed = time.perf_counter() - tic
            if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)) or not np.isfinite(energy):
                raise ValueError(f"Nonfinite/invalid predictions: {model.name}, {path.name}")
            state_path = path.with_suffix(".npz")
            if count == 1:
                mean, m2, energies = np.zeros_like(forces), np.zeros_like(forces), []
            else:
                with np.load(state_path) as state:
                    mean, m2, energies = state["mean"], state["m2"], state["energies"].tolist()
            mean, m2 = update_moments(mean, m2, forces, count)
            np.savez(state_path, mean=mean, m2=m2, energies=np.array(energies + [energy]))
            timings.append({"source_frame": int(atoms.info["source_frame"]), "timestep": atoms.info.get("timestep", ""), "n_atoms": len(atoms), "seconds": elapsed, "peak_cuda_allocated_MiB": peak, "energy_eV": energy})
            print(f"Model {count}/{len(models)} frame {atoms.info['source_frame']}: {elapsed:.3f} s", flush=True)
        write_csv(out / f"timing_model_{count}.csv", timings, list(timings[0]))
        metadata["models"][count - 1]["load_seconds"] = load_seconds
        del calc
        gc.collect()
        if str(cfg.DEVICE).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    summary, candidates = [], []
    for path in paths:
        atoms = read(path)
        with np.load(path.with_suffix(".npz")) as state:
            mean, m2, energies = state["mean"], state["m2"], state["energies"]
        std_xyz = np.sqrt(np.maximum(m2 / len(models), 0))
        force_std = np.linalg.norm(std_xyz, axis=1)
        local_mean = local_scores(atoms, force_std, cfg.LOCAL_RADIUS)
        scores = local_mean if cfg.RANK_BY == "local_mean" else force_std
        selected = select_centers(atoms, scores, cfg.TOP_CENTERS_PER_FRAME, cfg.MIN_CENTER_DISTANCE, cfg.CENTER_ELEMENTS, cfg.MIN_SCORE)
        atoms.set_array("committee_force_mean", mean)
        atoms.set_array("committee_force_std_xyz", std_xyz)
        atoms.set_array("force_std", force_std)
        atoms.set_array("local_mean", local_mean)
        mask = np.zeros(len(atoms), dtype=int)
        mask[selected] = 1
        atoms.set_array("candidate_center", mask)
        atoms.info.update(committee_energy_mean_eV=float(energies.mean()), committee_energy_std_eV=float(energies.std()))
        write(out / "uncertainty.extxyz", atoms, format="extxyz", append=True, write_results=False)
        frame = int(atoms.info["source_frame"])
        step = atoms.info.get("timestep", "")
        summary.append({"source_frame": frame, "timestep": step, "n_atoms": len(atoms), "force_std_max": float(force_std.max()), "force_std_mean": float(force_std.mean()), "force_std_p95": float(np.percentile(force_std, 95)), "local_mean_max": float(local_mean.max()), "energy_mean_eV": float(energies.mean()), "energy_std_eV": float(energies.std()), "energy_per_atom_std_eV": float(energies.std() / len(atoms)), "n_centers": len(selected)})
        for index in selected:
            candidates.append({"source_frame": frame, "timestep": step, "atom_id": int(atoms.arrays["atom_id"][index]), "atom_id_source": atoms.info["atom_id_source"], "ase_index": index, "element": atoms[index].symbol, "x": float(atoms.positions[index, 0]), "y": float(atoms.positions[index, 1]), "z": float(atoms.positions[index, 2]), "force_std": float(force_std[index]), "local_mean": float(local_mean[index]), "rank_score": float(scores[index]), "snapshot": str(path.relative_to(out))})
    candidates.sort(key=lambda row: (-row["rank_score"], row["source_frame"], row["atom_id"]))
    candidate_fields = ["source_frame", "timestep", "atom_id", "atom_id_source", "ase_index", "element", "x", "y", "z", "force_std", "local_mean", "rank_score", "snapshot"]
    write_csv(out / "candidates.csv", candidates, candidate_fields)
    write_csv(out / "summary.csv", summary, list(summary[0]))
    metadata.update(status="complete", selected_frames=len(paths), candidate_count=len(candidates), wall_seconds=time.perf_counter() - started)
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done: {out}\nScores are model disagreement, not calibrated DFT errors.", flush=True)
    return out


if __name__ == "__main__":
    import settings
    run(settings)
