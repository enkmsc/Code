"""CPU tests; synthetic calculators verify plumbing, not MACE model accuracy."""
import csv
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

import evaluate_committee as ec
import settings


def dump_text(step=0, length=10, coord="xu yu zu", rows="20 2 9.8 0 0\n5 1 0.2 0 0\n9 1 5 0 0\n"):
    return f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 {length}\n0 {length}\n0 {length}\nITEM: ATOMS id type {coord}\n{rows}"


def test_dump_ids_npt_and_unwrapped(tmp_path):
    path = tmp_path / "dump"
    path.write_text(dump_text() + dump_text(step=500, length=12, rows="20 2 15 0 0\n5 1 -0.2 0 0\n9 1 5 0 0\n"))
    frames = list(ec.dump_frames(path, {1: "Li", 2: "O"}))
    assert frames[0].arrays["atom_id"].tolist() == [5, 9, 20]
    assert frames[0].get_chemical_symbols() == ["Li", "Li", "O"]
    assert frames[1].info["timestep"] == 500
    np.testing.assert_allclose(frames[1].cell, np.eye(3) * 12)
    np.testing.assert_allclose(frames[1].positions[:, 0], [-0.2, 5, 15])


@pytest.mark.parametrize("coord", ["x y z", "xu yu zu", "xs ys zs", "xsu ysu zsu"])
def test_triclinic_and_origin_against_ase(tmp_path, coord):
    text = f"ITEM: TIMESTEP\n200\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS xy xz yz pp pp ff\n-1 12 -2\n2 15 1\n3 15 3\nITEM: ATOMS id type {coord}\n8 2 0.8 0.9 0.2\n2 1 0.2 0.3 0.4\n"
    path = tmp_path / "tilt"
    path.write_text(text)
    atoms = next(ec.dump_frames(path, {1: "Li", 2: "O"}))
    np.testing.assert_allclose(atoms.cell, [[10, 0, 0], [-2, 10, 0], [1, 3, 12]])
    np.testing.assert_allclose(atoms.info["lammps_origin"], [1, 2, 3])
    assert atoms.pbc.tolist() == [True, True, False]
    if coord.startswith("xs"):
        expected = np.array([[0.2, 0.3, 0.4], [0.8, 0.9, 0.2]]) @ atoms.cell + [1, 2, 3]
    else:
        expected = [[0.2, 0.3, 0.4], [0.8, 0.9, 0.2]]
    np.testing.assert_allclose(atoms.positions, expected)
    if coord != "xsu ysu zsu":
        reference = read(io.StringIO(text), format="lammps-dump-text", specorder=["Li", "O"])
        np.testing.assert_allclose(atoms.cell, reference.cell)


def test_periodic_local_average_and_spacing():
    atoms = Atoms("Li3", positions=[[0.2, 0, 0], [9.8, 0, 0], [5, 0, 0]], cell=[10, 10, 10], pbc=True)
    ec.ensure_ids(atoms)
    scores = ec.local_scores(atoms, np.array([1.0, 3.0, 9.0]), 1.0)
    np.testing.assert_allclose(scores, [2, 2, 9])
    assert ec.select_centers(atoms, scores, 3, 1) == [2, 0]
    assert ec.select_centers(atoms, scores, 3, 1, min_score=10) == []
    assert ec.select_centers(atoms, scores, 3, 1, elements=["O"]) == []


def test_neighbor_images_do_not_double_count():
    atoms = Atoms("LiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=True)
    np.testing.assert_allclose(ec.local_scores(atoms, np.array([1., 5.]), 3), [3, 3])


def test_statistics_match_explicit_committee():
    forces = np.array([[[1., 2., 3.], [4., -1., 2.]], [[3., 4., 5.], [6., -3., 0.]], [[5., 0., 1.], [2., 2., 4.]]])
    mean, m2 = np.zeros((2, 3)), np.zeros((2, 3))
    for count, frame in enumerate(forces, 1):
        mean, m2 = ec.update_moments(mean, m2, frame, count)
    np.testing.assert_allclose(mean, forces.mean(axis=0))
    np.testing.assert_allclose(m2 / 3, forces.var(axis=0))


def make_settings(tmp_path):
    for i in (1, 3):
        (tmp_path / f"model{i}").write_text(str(i))
    (tmp_path / "dump").write_text(dump_text() + dump_text(100, 11) + dump_text(200, 12))
    cfg = {k: v for k, v in vars(settings).items() if k.isupper()}
    cfg.update(MODEL_PATHS=["model1", "model3"], DEVICE="cpu", INPUT_FILE="dump", TYPE_MAP={1: "Li", 2: "O"}, OUTPUT_DIR="output", FRAME_STRIDE=2, MAX_FRAMES=2, LOCAL_RADIUS=1, MIN_CENTER_DISTANCE=1, TOP_CENTERS_PER_FRAME=2)
    return SimpleNamespace(**cfg)


class FakeCalculator:
    def __init__(self, scale):
        self.scale = scale

    def get_forces(self, atoms):
        return self.scale * atoms.positions

    def get_potential_energy(self, atoms):
        return self.scale * len(atoms)


def test_end_to_end_model_outer_loop(tmp_path):
    cfg = make_settings(tmp_path)
    order = []
    def factory(path, config):
        order.append(path.name)
        return FakeCalculator(float(path.read_text()))
    output = ec.run(cfg, tmp_path, factory)
    assert order == ["model1", "model3"]
    frames = read(output / "uncertainty.extxyz", ":")
    assert [at.info["source_frame"] for at in frames] == [0, 2]
    assert [at.info["timestep"] for at in frames] == [0, 200]
    for atoms in frames:
        np.testing.assert_allclose(atoms.arrays["committee_force_mean"], 2 * atoms.positions)
        np.testing.assert_allclose(atoms.arrays["force_std"], np.linalg.norm(atoms.positions, axis=1))
        assert atoms.arrays["atom_id"].tolist() == [5, 9, 20]
        assert atoms.info["committee_energy_std_eV"] == 3
    rows = list(csv.DictReader((output / "candidates.csv").open()))
    assert len(rows) == 4
    assert [float(row["rank_score"]) for row in rows] == sorted([float(row["rank_score"]) for row in rows], reverse=True)
    for row in rows:
        snapshot = read(output / row["snapshot"])
        assert snapshot.arrays["atom_id"][int(row["ase_index"])] == int(row["atom_id"])
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        ec.run(cfg, tmp_path, factory)


def test_empty_candidate_table(tmp_path):
    cfg = make_settings(tmp_path)
    cfg.MIN_SCORE = 100
    output = ec.run(cfg, tmp_path, lambda p, c: FakeCalculator(float(p.read_text())))
    assert list(csv.DictReader((output / "candidates.csv").open())) == []


def test_identical_models_rejected(tmp_path):
    cfg = make_settings(tmp_path)
    (tmp_path / "model3").write_text("1")
    with pytest.raises(ValueError, match="Identical"):
        ec.run(cfg, tmp_path)
    assert not (tmp_path / "output").exists()


def test_unknown_types_and_truncation(tmp_path):
    path = tmp_path / "dump"
    path.write_text(dump_text())
    with pytest.raises(ValueError, match="TYPE_MAP"):
        list(ec.dump_frames(path, {1: "Li"}))
    path.write_text(dump_text().rsplit("9 1", 1)[0])
    with pytest.raises(ValueError, match="Truncated"):
        list(ec.dump_frames(path, {1: "Li", 2: "O"}))


def test_max_frames_stops_before_trailing_invalid_data(tmp_path):
    cfg = make_settings(tmp_path)
    cfg.MAX_FRAMES = 1
    (tmp_path / "dump").write_text(dump_text() + "broken trailing data\n")
    ec.run(cfg, tmp_path, lambda p, c: FakeCalculator(float(p.read_text())))


def test_extxyz_preserves_ids(tmp_path):
    cfg = make_settings(tmp_path)
    cfg.INPUT_FORMAT, cfg.INPUT_FILE = "extxyz", "input.xyz"
    atoms = Atoms("LiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.new_array("atom_id", np.array([100, 7]))
    write(tmp_path / "input.xyz", atoms, format="extxyz")
    output = ec.run(cfg, tmp_path, lambda p, c: FakeCalculator(float(p.read_text())))
    result = read(output / "uncertainty.extxyz")
    assert result.arrays["atom_id"].tolist() == [100, 7]


def test_lammps_data_ids(tmp_path):
    cfg = make_settings(tmp_path)
    cfg.INPUT_FORMAT, cfg.INPUT_FILE = "lammps-data", "input.data"
    (tmp_path / "input.data").write_text("LAMMPS data\n\n2 atoms\n2 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nAtoms # atomic\n\n20 2 1 0 0\n5 1 0 0 0\n")
    atoms = next(ec.input_frames(cfg, tmp_path / "input.data"))
    ec.ensure_ids(atoms)
    assert atoms.arrays["atom_id"].tolist() == [5, 20]

