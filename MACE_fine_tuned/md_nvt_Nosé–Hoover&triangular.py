import os
import numpy as np
from ase.io import read
from ase import units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator

base_path = "BASEPATH"
model_path = "MODELPATH"
structure_path = os.path.join(base_path, "POSCAR")
trajectory_path = os.path.join(base_path, "md.traj")
log_path = os.path.join(base_path, "logfile")

temperature = TEMP
starting_temperature = TEMP
timestep_fs = 2.0
taut_fs = 100.0
pressure_gpa = 1.01325e-4
nsteps = 500000

atoms = read(structure_path)

atoms.calc = MACECalculator(
    model_paths=model_path,
    device="cuda"
)

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=starting_temperature,
    force_temp=True
)
Stationary(atoms)

def upper_triangular_cell(atoms):
    cell = atoms.get_cell()

    if not np.allclose(cell, np.triu(cell)):
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        angles = np.radians((alpha, beta, gamma))

        sin_a, sin_b, _sin_g = np.sin(angles)
        cos_a, cos_b, cos_g = np.cos(angles)

        cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
        cos_p = np.clip(cos_p, -1, 1)
        sin_p = (1 - cos_p**2) ** 0.5

        new_basis = [
            (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
            (0, b * sin_a, b * cos_a),
            (0, 0, c),
        ]

        atoms.set_cell(new_basis, scale_atoms=True)

upper_triangular_cell(atoms)

dyn = NPT(
    atoms=atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature,
    externalstress=pressure_gpa * units.GPa,
    ttime=taut_fs * units.fs,
    pfactor=None,
    trajectory=trajectory_path,
    logfile=log_path,
    loginterval=1,
    append_trajectory=False,
)

dyn.run(nsteps)
