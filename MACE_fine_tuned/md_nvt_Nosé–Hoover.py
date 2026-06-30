import os
import numpy as np
from ase.io import read, write
from ase import units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io.trajectory import Trajectory
from mace.calculators import MACECalculator

base_path = "BASEPATH" 
model_path = "MODELPATH"
structure_path = "STRUCPATH"
trajectory_path = os.path.join(base_path, "md.traj")
log_path = os.path.join(base_path, "logfile")
xdatcar_path = os.path.join(base_path, "XDATCAR")
contcar_path = os.path.join(base_path, "CONTCAR")

temperature = TEMP
starting_temperature = TEMP
timestep_fs = 2.0
taut_fs = 100.0
pressure_gpa = 1.01325e-4
nsteps = 150000

traj_interval = 1
log_interval = 1

atoms = read(structure_path)

atoms.calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    enable_cueq=True,
)

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=starting_temperature,
    force_temp=True,
)
Stationary(atoms)

dyn = NPT(
    atoms=atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature,
    externalstress=pressure_gpa * units.GPa,
    ttime=taut_fs * units.fs,
    pfactor=None,
    logfile=log_path,
    loginterval=log_interval,
)

traj = Trajectory(trajectory_path, "w", atoms)
dyn.attach(traj.write, interval=traj_interval)
dyn.run(nsteps)

traj.close()
traj_read = Trajectory(trajectory_path, "r")
write(xdatcar_path, traj_read, format="vasp-xdatcar")
traj_read.close()
write(contcar_path, atoms, format="vasp")
