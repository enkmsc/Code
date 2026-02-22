from DistMLIP.implementations.matgl import CHGNet_Dist, Potential_Dist, MolecularDynamics
import matgl
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.io import write
import os
import time

material = "LATP_LCO"
base_path = "/home/u9231474/chgnet_LATP_LCO"
log_path = os.path.join(base_path, "logfile")
xdatcar_path = os.path.join(base_path, "XDATCAR")
trajectory_path = os.path.join(base_path, f"{material}.traj")
fine_tune_path = "/home/u9231474/matgl_chgent_ft/LATP_LCO_finetuned_model"
structure = Structure.from_file(os.path.join(base_path, "POSCAR"))
print(f"There are {len(structure)} atoms.")

model = matgl.load_model(fine_tune_path).model
dist_model = CHGNet_Dist.from_existing(model)

dist_model.enable_distributed_mode([0, 1, 2, 3, 4, 5])
atoms = AseAtomsAdaptor().get_atoms(structure)

potential = Potential_Dist(model=dist_model, num_threads=72, calc_stresses=True)

MaxwellBoltzmannDistribution(atoms, temperature_K=300)

md = MolecularDynamics(
    atoms,
    potential=potential,
    ensemble="nvt",
    timestep=2,
    temperature=1500,
    loginterval=1,
    taut=10,
    trajectory=trajectory_path,
    logfile=log_path,
)

start = time.time()
md.run(100)
end = time.time()
total_time = end - start
print(f"Total MD wall time: {total_time:.5f} s")
print(f"Time per step: {total_time/100:.5f} s/step")

traj = Trajectory(trajectory_path)
final_structure = traj[-1]
write(base_path+ "/CONTCAR", final_structure, format="vasp")
