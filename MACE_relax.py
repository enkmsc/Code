from ase import Atoms
from pymatgen.core import Structure
from ase.optimize import FIRE
from mace.calculators import MACECalculator
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
import torch

base_path = "/data01/tian_02/MACE/test"
poscar_path = base_path + "/POSCAR"
input_structure = Structure.from_file(poscar_path)

atoms = AseAtomsAdaptor.get_atoms(input_structure)

# Use model_paths instead of model_path
calc = MACECalculator(
    model_paths=[base_path + "/2024-01-07-mace-128-L2_epoch-199.model"],
    enable_cueq=False,
    weights_only=False,  # Load only weights when loading the model
    device="cuda"
)

# Use the new syntax
atoms.calc = calc

opt = FIRE(atoms)
opt.run(fmax=0.01)

relaxed_structure = AseAtomsAdaptor.get_structure(atoms)
Poscar(relaxed_structure).write_file(base_path + "/CONTCAR")
