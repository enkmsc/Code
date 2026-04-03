from pymatgen.io.vasp import Poscar
from pymatgen.io.lammps.data import LammpsData

base_path = "/Users/emilydai/POSCAR_REV.vasp"
output_path = "/Users/emilydai/Downloads/LCO_LLZO_medium.data"

structure = Poscar.from_file(base_path).structure

ld = LammpsData.from_structure(
    structure,
    atom_style="atomic"
)

ld.write_file(output_path)
