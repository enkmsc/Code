#!/data00/software/python-envs/CHGNet-MD/bin/python
from pymatgen.io.vasp import Poscar
from pymatgen.io.lammps.data import LammpsData

input_path = "/data01/tian_02/CALYPSO/LLZO_interface/LCO_LLZO/interface/test/POSCAR_REV.vasp"
output_path = "LCO_LLZO_m.data"

structure = Poscar.from_file(input_path).structure

ld = LammpsData.from_structure(
    structure,
    atom_style="atomic"
)

ld.write_file(output_path)
