from ase.io import read, write
from ase.constraints import FixAtoms

cfg = read("POSCAR", format='vasp')
del cfg.constraints
write('POSCAR', cfg, format='vasp', direct=True)
