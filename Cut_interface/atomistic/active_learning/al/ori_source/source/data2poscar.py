import sys
from ase.io import read, write
import numpy as np
from ase.constraints import FixAtoms

name = str(sys.argv[1])
cfg = read(name, format='lammps-data', style='atomic', Z_of_type={1: 3, 2: 15, 3: 16, 4: 35})
cellpar = cfg.get_cell()

# constrain atoms
pos = cfg.get_positions(wrap=True)
idx = []
for i in range(len(pos)):
    dis_vec = np.absolute(pos[i] - cellpar[0,0]/2)
    if (dis_vec < 6.5).all():
        idx.append(i)
idx = np.array(idx)
constr = FixAtoms(indices=idx)
cfg.set_constraint(constr)

#double check min distance
dis = cfg.get_all_distances(mic=True)
min_dis = np.min(dis[np.nonzero(dis)])
if min_dis > 1.6:
    write('POSCAR_%s'%name, cfg, format='vasp', direct=True)
else:
    print("short distance after relaxation: %s"%min_dis)
    write('POSCAR_%s'%name, cfg, format='vasp', direct=True)
