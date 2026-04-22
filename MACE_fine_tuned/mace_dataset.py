
import os
import random
from ase.io import read, write

base_path = "/data01/tian_02/MACE/fine_tune_LATP"
dft_path = "/data01/tian_02/ML_LATP/MD/SP_spin/LATP_2Al"

SPLIT = (0.8, 0.1, 0.1)
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = SPLIT
SEED = 42

train_xyz = os.path.join(base_path, "train.xyz")
valid_xyz = os.path.join(base_path, "valid.xyz")
test_xyz  = os.path.join(base_path, "test.xyz")

all_atoms = []
folder_names = os.listdir(dft_path)

for folder in folder_names:
    outcar_path = os.path.join(dft_path, folder, "OUTCAR")
    print("Reading:", outcar_path)

    if not os.path.isfile(outcar_path):
        continue

    images = read(outcar_path, format="vasp-out", index=":")

    for atoms in images:
        atoms.info["vasp_energy"] = float(atoms.get_potential_energy())
        atoms.arrays["vasp_forces"] = atoms.get_forces()
        atoms.info["vasp_stress"] = atoms.get_stress(voigt=False)
        atoms.arrays["vasp_magmoms"] = atoms.get_magnetic_moments()
        all_atoms.append(atoms)

random.seed(SEED)
random.shuffle(all_atoms)

n_total = len(all_atoms)
n_train = int(TRAIN_RATIO * n_total)
n_valid = int(VALID_RATIO * n_total)
n_test = n_total - n_train - n_valid

train_atoms = all_atoms[:n_train]
valid_atoms = all_atoms[n_train:n_train + n_valid]
test_atoms  = all_atoms[n_train + n_valid:]

write(train_xyz, train_atoms, format="extxyz", write_results=False)
write(valid_xyz, valid_atoms, format="extxyz", write_results=False)
write(test_xyz,  test_atoms,  format="extxyz", write_results=False)

print("Total configs:", n_total)
print("Train:", len(train_atoms))
print("Valid:", len(valid_atoms))
print("Test :", len(test_atoms))
