import os
import random
from ase.io import read, write

base_path = "BASEPATH"
dft_path = "DFTPATH"
materials = ["LaCO", "LaZO", "LCO", "LCO_104", "LCO_LLZO", "LLZO", "LLZO_110"]
SPLIT = (0.8, 0.1, 0.1)
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = SPLIT
SEED = 42

train_xyz = os.path.join(base_path, "train.xyz")
valid_xyz = os.path.join(base_path, "valid.xyz")
test_xyz = os.path.join(base_path, "test.xyz")

train_atoms = []
valid_atoms = []
test_atoms = []

random.seed(SEED)

for material in materials:
    material_path = os.path.join(dft_path, material)
    folder_names = [
        folder for folder in os.listdir(material_path)
        if os.path.isdir(os.path.join(material_path, folder))
    ]
    random.shuffle(folder_names)

    n_total = len(folder_names)
    n_train = int(TRAIN_RATIO * n_total)
    n_valid = int(VALID_RATIO * n_total)

    train_folders = folder_names[:n_train]
    valid_folders = folder_names[n_train:n_train + n_valid]
    test_folders = folder_names[n_train + n_valid:]

    for folder in train_folders:
        outcar_path = os.path.join(material_path, folder, "OUTCAR")
        if not os.path.isfile(outcar_path):
            continue
        images = read(outcar_path, format="vasp-out", index=":")
        for atoms in images:
            atoms.info["vasp_energy"] = float(atoms.get_potential_energy())
            atoms.arrays["vasp_forces"] = atoms.get_forces()
            atoms.info["vasp_stress"] = atoms.get_stress(voigt=False)
            train_atoms.append(atoms)

    for folder in valid_folders:
        outcar_path = os.path.join(material_path, folder, "OUTCAR")
        if not os.path.isfile(outcar_path):
            continue
        images = read(outcar_path, format="vasp-out", index=":")
        for atoms in images:
            atoms.info["vasp_energy"] = float(atoms.get_potential_energy())
            atoms.arrays["vasp_forces"] = atoms.get_forces()
            atoms.info["vasp_stress"] = atoms.get_stress(voigt=False)
            valid_atoms.append(atoms)

    for folder in test_folders:
        outcar_path = os.path.join(material_path, folder, "OUTCAR")
        if not os.path.isfile(outcar_path):
            continue
        images = read(outcar_path, format="vasp-out", index=":")
        for atoms in images:
            atoms.info["vasp_energy"] = float(atoms.get_potential_energy())
            atoms.arrays["vasp_forces"] = atoms.get_forces()
            atoms.info["vasp_stress"] = atoms.get_stress(voigt=False)
            test_atoms.append(atoms)

write(train_xyz, train_atoms, format="extxyz", write_results=False)
write(valid_xyz, valid_atoms, format="extxyz", write_results=False)
write(test_xyz, test_atoms, format="extxyz", write_results=False)

print("Total configs:", len(train_atoms) + len(valid_atoms) + len(test_atoms))
print("Train:", len(train_atoms))
print("Valid:", len(valid_atoms))
print("Test :", len(test_atoms))
