

from ase.io import read, write
import numpy as np
import random

base_path = "/Users/emilydai/Downloads/LACTP/"

# 1. Read all ionic steps from XDATCAR
xdatcar_images = read(base_path + "XDATCAR", format="vasp-xdatcar", index=":")

# 2. Read all ionic steps from OUTCAR
outcar_images = read(base_path + "OUTCAR", format="vasp-out", index=":")

# Check how many frames are read
print("Number of XDATCAR images:", len(xdatcar_images))
print("Number of OUTCAR images:", len(outcar_images))

# 3. Combine data from each step
all_images = []
for i, (xdat_atoms, out_atoms) in enumerate(zip(xdatcar_images, outcar_images)):
    # Potential energy
    xdat_atoms.info["energy"] = out_atoms.get_potential_energy()

    # Stress (3x3 matrix returned by get_stress())
    xdat_atoms.info["stress"] = out_atoms.get_stress()

    # Forces
    xdat_atoms.arrays["forces"] = out_atoms.get_forces()

    # If there are magnetizations
    if "magmoms" in out_atoms.arrays:
        xdat_atoms.arrays["magmoms"] = out_atoms.arrays["magmoms"]

    # Energy per atom
    xdat_atoms.info["energy_per_atom"] = (
        xdat_atoms.info["energy"] / len(xdat_atoms)
    )

    # Lattice (cell)
    xdat_atoms.info["Lattice"] = out_atoms.get_cell()

    # Collect updated image
    all_images.append(xdat_atoms)

# -- Now RANDOMLY split into train, valid, test according to 0.8 : 0.1 : 0.1 --

# Optional: set a random seed for reproducibility
# random.seed(42)

# Shuffle the entire list in place
random.shuffle(all_images)

n_total = len(all_images)
train_end = int(n_total * 0.8)
valid_end = int(n_total * 0.9)

train_images = all_images[:train_end]
valid_images = all_images[train_end:valid_end]
test_images  = all_images[valid_end:]

print(f"Training frames: {len(train_images)}")
print(f"Validation frames: {len(valid_images)}")
print(f"Test frames: {len(test_images)}")

# 4. Write out the three sets into separate extended XYZ files
write(base_path + "train.xyz", train_images, format="extxyz")
write(base_path + "valid.xyz", valid_images, format="extxyz")
write(base_path + "test.xyz",  test_images,  format="extxyz")

print("Successfully generated train.xyz, valid.xyz, and test.xyz (RANDOM SPLIT)!")
