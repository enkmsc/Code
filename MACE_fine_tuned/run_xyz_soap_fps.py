#!/data01/tian_02/soap_env/bin/python

import os
import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP
from sklearn.metrics import pairwise_distances

traj_file = "md.traj"
frame = 50 # every 50 frames select one frame
skip_ps = 2
timestep_fs = 2
skip_steps = int(skip_ps * 1000 / timestep_fs)

n_select = 400
material = "LLZO"
species = ["Li", "La", "Zr", "O"]

frames = read(traj_file, index=f"{skip_steps}::{frame}")
write("md.xyz", frames, format="extxyz")

# SOAP
soap = SOAP(
    species=species,
    r_cut=5.0,
    n_max=8,
    l_max=6,
    periodic=True,
    average="inner",
    sparse=False,
)

X = soap.create(frames)
np.save("soap_vectors.npy", X)

# FPS
selected = [0]
dist_to_selected = pairwise_distances(X, X[selected], metric="euclidean").reshape(-1)

for i in range(1, n_select):
    idx = np.argmax(dist_to_selected)
    selected.append(idx)
    new_dist = pairwise_distances(X, X[[idx]], metric="euclidean").reshape(-1)
    dist_to_selected = np.minimum(dist_to_selected, new_dist)

selected = np.array(selected)
selected_frames = [frames[i] for i in selected]

for fps_id, frame_id, atoms in zip(range(1, len(selected_frames)+1), selected, selected_frames):
    atoms.info["fps_id"] = int(fps_id)
    atoms.info["original_frame"] = int(skip_steps + frame_id * frame)

write("fps_selected.xyz", selected_frames, format="extxyz")

os.makedirs("VASP", exist_ok=True)

for count, atoms in enumerate(selected_frames, start=1):
    write(f"VASP/{count:03d}_{material}.vasp", atoms, format="vasp", direct=True, sort=False)
