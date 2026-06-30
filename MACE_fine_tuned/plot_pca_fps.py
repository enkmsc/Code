#!/data01/tian_02/soap_env/bin/python

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (6, 5)

soap_file = "soap_vectors.npy"
fps_xyz = "fps_selected.xyz"

frame = 50
timestep_fs = 2
skip_ps = 2
skip_steps = int(skip_ps * 1000 / timestep_fs)

X = np.load(soap_file)

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

fps_frames = read(fps_xyz, ":")
selected = np.array([(atoms.info["original_frame"] - skip_steps) // frame for atoms in fps_frames])

time_ps = skip_ps + np.arange(len(X2)) * frame * timestep_fs / 1000

fig, ax = plt.subplots()

sc = ax.scatter(X2[:,0], X2[:,1], c=time_ps, cmap="viridis", s=8, vmin=0, vmax=300)
ax.scatter(X2[selected,0], X2[selected,1], marker="x", s=8, c="k", linewidths=0.5, alpha=0.8)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Simulation Time (ps)")
cbar.set_ticks([0, 50, 100, 150, 200, 250, 300])

ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.2f}%)")
ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.2f}%)")

plt.tight_layout()
plt.savefig("SOAP_PCA_FPS_Time.png", dpi=300, bbox_inches="tight")
