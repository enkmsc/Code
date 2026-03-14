#!/data01/tian_02/mech-mace/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ase.io import read

plt.rcParams["figure.figsize"] = (3.3, 3.3)
plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 600

base_path = "/data01/tian_02/MACE/fine_tune_LATP"
dft_xyz = base_path + "/test.xyz"
mace_xyz = base_path + "/eval_test.xyz"
force_png = base_path + "/force_parity.png"
energy_png = base_path + "/energy_parity.png"

cmap = matplotlib.colormaps["coolwarm"]

dft_frames = read(dft_xyz, index=":")
mace_frames = read(mace_xyz, index=":")

dft_force_all = np.concatenate([a.arrays["vasp_forces"].reshape(-1) for a in dft_frames]).astype(float)
pred_force_all = np.concatenate([a.arrays["MACE_forces"].reshape(-1) for a in mace_frames]).astype(float)

force_err = np.abs(dft_force_all - pred_force_all)
force_norm = (force_err - force_err.min()) / (np.ptp(force_err) + 1e-12)
force_colors = cmap(force_norm)

plt.figure()
plt.scatter(dft_force_all, pred_force_all, c=force_colors, s=50, alpha=0.5, edgecolors="none", label="Test set")

force_min = min(dft_force_all.min(), pred_force_all.min())
force_max = max(dft_force_all.max(), pred_force_all.max())
force_lims = [force_min, force_max]

plt.plot(force_lims, force_lims, "--", color="gray", label="y = x")

force_slope, force_intercept = np.polyfit(dft_force_all, pred_force_all, 1)
force_mae = np.mean(np.abs(dft_force_all - pred_force_all))

plt.plot(force_lims, force_slope * np.array(force_lims) + force_intercept, color="tab:red", lw=1, label=f"Fit: slope={force_slope:.3f}")

ax = plt.gca()
ax.set_box_aspect(1)
ax.set_xlim(force_min, force_max)
ax.set_ylim(force_min, force_max)

plt.xlabel("DFT force (eV/Å)")
plt.ylabel("MACE force (eV/Å)")
plt.text(0.45, 0.05, f"MAE: {force_mae:.3f} eV/Å", transform=ax.transAxes, fontsize=12)

plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()

os.makedirs(os.path.dirname(force_png), exist_ok=True)
plt.savefig(force_png, transparent=True)
plt.show()

dft_energy_all = np.array([float(a.info["vasp_energy"]) / len(a) for a in dft_frames], dtype=float)
pred_energy_all = np.array([float(a.info["MACE_energy"]) / len(a) for a in mace_frames], dtype=float)

energy_err = np.abs(dft_energy_all - pred_energy_all)
energy_norm = (energy_err - energy_err.min()) / (np.ptp(energy_err) + 1e-12)
energy_colors = cmap(energy_norm)

plt.figure()
plt.scatter(dft_energy_all, pred_energy_all, c=energy_colors, s=50, alpha=0.5, edgecolors="none", label="Test set")

energy_min = min(dft_energy_all.min(), pred_energy_all.min())
energy_max = max(dft_energy_all.max(), pred_energy_all.max())
energy_lims = [energy_min, energy_max]

plt.plot(energy_lims, energy_lims, "--", color="gray", label="y = x")

energy_slope, energy_intercept = np.polyfit(dft_energy_all, pred_energy_all, 1)
energy_mae = np.mean(np.abs(dft_energy_all - pred_energy_all))

plt.plot(energy_lims, energy_slope * np.array(energy_lims) + energy_intercept, color="tab:red", lw=1, label=f"Fit: slope={energy_slope:.3f}")

ax = plt.gca()
ax.set_box_aspect(1)
ax.set_xlim(energy_min, energy_max)
ax.set_ylim(energy_min, energy_max)

plt.xlabel("DFT energy (eV/atom)")
plt.ylabel("MACE energy (eV/atom)")
plt.text(0.31, 0.05, f"MAE: {energy_mae:.3f} eV/atom", transform=ax.transAxes, fontsize=12)

plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()

os.makedirs(os.path.dirname(energy_png), exist_ok=True)
plt.savefig(energy_png, transparent=True)
plt.show()

print("Saved:", force_png)
print("Saved:", energy_png)
