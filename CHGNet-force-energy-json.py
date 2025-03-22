*****************
FORCE-CHGNet-json 
*****************

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# Set plotting parameters
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams['font.family'] = "arial"
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300


file_path = "/Users/emilydai/Library/CloudStorage/OneDrive-個人/文件/Ncku lab/ML_LATP/test_json/test_result_LACTP_2Co.json"
# Load JSON file
with open(file_path, "r") as f:
    data = json.load(f)

# Prepare data
all_dft = []
all_chg = []

num_structures = len(data)
cmap = plt.colormaps.get_cmap('coolwarm').resampled(num_structures)  # Resample to match number of structures
index2color = {i: cmap(i) for i in range(num_structures)}

plt.figure()

# Parse data
for idx, item in enumerate(data):
    dft_forces = np.array(item["force"]["ground_truth"])
    chg_forces = np.array(item["force"]["prediction"])
    
    dft_flat = dft_forces.flatten()
    chg_flat = chg_forces.flatten()
    
    all_dft.append(dft_flat)
    all_chg.append(chg_flat)
    
    plt.scatter(chg_flat, dft_flat, alpha=0.6, color=index2color[idx])

# Concatenate all data
all_dft = np.concatenate(all_dft)
all_chg = np.concatenate(all_chg)

# Calculate MAE
force_mae = np.mean(np.abs(all_dft - all_chg))
print(f"Force MAE: {force_mae:.4f} eV/Å")

# Set plot limits
fmin = min(all_dft.min(), all_chg.min())
fmax = max(all_dft.max(), all_chg.max())
flims = [fmin, fmax]

# y = x reference line
plt.plot(flims, flims, ls="--", color="gray", label="y = x")

# Fit best linear line
slope, intercept = np.polyfit(all_chg, all_dft, 1)
best_fit_x = np.array(flims)
best_fit_y = slope * best_fit_x + intercept
plt.plot(best_fit_x, best_fit_y, color="red", label=f"Best Fit: slope={slope:.3f}")

# Set plot details
plt.xlim(flims)
plt.ylim(flims)
plt.xlabel("Predicted force (eV/Å)")
plt.ylabel("Target force (eV/Å)")
plt.title("Force Comparison: DFT vs CHGNet", fontsize=12)
plt.legend(fontsize=11)
plt.text(0.68, 0.05, f"MAE: {force_mae:.3f} eV/Å", transform=plt.gca().transAxes, fontsize=11)
plt.text(0.68, 0.1, f"Slope: {slope:.3f}", transform=plt.gca().transAxes, fontsize=11)
plt.tight_layout()
plt.savefig("force_plot_json.png")
plt.show()

******************
ENERGY-CHGNet-json 
******************
import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON data
# Replace with your file path
with open(file_path, "r") as f:
    data = json.load(f)

# Initialize lists
all_target_energy = []
all_pred_energy = []

# Ensure the data is a list
for item in data:
    target_energy = item['energy']['ground_truth']
    pred_energy = item['energy']['prediction']
    all_target_energy.append(target_energy)
    all_pred_energy.append(pred_energy)

# Convert to NumPy arrays
all_target_energy = np.array(all_target_energy)
all_pred_energy = np.array(all_pred_energy)

# Calculate global MAE
global_mae = np.mean(np.abs(all_target_energy - all_pred_energy))
# print(f"Global Energy MAE: {global_mae:.4f} eV/atom")

# Calculate plot range
emin = min(all_target_energy.min(), all_pred_energy.min())
emax = max(all_target_energy.max(), all_pred_energy.max())
lims = [emin, emax]

# Set plot parameters
plt.figure(figsize=(5, 5), dpi=300)
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 10

# Create scatter plot
plt.scatter(all_pred_energy, all_target_energy, alpha=0.6, label="Data Points")

# Draw y = x reference line
plt.plot(lims, lims, ls="--", color="gray", label="y = x")

# Fit best linear line
slope, intercept = np.polyfit(all_pred_energy, all_target_energy, 1)
best_fit_x = np.array(lims)
best_fit_y = slope * best_fit_x + intercept

# Plot best fit line
plt.plot(best_fit_x, best_fit_y, color="red", label=f"Best Fit: slope={slope:.3f}")

# Set axis limits
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Predicted energy (eV/atom)")
plt.ylabel("Target energy (eV/atom)")

# Add title
plt.title("Energy Comparison: DFT vs CHGNet", fontsize=12)

# Add MAE and slope info inside the plot
plt.text(0.6, 0.05, f"MAE: {global_mae:.3f} eV/atom", transform=plt.gca().transAxes, fontsize=11)
plt.text(0.6, 0.1, f"Slope: {slope:.3f}", transform=plt.gca().transAxes, fontsize=11)

# Add legend
plt.legend(fontsize=11)
plt.tight_layout()

# Save the figure
plt.savefig("energy_plot.png")

# Show the figure
plt.show()

