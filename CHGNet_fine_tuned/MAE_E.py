#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Plot settings
plt.rcParams["figure.figsize"] = (4, 4)
#plt.rcParams['font.family'] = "arial"
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 600

# Load JSON file
base_path = "/data01/tian_02/ML/LGPS_fine_tuning/test_results"
test_path = base_path + "/test_result.json"
with open(test_path, "r") as f:
    test_data = json.load(f)

# Use coolwarm colormap
cmap_full = matplotlib.colormaps["coolwarm"]

# Extract energy data and generate colored scatter plot
def extract_energy_colormap(data, label, cmap_range):
    target_list = []
    pred_list = []
    err_list = []
    for entry in data:
        e_target = entry["energy"]["ground_truth"]
        e_pred = entry["energy"]["prediction"]
        target_list.append(e_target)
        pred_list.append(e_pred)
        err_list.append(abs(e_target - e_pred))
    
    target_all = np.array(target_list)
    pred_all = np.array(pred_list)
    err_all = np.array(err_list)

    # Normalize error for color mapping
    norm_err = (err_all - err_all.min()) / (np.ptp(err_all) + 1e-8)
    mapped_vals = cmap_range[0] + norm_err * (cmap_range[1] - cmap_range[0])
    colors = cmap_full(mapped_vals)

    # Create scatter plot
    plt.scatter(pred_all, target_all, color=colors, s=50, alpha=0.5, label=label, edgecolors="none")
    return target_all, pred_all

# Begin plotting
plt.figure()

# Plot energy comparison for test set
test_target, test_pred = extract_energy_colormap(test_data, "Test set", (0.0, 1))

# Determine axis limits
emin = min(test_target.min(), test_pred.min())
emax = max(test_target.max(), test_pred.max())
lims = [emin, emax]

# Reference line y = x
plt.plot(lims, lims, "--", color="gray", label="y = x")

# Fit line and MAE calculation for test set
slope_test, intercept_test = np.polyfit(test_pred, test_target, 1)
mae_test = np.mean(np.abs(test_target - test_pred))
plt.plot(lims, slope_test * np.array(lims) + intercept_test,
         color="tab:red", linewidth=1, linestyle="-", label=f"Test fit: slope={slope_test:.3f}")

# Axis and title details
plt.xlim(lims)
plt.ylim(lims)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Predicted energy (eV/atom)")
plt.ylabel("Target energy (eV/atom)")
plt.title("Energy Comparison: DFT vs CHGNet", fontsize=12)

# Annotate MAE on plot
plt.text(0.4, 0.08, f"Test MAE: {mae_test:.3f} eV/atom", transform=plt.gca().transAxes, fontsize=10)

# Legend and export
plt.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(base_path + "/energy_plot_test_fit.png")
plt.show()

