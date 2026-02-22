#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Plotting parameters
plt.rcParams["figure.figsize"] = (4, 4)
#plt.rcParams['font.family'] = "arial"
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 600

# Load JSON file
base_path = "/data01/tian_02/ML/LGPS_fine_tuning/test_results"
test_path = base_path + "/test_result.json"

with open(test_path, "r") as f:
    test_data = json.load(f)

# Use colormap
cmap_full = matplotlib.colormaps["coolwarm"]
# cmap_full = matplotlib.colormaps["managua"]

# Data extraction and plotting function (for one dataset)
def extract_and_plot(data, label, cmap_range, reverse=False):
    dft_list, chg_list, err_list = [], [], []
    for entry in data:
        dft = np.array(entry["force"]["ground_truth"]).flatten()
        chg = np.array(entry["force"]["prediction"]).flatten()
        if dft.size and chg.size and dft.shape == chg.shape:
            dft_list.append(dft)
            chg_list.append(chg)
            err_list.append(np.abs(dft - chg))
    dft_all = np.concatenate(dft_list)
    chg_all = np.concatenate(chg_list)
    err_all = np.concatenate(err_list)

    # Normalize errors
    norm_err = (err_all - err_all.min()) / (np.ptp(err_all) + 1e-8)

    # Map normalized errors to colormap range
    mapped_vals = cmap_range[0] + norm_err * (cmap_range[1] - cmap_range[0])
    colors = cmap_full(mapped_vals)

    # Scatter plot
    plt.scatter(chg_all, dft_all, color=colors, s=50, alpha=0.5, label=label, edgecolors="none")
    return dft_all, chg_all

# Start plotting
plt.figure()

# Plot test set
test_dft, test_chg = extract_and_plot(test_data, "Test set", (0.0, 1))

# y = x reference line
fmin = min(test_dft.min(), test_chg.min())
fmax = max(test_dft.max(), test_chg.max())
flims = [fmin, fmax]
plt.plot(flims, flims, "--", color="gray", label="y = x")

# Test set fit line and MAE
slope_test, intercept_test = np.polyfit(test_chg, test_dft, 1)
mae_test = np.mean(np.abs(test_dft - test_chg))
plt.plot(flims, slope_test * np.array(flims) + intercept_test,
         color="tab:red", linewidth=1, linestyle="-", label=f"Test fit: slope={slope_test:.3f}")

# Axis and title settings
plt.xlim(flims)
plt.ylim(flims)
plt.xlabel("Predicted force (eV/Å)")
plt.ylabel("Target force (eV/Å)")
plt.title("Force Comparison: DFT vs CHGNet", fontsize=12)

# MAE annotation
plt.text(0.4, 0.08, f"Test MAE: {mae_test:.3f} eV/Å", transform=plt.gca().transAxes, fontsize=12)

# Legend and layout
plt.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(base_path + "/force_plot_test_fit.png")
plt.show()

