"""
Verify the fitted parameters against the data:
  - reports the L1-distance-based fit quality metric
  - saves a plot comparing the fitted curve to the raw data cloud

Usage:
    python verify.py [path_to_csv]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fit import fit, uniform_l1_distance, Y_OFFSET, OMEGA

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()

    result = fit(x, y)
    theta, M, X = result.x

    mean_l1, max_l1 = uniform_l1_distance(theta, M, X, x, y)
    print(f"theta = {np.rad2deg(theta):.4f} deg, M = {M:.6f}, X = {X:.6f}")
    print(f"mean nearest-neighbor L1 distance: {mean_l1:.4f}")
    print(f"max  nearest-neighbor L1 distance: {max_l1:.4f}")

    # Plot fitted curve over the data
    t = np.linspace(6, 60, 2000)
    ct, st = np.cos(theta), np.sin(theta)
    env = np.exp(M * np.abs(t)) * np.sin(OMEGA * t)
    xf = t * ct - env * st + X
    yf = Y_OFFSET + t * st + env * ct

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=6, alpha=0.4, label="data points", color="tab:gray")
    plt.plot(xf, yf, color="tab:red", linewidth=1.5, label="fitted curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Fit: theta={np.rad2deg(theta):.2f} deg, M={M:.4f}, X={X:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fit_verification.png", dpi=150)
    print("Saved plot to fit_verification.png")


if __name__ == "__main__":
    main()
