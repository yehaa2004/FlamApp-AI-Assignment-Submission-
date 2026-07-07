"""
Parameter recovery for a rotated/translated parametric curve.

    x(t) = t*cos(theta) - e^{M|t|} * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^{M|t|} * sin(0.3t) * cos(theta)

Unknowns: theta, M, X
Known:    parameter domain 6 < t < 60
Data:     xy_data.csv  (columns: x, y -- unordered, t not given)

Usage:
    python fit.py [path_to_csv]
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"

# Search bounds given by the assignment
THETA_BOUNDS = (0.0, np.deg2rad(50.0))
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.0, 100.0)
Y_OFFSET = 42.0
OMEGA = 0.3  # coefficient inside sin(0.3 t)


def load_data(path):
    df = pd.read_csv(path)
    return df["x"].to_numpy(), df["y"].to_numpy()


def residuals(params, x, y):
    """
    Key idea: the (x, y) equations are just a rotation by theta of the
    point (t, e^{M|t|} sin(0.3 t)) followed by translating x by X and
    y by 42. So for ANY point, undoing that rotation/translation with
    the correct (theta, X) must recover:

        p := (x - X) cos(theta) + (y - 42) sin(theta)   ==  t
        q := -(x - X) sin(theta) + (y - 42) cos(theta)  ==  e^{M|t|} sin(0.3 t)

    This holds pointwise (no correspondence / ordering between points
    needed), so we can fit (theta, M, X) directly by minimizing the
    gap between q and the envelope model evaluated at p.
    """
    theta, M, X = params
    ct, st = np.cos(theta), np.sin(theta)
    xs = x - X
    ys = y - Y_OFFSET
    p = xs * ct + ys * st          # recovered t
    q = -xs * st + ys * ct         # recovered envelope value
    model_q = np.exp(M * np.abs(p)) * np.sin(OMEGA * p)
    return q - model_q


def fit(x, y, seed=42):
    bounds = [THETA_BOUNDS, M_BOUNDS, X_BOUNDS]

    # 1) Global search to avoid local minima (the sin(0.3t) term is
    #    periodic-ish, so a purely local optimizer can get stuck).
    de_result = differential_evolution(
        lambda p: np.sum(residuals(p, x, y) ** 2),
        bounds,
        seed=seed,
        tol=1e-12,
        maxiter=2000,
        popsize=40,
        polish=True,
    )

    # 2) Local refinement (Levenberg-Marquardt style) from the DE result.
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    ls_result = least_squares(
        residuals,
        x0=de_result.x,
        args=(x, y),
        bounds=(lo, hi),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    return ls_result


def uniform_l1_distance(theta, M, X, x, y, n_samples=2000):
    """
    Sample the fitted curve uniformly over t in (6, 60) and report the
    L1 distance to the nearest data point for each sample (a proxy for
    how well the fitted curve matches the given point cloud), per the
    assignment's stated evaluation metric.
    """
    t = np.linspace(6, 60, n_samples)
    ct, st = np.cos(theta), np.sin(theta)
    env = np.exp(M * np.abs(t)) * np.sin(OMEGA * t)
    xf = t * ct - env * st + X
    yf = Y_OFFSET + t * st + env * ct

    # nearest-neighbor L1 distance from each fitted sample to the data cloud
    dists = np.min(
        np.abs(xf[:, None] - x[None, :]) + np.abs(yf[:, None] - y[None, :]),
        axis=1,
    )
    return dists.mean(), dists.max()


def to_desmos_string(theta, M, X):
    return (
        f"\\left(t*\\cos({theta:.6f})-e^{{{M:.6f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta:.6f})+{X:.6f},"
        f"42+t*\\sin({theta:.6f})+e^{{{M:.6f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta:.6f})\\right)"
    )


def main():
    x, y = load_data(CSV_PATH)
    result = fit(x, y)
    theta, M, X = result.x

    print(f"theta = {theta:.6f} rad  ({np.rad2deg(theta):.4f} deg)")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"sum of squared residuals = {np.sum(result.fun ** 2):.3e}")
    print(f"max |residual|           = {np.max(np.abs(result.fun)):.3e}")

    mean_l1, max_l1 = uniform_l1_distance(theta, M, X, x, y)
    print(f"mean nearest-neighbor L1 distance (fit vs data) = {mean_l1:.4f}")
    print(f"max  nearest-neighbor L1 distance (fit vs data) = {max_l1:.4f}")

    print("\nDesmos / LaTeX submission string:")
    print(to_desmos_string(theta, M, X))


if __name__ == "__main__":
    main()
