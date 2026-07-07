# Parametric Curve Parameter Recovery

## Problem

Recover the unknown parameters `θ`, `M`, `X` in:

```
x(t) = t*cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
y(t) = 42 + t*sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)
```

given a cloud of `(x, y)` points sampled from the curve for `6 < t < 60`, with
search ranges `0° < θ < 50°`, `-0.05 < M < 0.05`, `0 < X < 100`.

## Result

| Parameter | Value |
|---|---|
| θ | **30°** (0.523599 rad) |
| M | **0.03** |
| X | **55** |

**Desmos / LaTeX string:**

```
\left(t*\cos(0.523599)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)+55,42+t*\sin(0.523599)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)\right)
```
domain: `6 ≤ t ≤ 60`

**Fit quality:** sum of squared residuals ≈ 1.8×10⁻⁸ over all 1500 data
points (see [Verification](#verification) below) — the recovered curve
passes through the data essentially exactly, which is strong evidence these
are the true generating parameters rather than a nearby local optimum.

## Approach

### 1. Exploit the geometry — decouple rotation/translation from the envelope

The two equations are not independent in `x` and `y` — together they describe
a single 2D transform. Writing:

```
u = t
v = e^(M|t|) * sin(0.3t)
```

the equations become:

```
x - X = u*cos(θ) - v*sin(θ)
y - 42 = u*sin(θ) + v*cos(θ)
```

This is exactly a **rotation by θ** of the point `(u, v)`, plus a
**translation** of `(X, 42)`. So the transform is invertible: for *any*
candidate `(θ, X)`, undoing the rotation and translation on a raw data point
recovers what `(u, v)` — i.e. `(t, envelope(t))` — would have to be:

```
p = (x - X)*cos(θ) + (y - 42)*sin(θ)     ->  should equal t
q = -(x - X)*sin(θ) + (y - 42)*cos(θ)    ->  should equal e^(M|t|) * sin(0.3t)
```

Crucially, this holds **pointwise** — no correspondence between rows of the
CSV and a specific `t` value is needed, and no ordering/sorting of the data
is required, since each `(x_i, y_i)` maps to its own `(p_i, q_i)` under a
candidate `(θ, X)`.

### 2. Turn it into a least-squares problem

For the *correct* `(θ, M, X)`:

```
q_i ≈ exp(M * |p_i|) * sin(0.3 * p_i)   for every point i
```

So the fitting problem reduces to minimizing, over all 1500 points at once:

```
sum_i ( q_i - exp(M*|p_i|)*sin(0.3*p_i) )^2
```

This is a clean 3-parameter nonlinear least-squares problem.

### 3. Optimize in two stages

- **Global search** (`scipy.optimize.differential_evolution`) over the given
  bounds `θ ∈ (0°, 50°)`, `M ∈ (-0.05, 0.05)`, `X ∈ (0, 100)` — used because
  the `sin(0.3t)` term is oscillatory, so a naive local optimizer started
  from a poor guess can lock onto a wrong local minimum.
- **Local refinement** (`scipy.optimize.least_squares`, Levenberg–Marquardt
  style trust-region) starting from the global search result, tightened to
  `xtol = ftol = gtol = 1e-15` for high precision.

This converged to `θ = 30°`, `M = 0.03`, `X = 55` with near-zero residual,
and the values are suspiciously round — consistent with these being the
exact ground-truth parameters used to generate the dataset.

### 4. Sanity checks

- Recovering `t` from the data via `p = (x-X)cos(θ) + (y-42)sin(θ)` at the
  fitted parameters gives a range of **[6.05, 59.995]**, matching the stated
  domain `6 < t < 60` almost exactly.
- Re-sampling the fitted curve uniformly over `t ∈ [6, 60]` and computing the
  nearest-neighbor L1 distance to the data cloud gives a mean of **~0.027**
  and a max of **~0.34** (see `verify.py`), on data whose x, y values range
  over ~60 and ~24 units respectively — i.e. sub-percent error, consistent
  with points being sampled from the fitted curve plus/minus small numerical
  noise.

## Files

- `fit.py` — main solver; run `python fit.py xy_data.csv`
- `verify.py` — reproduces the fit quality / L1 distance metric and plots the
  fitted curve against the data
- `xy_data.csv` — input data (as provided in the assignment)
- `requirements.txt` — Python dependencies

## Usage

```bash
pip install -r requirements.txt
python fit.py xy_data.csv
python verify.py
```

## Notes on the search / assessment criteria

- The **L1 distance between uniformly sampled points** on the expected vs.
  predicted curve (assessment criterion #1) is what `verify.py` reports via
  nearest-neighbor distance from fitted-curve samples to the raw data cloud,
  since the "expected" curve itself isn't directly available — only points
  sampled from it.
- No part of this approach required brute-force grid search over all three
  parameters simultaneously — the geometric decoupling (rotation/translation
  vs. envelope) turns it into a much better-conditioned 3-parameter fit that
  converges reliably and precisely.
