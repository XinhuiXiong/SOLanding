# SOLanding

This repository contains the reference Python implementation for the paper
**"A Second-Order Method Landing on the Stiefel Manifold via Newton–Schulz
Iteration"** by Xinhui Xiong, Bin Gao, and P.-A. Absil.

The code implements the two second-order landing methods from Algorithm 5.1:

- `SOL`: solves the projection-free approximate Newton equation (4.11).
- `SOL-sym`: solves the modified Newton equation (4.9) with the full
  Riemannian Hessian under the metric `g`.

Both methods use the order-1 Newton–Schulz normal component `N(X)` from
equation (3.5) and the second-order landing update
`Lambda(X) = T(X) + N(X)`.

## Repository Layout

```text
optimizer/
  optimizer.py        # SOL, SOL-sym, landing field components, safe step rule
  linear_solvers.py   # Krylov solvers under the extended canonical metric

experiments/
  Procrustes/         # Orthogonal Procrustes experiment
  PCA/                # Principal component analysis experiment
  ICA/                # Real-data independent component analysis experiment
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The ICA experiment uses the MNE-Python sample EEG dataset. Install `mne`
separately if you want to run that experiment:

```bash
pip install mne
```

## Run Experiments

Orthogonal Procrustes:

```bash
python -m experiments.Procrustes.run
```

Principal component analysis:

```bash
python -m experiments.PCA.run
```

Independent component analysis:

```bash
python -m experiments.ICA.run
```

Each script accepts `--quiet` to suppress per-iteration progress output and
`--out` to choose the output directory. The scripts write `summary.json` and
`logs.json` with the metrics listed above, so plotting code can read the full
per-iteration histories directly from `logs.json`.
