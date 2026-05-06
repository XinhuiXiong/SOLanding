# SOLanding

This repository contains the reference Python implementation for the paper
[A Second-Order Method Landing on the Stiefel Manifold via Newton–Schulz
Iteration](https://arxiv.org/abs/2605.02838) by Xinhui Xiong, Bin Gao, and P.-A. Absil.

## Repository Layout

```text
optimizer/
  optimizer.py        # SOL, SOL-sym, first-order landing, safe step rule,...
  linear_solvers.py   # Krylov solvers under the metric g

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

## Authors

- Xinhui Xiong (AMSS, China)

## Copyright

Copyright (C) 2026, Xinhui Xiong, Bin Gao, P.-A. Absil.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/
