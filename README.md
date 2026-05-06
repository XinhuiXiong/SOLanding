# SOLanding

This is the code to reproduce the experiments in the following paper:

> *A second-order method landing on the Stiefel manifold via Newton&ndash;Schulz iteration*
>
> Xinhui Xiong, Bin Gao, and P.-A. Absil
>
> <https://arxiv.org/abs/2605.02838>

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
## Dependencies

- Ubuntu 22.04
- Python 3.12.1
- NumPy 2.4.3
- SciPy 1.17.1
- MNE-Python 1.11.0 (required only for ICA data loading and preprocessing)

## Get Started

You can create a conda environment by running the following commands.

```bash
conda create -n SOLanding_env python=3.12.1
pip install numpy==2.4.3 scipy==1.17.1
pip install mne==1.11.0
```

### Running the Experiments

First, ensure your environment is activated:
```bash
conda activate SOLanding_env
```

Then, you can execute the individual experiments by running their respective modules:

#### Orthogonal Procrustes

```bash
python -m experiments.Procrustes.run
```

#### Principal Component Analysis (PCA)

```bash
python -m experiments.PCA.run
```

#### Independent Component Analysis (ICA)

```bash
python -m experiments.ICA.run
```

Each script accepts `--quiet` to suppress per-iteration progress output and
`--out` to choose the output directory.

## Authors

- Xinhui Xiong (AMSS, China)

## Copyright

Copyright (C) 2026, Xinhui Xiong, Bin Gao, P.-A. Absil.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/
