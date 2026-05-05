"""ICA problem definition using the notation from the paper.

The experiment solves

    min_{X in St(d,d)} f(X) = -1/N sum_{i=1}^N sum_{j=1}^d
        log(cosh((W X)_{ij})),

where W in R^{N x d} is the whitened EEG data matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mne
except Exception:  # pragma: no cover
    mne = None

Array = np.ndarray


@dataclass(frozen=True)
class ICAProblem:
    """Real-data ICA instance from Section 6.3.

    The objective uses the log-cosh contrast

        f(X) = -1/N sum_{i=1}^N sum_{j=1}^d log(cosh((W X)_{ij})),
        X in St(d, d),

    where ``W`` is the whitened EEG data matrix. Methods return the Euclidean
    gradient and Hessian action used by SOL and SOL-sym.
    """

    W: Array
    metadata: dict[str, Any]

    @property
    def N(self) -> int:
        return int(self.W.shape[0])

    @property
    def d(self) -> int:
        return int(self.W.shape[1])

    def cost(self, X: Array) -> float:
        """Return the objective value f(X)."""
        Z = self.W @ X
        return float(-np.sum(np.log(np.cosh(Z))) / float(self.N))

    def grad(self, X: Array) -> Array:
        """Return the Euclidean gradient ∇f(X)."""
        Z = self.W @ X
        return -(1.0 / float(self.N)) * (
            self.W.T @ np.tanh(Z)
        )

    def hess(self, X: Array, V: Array) -> Array:
        """Return the Euclidean Hessian action ∇²f(X)[V]."""
        Z = self.W @ X
        WV = self.W @ V
        sech2 = 1.0 - np.tanh(Z) ** 2
        return -(1.0 / float(self.N)) * (
            self.W.T @ (sech2 * WV)
        )


def load_real_ica_data(
    *,
    d: int = 60,
    tmin: float = 0.0,
    tmax: float = 60.0,
    resample_sfreq: float = 100.0,
    eeg_only: bool = True,
) -> tuple[Array, dict[str, Any]]:
    """Load and whiten MNE sample EEG data as W in R^{N x d}."""
    if mne is None:
        raise ImportError(
            "mne is required to run the ICA experiment because the original "
            "experiment uses the MNE sample EEG dataset."
        )

    data_path = Path(mne.datasets.sample.data_path(download=True))
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    if eeg_only:
        raw.pick("eeg")
    raw.crop(tmin=tmin, tmax=tmax)
    raw.load_data()
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    raw.resample(resample_sfreq, verbose=False)

    data = raw.get_data().T
    data = data - data.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(data, full_matrices=False)
    d_eff = min(d, Vt.shape[0])
    N = data.shape[0]
    W = np.sqrt(max(N - 1, 1.0)) * U[:, :d_eff]

    metadata: dict[str, Any] = {
        "raw_file": str(raw_fname),
        "requested_d": int(d),
        "d": int(W.shape[1]),
        "N": int(W.shape[0]),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "resample_sfreq": float(resample_sfreq),
        "eeg_only": bool(eeg_only),
    }
    return W.astype(np.float64), metadata


def make_ica_problem(
    *,
    d: int,
    tmin: float,
    tmax: float,
    resample_sfreq: float,
    eeg_only: bool = True,
) -> ICAProblem:
    """Build the real-data ICA problem used in Section 6.3."""
    W, metadata = load_real_ica_data(
        d=d,
        tmin=tmin,
        tmax=tmax,
        resample_sfreq=resample_sfreq,
        eeg_only=eeg_only,
    )
    return ICAProblem(W=W, metadata=metadata)


def random_orthogonal(d: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q.astype(np.float64)


def feasibility_violation(X: Array) -> float:
    """Return the feasibility violation ||X^T X - I||_F."""
    d = X.shape[1]
    return float(np.linalg.norm(X.T @ X - np.eye(d), ord="fro"))


def component_alignment(X: Array, X_ref: Array) -> float:
    """Permutation-invariant basis alignment error between two orthogonal matrices."""
    corr = np.abs(X_ref.T @ X)
    best_match = np.max(corr, axis=0)
    return float(np.linalg.norm(1.0 - best_match))
