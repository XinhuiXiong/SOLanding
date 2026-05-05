"""PCA problem definition for the SOL vs SOL-sym experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PCAProblem:
    """PCA instance from Section 6.2.

    The objective is

        f(X) = -1/2 tr(X^T C X),  X in St(p, n),

    where ``C = A_data^T A_data / N`` is the sample covariance matrix. Methods
    return the Euclidean gradient and Hessian action used by SOL and SOL-sym.
    """

    A_data: Array
    U_true: Array
    covariance: Array

    @property
    def n(self) -> int:
        return int(self.A_data.shape[1])

    @property
    def p(self) -> int:
        return int(self.U_true.shape[1])

    @property
    def num_samples(self) -> int:
        return int(self.A_data.shape[0])

    def cost(self, X: Array) -> float:
        """Return the objective value f(X)."""
        return float(-0.5 * np.sum(X * (self.covariance @ X)))

    def grad(self, X: Array) -> Array:
        """Return the Euclidean gradient ∇f(X)."""
        return -(self.covariance @ X)

    def hess(self, X: Array, V: Array) -> Array:
        """Return the Euclidean Hessian action ∇²f(X)[V]."""
        _ = X
        return -(self.covariance @ V)



def make_online_pca_data(
    n: int,
    p: int,
    *,
    num_samples: int = 15000,
    sigma: float = 0.1,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Generate synthetic online-PCA samples.

    Samples are generated row-wise as

        a_i ~ N(0, U U^T + sigma I_n),

    where U in R^{n x p} is Haar-distributed on the Stiefel manifold.
    """
    rng = np.random.default_rng(seed)
    U_true, _ = np.linalg.qr(rng.standard_normal((n, p)))
    scores = rng.standard_normal((num_samples, p))
    noise = rng.standard_normal((num_samples, n))
    A_data = scores @ U_true.T + float(np.sqrt(sigma)) * noise
    return A_data.astype(np.float64), U_true.astype(np.float64)


def make_pca_problem(
    *,
    n: int,
    p: int,
    num_samples: int,
    sigma: float,
    seed: int,
) -> PCAProblem:
    """Build the synthetic PCA problem used in Section 6.2."""
    A_data, U_true = make_online_pca_data(
        n,
        p,
        num_samples=num_samples,
        sigma=sigma,
        seed=seed,
    )
    covariance = (A_data.T @ A_data) / float(num_samples)
    return PCAProblem(A_data=A_data, U_true=U_true, covariance=covariance)


def random_stiefel(n: int, p: int, rng: np.random.Generator) -> Array:
    Q, _ = np.linalg.qr(rng.standard_normal((n, p)))
    return Q.astype(np.float64)


def pca_optimum(covariance: Array, p: int) -> tuple[float, Array]:
    """Return the objective optimum and leading eigenspace representative."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    X_star = eigenvectors[:, -p:]
    f_star = float(-0.5 * np.sum(eigenvalues[-p:]))
    return f_star, X_star
