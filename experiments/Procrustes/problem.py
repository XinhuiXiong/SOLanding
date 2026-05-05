"""Orthogonal Procrustes problem using the notation from the paper.

The experiment solves

    min_{X in St(d,d)} 1/(2n) ||A X - B||_F^2,

where A, B in R^{n x d}. Synthetic instances are generated as

    B = A X_true + sigma Xi,

with X_true in St(d,d) and Xi a standard Gaussian noise matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ProcrustesProblem:
    """Orthogonal Procrustes instance from Section 6.1.

    The objective is

        f(X) = 1/(2n) ||A X - B||_F^2,  X in St(d, d),

    with synthetic data ``B = A X_true + sigma Xi``. Methods return the
    Euclidean gradient and Hessian action used by SOL and SOL-sym.
    """

    A: Array
    B: Array
    X_true: Array
    sigma: float

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def d(self) -> int:
        return int(self.A.shape[1])

    def cost(self, X: Array) -> float:
        """Return the objective value f(X)."""
        residual = self.A @ X - self.B
        return float(0.5 * np.sum(residual * residual) / self.n)

    def grad(self, X: Array) -> Array:
        """Return the Euclidean gradient ∇f(X)."""
        return (self.A.T @ (self.A @ X - self.B)) / float(self.n)

    def hess(self, X: Array, V: Array) -> Array:
        """Return the Euclidean Hessian action ∇²f(X)[V]."""
        _ = X
        return (self.A.T @ (self.A @ V)) / float(self.n)


def random_stiefel(d: int, rng: np.random.Generator) -> Array:
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q.astype(np.float64)


def make_procrustes_problem(
    *,
    n: int,
    d: int,
    sigma: float,
    seed: int,
) -> ProcrustesProblem:
    """Generate the synthetic Procrustes instance used in Section 6.1."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d)).astype(np.float64)
    X_true = random_stiefel(d, rng)
    Xi = rng.standard_normal((n, d)).astype(np.float64)
    B = (A @ X_true + sigma * Xi).astype(np.float64)
    return ProcrustesProblem(A=A, B=B, X_true=X_true, sigma=float(sigma))


def procrustes_solution(A: Array, B: Array) -> Array:
    """Return the closed-form Procrustes optimizer X_star."""
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def feasibility_violation(X: Array) -> float:
    """Return the feasibility violation ||X^T X - I||_F."""
    d = X.shape[1]
    return float(np.linalg.norm(X.T @ X - np.eye(d), ord="fro"))


def distance_to_solution(X: Array, X_star: Array) -> float:
    return float(np.linalg.norm(X - X_star, ord="fro"))
