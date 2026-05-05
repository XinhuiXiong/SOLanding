"""Second-Order Landing (SOL) method on the Stiefel manifold.

This module implements the **Second-Order Landing (SOL)** algorithm described in the
paper *"A second-order method landing on the Stiefel manifold via Newton–Schulz
iteration"*.

The algorithm targets problems of the form

    min f(X)  s.t.  X^T X = I_p.

SOL uses the decomposition

    Λ(X) = T(X) + N(X),

where:
    - N(X) is the normal feasibility restoration component, chosen as the order-1
      Newton–Schulz update

            N(X) = -1/2 ∇N(X),   ∇N(X) = X(X^T X - I).

    - T(X) is the tangent component obtained by solving

            A_T(X)[T(X)] = -grad f(X) - A_N(X)[N(X)],


Dependencies:
    numpy, scipy (for lgmres, bicgstab + LinearOperator)

Compatibility:
    Works with or without Pymanopt. If you pass a Pymanopt Problem, it will use
    problem.manifold.random_point() for initialization. Otherwise you can pass
    explicit callables for f, ∇f, and ∇^2 f.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from .linear_solvers import g_metric_minres

try:
    from scipy.sparse.linalg import LinearOperator, gmres, bicgstab, lgmres
    from scipy.linalg import cho_factor, cho_solve
except Exception as e:  # pragma: no cover
    raise ImportError(
        "optimizer.optimizer requires scipy."
    ) from e



Array = np.ndarray




def sym(A: Array) -> Array:
    return 0.5 * (A + A.T)

def grad_N(X: Array) -> Array:
    """Penalty gradient ∇N(X) with N(X)=1/4||X^T X - I||_F^2."""
    p = X.shape[1]
    return X @ (X.T @ X - np.eye(p))

def NS_displacement_r(X: Array, r: int) -> Array:
    """
    Order-r truncated Newton–Schulz update:
        N_r(X) = X @ (q_r(E) - I),
    where
        E = X.T @ X - I
    and
        q_r(E) = sum_{j=0}^r (-1)^j * C(2j,j) / 4^j * E^j.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Current iterate.
    r : int
        Truncation order, must satisfy r >= 1.

    Returns
    -------
    np.ndarray, shape (n, p)
        The displacement N_r(X).
    """
    E = X.T @ X - np.eye(X.shape[1], dtype=X.dtype)

    if r == 1:
        P = -0.5 * E
    elif r == 2:
        E2 = E @ E
        P = -0.5 * E + 3.0 / 8.0 * E2
    elif r == 3:
        E2 = E @ E
        E3 = E2 @ E
        P = -0.5 * E + 3.0 / 8.0 * E2 - 5.0 / 16.0 * E3
    else:
        raise ValueError("Only r=1,2,3 are supported in this implementation.")

    return X @ P


def T1(X: Array, grad_f: Callable[[Array], Array]) -> Array:
    """Riemannian gradient of f
    grad f(X) = T1(X) = 2*skew(∇f(X) X^T) X.
    Algebraically: ∇f(X)(X^T X) - X(∇f(X)^T X).
    """
    egrad = grad_f(X)
    XTX = X.T @ X
    XTegrad = X.T @ egrad
    return egrad @ XTX - X @ XTegrad.T



def proj_tangent(X: Array, V: Array, XTX_inv: Optional[Array] = None) -> Array:
    """Project V onto T_X St_{X^T X}(p,n) under the metric g.

    Π_X^T(V) = V - X (X^T X)^{-1} sym(X^T V).
    """
    if XTX_inv is None:
        XTX_inv = np.linalg.inv(X.T @ X)
    return V - X @ (XTX_inv @ sym(X.T @ V))



def AT_action(
    X: Array,
    V: Array,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    XTX: Optional[Array] = None,
    egrad: Optional[Array] = None,
    Xtegrad: Optional[Array] = None,
) -> Array:
    """
    A_T(X)[V] = HV @ (X^T X) - X @ (HV^T X) + ∇f(X) @ (V^T X) - V @ (∇f(X)^T X)
    """
    Xt = X.T

    if XTX is None:
        XTX = Xt @ X
    if egrad is None:
        egrad = grad_f(X)
    if Xtegrad is None:
        Xtegrad = Xt @ egrad  # p×p

    HV = hess_f(X, V)

    # Small matrices (p×p)
    HVtX = HV.T @ X        # p×p
    VtX  = V.T @ X         # p×p

    # Terms: all n×p
    term2 = HV @ XTX - X @ HVtX
    term3 = egrad @ VtX  - V @ Xtegrad.T

    return term2 + term3


# Adjoint of the tractable tangential operator with respect to Euclidean inner product, prepared for BiCGSTAB
def AT_adjoint_action(
    X, W, grad_f, hess_f, *, XTX=None, egrad=None, Xtegrad=None
):
    """
    A_T(X)^*[W] = H(W XTX) - H(X (W^T X)) + X (W^T ∇f(X)) - W (X^T ∇f(X))
    """
    if XTX is None:
        XTX = X.T @ X
    if egrad is None:
        egrad = grad_f(X)
    if Xtegrad is None:
        Xtegrad = X.T @ egrad  # p×p

    # small p×p
    WtX = W.T @ X
    Wtegrad = W.T @ egrad

    D = W @ XTX - X @ WtX            # n×p
    HD = hess_f(X, D)

    term_g1 = X @ Wtegrad                # n×p
    term_g2 = W @ Xtegrad                # n×p

    return HD + (term_g1 - term_g2)


def AN_action(
    X: Array,
    V_normal: Array,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    XTX: Optional[Array] = None,
    egrad: Optional[Array] = None,
    Xtegrad: Optional[Array] = None,
) -> Array:
    """Normal-to-tangent operator A_N(X)[V].

    A_N(X)[V] = 2*skew(∇^2 f(X)[V] X^T + ∇f(X) V^T) X.

    Same algebra as AT_action, but V is the normal displacement.
    """
    if XTX is None:
        XTX = X.T @ X
    if egrad is None:
        egrad = grad_f(X)
    if Xtegrad is None:
        Xtegrad = X.T @ egrad  # p×p

    return AT_action(
        X,
        V_normal,
        grad_f,
        hess_f,
        XTX=XTX,
        egrad=egrad,
        Xtegrad=Xtegrad,
    )



def eta_safe(d: float, g: float, epsilon: float, lam: float = 0.5) -> float:
    """Safeguard stepsize η_safe from Lemma 5.1 (paper).

    Inputs:
        d = ||X^T X - I||_F
        g = ||Λ(X)||_F
        epsilon: the parameter of the safe region (paper's Definition 2.1)
        lam: fixed parameter λ
    """
    if g <= 0:
        return 1.0

    # the paper's Lemma 5.1 (Eq. (5.2)):
    #
    #   η_safe(X) = min{ ( -1/2 d(1-d) + sqrt( 1/4 d^2 (1-d)^2 + g^2 (ε-d) ) ) / g^2 , 1 / (2 * λ) }.
    #
    # This has the correct limiting behavior at d -> 0:
    #     η_safe -> sqrt(ε)/g.
    denom = g * g

    one_minus_d = 1.0 - d
    if one_minus_d <= 0:
        return 0.0

    b = 0.5 * d * one_minus_d
    inside = (b * b) + denom * (epsilon - d)
    if inside <= 0:
        return 0.0

    eta = (-b + np.sqrt(inside)) / denom

    if lam > 0.0:
        eta_safe_size = float(min(max(eta, 0.0), 1.0 / (2 * lam )))
    else:
        eta_safe_size = float(max(eta, 0.0))
    return eta_safe_size


# --------- First-order landing optimizer and result ----------


@dataclass
class Landing1Result:
    """Return object for the first-order landing warm-start helper.

    The public experiments use this helper only to enter the local regime described
    in Section 6 of the paper. It is not reported as a compared method.
    """

    X: Array
    iterations: int
    stopping_reason: str
    log: Dict[str, list]


class FirstOrderLanding:
    """First-order landing helper used to generate warm-start points.

    This implements the first-order landing field Λ1 from equation (2.1). The
    experiment scripts use it only to construct an initial point in the local
    regime before running SOL or SOL-sym.

    The stopping measure is the paper's first-order optimality violation,

        ||grad f(X)||_F + ||X^T X - I||_F <= tol,

    where ``grad f(X) = T1(X)`` is the Riemannian gradient on the layered
    manifold.

    Update:
        X_{k+1} = X_k - η_k Λ1(X_k),
    with a safeguard step size to stay inside the safe region
        St(p,n)_ε = { X : ||X^T X - I||_F <= ε }.

    Parameters:
        epsilon: safe region radius ε.
        lam: hyperparameter λ.
        eta: nominal step size η.
        tol: stopping tolerance on ||Λ1(X)||_F.
        max_iter: maximum iterations.
        verbosity: 0 silent; >=2 prints per-iteration info.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        lam: float = 0.5,
        eta: float = 0.1,
        tol: float = 1e-3,
        max_iter: int = 2000,
        verbosity: int = 0,
    ):
        if not (0 < epsilon <= 1.0):
            raise ValueError("epsilon is too big.")
        if lam <= 0:
            raise ValueError("lam must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        self.epsilon = float(epsilon)
        self.lam = float(lam)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbosity = int(verbosity)

    def run(
        self,
        *,
        n: int,
        p: int,
        grad_f: Callable[[Array], Array],
        cost: Optional[Callable[[Array], float]] = None,
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> Landing1Result:
        """Run the warm-start phase.

        Parameters
        ----------
        n, p:
            Matrix dimensions for X in R^{n x p}.
        grad_f:
            Euclidean gradient ∇f(X) of the objective.
        cost:
            Optional objective evaluator used only for logging.
        X0:
            Optional initial matrix. If omitted, a random point is generated.
        manifold:
            Optional object exposing ``random_point()`` for initialization.
        seed:
            Random seed used when ``X0`` is omitted.
        callback:
            Optional hook ``callback(X, k)``. Returning True or ``"stop"`` stops
            the iteration.

        Returns
        -------
        Landing1Result
            Final iterate, iteration count, stopping reason, and a log containing
            ``objective``, ``Rgrad_norm``, ``feasibility_violation``, ``step_size``,
            and cumulative ``time``.
        """
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],  # cumulative seconds since start
            "objective": [],
            "feasibility_violation": [],  # ||X^T X - I||_F
            "Rgrad_norm": [],  # ||grad f(X)||_F = ||T1(X)||_F
            "step_size": [],
        }

        t_start = time.perf_counter()
        stopping_reason = "max_iter"


        for k in range(1, self.max_iter + 1):
            XTX = X.T @ X
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))

            lam1 = t1 + self.lam * grad_N(X)
            lam1_norm = float(np.linalg.norm(lam1, ord="fro"))

            f_val = float(cost(X)) if cost is not None else float("nan")
            elapsed = float(time.perf_counter() - t_start)

            log["time"].append(elapsed)
            log["objective"].append(f_val)
            log["feasibility_violation"].append(ortho)
            log["Rgrad_norm"].append(t1_norm)

            if self.verbosity >= 2:
                print(
                    f"[warm-up {k:4d}] t={elapsed:8.3f}s  objective={f_val:.6e}  "
                    f"||grad f||={t1_norm:.3e}  "
                    f"||X^T X - I||={ortho:.3e}  "
                    f"||Λ1||={lam1_norm:.3e}  (λ={self.lam:.2g})"
                )

            # Per-iteration callback hook (only expose current iterate X)
            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(f"Callback raised an exception at iteration {k}: {ex}") from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if  t1_norm + ortho <= self.tol:
                stopping_reason = "tol"
                break

            
            step = min(eta_safe(ortho, lam1_norm, self.epsilon, self.lam), self.eta)
            log["step_size"].append(float(step))
            X = X - step * lam1

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return Landing1Result(
            X=X,
            iterations=len(log["objective"]),
            stopping_reason=stopping_reason,
            log=log,
        )

@dataclass
class SOLResult:
    """Result returned by SOL and SOL-sym.

    ``log`` uses the paper's metric names: ``objective`` stores f(X),
    ``feasibility_violation`` is ``||X^T X - I||_F``, and ``Rgrad_norm`` is
    ``||grad f(X)||_F`` with ``grad f(X) = T1(X)`` as in equation (2.5).
    """

    X: Array
    iterations: int
    stopping_reason: str
    log: Dict[str, list]


class SecondOrderLanding:
    """Second-Order Landing (SOL) solver from Algorithm 5.1.

    SOL computes the normal component ``N(X)`` by the order-1 Newton-Schulz
    update (equation (3.5)) and computes the tangent component ``T(X)`` by the
    projection-free approximate Newton equation (4.11),

        A_T(X)[T(X)] = -grad f(X) - A_N(X)[N(X)].

    The update is ``X <- X + eta * (T(X) + N(X))`` when the unit trial point stays
    in the safe region; otherwise the safeguard from Lemma 5.1 is used.

    Parameters
    ----------
    epsilon:
        Safe-region radius in Definition 2.1. Iterates are kept in
        ``St(p,n)^epsilon = {X : ||X^T X - I||_F <= epsilon}`` by first trying
        the nominal step and then applying the safeguard from Lemma 5.1 if the
        trial point leaves the safe region. This implementation requires
        ``0 < epsilon < 1.0``.
    eta:
        Nominal step size for the second-order landing update
        ``X <- X + eta * Lambda(X)``. The paper's local method uses the unit
        step near the solution; setting ``eta=1`` follows that convention. If
        the trial point violates the safe region, the actual step size is
        replaced by the safeguarded value.
    tol:
        Stopping tolerance for the paper's first-order optimality violation,
        ``||grad f(X)||_F + ||X^T X - I||_F``. Here ``grad f(X)`` is the
        Riemannian gradient on the layered manifold, implemented as ``T1(X)``.
    max_iter:
        Maximum number of outer SOL iterations.
    linear_solver:
        Krylov solver for the non-symmetric approximate Newton equation (4.11).
        Supported values are ``"lgmres"``, ``"gmres"``, and ``"bicgstab"``.
        The default value is ``"bicgstab"``.
    linear_rtol:
        Relative tolerance passed to the Krylov solver for the tangent linear
        system. If a number is supplied, it is used directly.
    linear_atol:
        Absolute tolerance passed to the Krylov solver. ``None`` is treated as
        ``0.0``.
    proj_to_tangent_space:
        If true, projects the right-hand side
        ``-grad f(X) - A_N(X)[N(X)]`` onto the tangent space before solving
        (4.11). This is a numerical cleanup option used by some experiments; it
        does not change the definition of the SOL tangent equation.
    theta:
        Exponent ``theta`` in the adaptive forcing rule
        ``rtol = min(zeta_max, ||b(X)||_F ** theta)``. This only matters when
        the implementation is allowed to use that adaptive rule.
    zeta_max:
        Upper bound ``zeta_max`` for the adaptive forcing tolerance in the
        inexact tangent solve. It must lie in ``(0, 1)``.
    linear_maxiter:
        Default maximum number of Krylov iterations for the tangent solve,
        unless ``linear_solver_options["maxiter"]`` is provided.
    verbosity:
        Progress output level. ``0`` is silent, ``1`` prints a final message,
        and ``2`` prints per-iteration objective, Riemannian gradient norm, and
        feasibility violation.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        eta: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 200,
        linear_solver: str = "bicgstab",
        linear_rtol: Optional[float] = None,
        linear_atol: Optional[float] = None,
        proj_to_tangent_space: Optional[bool] = False,
        theta: float = 1.0,
        zeta_max: float = 1e-2,
        linear_maxiter: int = 200,
        verbosity: int = 1,
    ):
        if not (0 < epsilon < 1.0):
            raise ValueError("epsilon is too big.")
        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.linear_solver = str(linear_solver).lower()
        self.linear_rtol = linear_rtol
        self.linear_atol = 0.0 if linear_atol is None else float(linear_atol)
        self.proj_to_tangent_space = proj_to_tangent_space
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if not (0.0 < zeta_max < 1.0):
            raise ValueError("zeta_max must be in (0, 1).")
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.linear_maxiter = int(linear_maxiter)
        self.verbosity = int(verbosity)

    def _solve_linear_system(self, Aop, b_flat, rtol: float, atol: float):
        maxiter = self.linear_maxiter

        solver = self.linear_solver

        if solver == "lgmres":
            x, info = lgmres(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter)

        elif solver == "gmres":
            x, info = gmres(
                Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter
            )

        elif solver == "bicgstab":
            x, info = bicgstab(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter)

        else:
            raise ValueError(f"Unknown linear_solver: {solver}")

        return x, info


    def run(
        self,
        *,
        n: int,
        p: int,
        cost: Callable[[Array], float],
        grad_f: Callable[[Array], Array],
        hess_f: Callable[[Array, Array], Array],
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        NS_order: int = 1,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> SOLResult:
        """Run SOL on a Stiefel-constrained problem.

        Parameters
        ----------
        n, p:
            Matrix dimensions for X in R^{n x p}.
        cost, grad_f, hess_f:
            Objective f, Euclidean gradient ∇f, and Euclidean Hessian action
            ∇²f(X)[V].
        X0:
            Optional initial point in the safe region.
        manifold:
            Optional object exposing ``random_point()`` for initialization.
        NS_order:
            Newton-Schulz truncation order. The paper uses order 1 for SOL.
        seed:
            Random seed used when ``X0`` is omitted.
        callback:
            Optional hook ``callback(X, k)``. Returning True or ``"stop"`` stops
            the iteration.

        Returns
        -------
        SOLResult
            Final iterate, stopping information, and per-iteration logs for the
            objective, feasibility violation, Riemannian gradient norm, step size,
            Krylov solver information, and forcing tolerance ``rtol``.
        """
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],
            "objective": [],
            "feasibility_violation": [],
            "Rgrad_norm": [],  # ||grad f(X)||_F = ||T1(X)||_F
            "step_size": [],
            "solver_flag": [],
            "rtol": [],
        }
        t_start = time.perf_counter()

        stopping_reason = "max_iter"


        for k in range(1, self.max_iter + 1):
            XTX = X.T @ X
            egrad = grad_f(X)
            Xtegrad = X.T @ egrad

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))


            log["objective"].append(float(cost(X)))
            log["feasibility_violation"].append(ortho)
            log["Rgrad_norm"].append(t1_norm)

            elapsed = float(time.perf_counter() - t_start)
            log["time"].append(elapsed)

            if self.verbosity >= 2:
                print(
                    f"[SOL {k:4d}] t={elapsed:8.3f}s  objective={log['objective'][-1]:.6e}  "
                    f"||grad f||={t1_norm:.3e}  "
                    f"||X^T X - I||={ortho:.3e}"
                )

            # Per-iteration callback hook (only expose current iterate X)
            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(f"Callback raised an exception at iteration {k}: {ex}") from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if t1_norm + ortho <= self.tol:
                stopping_reason = "tol"
                break

            # Normal step (order-r Newton–Schulz displacement)
            N = NS_displacement_r(X, NS_order)

            # Right-hand side b(X) = -T1(X) - A_N(X)[N(X)]
            b = -t1 - AN_action(
                X,
                N,
                grad_f,
                hess_f,
                XTX=XTX,
                egrad=egrad,
                Xtegrad=Xtegrad,
            )

            if self.proj_to_tangent_space:
                chol_G = cho_factor(XTX, overwrite_a=False, check_finite=False)
                Qx = cho_solve(chol_G, np.eye(p))
                XQx = X @ Qx
                b = proj_tangent_cached(X, b, Qx=Qx, XQx=XQx)

            b_flat = b.ravel()
            dim = n * p
            def matvec(v_flat: Array) -> Array:
                V = v_flat.reshape(n, p)
                AV = AT_action(
                    X,
                    V,
                    grad_f,
                    hess_f,
                    XTX=XTX,
                    egrad=egrad,
                    Xtegrad=Xtegrad,
                )
                return AV.ravel()

            def rmatvec(v_flat: Array) -> Array:
                V = v_flat.reshape(n, p)
                AstarV = AT_adjoint_action(
                    X,
                    V,
                    grad_f,
                    hess_f,
                    XTX=XTX,
                    egrad=egrad,
                    Xtegrad=Xtegrad,
                )
                return AstarV.ravel()

            Aop = LinearOperator((dim, dim), matvec=matvec, rmatvec=rmatvec, dtype=float)

            if self.linear_rtol is None:
                b_norm = float(np.linalg.norm(b_flat))
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)
            log["rtol"].append(float(rtol))


            T_flat, info = self._solve_linear_system(Aop, b_flat, rtol, atol=self.linear_atol)
            log["solver_flag"].append(info)


            T = T_flat.reshape(n, p)

            # SOL update Λ(X) = T(X) + N(X)
            Lambda = T + N

            # Try full step first
            X_trial = X + self.eta * Lambda
            ortho_trial = float(np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro"))
            if ortho_trial <= self.epsilon:
                step = self.eta
                X = X_trial
            else:
                # Safeguard
                g_norm = float(np.linalg.norm(Lambda, ord="fro"))
                step = eta_safe(ortho, g_norm, self.epsilon)
                X = X + step * Lambda

            log["step_size"].append(float(step))

        if self.verbosity >= 1:
            print(
                f"SOL finished: {stopping_reason} after {len(log['objective'])} iterations."
            )

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return SOLResult(X=X, iterations=len(log["objective"]), stopping_reason=stopping_reason, log=log)


# --------- Symmetric SOL helpers for Section 4.2 ----------


def proj_tangent_cached(
    X: Array,
    V: Array,
    *,
    Qx: Array,
    XQx: Optional[Array] = None,
) -> Array:
    """Project V onto T_X St_{X^T X}(p,n) using cached (X^T X)^{-1}."""
    if XQx is None:
        XQx = X @ Qx
    return V - XQx @ sym(X.T @ V)



def exact_tangent_hessian_action_cached(
    X: Array,
    V: Array,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    XTX: Array,
    Qx: Array,
    XQx: Array,
    egrad: Array,
    Xtegrad: Array,
    project_input: bool = False,
) -> Array:
    r"""Exact Riemannian Hessian action from the paper's Proposition 4.1.

    For tangent ``V``,

        Hess f(X)[V]
          = Π_X^T(
                2*skew(∇²f(X)[V] X^T) X
              + 2*skew(∇f(X) V^T) X
              + 2*skew(∇f(X) X^T) V
              - (I + P_X) Xi_X(V)
            ),

    where ``Qx = (X^T X)^{-1}``, ``P_X = X Qx X^T``, and ``Xi_X(V)`` is
    the Levi-Civita connection correction term from Proposition 4.1.
    ``egrad`` denotes the ambient Euclidean gradient ``∇f(X)`` and
    ``Xtegrad`` caches ``X^T ∇f(X)``.
    """
    if project_input:
        V = proj_tangent_cached(X, V, Qx=Qx, XQx=XQx)

    HV = hess_f(X, V)
    XtV = X.T @ V
    VtX = XtV.T
    egradtV = egrad.T @ V
    HVtX = HV.T @ X

    # 2*skew(HV X^T)X + 2*skew(egrad V^T)X + 2*skew(egrad X^T)V.
    # Group the two gradient terms so that egrad @ (XtV + VtX) vanishes
    # automatically for tangent V (XtV skew-symmetric).
    hv_term = HV @ XTX - X @ HVtX
    grad_terms = -V @ Xtegrad.T - X @ egradtV

    # Riemannian gradient grad f(X) = 2*skew(∇f(X) X^T)X.
    # Compute all p×p ingredients from cached small matrices, avoiding an
    # explicit construction of rgrad = egrad @ XTX - X @ Xtegrad.T.
    Xt_rgrad = Xtegrad @ XTX - XTX @ Xtegrad.T
    VQXt_rgrad = V @ (Qx @ Xt_rgrad)
    rgradQXtV = egrad @ XtV - X @ (Xtegrad.T @ (Qx @ XtV))
    Vt_rgrad = egradtV.T @ XTX - VtX @ Xtegrad.T
    xi = (
        0.5 * (VQXt_rgrad + rgradQXtV)
        + 0.25 * XQx @ (Vt_rgrad + Vt_rgrad.T)
    )
    connection = xi + XQx @ (X.T @ xi)

    ambient_hess = hv_term + grad_terms - connection
    return proj_tangent_cached(X, ambient_hess, Qx=Qx, XQx=XQx)


class SecondOrderLandingSymmetric:
    """SOL-sym solver from Algorithm 5.1.

    SOL-sym uses the same normal component as SOL but computes the tangent
    component by solving the modified Newton equation (4.9) with the full
    Riemannian Hessian under the metric ``g``. The suffix ``sym`` follows the
    paper: the complete Hessian operator is symmetric with respect to ``g``.

    Parameters are aligned with :class:`SecondOrderLanding`. The linear solve
    uses ``linear_rtol`` if provided; otherwise it uses the adaptive forcing
    tolerance from equation (5.11), controlled by ``theta`` and ``zeta_max``.
    The selected tolerance is logged as ``rtol``.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        eta: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 200,
        linear_rtol: Optional[float] = None,
        linear_atol: Optional[float] = 0.0,
        theta: float = 1.0,
        zeta_max: float = 1e-1,
        linear_maxiter: int = 200,
        verbosity: int = 1,
    ):
        if not (0 < epsilon < 1.0):
            raise ValueError("epsilon is too big.")
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if not (0.0 < zeta_max < 1.0):
            raise ValueError("zeta_max must be in (0, 1).")

        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.linear_rtol = linear_rtol
        self.linear_atol = 0.0 if linear_atol is None else float(linear_atol)
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.linear_maxiter = int(linear_maxiter)
        self.verbosity = int(verbosity)

    def run(
        self,
        *,
        n: int,
        p: int,
        cost: Callable[[Array], float],
        grad_f: Callable[[Array], Array],
        hess_f: Callable[[Array, Array], Array],
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        NS_order: int = 1,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> SOLResult:
        """Run SOL-sym on a Stiefel-constrained problem.

        Parameters are the same as :meth:`SecondOrderLanding.run`, except that the
        tangent linear system is solved by ``g_metric_minres`` under the extended
        canonical metric. The returned log includes ``rtol`` for the inexact solve,
        matching the enforcing condition (5.11).
        """
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],
            "objective": [],
            "feasibility_violation": [],
            "Rgrad_norm": [],
            "step_size": [],
            "solver_info": [],
            "rtol": [],
        }
        t_start = time.perf_counter()
        stopping_reason = "max_iter"

        for k in range(1, self.max_iter + 1):
            XTX = X.T @ X
            chol_G = cho_factor(XTX, overwrite_a=False, check_finite=False)
            Qx = cho_solve(chol_G, np.eye(p))
            XQx = X @ Qx
            egrad = grad_f(X)
            Xtegrad = X.T @ egrad

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))

            elapsed = float(time.perf_counter() - t_start)
            log["time"].append(elapsed)
            log["objective"].append(float(cost(X)))
            log["feasibility_violation"].append(ortho)
            log["Rgrad_norm"].append(t1_norm)

            if self.verbosity >= 2:
                print(
                    f"[SOL-sym {k:4d}] t={elapsed:8.3f}s  objective={log['objective'][-1]:.6e}  "
                    f"||grad f||={t1_norm:.3e}  "
                    f"||X^T X - I||={ortho:.3e}"
                )

            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(
                        f"Callback raised an exception at iteration {k}: {ex}"
                    ) from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if t1_norm + ortho <= self.tol:
                stopping_reason = "tol"
                break

            # normal step
            N = NS_displacement_r(X, NS_order)


            b = -t1 - AN_action(
                X,
                N,
                grad_f,
                hess_f,
                XTX=XTX,
                egrad=egrad,
                Xtegrad=Xtegrad,
            )
            b = proj_tangent_cached(X, b, Qx=Qx, XQx=XQx)

            def Hessop(V: Array) -> Array:
                HessV = exact_tangent_hessian_action_cached(
                    X,
                    V,
                    grad_f,
                    hess_f,
                    XTX=XTX,
                    Qx=Qx,
                    XQx=XQx,
                    egrad=egrad,
                    Xtegrad=Xtegrad,
                )
                return HessV

            b_norm = float(np.linalg.norm(b, ord="fro"))

            if self.linear_rtol is None:
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)

            log["rtol"].append(rtol)
            T, info = g_metric_minres(
                X,
                Hessop,
                b,
                rtol=rtol,
                atol=self.linear_atol,
                maxiter=self.linear_maxiter,
            )
            log["solver_info"].append(info)

            Lambda = T + N

            X_trial = X + self.eta * Lambda
            ortho_trial = float(
                np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro")
            )
            if ortho_trial <= self.epsilon:
                step = self.eta
                X = X_trial
            else:
                g_norm = float(np.linalg.norm(Lambda, ord="fro"))
                step = eta_safe(ortho, g_norm, self.epsilon)
                X = X + step * Lambda

            log["step_size"].append(float(step))

        if self.verbosity >= 1:
            print(
                f"SOL-sym finished: {stopping_reason} after {len(log['objective'])} iterations."
            )

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return SOLResult(
            X=X,
            iterations=len(log["objective"]),
            stopping_reason=stopping_reason,
            log=log,
        )
