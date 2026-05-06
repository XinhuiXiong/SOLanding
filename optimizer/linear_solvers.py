"""Linear solvers for SOL-sym under the extended canonical metric.

This module implements matrix-variable Krylov solvers for linear systems

    A[T] = b,

where the unknown ``T`` and the right-hand side ``b`` are matrices in the
tangent space of a layered Stiefel manifold, and all inner products are taken
with respect to the metric

    g_X(U, V) = <U, (I - 0.5 * X (X^T X)^{-1} X^T) V (X^T X)^{-1}>_F.

The main entry points are:

    - g_metric_minres(...)
    - g_metric_cg(...)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np


Array = np.ndarray
LinearMatrixOperator = Callable[[Array], Array]
Callback = Callable[[int, Array, Array, float, float], None]
MinresCallback = Callable[[Array], None]


def _prepare_metric_cache(
    X: Array,
    XTX: Optional[Array] = None,
    invXTX: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Return cached Gram matrix and its inverse for the metric at ``X``."""
    if XTX is None:
        XTX = X.T @ X
    if invXTX is None:
        invXTX = np.linalg.inv(XTX)
    return XTX, invXTX


def metric_map(
    X: Array,
    V: Array,
    *,
    XTX: Optional[Array] = None,
    invXTX: Optional[Array] = None,
) -> Array:
    """Apply the metric map M_X to a matrix V.

    This implements

        M_X[V] = (I - 0.5 * X (X^T X)^{-1} X^T) V (X^T X)^{-1}

    using only matrix multiplications. No n x n identity matrix is formed.
    """
    _, invXTX = _prepare_metric_cache(X, XTX=XTX, invXTX=invXTX)

    VGinv = V @ invXTX
    Xt_VGinv = X.T @ VGinv
    correction = 0.5 * X @ (invXTX @ Xt_VGinv)
    return VGinv - correction


def g_inner(
    X: Array,
    U: Array,
    V: Array,
    *,
    XTX: Optional[Array] = None,
    invXTX: Optional[Array] = None,
) -> float:
    """Inner product g_X(U, V).

    The formula is

        g_X(U, V) = <U, M_X[V]>_F,

    where M_X is the metric map defined above and <.,.>_F is the Frobenius
    inner product.
    """
    MV = metric_map(X, V, XTX=XTX, invXTX=invXTX)
    return float(np.sum(U * MV))


def g_norm(
    X: Array,
    U: Array,
    *,
    XTX: Optional[Array] = None,
    invXTX: Optional[Array] = None,
) -> float:
    """Return sqrt(g_X(U, U))."""
    value = g_inner(X, U, U, XTX=XTX, invXTX=invXTX)
    return float(np.sqrt(max(value, 0.0)))


def euc_inner(U: Array, V: Array) -> float:
    """Euclidean/Frobenius inner product."""
    return float(np.sum(U * V))


def euc_norm(U: Array) -> float:
    """Euclidean/Frobenius norm."""
    return float(np.linalg.norm(U, ord="fro"))


def _frobenius_zero_like(reference: Array) -> Array:
    return np.zeros_like(reference, dtype=float)


def _default_max_iter(reference: Array, maxiter: Optional[int]) -> int:
    """Return a practical default iteration limit."""
    if maxiter is None:
        return int(reference.size)
    if maxiter <= 0:
        raise ValueError("maxiter must be positive or None.")
    return int(maxiter)


def _build_solver_info(
    *,
    b_norm_g: float,
    initial_residual_norm_g: float,
    residual_norms_g: List[float],
    stopping_threshold: float,
    atol: float,
    rtol: float,
    converged: bool,
    num_iter: int,
    stop_reason: str,
    breakdown_kind: Optional[str] = None,
) -> Dict[str, object]:
    """Assemble a consistent info dictionary for all solvers."""
    info: Dict[str, object] = {
        "b_norm_g": b_norm_g,
        "initial_residual_norm_g": initial_residual_norm_g,
        "final_residual_norm_g": residual_norms_g[-1],
        "residual_norms_g": residual_norms_g,
        "stopping_threshold": stopping_threshold,
        "atol": float(atol),
        "rtol": float(rtol),
        "converged": bool(converged),
        "num_iter": int(num_iter),
        "stop_reason": stop_reason,
    }
    if breakdown_kind is not None:
        info["breakdown_kind"] = breakdown_kind
    return info


def _sym_ortho(a: float, b: float) -> Tuple[float, float, float]:
    """Stable symmetric Givens rotation.

    Returns c, s, r such that

        [c  s] [a] = [r]
        [s -c] [b]   [0]

    up to the sign convention used in MINRES.
    """
    if b == 0.0:
        c = 1.0 if a == 0.0 else float(np.sign(a))
        s = 0.0
        r = abs(a)
    elif a == 0.0:
        c = 0.0
        s = float(np.sign(b))
        r = abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = float(np.sign(b)) / np.sqrt(1.0 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = float(np.sign(a)) / np.sqrt(1.0 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, abs(r)


def _check_g_self_adjointness(
    A_action: LinearMatrixOperator,
    X: Array,
    *,
    XTX: Array,
    invXTX: Array,
    num_tests: int = 2,
    seed: int = 0,
) -> float:
    """Return a lightweight relative symmetry check error for A under g."""
    rng = np.random.default_rng(seed)
    max_rel_err = 0.0
    for _ in range(num_tests):
        U = rng.standard_normal(X.shape)
        V = rng.standard_normal(X.shape)
        lhs = g_inner(X, U, A_action(V), XTX=XTX, invXTX=invXTX)
        rhs = g_inner(X, A_action(U), V, XTX=XTX, invXTX=invXTX)
        scale = max(1.0, abs(lhs), abs(rhs))
        rel_err = abs(lhs - rhs) / scale
        max_rel_err = max(max_rel_err, rel_err)
    return max_rel_err


def g_metric_cg(
    X: Array,
    operator: LinearMatrixOperator,
    b: Array,
    *,
    x0: Optional[Array] = None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    callback: Optional[Callback] = None,
    verbose: bool = False,
    XTX: Optional[Array] = None,
    invXTX: Optional[Array] = None,
) -> Tuple[Array, Dict[str, object]]:
    """Conjugate Gradient under the metric g.

    This solver is intended for linear operators that are:

    - g-self-adjoint, and
    - g-positive definite

    on the tangent space at ``X``.
    """
    if rtol < 0.0:
        raise ValueError("rtol must be nonnegative.")
    if atol < 0.0:
        raise ValueError("atol must be nonnegative.")

    XTX, invXTX = _prepare_metric_cache(X, XTX=XTX, invXTX=invXTX)
    maxiter = _default_max_iter(b, maxiter)

    if x0 is None:
        x = _frobenius_zero_like(b)
    else:
        x = np.array(x0, dtype=float, copy=True)

    b_norm_g = g_norm(X, b, XTX=XTX, invXTX=invXTX)
    stopping_threshold = max(rtol * b_norm_g, atol)

    r = b - operator(x)
    residual_norm_g = g_norm(X, r, XTX=XTX, invXTX=invXTX)
    initial_residual_norm_g = residual_norm_g
    residual_norms_g = [residual_norm_g]

    converged = residual_norm_g <= stopping_threshold
    if converged:
        info = _build_solver_info(
            b_norm_g=b_norm_g,
            initial_residual_norm_g=initial_residual_norm_g,
            residual_norms_g=residual_norms_g,
            stopping_threshold=stopping_threshold,
            atol=atol,
            rtol=rtol,
            converged=True,
            num_iter=0,
            stop_reason="converged_atol_rtol",
        )
        return x, info

    p = np.array(r, copy=True)
    rr_old = g_inner(X, r, r, XTX=XTX, invXTX=invXTX)
    stop_reason = "maxiter"

    for iteration in range(1, maxiter + 1):
        Ap = operator(p)
        pAp = g_inner(X, p, Ap, XTX=XTX, invXTX=invXTX)
        if pAp <= 0.0:
            stop_reason = "negative_curvature"
            info = _build_solver_info(
                b_norm_g=b_norm_g,
                initial_residual_norm_g=initial_residual_norm_g,
                residual_norms_g=residual_norms_g,
                stopping_threshold=stopping_threshold,
                atol=atol,
                rtol=rtol,
                converged=False,
                num_iter=iteration - 1,
                stop_reason=stop_reason,
            )
            return x, info

        alpha = rr_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        residual_norm_g = g_norm(X, r, XTX=XTX, invXTX=invXTX)
        residual_norms_g.append(residual_norm_g)

        if verbose:
            print(
                f"[g-CG {iteration:4d}] "
                f"||r||_g={residual_norm_g:.3e}  "
                f"threshold={stopping_threshold:.3e}"
            )

        if callback is not None:
            callback(iteration, x, r, residual_norm_g, stopping_threshold)

        if residual_norm_g <= stopping_threshold:
            converged = True
            stop_reason = "converged_atol_rtol"
            break

        rr_new = g_inner(X, r, r, XTX=XTX, invXTX=invXTX)
        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new
    else:
        converged = False

    info = _build_solver_info(
        b_norm_g=b_norm_g,
        initial_residual_norm_g=initial_residual_norm_g,
        residual_norms_g=residual_norms_g,
        stopping_threshold=stopping_threshold,
        atol=atol,
        rtol=rtol,
        converged=converged,
        num_iter=len(residual_norms_g) - 1,
        stop_reason=stop_reason,
    )
    return x, info


def g_metric_minres(
    X: Array,
    A_action: LinearMatrixOperator,
    b: Array,
    *,
    x0: Optional[Array] = None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    check: bool = False,
    callback: Optional[MinresCallback] = None,
    verbose: bool = False,
    XTX: Optional[Array] = None,
    invXTX: Optional[Array] = None,
) -> Tuple[Array, Dict[str, object]]:
    """MINRES under the metric g.

    This solver targets g-self-adjoint operators that may be indefinite.

    The Lanczos process is carried out in a g-orthonormal basis, so if

        A V_k = V_{k+1} Tbar_k,

    then for any small least-squares coefficient vector y_k,

        r_k = b - A x_k = V_{k+1} (beta_1 e_1 - Tbar_k y_k).

    Because the columns of V_{k+1} are g-orthonormal, the g-norm of the
    residual equals the Euclidean 2-norm of the small least-squares residual:

        ||r_k||_g = ||beta_1 e_1 - Tbar_k y_k||_2.

    MINRES exploits this identity and uses Givens rotations to update the small
    residual norm estimate by short recurrences, without forming Tbar_k
    explicitly and without recomputing the true residual each iteration.
    """
    if rtol < 0.0:
        raise ValueError("rtol must be nonnegative.")
    if atol < 0.0:
        raise ValueError("atol must be nonnegative.")

    XTX, invXTX = _prepare_metric_cache(X, XTX=XTX, invXTX=invXTX)
    maxiter = _default_max_iter(b, maxiter)

    symmetry_check_relerr: Optional[float] = None
    if check:
        symmetry_check_relerr = _check_g_self_adjointness(
            A_action,
            X,
            XTX=XTX,
            invXTX=invXTX,
        )
        if symmetry_check_relerr > 1e-8:
            warnings.warn(
                "A_action may not be g-self-adjoint: "
                f"relative check error {symmetry_check_relerr:.3e}",
                RuntimeWarning,
                stacklevel=2,
            )

    if x0 is None:
        x = _frobenius_zero_like(b)
    else:
        x = np.array(x0, dtype=float, copy=True)

    b_norm_g = g_norm(X, b, XTX=XTX, invXTX=invXTX)
    stopping_threshold = max(rtol * b_norm_g, atol)

    r0 = b - A_action(x)
    residual_norm0_g = g_norm(X, r0, XTX=XTX, invXTX=invXTX)
    initial_residual_norm_g = residual_norm0_g
    residual_norms_g = [residual_norm0_g]

    if residual_norm0_g <= stopping_threshold:
        info = _build_solver_info(
            b_norm_g=b_norm_g,
            initial_residual_norm_g=initial_residual_norm_g,
            residual_norms_g=residual_norms_g,
            stopping_threshold=stopping_threshold,
            atol=atol,
            rtol=rtol,
            converged=True,
            num_iter=0,
            stop_reason="converged_atol_rtol",
        )
        if symmetry_check_relerr is not None:
            info["self_adjoint_check_relative_error"] = symmetry_check_relerr
        return x, info

    # Lanczos state:
    # r_curr = beta_k * v_k, r_prev = beta_{k-1} * v_{k-1}.
    beta_curr = residual_norm0_g
    beta_prev = 0.0
    r_prev = _frobenius_zero_like(b)
    r_curr = np.array(r0, copy=True)

    # QR / MINRES short-recurrence state.
    cs_prev = -1.0
    sn_prev = 0.0
    dbar_prev = 0.0
    eps_prev = 0.0
    phibar = beta_curr

    # Short-recurrence basis for solution updates.
    m_prevprev = _frobenius_zero_like(b)
    m_prev = _frobenius_zero_like(b)

    converged = False
    stop_reason = "maxiter"
    breakdown_kind: Optional[str] = None
    breakdown_tol = 10.0 * np.finfo(float).eps

    for iteration in range(1, maxiter + 1):
        if beta_curr <= breakdown_tol:
            stop_reason = "breakdown"
            breakdown_kind = "initial_lanczos_breakdown"
            break

        v_curr = r_curr / beta_curr
        Av = A_action(v_curr)

        if iteration > 1:
            Av = Av - (beta_curr / beta_prev) * r_prev

        alpha_curr = g_inner(X, v_curr, Av, XTX=XTX, invXTX=invXTX)
        Av = Av - (alpha_curr / beta_curr) * r_curr

        r_prev = r_curr
        r_curr = Av
        beta_next = g_norm(X, r_curr, XTX=XTX, invXTX=invXTX)

        old_eps = eps_prev
        delta_curr = cs_prev * dbar_prev + sn_prev * alpha_curr
        gbar_curr = sn_prev * dbar_prev - cs_prev * alpha_curr
        eps_curr = sn_prev * beta_next
        dbar_curr = -cs_prev * beta_next

        # In MINRES, the previous rotation first transforms the tridiagonal
        # Lanczos relation, and the current rotation then acts on [gbar_k, beta_{k+1}].
        # The residual estimate is |phibar| after this second rotation.
        cs_curr, sn_curr, gamma_curr = _sym_ortho(gbar_curr, beta_next)
        gamma_curr = max(gamma_curr, np.finfo(float).eps)

        phi_curr = cs_curr * phibar
        phibar = sn_curr * phibar

        m_curr = (v_curr - old_eps * m_prevprev - delta_curr * m_prev) / gamma_curr
        x = x + phi_curr * m_curr

        residual_estimate_g = abs(phibar)
        residual_norms_g.append(residual_estimate_g)

        if verbose:
            print(
                f"[g-MINRES {iteration:4d}] "
                f"||r||_g(est)={residual_estimate_g:.3e}  "
                f"threshold={stopping_threshold:.3e}"
            )

        if callback is not None:
            callback(x)

        if residual_estimate_g <= stopping_threshold:
            converged = True
            stop_reason = "converged_atol_rtol"
            if beta_next <= breakdown_tol:
                breakdown_kind = "happy_breakdown"
            break

        if not np.isfinite(beta_next):
            stop_reason = "breakdown"
            breakdown_kind = "numerical_breakdown"
            break

        if beta_next <= breakdown_tol:
            stop_reason = "breakdown"
            breakdown_kind = "happy_breakdown_nonconverged"
            break

        m_prevprev = m_prev
        m_prev = m_curr
        beta_prev = beta_curr
        beta_curr = beta_next
        cs_prev = cs_curr
        sn_prev = sn_curr
        dbar_prev = dbar_curr
        eps_prev = eps_curr

    info = _build_solver_info(
        b_norm_g=b_norm_g,
        initial_residual_norm_g=initial_residual_norm_g,
        residual_norms_g=residual_norms_g,
        stopping_threshold=stopping_threshold,
        atol=atol,
        rtol=rtol,
        converged=converged,
        num_iter=len(residual_norms_g) - 1,
        stop_reason=stop_reason,
        breakdown_kind=breakdown_kind,
    )
    if symmetry_check_relerr is not None:
        info["self_adjoint_check_relative_error"] = symmetry_check_relerr
    return x, info




