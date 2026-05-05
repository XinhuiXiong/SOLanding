"""Run the fixed PCA experiment from the paper.

Run from the project root with:

    python experiments/PCA/run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer import SecondOrderLanding, SecondOrderLandingSymmetric
from optimizer.optimizer import FirstOrderLanding

from experiments.PCA.problem import (
    make_pca_problem,
    pca_optimum,
)

Array = np.ndarray


@dataclass(frozen=True)
class PCAConfig:
    n: int = 10000
    p: int = 500
    num_samples: int = 30000
    sigma: float = 0.1
    seed: int = 40
    init_seed: int = 0
    warm_tol: float = 1e-2
    tol: float = 1e-12
    theta: float = 0.5
    zeta_max: float = 1e-1
    linear_atol: float = 0.0
    linear_maxiter: int = 200
    sol_max_iter: int = 200
    sol_sym_max_iter: int = 200
    out: str = "results/PCA"
    verbosity: int = 2


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def final_value(values: list[float]) -> float:
    return float(values[-1]) if values else float("nan")


def summarize_method(result: Any, f_star: float) -> dict[str, Any]:
    """Summarize one SOL/SOL-sym run for the PCA experiment."""
    objective = final_value(result.log["objective"])
    return {
        "stopping_reason": result.stopping_reason,
        "iterations": int(result.iterations),
        "objective": objective,
        "gap": float(objective - f_star),
        "feasibility_violation": final_value(result.log["feasibility_violation"]),
        "Rgrad_norm": final_value(result.log["Rgrad_norm"]),
        "time": final_value(result.log["time"]),
    }


def make_initial_point(config: PCAConfig, problem: Any) -> tuple[Array, Any]:
    """Construct the warm-start point used before the SOL and SOL-sym runs."""
    rng = np.random.default_rng(config.init_seed)
    X_init, _ = np.linalg.qr(rng.standard_normal((config.n, config.p)))
    warm = FirstOrderLanding(
        epsilon=0.75,
        lam=5.0,
        eta=0.1,
        tol=config.warm_tol,
        max_iter=10000,
        verbosity=config.verbosity,
    )
    warm_result = warm.run(
        n=config.n,
        p=config.p,
        grad_f=problem.grad,
        cost=problem.cost,
        X0=X_init,
    )
    return warm_result.X, warm_result


def run_experiment(config: PCAConfig = PCAConfig()) -> dict[str, Any]:
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    problem = make_pca_problem(
        n=config.n,
        p=config.p,
        num_samples=config.num_samples,
        sigma=config.sigma,
        seed=config.seed,
    )
    f_star, _X_star = pca_optimum(problem.covariance, config.p)

    if config.verbosity > 0:
        print("\n================== PCA experiment ==================")
        print("problem: min_{X in St(p,n)} -1/N tr(X^T A^T A X)")
        print(f"n:           {config.n}")
        print(f"p:           {config.p}")
        print(f"num_samples: {config.num_samples}")
        print(f"sigma:       {config.sigma}")
        print(f"seed:        {config.seed}")
        print(f"init_seed:   {config.init_seed}")
        print(f"out:         {out}")
        print(f"f_star:      {f_star:.6e}")
        print("====================================================\n")

    X0, warm_result = make_initial_point(config, problem)

    # The paper initializes the infeasible second-order methods near the local
    # regime; this scaling makes the warm-start point slightly infeasible.
    X0 = 1.01 * X0

    sol = SecondOrderLanding(
        epsilon=0.75,
        eta=1.0,
        tol=config.tol,
        max_iter=config.sol_max_iter,
        linear_solver="bicgstab",
        linear_maxiter=config.linear_maxiter,
        linear_atol=config.linear_atol,
        theta=config.theta,
        zeta_max=config.zeta_max,
        proj_to_tangent_space=True,
        verbosity=config.verbosity,
    )
    sol_res = sol.run(
        n=config.n,
        p=config.p,
        cost=problem.cost,
        grad_f=problem.grad,
        hess_f=problem.hess,
        X0=X0,
    )

    sol_sym = SecondOrderLandingSymmetric(
        epsilon=0.75,
        eta=1.0,
        tol=config.tol,
        max_iter=config.sol_sym_max_iter,
        linear_maxiter=config.linear_maxiter,
        linear_atol=config.linear_atol,
        theta=config.theta,
        zeta_max=config.zeta_max,
        verbosity=config.verbosity,
    )
    sol_sym_res = sol_sym.run(
        n=config.n,
        p=config.p,
        cost=problem.cost,
        grad_f=problem.grad,
        hess_f=problem.hess,
        X0=X0,
    )

    summary = {
        "config": asdict(config),
        "paper_notation": {
            "objective": "min_{X in St(p,n)} -1/2 tr(X^T C X)",
            "sample_covariance": "C = A^T A / N",
            "data_model": "a_i ~ N(0, U U^T + sigma I_n)",
            "A_shape": [problem.num_samples, problem.n],
            "X_shape": [problem.n, problem.p],
            "sigma": config.sigma,
        },
        "objective_star": f_star,
        "initial_point": {
            "X0_shape": [config.n, config.p],
            "warm_start_iterations": warm_result.iterations,
            "warm_start_stopping_reason": warm_result.stopping_reason,
            "objective": problem.cost(X0),
            "feasibility_violation": float(
                np.linalg.norm(X0.T @ X0 - np.eye(config.p), ord="fro")
            ),
            "Rgrad_norm": final_value(warm_result.log["Rgrad_norm"]),
        },
        "methods": {
            "SOL": summarize_method(sol_res, f_star),
            "SOL-sym": summarize_method(sol_sym_res, f_star),
        },
    }

    logs = {
        "SOL": sol_res.log,
        "SOL-sym": sol_sym_res.log,
    }
    save_json(out / "summary.json", summary)
    save_json(out / "logs.json", logs)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed PCA experiment with n=10000, p=500, "
            "num_samples=30000, sigma=0.1."
        )
    )
    parser.add_argument("--out", default=PCAConfig.out, help="Output directory.")
    parser.add_argument("--seed", type=int, default=PCAConfig.seed, help="Data random seed.")
    parser.add_argument(
        "--init-seed",
        type=int,
        default=PCAConfig.init_seed,
        help="Warm-start random seed.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable algorithm progress output.")
    return parser.parse_args()


def print_summary(results: dict[str, Any]) -> None:
    config = results["config"]
    print("\n================== PCA experiment summary ==================")
    print(
        f"n={config['n']}, p={config['p']}, "
        f"num_samples={config['num_samples']}, sigma={config['sigma']}"
    )
    print(f"f_star={results['objective_star']:.6e}")
    for method_name in ("SOL", "SOL-sym"):
        final = results["methods"][method_name]
        print(
            f"{method_name}: objective={final['objective']:.6e}, "
            f"gap={final['gap']:.3e}, "
            f"feasibility violation={final['feasibility_violation']:.3e}, "
            f"iters={final['iterations']}, "
            f"time={final['time']:.3f}s"
        )
    print("============================================================\n")


def main() -> None:
    args = parse_args()
    config = PCAConfig(
        seed=args.seed,
        init_seed=args.init_seed,
        out=args.out,
        verbosity=0 if args.quiet else PCAConfig.verbosity,
    )
    print_summary(run_experiment(config))


if __name__ == "__main__":
    main()
