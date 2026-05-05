"""Run the Procrustes experiment from the paper."""

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

from experiments.Procrustes.problem import (
    distance_to_solution,
    feasibility_violation,
    make_procrustes_problem,
    procrustes_solution,
)


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 10000
    d: int = 1000
    sigma: float = 0.02
    seed: int = 0
    out: str = "results/Procrustes"
    tol: float = 1e-12
    linear_atol: float = 1e-14
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


def summarize_method(result: Any, X_star: np.ndarray, objective_star: float) -> dict[str, Any]:
    """Summarize one SOL/SOL-sym run using the paper's metric names."""
    X = result.X
    log = result.log
    objective = final_value(log["objective"])
    return {
        "stopping_reason": result.stopping_reason,
        "iterations": int(result.iterations),
        "time": final_value(log["time"]),
        "objective": objective,
        "gap": float(objective - objective_star),
        "feasibility_violation": feasibility_violation(X),
        "Rgrad_norm": final_value(log["Rgrad_norm"]),
        "distance_to_X_star": distance_to_solution(X, X_star),
    }


def make_initial_point(problem: Any, *, seed: int, verbosity: int) -> tuple[np.ndarray, Any]:
    """Construct the warm-start point used before the SOL and SOL-sym runs."""
    rng = np.random.default_rng(seed)
    X_init, _ = np.linalg.qr(rng.standard_normal((problem.d, problem.d)))
    warm = FirstOrderLanding(
        epsilon=0.75,
        lam=5.0,
        eta=0.1,
        tol=1e-2,
        max_iter=10000,
        verbosity=verbosity,
    )
    warm_result = warm.run(
        n=problem.d,
        p=problem.d,
        grad_f=problem.grad,
        cost=problem.cost,
        X0=X_init,
    )
    return warm_result.X, warm_result


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    problem = make_procrustes_problem(
        n=config.n,
        d=config.d,
        sigma=config.sigma,
        seed=config.seed,
    )
    X_star = procrustes_solution(problem.A, problem.B)
    objective_star = problem.cost(X_star)

    if config.verbosity > 0:
        print("\n================== Procrustes experiment ==================")
        print("problem: min_{X in St(d,d)} 1/(2n)||A X - B||_F^2")
        print("model:   B = A X_true + sigma Xi")
        print(f"n:       {config.n}")
        print(f"d:       {config.d}")
        print(f"sigma:   {config.sigma}")
        print(f"seed:    {config.seed}")
        print(f"out:     {out}")
        print(f"f_star:  {objective_star:.6e}")
        print("===========================================================\n")

    X0, warm_result = make_initial_point(
        problem,
        seed=config.seed,
        verbosity=config.verbosity,
    )

    # The paper initializes the infeasible second-order methods near the local
    # regime; this scaling makes the warm-start point slightly infeasible.
    X0 = 1.01 * X0

    sol = SecondOrderLanding(
        epsilon=0.75,
        eta=1.0,
        tol=config.tol,
        max_iter=200,
        linear_solver="bicgstab",
        linear_maxiter=1000,
        linear_atol=config.linear_atol,
        theta=1.0,
        zeta_max=1e-1,
        verbosity=config.verbosity,
    )
    sol_result = sol.run(
        n=problem.d,
        p=problem.d,
        cost=problem.cost,
        grad_f=problem.grad,
        hess_f=problem.hess,
        NS_order=1,
        X0=X0,
    )

    sol_sym = SecondOrderLandingSymmetric(
        epsilon=0.75,
        eta=1.0,
        tol=config.tol,
        max_iter=200,
        linear_maxiter=1000,
        linear_atol= config.linear_atol,
        theta=1.0,
        zeta_max=1e-1,
        verbosity=config.verbosity,
    )
    sol_sym_result = sol_sym.run(
        n=problem.d,
        p=problem.d,
        cost=problem.cost,
        grad_f=problem.grad,
        hess_f=problem.hess,
        NS_order=1,
        X0=X0,
    )

    summary = {
        "config": asdict(config),
        "paper_notation": {
            "objective": "min_{X in St(d,d)} 1/(2n)||A X - B||_F^2",
            "data_model": "B = A X_true + sigma Xi",
            "A_shape": [problem.n, problem.d],
            "B_shape": [problem.n, problem.d],
            "X_true_shape": [problem.d, problem.d],
            "sigma": config.sigma,
        },
        "objective_star": objective_star,
        "initial_point": {
            "X0_shape": [problem.d, problem.d],
            "warm_start_iterations": warm_result.iterations,
            "warm_start_stopping_reason": warm_result.stopping_reason,
            "objective": problem.cost(X0),
            "feasibility_violation": feasibility_violation(X0),
            "Rgrad_norm": final_value(warm_result.log["Rgrad_norm"]),
            "ambient_grad_norm": float(np.linalg.norm(problem.grad(X0), ord="fro")),
        },
        "methods": {
            "SOL": summarize_method(sol_result, X_star, objective_star),
            "SOL-sym": summarize_method(sol_sym_result, X_star, objective_star),
        },
    }

    logs = {
        "SOL": sol_result.log,
        "SOL-sym": sol_sym_result.log,
    }
    save_json(out / "summary.json", summary)
    save_json(out / "logs.json", logs)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Procrustes experiment with n=10000, d=1000, sigma=0.02."
    )
    parser.add_argument("--out", default=ExperimentConfig.out, help="Output directory.")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed, help="Random seed.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-iteration output.")
    return parser.parse_args()


def print_summary(results: dict[str, Any]) -> None:
    config = results["config"]
    print("\n================== Procrustes experiment summary ==================")
    print(
        f"n={config['n']}, d={config['d']}, "
        f"sigma={config['sigma']}, seed={config['seed']}"
    )
    print(f"f_star={results['objective_star']:.6e}")
    for method_name in ("SOL", "SOL-sym"):
        final = results["methods"][method_name]
        print(
            f"{method_name}: objective={final['objective']:.6e}, "
            f"gap={final['gap']:.3e}, "
            f"feasibility violation={final['feasibility_violation']:.3e}, "
            f"Rgrad_norm={final['Rgrad_norm']:.3e}, "
            f"distance_to_X_star={final['distance_to_X_star']:.3e}, "
            f"iters={final['iterations']}, "
            f"time={final['time']:.3f}s"
        )
    print("===================================================================\n")


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        out=args.out,
        verbosity=0 if args.quiet else 2,
    )
    print_summary(run_experiment(config))


if __name__ == "__main__":
    main()
