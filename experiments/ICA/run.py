"""Run the fixed ICA experiment from the paper.

Run from the project root with:

    python experiments/ICA/run.py
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

from experiments.ICA.problem import (
    component_alignment,
    feasibility_violation,
    make_ica_problem,
)

Array = np.ndarray


@dataclass(frozen=True)
class ICAConfig:
    d: int = 60
    tmin: float = 0.0
    tmax: float = 60.0
    resample_sfreq: float = 100.0
    eeg_only: bool = True
    init_seed: int = 0
    warm_tol: float = 1e-4
    warm_max_iter: int = 100000
    tol: float = 1e-13
    theta: float = 1.0
    zeta_max: float = 1e-1
    linear_atol: float = 1e-14
    linear_maxiter: int = 200
    max_iter: int = 200
    out: str = "results/ICA"
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


def make_initial_point(config: ICAConfig, problem: Any) -> tuple[Array, Any]:
    """Construct the warm-start point used before the SOL and SOL-sym runs."""
    rng = np.random.default_rng(config.init_seed)
    X_init, _ = np.linalg.qr(rng.standard_normal((config.d, config.d)))
    warm = FirstOrderLanding(
        epsilon=0.75,
        lam=0.5,
        eta=1.0,
        tol=config.warm_tol,
        max_iter=config.warm_max_iter,
        verbosity=config.verbosity,
    )
    warm_result = warm.run(
        n=config.d,
        p=config.d,
        grad_f=problem.grad,
        cost=problem.cost,
        X0=X_init,
    )
    return warm_result.X, warm_result


def summarize_method(
    result: Any,
    *,
    problem: Any,
    X_ref: Array,
) -> dict[str, Any]:
    """Summarize one SOL/SOL-sym run for the real-data ICA experiment."""
    objective = final_value(result.log["objective"])
    return {
        "stopping_reason": result.stopping_reason,
        "iterations": int(result.iterations),
        "objective": objective,
        "feasibility_violation": feasibility_violation(result.X),
        "alignment_error": component_alignment(result.X, X_ref),
        "amari_distance": None,
        "Rgrad_norm": final_value(result.log["Rgrad_norm"]),
        "time": final_value(result.log["time"]),
    }


def run_experiment(config: ICAConfig = ICAConfig()) -> dict[str, Any]:
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    problem = make_ica_problem(
        d=config.d,
        tmin=config.tmin,
        tmax=config.tmax,
        resample_sfreq=config.resample_sfreq,
        eeg_only=config.eeg_only,
    )
    d = problem.d

    if config.verbosity > 0:
        print("\n================== ICA experiment ==================")
        print("problem: min_{X in St(d,d)} -1/N sum_ij log(cosh((W X)_ij))")
        print(f"requested d:      {config.d}")
        print(f"effective d:      {d}")
        print(f"N:                {problem.N}")
        print(f"tmax:             {config.tmax}")
        print(f"resample_sfreq:   {config.resample_sfreq}")
        print(f"out:              {out}")
        print("====================================================\n")

    X0, warm_result = make_initial_point(config, problem)

    # The paper initializes the infeasible second-order methods near the local
    # regime; this scaling makes the warm-start point slightly infeasible.
    X0 = 1.01 * X0


    sol = SecondOrderLanding(
        epsilon=0.75,
        eta=1.0,
        tol=config.tol,
        max_iter=config.max_iter,
        linear_solver="bicgstab",
        linear_maxiter=config.linear_maxiter,
        linear_atol=config.linear_atol,
        theta=config.theta,
        zeta_max=config.zeta_max,
        proj_to_tangent_space=True,
        verbosity=config.verbosity,
    )
    sol_res = sol.run(
        n=d,
        p=d,
        cost=problem.cost,
        grad_f=problem.grad,
        hess_f=problem.hess,
        X0=X0,
    )

    sol_sym = SecondOrderLandingSymmetric(
        epsilon=0.75,
        eta=1.0,
        tol=config.tol,
        max_iter=config.max_iter,
        linear_maxiter=config.linear_maxiter,
        linear_atol=config.linear_atol,
        theta=config.theta,
        zeta_max=config.zeta_max,
        verbosity=config.verbosity,
    )
    sol_sym_res = sol_sym.run(
        n=d,
        p=d,
        cost=problem.cost,
        grad_f=problem.grad,
        hess_f=problem.hess,
        X0=X0,
    )

    X_ref = (
        sol_res.X
        if final_value(sol_res.log["objective"]) <= final_value(sol_sym_res.log["objective"])
        else sol_sym_res.X
    )

    summary = {
        "config": asdict(config),
        "paper_notation": {
            "objective": (
                "min_{X in St(d,d)} -1/N sum_{i=1}^N sum_{j=1}^d "
                "log(cosh((W X)_{ij}))"
            ),
            "data_matrix": "W is the whitened EEG data matrix",
            "W_shape": [problem.N, d],
            "X_shape": [d, d],
        },
        "problem": {
            "N": int(problem.N),
            "d": int(d),
            "requested_d": int(config.d),
            "tmin": float(config.tmin),
            "tmax": float(config.tmax),
            "resample_sfreq": float(config.resample_sfreq),
            "eeg_only": bool(config.eeg_only),
            "init_seed": int(config.init_seed),
            "metadata": problem.metadata,
        },
        "initial_point": {
            "X0_shape": [d, d],
            "warm_start_iterations": int(warm_result.iterations),
            "warm_start_stopping_reason": warm_result.stopping_reason,
            "objective": problem.cost(X0),
            "feasibility_violation": feasibility_violation(X0),
            "Rgrad_norm": final_value(warm_result.log["Rgrad_norm"]),
        },
        "methods": {
            "SOL": summarize_method(
                sol_res,
                problem=problem,
                X_ref=X_ref,
            ),
            "SOL-sym": summarize_method(
                sol_sym_res,
                problem=problem,
                X_ref=X_ref,
            ),
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
        description="Run the fixed ICA experiment with N=6000, d=60, tmax=60, resample_sfreq=100."
    )
    parser.add_argument("--out", default=ICAConfig.out, help="Output directory.")
    parser.add_argument(
        "--init-seed",
        type=int,
        default=ICAConfig.init_seed,
        help="Warm-start random seed.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable algorithm progress output.")
    return parser.parse_args()


def print_summary(results: dict[str, Any]) -> None:
    problem = results["problem"]
    print("\n================== ICA experiment summary ==================")
    print(
        f"N={problem['N']}, d={problem['d']}, "
        f"tmax={problem['tmax']}, "
        f"resample_sfreq={problem['resample_sfreq']}"
    )
    for method_name in ("SOL", "SOL-sym"):
        final = results["methods"][method_name]
        print(
            f"{method_name}: objective={final['objective']:.6e}, "
            f"feasibility violation={final['feasibility_violation']:.3e}, "
            f"alignment={final['alignment_error']:.3e}, "
            f"iters={final['iterations']}, "
            f"time={final['time']:.3f}s"
        )
    print("============================================================\n")


def main() -> None:
    args = parse_args()
    config = ICAConfig(
        init_seed=args.init_seed,
        out=args.out,
        verbosity=0 if args.quiet else ICAConfig.verbosity,
    )
    print_summary(run_experiment(config))


if __name__ == "__main__":
    main()
