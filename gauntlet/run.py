"""Run gauntlet cells: every sampler on every problem at a matched budget.

For the problem's home sampler the paper's hyperparameters are used verbatim
(labelled ``paper``); every other (problem, sampler) cell evaluates a small
fixed grid and all configurations' metrics are stored (table.py reports the
best). Results go to results/<problem>/<sampler>.json.

Usage:
  python -m gauntlet.run                        # everything
  python -m gauntlet.run --problem mog25        # one row
  python -m gauntlet.run --sampler amagold      # one column
"""

import argparse
import json
import os
import time

import jax
import numpy as np

from .doctor_bridge import chain_diagnostics, save_chain
from .problems import PROBLEMS
from .samplers import SAMPLERS


def run_cell(problem, sampler_name, seed=0, out_dir="results", save_chains=False):
    run, grid = SAMPLERS[sampler_name]
    configs = dict(grid(problem))
    if problem.home == sampler_name:
        configs = {"paper": dict(problem.home_hp)}
    results = {}
    best_cfg, best_val, best_samples = None, None, None
    for cname, hp in configs.items():
        key = jax.random.key(seed)
        t0 = time.perf_counter()
        try:
            samples = run(problem, hp, key)
            metrics = problem.evaluate(samples)
            metrics = {k: (None if not np.isfinite(v) else v)
                       for k, v in metrics.items()}
            metrics.update(chain_diagnostics(samples))
            primary_val = next(iter(metrics.values()))
            if primary_val is not None and (best_val is None or primary_val < best_val):
                best_cfg, best_val, best_samples = cname, primary_val, samples
        except FloatingPointError:
            metrics = {"error": "diverged"}
        results[cname] = {"hp": hp, "metrics": metrics,
                          "wall_s": round(time.perf_counter() - t0, 2)}
        primary = next(iter(metrics.values()))
        print(f"  {problem.name} / {sampler_name} / {cname}: "
              f"{list(metrics)[0]}={primary}")
    os.makedirs(os.path.join(out_dir, problem.name), exist_ok=True)
    payload = {
        "problem": problem.name, "origin": problem.origin,
        "sampler": sampler_name, "is_home": problem.home == sampler_name,
        "budget": problem.budget, "seed": seed, "configs": results,
    }
    path = os.path.join(out_dir, problem.name, f"{sampler_name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    if save_chains and best_samples is not None:
        save_chain(
            os.path.join(out_dir, problem.name, "chains", f"{sampler_name}.npz"),
            best_samples, sampler=sampler_name, config=best_cfg,
            grad_evals=problem.budget, seed=seed)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None, choices=sorted(PROBLEMS))
    ap.add_argument("--sampler", default=None, choices=sorted(SAMPLERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-chains", action="store_true",
                    help="save each cell's best-config chain to "
                         "results/<problem>/chains/<sampler>.npz (for mcmc-doctor)")
    args = ap.parse_args()

    problems = [PROBLEMS[args.problem]] if args.problem else list(PROBLEMS.values())
    samplers = [args.sampler] if args.sampler else list(SAMPLERS)
    for problem in problems:
        for sampler_name in samplers:
            run_cell(problem, sampler_name, args.seed, args.out, args.save_chains)


if __name__ == "__main__":
    main()
