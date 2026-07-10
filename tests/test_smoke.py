"""Smoke tests: every (problem, sampler) cell runs at a tiny budget.

Run:  JAX_PLATFORMS=cpu python tests/test_smoke.py
"""

import dataclasses

import jax
import numpy as np

from gauntlet.problems import PROBLEMS
from gauntlet.samplers import SAMPLERS


def test_every_cell_runs():
    for pname, problem in PROBLEMS.items():
        tiny = dataclasses.replace(problem, budget=1200)
        for sname, (run, grid) in SAMPLERS.items():
            hp = (dict(problem.home_hp) if problem.home == sname
                  else next(iter(grid(tiny).values())))
            samples = run(tiny, hp, jax.random.key(0))
            assert samples.ndim == 2 and samples.shape[1] == problem.dim, (pname, sname)
            assert np.isfinite(samples).all(), (pname, sname)
            metrics = tiny.evaluate(samples)
            assert len(metrics) >= 1, (pname, sname)
            print(f"  {pname} x {sname}: ok ({samples.shape[0]} samples)")


def test_home_samplers_exist():
    for problem in PROBLEMS.values():
        assert problem.home in SAMPLERS, problem.name


def test_grids_are_bounded():
    for problem in PROBLEMS.values():
        for sname, (run, grid) in SAMPLERS.items():
            assert 1 <= len(grid(problem)) <= 8, (problem.name, sname)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
