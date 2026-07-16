"""Generate step-size ladder rungs for mcmc-doctor's tune (Mode B input).

Runs multi-chain rungs of one sampler on one gauntlet problem across a
log-spaced step ladder, writing one npz per rung with the metadata tune
requires (step_size, sampler, grad_evals). Adjudicate with:

  python3 ~/.claude/skills/mcmc-doctor/scripts/mcmc_doctor.py tune \
      --chains <out>/rung_*.npz

Validation result (2026-07-16): sghmc on gaussian2d with the default ladder
recommends step = 0.05 — exactly the source paper's published setting —
with the bias probe jumping from z=1.3 (0.05) to z=69 (0.1).

Usage: python -m gauntlet.ladder --problem gaussian2d --sampler sghmc \
         [--steps 0.003,0.006,...] [--n 100000] [--chains 4] [--out ladders/...]
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import samplax

from .problems import PROBLEMS


def make_kernel(sampler, step, alpha):
    if sampler == "sgld":
        return samplax.sgld()
    if sampler == "psgld":
        return samplax.sgld(preconditioner=samplax.rmsprop())
    if sampler == "sghmc":
        # v_hat correction only where feasible at this step
        v_hat = 1.0 if 0.5 * 1.0 * step < alpha else 0.0
        return samplax.sghmc(alpha=alpha, v_hat=v_hat)
    raise ValueError(f"ladder supports sgld/psgld/sghmc, got {sampler!r}")


def run_rung(problem, sampler, step, n, n_chains, alpha, seed):
    kernel = make_kernel(sampler, step, alpha)

    def one_chain(chain_seed):
        key = jax.random.key(chain_seed)
        key, k0 = jax.random.split(key)
        pos0 = 2.0 * jax.random.normal(k0, (problem.dim,))
        state = kernel.init(k0, pos0)

        def body(state, subkey):
            kg, ks = jax.random.split(subkey)
            g = problem.grad_fn(kg, state.position)
            state = kernel.step(ks, state, g, step, 1.0)
            return state, state.position

        _, xs = jax.lax.scan(body, state, jax.random.split(key, n))
        return np.asarray(xs)

    chains = np.stack([one_chain(seed * 1000 + i) for i in range(n_chains)])
    return chains[:, n // 10:]  # drop 10% burn-in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--sampler", default="sghmc",
                    choices=("sgld", "psgld", "sghmc"))
    ap.add_argument("--steps", default=None,
                    help="comma-separated; default: 8 rungs x2 from step_scale/16")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    problem = PROBLEMS[args.problem]
    if args.steps:
        steps = [float(s) for s in args.steps.split(",")]
    else:
        base = problem.hints["step_scale"] / 16.0
        steps = [base * 2**i for i in range(8)]
    out_dir = args.out or os.path.join("ladders", f"{args.problem}_{args.sampler}")
    os.makedirs(out_dir, exist_ok=True)
    for step in steps:
        chains = run_rung(problem, args.sampler, step, args.n, args.chains,
                          args.alpha, args.seed)
        path = os.path.join(out_dir, f"rung_{step:g}.npz")
        np.savez_compressed(path, chains=chains.astype(np.float32),
                            step_size=step, sampler=args.sampler,
                            grad_evals=args.chains * args.n)
        print(f"wrote {path} ({chains.shape})")
    print(f"\nadjudicate with:\n  python3 ~/.claude/skills/mcmc-doctor/scripts/"
          f"mcmc_doctor.py tune --chains {out_dir}/rung_*.npz")


if __name__ == "__main__":
    main()
