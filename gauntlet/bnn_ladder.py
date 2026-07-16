"""BNN-scale ladder rungs for mcmc-doctor tune (the cluster tier).

Runs SGHMC on the mnist_sghmc row's model (784-100-10 sigmoid, batch 500,
fixed weight decay) across a step-multiplier ladder, multiple chains per
rung. Full weight chains are too large to ship, so each rung records
decision-relevant FUNCTIONAL chains per step: [subset test NLL,
mean(w1^2), mean(w2^2), mean(b1^2), mean(b2^2)] — exactly the projection
the mcmc-doctor ecosystem notes prescribe for large models. The functionals
double as bias probes: tune's mean/x2 checks on these chains test the
posterior functionals a BNN user actually cares about.

Usage (cluster): PYTHONPATH=. python3 -m gauntlet.bnn_ladder \
    --data_dir data/mnist --epochs 200 --chains 3 --out ladders/bnn
Adjudicate: mcmc_doctor.py tune --chains ladders/bnn/rung_*.npz
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import samplax

from .mnist import evaluate_fn, make_grad_fns, make_model, ROWS
from .mnist_data import load_mnist


def make_kernel(sampler, alpha):
    if sampler == "sghmc":
        return samplax.sghmc(alpha=alpha)
    if sampler == "psgld":
        return samplax.sgld(preconditioner=samplax.rmsprop())
    if sampler == "sgld":
        return samplax.sgld()
    raise ValueError(f"bnn_ladder supports sgld/psgld/sghmc, got {sampler!r}")


def run_rung(step_mult, epochs, n_chains, data_dir, seed, alpha=0.01,
             n_probe=2000, sampler="sghmc"):
    row = ROWS["mnist_sghmc"]
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir)
    x_train, x_test = x_train / 256.0, x_test / 256.0
    num_train = x_train.shape[0]
    batch = row["batch"]
    nb = num_train // batch
    eta = row["base_eta"] * step_mult
    step = eta / num_train
    wd0 = 2e-5
    x_probe = jnp.asarray(x_test[:n_probe])
    y_probe = jnp.asarray(y_test[:n_probe])

    def one_chain(chain_seed):
        key = jax.random.key(chain_seed)
        key, k_model = jax.random.split(key)
        params, forward = make_model(row, k_model)
        ascent_grad, grad_nll, _, mean_nll = make_grad_fns(
            row, forward, jnp.asarray(x_train), jnp.asarray(y_train), num_train)
        kernel = make_kernel(sampler, alpha)
        state = kernel.init(key, params)
        wd = jax.tree_util.tree_map(lambda _: jnp.asarray(wd0), params)
        rng = np.random.default_rng(chain_seed)

        def functionals(p):
            return jnp.stack([
                mean_nll(p, x_probe, y_probe),
                jnp.mean(p["w1"] ** 2), jnp.mean(p["w2"] ** 2),
                jnp.mean(p["b1"] ** 2), jnp.mean(p["b2"] ** 2)])

        @jax.jit
        def epoch_fn(key, state, xb, yb):
            def body(carry, inp):
                state, key = carry
                x, y = inp
                key, ks = jax.random.split(key)
                g = ascent_grad(state.position, x, y, wd)
                state = kernel.step(ks, state, g, step, 1.0)
                return (state, key), functionals(state.position)

            (state, _), fs = jax.lax.scan(body, (state, key), (xb, yb))
            return state, fs

        chunks = []
        for _ in range(epochs):
            perm = rng.permutation(num_train)[: nb * batch].reshape(nb, batch)
            key, k_ep = jax.random.split(key)
            state, fs = epoch_fn(k_ep, state,
                                 jnp.asarray(x_train[perm]),
                                 jnp.asarray(y_train[perm]))
            chunks.append(np.asarray(fs))
        return np.concatenate(chunks, axis=0)  # (epochs * nb, 5)

    chains = np.stack([one_chain(seed * 100 + i) for i in range(n_chains)])
    burn = chains.shape[1] // 5
    return chains[:, burn:], step, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/mnist")
    ap.add_argument("--sampler", default="sghmc",
                    choices=("sgld", "psgld", "sghmc"))
    ap.add_argument("--step_mults", default="0.1,0.3,1.0,3.0,10.0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ladders/bnn")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for m in (float(s) for s in args.step_mults.split(",")):
        chains, step, nb = run_rung(m, args.epochs, args.chains,
                                    args.data_dir, args.seed, args.alpha,
                                    sampler=args.sampler)
        path = os.path.join(args.out, f"rung_{m:g}.npz")
        np.savez_compressed(
            path, chains=chains.astype(np.float32), step_size=step,
            sampler=args.sampler, grad_evals=args.chains * args.epochs * nb,
            step_mult=m)
        print(f"wrote {path} {chains.shape} (step={step:g})", flush=True)


if __name__ == "__main__":
    main()
