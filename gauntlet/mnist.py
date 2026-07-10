"""MNIST BNN rows of the gauntlet (GPU tier).

Two rows, each following its origin paper's model and evaluation protocol;
all seven samplers run on both at a matched budget of minibatch gradient
steps. AMAGOLD's full-data M-H evaluations are reported as overhead (wall
clock) rather than deducted, matching its paper's own accounting.

- ``mnist_sghmc``   (SGHMC / ML-SGHMC bayesnn): 784-100-10 sigmoid MLP,
  batch 500, Gibbs-resampled per-group weight decay (the model's Gamma
  hyperprior — applied for every sampler), running posterior-averaged (BMA)
  test error after 50 burn-in epochs. Budget: 800 epochs = 80,000 steps.
  Home hyperparameters: eta 0.1, mdecay 0.01 (paper demo).
- ``mnist_amagold`` (AMAGOLD bnn): 784-500-256-10 ReLU MLP, batch 2000,
  fixed weight decay 5e-4, SGD-init checkpoint, current-sample test error
  (no averaging, as the paper evaluates). Budget: 22,500 steps (= the
  paper's 2500 outer iterations x 9 gradient steps).

Usage: python -m gauntlet.mnist --row mnist_sghmc [--sampler sghmc] [--out results]
"""

import argparse
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import samplax
from samplax.transforms.quant import fixed_point_quantize

from .mnist_data import load_mnist

ROWS = {
    "mnist_sghmc": dict(
        origin="SGHMC (Chen et al. 2014)", home="sghmc",
        hidden=(100,), act="sigmoid", batch=500, epochs=800, burn=50,
        weight_decay="gibbs", init="random", bma=True,
        home_hp={"step_mult": 1.0, "alpha": 0.01}, base_eta=0.1,
    ),
    "mnist_amagold": dict(
        origin="AMAGOLD (Zhang et al. 2020)", home="amagold",
        hidden=(500, 256), act="relu", batch=2000, epochs=750, burn=0,
        weight_decay=5e-4, init="checkpoint", bma=False,
        home_hp={"step_mult": 1.0, "T": 10, "beta": 5e-6}, base_eta=0.0005,
    ),
}


def make_model(row, key):
    sizes = (784, *row["hidden"], 10)
    act = jax.nn.sigmoid if row["act"] == "sigmoid" else jax.nn.relu
    if row["init"] == "checkpoint":
        d = np.load(os.path.join(os.path.dirname(__file__), "..",
                                 "checkpoints", "sgd_init_epoch3.npz"))
        params = {k: jnp.asarray(d[k]) for k in d.files}
    else:
        params = {}
        for i in range(len(sizes) - 1):
            key, kw = jax.random.split(key)
            params[f"w{i+1}"] = 0.01 * jax.random.normal(kw, (sizes[i], sizes[i+1]))
            params[f"b{i+1}"] = jnp.zeros(sizes[i+1])
    n_layers = len(sizes) - 1

    def forward(p, x):
        h = x
        for i in range(1, n_layers):
            h = act(h @ p[f"w{i}"] + p[f"b{i}"])
        return h @ p[f"w{n_layers}"] + p[f"b{n_layers}"]

    return params, forward


def make_grad_fns(row, forward, x_train, y_train, num_train):
    def mean_nll(p, x, y):
        logp = jax.nn.log_softmax(forward(p, x))
        return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

    grad_nll = jax.grad(mean_nll)

    def ascent_grad(p, x, y, wd):
        g = grad_nll(p, x, y)
        return jax.tree_util.tree_map(
            lambda g_, p_, w_: -num_train * g_ - w_ * num_train * p_, g, p, wd)

    def full_potential(p):  # sum-scale NLL (AMAGOLD's M-H energy, likelihood only)
        return mean_nll(p, x_train, y_train) * num_train

    return ascent_grad, grad_nll, full_potential, mean_nll


def evaluate_fn(forward, x_test, y_test):
    @jax.jit
    def current(p):
        logits = forward(p, x_test)
        err = jnp.mean((jnp.argmax(logits, 1) != y_test).astype(jnp.float32))
        return err, jax.nn.softmax(logits)

    return current


def run_cell(row_name, sampler, hp, seed, data_dir):
    row = ROWS[row_name]
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir)
    if row["act"] == "sigmoid":       # ML-SGHMC bayesnn normalization
        x_train, x_test = x_train / 256.0, x_test / 256.0
    else:                             # AMAGOLD/torchvision normalization
        x_train = (x_train / 255.0 - 0.1307) / 0.3081
        x_test = (x_test / 255.0 - 0.1307) / 0.3081
    num_train = x_train.shape[0]
    batch, epochs = row["batch"], row["epochs"]
    nb = num_train // batch
    eta = row["base_eta"] * hp.get("step_mult", 1.0)
    step = eta / num_train
    key = jax.random.key(seed)
    key, k_model = jax.random.split(key)
    params, forward = make_model(row, k_model)
    ascent_grad, grad_nll, full_potential, mean_nll = make_grad_fns(
        row, forward, jnp.asarray(x_train), jnp.asarray(y_train), num_train)
    current_eval = evaluate_fn(forward, jnp.asarray(x_test), jnp.asarray(y_test))
    rng = np.random.default_rng(seed)

    gibbs = row["weight_decay"] == "gibbs"
    wd0 = 2e-5 if gibbs else row["weight_decay"]
    wd = jax.tree_util.tree_map(lambda _: jnp.asarray(wd0), params)

    # --- build the per-epoch driver for each sampler family -------------------
    if sampler == "amagold":
        T = hp.get("T", 10)
        warmup_steps = 0
        if row["init"] == "random":
            # AMAGOLD's own protocol SGD-initializes before sampling; on
            # random-init rows the visitor gets an equivalent noise-free
            # warmup, deducted from its gradient budget
            warmup_steps = 3 * nb
            warm_step = row["base_eta"] / num_train  # row scale, not AMAGOLD's
            warm_kernel = samplax.sghmc(alpha=0.1)
            wstate = warm_kernel.init(key, params)

            @jax.jit
            def warm_epoch(key, wstate, xb, yb):
                def body(carry, inp):
                    wstate, key = carry
                    x, y = inp
                    key, ks = jax.random.split(key)
                    g = ascent_grad(wstate.position, x, y, wd)
                    wstate = warm_kernel.step(ks, wstate, g, warm_step, 0.0)
                    return (wstate, key), None

                (wstate, _), _ = jax.lax.scan(body, (wstate, key), (xb, yb))
                return wstate

            for _ in range(3):
                perm = rng.permutation(num_train)[: nb * batch].reshape(nb, batch)
                key, k_ep = jax.random.split(key)
                wstate = warm_epoch(k_ep, wstate,
                                    jnp.asarray(x_train[perm]),
                                    jnp.asarray(y_train[perm]))
            params = wstate.position
        init, outer = samplax.amagold_minibatch(
            T=T, beta=hp.get("beta", 5e-6), step_size=step)
        state = init(key, params)

        def grad_fn(p, batch_xy):
            x, y = batch_xy
            g = grad_nll(p, x, y)
            return jax.tree_util.tree_map(
                lambda g_, p_: num_train * g_ + wd0 * num_train * p_, g, p)

        outer_jit = jax.jit(lambda k, s, xb, yb: outer(
            k, s, grad_fn, full_potential, (xb, yb)))
        n_outer = (epochs * nb - warmup_steps) // (T - 1)
        accepts = 0
        for it in range(n_outer):
            perm = rng.permutation(num_train)[: T * batch].reshape(T, batch)
            xb = jnp.asarray(x_train[perm])
            yb = jnp.asarray(y_train[perm])
            key, sub = jax.random.split(key)
            state, acc, _ = outer_jit(sub, state, xb, yb)
            accepts += int(acc)
        params_final = state.position
        err, _ = current_eval(params_final)
        return {"test_err": float(err), "accept_rate": accepts / max(1, n_outer)}

    # gradient-per-step samplers
    if sampler in ("sgld", "psgld"):
        precond = samplax.rmsprop() if sampler == "psgld" else None
        kernel = samplax.sgld(preconditioner=precond)
        sched = samplax.constant(step)
    elif sampler == "sghmc":
        kernel = samplax.sghmc(alpha=hp.get("alpha", 0.01))
        sched = samplax.constant(step)
    elif sampler in ("csgld", "csghmc"):
        kernel = (samplax.sgld() if sampler == "csgld"
                  else samplax.sghmc(alpha=hp.get("alpha", 0.01)))
        sched = samplax.cyclical(epochs * nb, hp.get("num_cycles", 4), step)
    elif sampler == "vc_lp_sgld":
        kernel = samplax.lp_sgld("vc", hp.get("wl", 8), hp["fl"],
                                 datasize=num_train, weight_decay=wd0)
        sched = None
    else:
        raise ValueError(sampler)

    if sampler == "vc_lp_sgld":
        state = kernel.init(key, params)

        @jax.jit
        def epoch_fn(key, state, xb, yb, wd_):
            def body(carry, inp):
                state, key = carry
                x, y = inp
                key, kq, ks = jax.random.split(key, 3)
                p_fwd = kernel.forward_quant(kq, state)
                g = grad_nll(p_fwd, x, y)  # descent, mean scale (native convention)
                state = kernel.step(ks, state, g, eta, 1.0)
                return (state, key), None

            (state, _), _ = jax.lax.scan(body, (state, key), (xb, yb))
            return state

        get_pos = lambda s: s.position
    else:
        state = kernel.init(key, params)
        t0 = jnp.asarray(0)

        @jax.jit
        def epoch_fn(key, state, xb, yb, wd_, t):
            def body(carry, inp):
                state, key, t = carry
                x, y = inp
                key, ks = jax.random.split(key)
                g = ascent_grad(state.position, x, y, wd_)
                s = sched(t)
                temp = jnp.where(s.do_sample, 1.0, 0.0)
                state = kernel.step(ks, state, g, s.step_size, temp)
                return (state, key, t + 1), None

            (state, _, t), _ = jax.lax.scan(body, (state, key, t), (xb, yb))
            return state, t

        get_pos = lambda s: s.position

    # --- epoch loop with Gibbs + evaluation -----------------------------------
    o_pred = jnp.zeros((y_test.shape[0], 10))
    sum_w = 0.0
    err_hist = []
    t = jnp.asarray(0)
    t_start = time.perf_counter()
    for ep in range(epochs):
        perm = rng.permutation(num_train)[: nb * batch].reshape(nb, batch)
        xb, yb = jnp.asarray(x_train[perm]), jnp.asarray(y_train[perm])
        key, k_ep, k_gibbs = jax.random.split(key, 3)
        if sampler == "vc_lp_sgld":
            state = epoch_fn(k_ep, state, xb, yb, wd)
        else:
            state, t = epoch_fn(k_ep, state, xb, yb, wd, t)
        if gibbs and ep >= 1:
            lams = samplax.gibbs_precision(k_gibbs, get_pos(state))
            wd = jax.tree_util.tree_map(lambda l: l / num_train, lams)
        err, probs = current_eval(get_pos(state))
        if row["bma"]:
            if ep < row["burn"]:
                o_pred, sum_w = probs, 1.0
            else:
                sum_w += 1.0
                o_pred = o_pred * (1 - 1 / sum_w) + probs / sum_w
            bma_err = float(jnp.mean(
                (jnp.argmax(o_pred, 1) != jnp.asarray(y_test)).astype(jnp.float32)))
            err_hist.append(bma_err)
        else:
            err_hist.append(float(err))
        if not np.isfinite(err_hist[-1]):
            return {"test_err": None, "diverged_epoch": ep}
    return {"test_err": err_hist[-1], "wall_s": round(time.perf_counter() - t_start, 1)}


GRIDS = {
    "sgld": lambda row: [{"step_mult": m} for m in (0.3, 1.0, 3.0)],
    "psgld": lambda row: [{"step_mult": m} for m in (0.03, 0.1, 0.3)],
    "sghmc": lambda row: [{"step_mult": m, "alpha": a}
                          for m in (0.3, 1.0) for a in (0.01, 0.1)],
    "csgld": lambda row: [{"step_mult": m, "num_cycles": c}
                          for m in (1.0, 3.0) for c in (4, 8)],
    "csghmc": lambda row: [{"step_mult": m, "num_cycles": 4, "alpha": 0.01}
                           for m in (0.3, 1.0, 3.0)],
    # AMAGOLD's step size is capped by leapfrog error against the full-data
    # M-H energy, far below SGLD-scale steps: its visitor grid spans its own
    # regime rather than the row's
    "amagold": lambda row: [{"step_mult": m, "T": 10, "beta": 5e-6}
                            for m in (0.003, 0.01, 0.03)],
    "vc_lp_sgld": lambda row: [{"step_mult": m, "wl": 8, "fl": fl}
                               for m in (0.3, 1.0) for fl in (6, 8)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=sorted(ROWS))
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="data/mnist")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    row = ROWS[args.row]
    samplers = [args.sampler] if args.sampler else list(GRIDS)
    for sampler in samplers:
        is_home = row["home"] == sampler
        configs = ({"paper": dict(row["home_hp"])} if is_home
                   else {f"cfg{i}": hp for i, hp in enumerate(GRIDS[sampler](row))})
        results = {}
        for cname, hp in configs.items():
            t0 = time.perf_counter()
            m = run_cell(args.row, sampler, hp, args.seed, args.data_dir)
            m["wall_s"] = round(time.perf_counter() - t0, 1)
            results[cname] = {"hp": hp, "metrics": m}
            print(f"  {args.row} / {sampler} / {cname}: {m}", flush=True)
        os.makedirs(os.path.join(args.out, args.row), exist_ok=True)
        with open(os.path.join(args.out, args.row, f"{sampler}.json"), "w") as f:
            json.dump({"problem": args.row, "origin": row["origin"],
                       "sampler": sampler, "is_home": is_home,
                       "budget": row["epochs"] * (60000 // row["batch"]),
                       "seed": args.seed, "configs": results}, f, indent=1)


if __name__ == "__main__":
    main()
