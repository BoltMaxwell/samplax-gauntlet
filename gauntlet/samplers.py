"""Sampler adapters and per-cell tuning grids.

Each sampler exposes ``run(problem, hp, key) -> samples`` (post-burn, thinned
to <= MAX_KEPT) and ``grid(problem) -> {config_name: hp}``. Budget accounting:
SGLD-family kernels cost one gradient per step (n_steps = budget); AMAGOLD
costs nstep gradients + 2 potential evaluations per outer step.

Diagonal cells use the paper's hyperparameters verbatim (``problem.home_hp``);
visitor cells search a small fixed grid around the problem's scale hints and
report the best configuration (see run.py).
"""

import jax
import jax.numpy as jnp
import numpy as np

import samplax
from samplax.transforms.lp_sgld import LPSGLDState
from samplax.transforms.quant import fixed_point_quantize

MAX_KEPT = 400_000
BURN_FRAC = 0.1


def _thin(n):
    return max(1, int(np.ceil(n / MAX_KEPT)))


def _run_kernel(problem, kernel, schedule, n_steps, key, quantize=None):
    """Drive a samplax Kernel for n_steps; collect sampling-phase positions."""
    thin = _thin(n_steps)
    key, k_init, k_pos = jax.random.split(key, 3)
    state = kernel.init(k_init, problem.init_fn(k_pos))

    def body(carry, inp):
        state, = carry
        t, subkey = inp
        k_grad, k_step, k_q = jax.random.split(subkey, 3)
        sched = schedule(t)
        g = problem.grad_fn(k_grad, state.position)
        temp = jnp.where(sched.do_sample, 1.0, 0.0)
        state = kernel.step(k_step, state, g, sched.step_size, temp)
        if quantize is not None:
            wl, fl = quantize
            state = state._replace(position=fixed_point_quantize(
                state.position, wl, fl, "stochastic", k_q))
        return (state,), (state.position, sched.do_sample)

    @jax.jit
    def run_all(state, keys):
        (_,), (pos, mask) = jax.lax.scan(body, (state,), (jnp.arange(n_steps), keys))
        return pos, mask

    pos, mask = run_all(state, jax.random.split(key, n_steps))
    pos = np.asarray(pos[:: thin]).reshape(-1, problem.dim)
    mask = np.asarray(mask[:: thin])
    burn = int(BURN_FRAC * pos.shape[0])
    keep = mask.copy()
    keep[:burn] = False
    return pos[keep]


def _quantize_of(problem, hp):
    return problem.hints.get("quantize") if hp.get("respect_quantize", True) else None


# --- SGLD family ----------------------------------------------------------------

def _sgld_like(preconditioner):
    def run(problem, hp, key):
        precond = samplax.rmsprop() if preconditioner else None
        kernel = samplax.sgld(preconditioner=precond,
                              v_hat=hp.get("v_hat", 0.0))
        sched = samplax.constant(hp["step_size"])
        return _run_kernel(problem, kernel, sched, problem.budget, key,
                           quantize=problem.hints.get("quantize"))

    def grid(problem):
        s = problem.hints["step_scale"]
        return {f"step{m}": {"step_size": s * m} for m in (0.1, 0.3, 1.0, 3.0, 10.0)}

    return run, grid


def _sghmc():
    def run(problem, hp, key):
        kernel = samplax.sghmc(alpha=hp["alpha"], v_hat=hp.get("v_hat", 0.0))
        sched = samplax.constant(hp["step_size"])
        return _run_kernel(problem, kernel, sched, problem.budget, key,
                           quantize=problem.hints.get("quantize"))

    def grid(problem):
        s = problem.hints["step_scale"]
        v = problem.hints.get("grad_noise_var", 0.0)
        out = {}
        for m in (0.3, 1.0, 3.0):
            for alpha in (0.1, 0.3):
                # keep the v_hat correction feasible: 0.5 v s m < alpha
                vh = v if 0.5 * v * s * m < alpha else 0.0
                out[f"step{m}_a{alpha}"] = {"step_size": s * m, "alpha": alpha,
                                            "v_hat": vh}
        return out

    return run, grid


def _cyclical(base):  # base in {"sgld", "sghmc"}
    def run(problem, hp, key):
        if base == "sgld":
            kernel = samplax.sgld()
        else:
            kernel = samplax.sghmc(alpha=hp.get("alpha", 0.1))
        sched = samplax.cyclical(problem.budget, hp["num_cycles"],
                                 hp["step_size"],
                                 hp.get("exploration_ratio", 0.25))
        return _run_kernel(problem, kernel, sched, problem.budget, key,
                           quantize=problem.hints.get("quantize"))

    def grid(problem):
        s = problem.hints["step_scale"]
        cycles = int(np.clip(30 * problem.budget / 50_000, 4, 60))
        out = {}
        for m in (0.3, 1.0, 3.0):
            for c in (cycles // 3, cycles):
                out[f"step{m}_c{c}"] = {"step_size": s * m, "num_cycles": max(2, c)}
        return out

    return run, grid


# --- AMAGOLD ----------------------------------------------------------------------

def _amagold():
    def run(problem, hp, key):
        nstep = hp.get("nstep", 10)
        n_outer = problem.budget // (nstep + 2)
        thin = _thin(n_outer)
        step = samplax.amagold(problem.u_fn,
                               lambda k, x: -problem.grad_fn(k, x),
                               dt=hp["dt"], nstep=nstep, C=hp["C"], mh=True)
        key, k_pos = jax.random.split(key)
        x0 = problem.init_fn(k_pos)
        quantize = problem.hints.get("quantize")

        def body(x, subkey):
            k_step, k_q = jax.random.split(subkey)
            x, _ = step(k_step, x)
            if quantize is not None:
                # low-precision storage between outer steps (the inner
                # leapfrog stays full-precision, i.e. an "-F" treatment)
                x = fixed_point_quantize(x, quantize[0], quantize[1],
                                         "stochastic", k_q)
            return x, x

        @jax.jit
        def run_all(x0, keys):
            _, xs = jax.lax.scan(body, x0, keys)
            return xs

        xs = run_all(x0, jax.random.split(key, n_outer))
        xs = np.asarray(xs[:: thin]).reshape(-1, problem.dim)
        return xs[int(BURN_FRAC * xs.shape[0]):]

    def grid(problem):
        d = problem.hints["dt_scale"]
        return {f"dt{m}_C{c}": {"dt": d * m, "C": c, "nstep": 10}
                for m in (0.3, 1.0, 3.0) for c in (0.5, 3.0)}

    return run, grid


# --- VC low-precision SGLD ---------------------------------------------------------

def _vc_lp_sgld():
    def run(problem, hp, key):
        kernel = samplax.lp_sgld("vc", hp["wl"], hp["fl"], datasize=1)
        n_steps = problem.budget
        thin = _thin(n_steps)
        key, k_pos = jax.random.split(key)
        state = kernel.init(key, problem.init_fn(k_pos))

        def body(state, subkey):
            k_grad, k_step = jax.random.split(subkey)
            g = -problem.grad_fn(k_grad, state.position)  # descent convention
            state = kernel.step(k_step, state, g, hp["lr"], 1.0)
            return state, state.position

        @jax.jit
        def run_all(state, keys):
            _, xs = jax.lax.scan(body, state, keys)
            return xs

        xs = run_all(state, jax.random.split(key, n_steps))
        xs = np.asarray(xs[:: thin]).reshape(-1, problem.dim)
        return xs[int(BURN_FRAC * xs.shape[0]):]

    def grid(problem):
        s = problem.hints["step_scale"]
        # fl chosen so the 8-bit range covers the problem domain
        return {f"lr{m}_fl{fl}": {"lr": s * m, "wl": 8, "fl": fl}
                for m in (0.3, 1.0, 3.0) for fl in (3, 4)}

    return run, grid


SAMPLERS = {
    "sgld": _sgld_like(preconditioner=False),
    "psgld": _sgld_like(preconditioner=True),
    "sghmc": _sghmc(),
    "csgld": _cyclical("sgld"),
    "csghmc": _cyclical("sghmc"),
    "amagold": _amagold(),
    "vc_lp_sgld": _vc_lp_sgld(),
}
