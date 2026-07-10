"""The five simulated problems, each from one of the source papers.

Every problem provides the ascent gradient (stochastic where the paper's is),
the potential (for AMAGOLD's M-H test), an evaluate() whose FIRST metric is
primary (lower = better), a total gradient-evaluation budget, the paper's own
hyperparameters for its home sampler (the diagonal of the table), and scale
hints for the visitors' tuning grids.

Problem definitions are taken from the verified ports:
SGHMC-jax (figures 1/3), csgmcmc-jax (toy_mog), low-precision-sgld-jax
(gaussian toy), amagold-jax (doublewell).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    origin: str
    home: str                       # home sampler key in SAMPLERS
    dim: int
    budget: int                     # total gradient evaluations per cell
    init_fn: Callable               # init_fn(key) -> position
    grad_fn: Callable               # grad_fn(key, x) -> ascent log-density grad
    u_fn: Callable                  # potential U(x) (for AMAGOLD's M-H)
    evaluate: Callable              # evaluate(samples (n, dim)) -> {metric: val}
    home_hp: dict                   # paper-verbatim hyperparameters (diagonal)
    hints: dict = field(default_factory=dict)  # step_scale, dt_scale, quantize


def _hist_l1(samples, grid, xstep, u_fn):
    edges = np.concatenate([[-np.inf], (grid[:-1] + grid[1:]) / 2, [np.inf]])
    y, _ = np.histogram(samples, bins=edges)
    dens = y / y.sum() / xstep
    true = np.exp(-u_fn(grid))
    true = true / true.sum() / xstep
    return float(np.abs(dens - true).sum() * xstep)


# --- 1. double-well density, SGHMC figure 1 (Chen et al. 2014) -----------------

def _dw_sghmc():
    u = lambda x: -2.0 * x**2 + x**4
    grad_u = lambda x: -4.0 * x + 4.0 * x**3

    def grad_fn(key, x):
        return -(grad_u(x) + 2.0 * jax.random.normal(key, jnp.shape(x)))

    grid = np.arange(-3.0, 3.0 + 1e-9, 0.1)

    def evaluate(samples):
        return {"density_l1": _hist_l1(samples[:, 0], grid, 0.1, u)}

    return Problem(
        name="doublewell_sghmc", origin="SGHMC (Chen et al. 2014)",
        home="sghmc", dim=1, budget=4_000_000,
        init_fn=lambda key: jnp.zeros(1),
        grad_fn=grad_fn, u_fn=lambda x: jnp.sum(u(x)), evaluate=evaluate,
        # figure-1 physical form (m=1, dt=0.1, C=3, V=4) mapped to the buffer
        # form: step = dt^2, alpha = C dt, v_hat = V
        home_hp={"step_size": 0.01, "alpha": 0.3, "v_hat": 4.0},
        hints={"step_scale": 0.01, "dt_scale": 0.1, "grad_noise_var": 4.0},
    )


# --- 2. correlated 2D Gaussian, SGHMC figure 3 (Chen et al. 2014) --------------

def _gauss2d():
    rho = 0.9
    inv_s = jnp.asarray(np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]])))
    cov = np.array([[1.0, rho], [rho, 1.0]])

    def grad_fn(key, x):
        return -(inv_s @ x + jax.random.normal(key, (2,)))

    def evaluate(samples):
        s = samples - samples.mean(axis=0)
        cov_e = s.T @ s / s.shape[0]
        return {"cov_err": float(np.abs(cov_e - cov).sum() / 4.0)}

    return Problem(
        name="gaussian2d", origin="SGHMC (Chen et al. 2014)",
        home="sghmc", dim=2, budget=2_000_000,
        init_fn=lambda key: jnp.zeros(2),
        grad_fn=grad_fn,
        u_fn=lambda x: 0.5 * x @ inv_s @ x, evaluate=evaluate,
        home_hp={"step_size": 0.05, "alpha": 0.05, "v_hat": 1.0},
        hints={"step_scale": 0.05, "dt_scale": 0.22, "grad_noise_var": 1.0},
    )


# --- 3. 25-Gaussian grid, cSG-MCMC (Zhang et al. 2020) --------------------------

def _mog25():
    g = np.asarray([-4, -2, 0, 2, 4], dtype=np.float32)
    mu = jnp.asarray(np.stack(np.meshgrid(g, g), -1).reshape(-1, 2))
    sigma2 = 0.03

    def logprob(x):
        comp = -0.5 * jnp.sum((x - mu) ** 2, axis=-1) / sigma2 - jnp.log(
            2 * jnp.pi * sigma2)
        return jsp.special.logsumexp(comp) - jnp.log(mu.shape[0])

    grad_logprob = jax.grad(logprob)

    def grad_fn(key, x):
        del key  # the toy uses the exact gradient (as in the original)
        return grad_logprob(x)

    mu_np = np.asarray(mu)

    def evaluate(samples):
        d2 = ((samples[:, None, :] - mu_np[None]) ** 2).sum(-1)
        near = d2.min(axis=1) < (3.0 * np.sqrt(sigma2)) ** 2
        assign = d2.argmin(axis=1)[near]
        visited = np.unique(assign).size
        # primary: TV between the empirical mode-visit distribution and the
        # true uniform 1/25 (mass falling outside all modes counts as error);
        # a missed mode contributes 1/25, so this subsumes coverage
        counts = np.bincount(assign, minlength=25).astype(float)
        weights = counts / max(1, samples.shape[0])
        tv = 0.5 * (np.abs(weights - 1.0 / 25).sum() + (1.0 - weights.sum()))
        return {"mode_tv": float(tv), "modes_visited": float(visited)}

    return Problem(
        name="mog25", origin="cSG-MCMC (Zhang et al. 2020)",
        home="csgld", dim=2, budget=50_000,
        init_fn=lambda key: -10.0 + 20.0 * jax.random.uniform(key, (2,)),
        grad_fn=grad_fn, u_fn=lambda x: -logprob(x), evaluate=evaluate,
        home_hp={"step_size": 0.09, "num_cycles": 30, "exploration_ratio": 0.25},
        hints={"step_scale": 0.03, "dt_scale": 0.17},
    )


# --- 4. low-precision standard Gaussian (Zhang, Wilson, De Sa 2022) ------------

def _lp_gauss():
    sigma = 0.1  # gradient noise of the original toy

    def grad_fn(key, x):
        return -(x + sigma * jax.random.normal(key, jnp.shape(x)))

    def evaluate(samples):
        return {"std_err": float(abs(samples[:, 0].std() - 1.0)),
                "std": float(samples[:, 0].std())}

    return Problem(
        name="lp_gaussian", origin="low-precision SGLD (Zhang et al. 2022)",
        home="vc_lp_sgld", dim=1, budget=1_000_000,
        init_fn=lambda key: jnp.zeros(1),
        grad_fn=grad_fn, u_fn=lambda x: jnp.sum(0.5 * x**2), evaluate=evaluate,
        home_hp={"lr": 2e-3, "wl": 8, "fl": 3},
        # every visitor's position is stochastically quantized to the same
        # 8-bit fixed-point grid after each step (the naive low-precision
        # treatment); the home method replaces that with VC quantization
        hints={"step_scale": 2e-3, "dt_scale": 0.045, "quantize": (8, 3)},
    )


# --- 5. asymmetric double-well, AMAGOLD (Zhang, Cooper, De Sa 2020) ------------

def _dw_amagold():
    u = lambda x: (x + 4.0) * (x + 1.0) * (x - 1.0) * (x - 3.0) / 14.0 + 0.5
    grad_u = lambda x: (4.0 * x**3 + 3.0 * x**2 - 26.0 * x - 1.0) / 14.0

    def grad_fn(key, x):
        return -(grad_u(x) + jax.random.normal(key, jnp.shape(x)))

    grid = np.arange(-6.0, 6.0 + 1e-9, 0.1)

    def evaluate(samples):
        return {"density_l1": _hist_l1(samples[:, 0], grid, 0.1, u)}

    return Problem(
        name="doublewell_amagold", origin="AMAGOLD (Zhang et al. 2020)",
        home="amagold", dim=1, budget=1_212_000,  # 101k outer x (10 + 2)
        init_fn=lambda key: jnp.zeros(1),
        grad_fn=grad_fn, u_fn=lambda x: jnp.sum(u(x)), evaluate=evaluate,
        home_hp={"dt": 0.25, "C": 0.5, "nstep": 10},
        hints={"step_scale": 0.0625, "dt_scale": 0.25, "grad_noise_var": 1.0},
    )


PROBLEMS = {p.name: p for p in (
    _dw_sghmc(), _gauss2d(), _mog25(), _lp_gauss(), _dw_amagold())}
