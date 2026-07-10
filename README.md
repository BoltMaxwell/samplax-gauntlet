samplax-gauntlet
================

Every [samplax](https://github.com/BoltMaxwell/samplax) sampler on every
source paper's demo problem — the cross-matrix the individual papers never
ran. Each problem is tagged with the paper it comes from; the expectation is
that each problem's best method is the one whose paper introduced it (bold
diagonal, run with the paper's own hyperparameters), and the interesting
findings are the exceptions.

Method
------

- **Matched budgets.** Every cell gets the same number of gradient
  evaluations on a given problem (AMAGOLD's leapfrog gradients and full-data
  M-H energy evaluations are counted against its budget).
- **Diagonal = paper settings.** The home cell uses the original paper's
  hyperparameters verbatim (as verified in the port repos).
- **Visitors get a budgeted grid.** Every off-diagonal cell reports the best
  of a small fixed grid (<= 8 configurations) around the problem's scale —
  so a visitor losing is meaningful, not just untuned.
- **Low-precision ground rules.** On the 8-bit problem, visitors' positions
  are stochastically quantized to the same fixed-point grid each step (the
  naive treatment); the home method replaces that with variance-corrected
  quantization. AMAGOLD quantizes between outer steps only (an "-F"
  treatment; its inner leapfrog is full-precision).

Results
-------

<!-- GAUNTLET-TABLE-BEGIN -->
| problem | origin | metric | sgld | psgld | sghmc | csgld | csghmc | amagold | vc_lp_sgld |
|---|---|---|---|---|---|---|---|---|---|
| doublewell_sghmc | SGHMC (Chen et al. 2014) | density_l1 | 0.01621 | 0.04037 | **0.0193** | 0.03504 | 0.02548 | 0.01054 ⭐ | 0.3012 |
| gaussian2d | SGHMC (Chen et al. 2014) | cov_err | 0.009223 | 0.06389 | **0.007221** | 0.00648 | 0.03225 | 0.1918 | 0.006134 ⭐ |
| mog25 | cSG-MCMC (Zhang et al. 2020) | mode_tv | 0.8348 | 0.88 | 0.96 | **0.4952** ⭐ | 0.88 | 0.96 | 0.5241 |
| lp_gaussian | low-precision SGLD (Zhang et al. 2022) | std_err | 0.04836 | 0.1571 | 0.02073 | 0.3667 | 0.07744 | 0.0002023 ⭐ | **0.002301** |
| doublewell_amagold | AMAGOLD (Zhang et al. 2020) | density_l1 | 0.03902 | 0.0506 | 0.0204 ⭐ | 0.03776 | 0.03703 | **0.02996** | 0.3008 |
| mnist_amagold | AMAGOLD (Zhang et al. 2020) | test_err | 0.0343 | 0.0387 | 0.0318 | 0.0338 | 0.0291 | **0.0233** ⭐ | 0.0382 |
| mnist_sghmc | SGHMC (Chen et al. 2014) | test_err | 0.0235 | 0.4029 | **0.0161** ⭐ | 0.0497 | 0.0201 | 0.0357 | 0.0319 |

Bold = the paper's own demo run with the paper's hyperparameters (the diagonal); ⭐ = row winner; every other cell is the best of a small tuning grid at the same gradient budget. Lower is better for all metrics.
<!-- GAUNTLET-TABLE-END -->

Compute cost
------------

Wall-clock seconds for the best configuration of each cell (single chain,
includes JIT compile; toys on a laptop CPU, MNIST rows on an H100 — read
within rows, not across them):

| problem | device | sgld | psgld | sghmc | csgld | csghmc | amagold | vc_lp_sgld |
|---|---|---|---|---|---|---|---|---|
| doublewell_sghmc | M-series CPU | 9.2 | 9.7 | 9.5 | 8.3 | 8.0 | 3.9 | 23.6 |
| gaussian2d | M-series CPU | 4.4 | 4.6 | 4.6 | 5.1 | 5.0 | 2.4 | 10.1 |
| mog25 | M-series CPU | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.7 |
| lp_gaussian | M-series CPU | 2.6 | 2.9 | 2.6 | 2.9 | 2.7 | 1.3 | 9.3 |
| doublewell_amagold | M-series CPU | 2.8 | 2.7 | 2.7 | 2.5 | 2.5 | 1.3 | 7.0 |
| mnist_sghmc | H100 | 53.8 | 53.3 | 53.6 | 53.4 | 53.0 | 58.2 | 57.2 |
| mnist_amagold | H100 | 47.9 | 48.1 | 47.8 | 47.5 | 47.9 | 60.8 | 50.6 |

Two consistent patterns: at matched gradient budgets the SGLD/SGHMC-family
kernels cost the same (the gradient dominates; schedules and preconditioning
are free), **vc_lp_sgld pays ~2-3x on CPU** for simulated quantization
(dominated by the VC branch arithmetic — on real low-precision hardware this
inverts into a saving, which is the paper's point), and **AMAGOLD is fastest
per gradient on the toys** (its inner leapfrog fuses 10 gradients per scan
step) but pays its full-data M-H tax on MNIST (+15-25% wall vs the
SGLD-family cells at the same budget, plus the ~100x step-size cap noted
above — its real cost is statistical, not wall-clock).

Notes on the MNIST rows
-----------------------

- Both diagonals hold: SGHMC at its paper settings wins its Gibbs-hyperprior
  BNN (1.61% BMA test error), AMAGOLD at its paper settings wins its BNN
  (2.33% current-sample error, acceptance 0.18).
- AMAGOLD as a visitor needs two protocol courtesies its own paper grants it:
  an SGD warmup on random-init rows (its M-H rejects everything from a cold
  start) and a step-size grid in **its own regime** — its usable step is
  capped by leapfrog error against the full-data energy at roughly 1/100 of
  the SGLD-scale steps. Amortized M-H buys unbiasedness at the price of
  step size; the matrix makes that cost visible (3.57% vs home 1.61%).
- pSGLD's best grid point on the sigmoid BNN is far off (40% error): the
  RMSprop-preconditioned regime needs its own step scale, which the current
  grid misses — a known grid limitation, not a verdict on pSGLD.
- The current-sample evaluation of the AMAGOLD row is inherently noisy
  (+/- a few tenths of a percent); margins below that are not meaningful.
  Seed replicates are the next hardening step for both MNIST rows.

Problems
--------

| problem | origin | what it stresses |
|---|---|---|
| `doublewell_sghmc` | SGHMC (Chen et al. 2014, fig. 1) | stochastic-gradient noise without M-H |
| `gaussian2d` | SGHMC (Chen et al. 2014, fig. 3) | correlated targets, sampler bias |
| `mog25` | cSG-MCMC (Zhang et al. 2020) | multimodality / mode coverage |
| `lp_gaussian` | low-precision SGLD (Zhang et al. 2022) | 8-bit fixed-point storage |
| `doublewell_amagold` | AMAGOLD (Zhang et al. 2020) | large step sizes, skewed target |

Planned rows (cluster): MNIST BNNs (AMAGOLD's 784-500-256-10 and ML-SGHMC's
784-100-10 + Gibbs hyperpriors), ml-1m matrix factorization (SGHMC),
CIFAR-10 ResNet18 BMA (cSG-MCMC / lp-SGLD).

Run it
------

```bash
pip install -e .
python -m gauntlet.run                    # all cells (~20-40 min on a laptop CPU)
python -m gauntlet.table --update-readme  # render the matrix into this README
JAX_PLATFORMS=cpu python tests/test_smoke.py
```
