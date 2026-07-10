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

Bold = the paper's own demo run with the paper's hyperparameters (the diagonal); ⭐ = row winner; every other cell is the best of a small tuning grid at the same gradient budget. Lower is better for all metrics.
<!-- GAUNTLET-TABLE-END -->

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
