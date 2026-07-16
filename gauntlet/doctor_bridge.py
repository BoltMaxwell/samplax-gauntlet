"""Bridge to the mcmc-doctor skill's diagnostics (optional dependency).

The skill lives at ~/.claude/skills/mcmc-doctor and is not a package; load
its script by path. When absent, chain diagnostics are silently skipped so
the gauntlet stays self-contained.
"""

import importlib.util
import os

import numpy as np

_SKILL = os.path.expanduser(
    "~/.claude/skills/mcmc-doctor/scripts/mcmc_doctor.py")


def _load():
    if not os.path.exists(_SKILL):
        return None
    spec = importlib.util.spec_from_file_location("mcmc_doctor", _SKILL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MD = _load()


def chain_diagnostics(samples):
    """min bulk ESS, max R-hat (split, single chain), max tau for (N, D) samples.

    Returns {} when the skill is unavailable or the chain is too short.
    """
    if _MD is None or samples.shape[0] < 16:
        return {}
    arr = np.asarray(samples, np.float64)[None, ...]  # (1, N, D)
    worst, _ = _MD.diagnose_panel(arr)
    return {"ess_bulk_min": round(worst["ess_bulk_min"], 1),
            "rhat_split_max": round(worst["rhat_max"], 4),
            "tau_max": round(worst["tau_max"], 1)}


def save_chain(path, samples, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, samples=np.asarray(samples, np.float32),
                        **{k: v for k, v in meta.items() if v is not None})
