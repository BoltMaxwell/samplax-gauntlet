"""Render the gauntlet matrix from results/ as a markdown table.

Rows = problems (with their origin paper and primary metric), columns =
samplers. Each cell shows the best configuration's primary metric (lower is
better). The home cell (the paper's own demo, run with the paper's
hyperparameters) is bold; the row winner gets a star.

Usage: python -m gauntlet.table [--results results] [--update-readme]
"""

import argparse
import glob
import json
import os

from .problems import PROBLEMS
from .samplers import SAMPLERS

MARKER_BEGIN = "<!-- GAUNTLET-TABLE-BEGIN -->"
MARKER_END = "<!-- GAUNTLET-TABLE-END -->"


def _best(payload):
    best_name, best_val = None, None
    for cname, r in payload["configs"].items():
        m = r["metrics"]
        if "error" in m:
            continue
        val = next(iter(m.values()))
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_name, best_val = cname, val
    return best_name, best_val


def render(results_dir="results"):
    lines = []
    header = ["problem", "origin", "metric"] + list(SAMPLERS)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    known = list(PROBLEMS)
    extra = sorted(d for d in os.listdir(results_dir)
                   if os.path.isdir(os.path.join(results_dir, d)) and d not in known)
    origins = {}
    for pname in known + extra:
        cells, vals = {}, {}
        for path in glob.glob(os.path.join(results_dir, pname, "*.json")):
            payload = json.load(open(path))
            cname, val = _best(payload)
            origins[pname] = payload["origin"]
            if val is not None:
                cells[payload["sampler"]] = (cname, val, payload["is_home"])
                vals[payload["sampler"]] = val
        if not cells:
            continue
        winner = min(vals, key=vals.get) if vals else None
        row = []
        for sname in SAMPLERS:
            if sname not in cells:
                row.append("—")
                continue
            cname, val, is_home = cells[sname]
            txt = f"{val:.4g}"
            if is_home:
                txt = f"**{txt}**"
            if sname == winner:
                txt += " ⭐"
            row.append(txt)
        any_payload = json.load(open(glob.glob(
            os.path.join(results_dir, pname, "*.json"))[0]))
        metric_name = next(iter(next(iter(
            any_payload["configs"].values()))["metrics"]))
        lines.append("| " + " | ".join(
            [pname, origins[pname], metric_name] + row) + " |")
    lines.append("")
    lines.append("Bold = the paper's own demo run with the paper's "
                 "hyperparameters (the diagonal); ⭐ = row winner; every other "
                 "cell is the best of a small tuning grid at the same "
                 "gradient budget. Lower is better for all metrics.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()
    table = render(args.results)
    print(table)
    if args.update_readme:
        with open("README.md") as f:
            text = f.read()
        pre, _, rest = text.partition(MARKER_BEGIN)
        _, _, post = rest.partition(MARKER_END)
        with open("README.md", "w") as f:
            f.write(pre + MARKER_BEGIN + "\n" + table + "\n" + MARKER_END + post)
        print("\nREADME.md updated")


if __name__ == "__main__":
    main()
