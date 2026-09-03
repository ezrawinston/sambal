#!/usr/bin/env python3
"""Figure 1 — swap-probe scatter: semantics margin ΔK vs syntax margin ΔG.

Reads the committed probe output (records/swap_probe/paired_probe.json,
produced by evals/swap_probe/paired_knowledge_probe.py) and renders the
scatter with the same geometry the probe script uses.
"""
import sys
from pathlib import Path

from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import FIGURES_OUT, check_swap_probe_protocol, load_json  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "axes.linewidth": 1,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "figure.figsize": (3.5, 2.5),
})

RESULTS = "swap_probe/paired_probe.json"
OUT = FIGURES_OUT


def main():
    data = load_json(RESULTS)
    check_swap_probe_protocol(data, RESULTS)
    dp = data["dual_pair"]
    f_pts = [(float(r["knowledge_margin"]), float(r["grammar_margin"]))
             for r in dp["sambal"]["rows"] if r.get("grammar_margin") is not None]
    n_pts = [(float(r["knowledge_margin"]), float(r["grammar_margin"]))
             for r in dp["normal"]["rows"] if r.get("grammar_margin") is not None]
    fx, fy = zip(*f_pts)
    nx, ny = zip(*n_pts)

    plt.figure()
    plt.scatter(nx, ny, alpha=0.7, label="Normal LM")
    plt.scatter(fx, fy, alpha=0.7, label="SAMBAL")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Semantics margin ΔK ")
    plt.ylabel("Syntax margin ΔG")
    plt.legend()
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"paired_probe_scatter.{ext}", dpi=200)
    plt.close()
    print(f"wrote {OUT}/paired_probe_scatter.{{pdf,png}} "
          f"({len(n_pts)} baseline pts, {len(f_pts)} SAMBAL pts)")


if __name__ == "__main__":
    main()
