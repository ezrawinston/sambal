#!/usr/bin/env python3
"""Figure 6 — SAMBAL's BLiMP advantage vs data-utilization.

Each point is one (data budget, model size) configuration; x = the baseline's
accuracy as a percentage of the best SAMBAL accuracy at the same data budget,
y = SAMBAL's advantage in percentage points.

Per-configuration accuracies are the 3-seed mean end-of-training BLiMP for
each arm, read from records/scaling/scaling_results.json and rounded to one
decimal (the precision the figure is drawn at).
"""
import statistics
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import FIGURES_OUT, load_json  # noqa: E402

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

RESULTS_JSON = "scaling/scaling_results.json"
OUT = FIGURES_OUT

# The (data budget, model size) configurations the figure plots, in order.
CELLS = [
    ("5%", "5M"), ("5%", "10M"), ("5%", "14M"),
    ("10%", "5M"), ("10%", "10M"), ("10%", "14M"), ("10%", "30M"),
    ("15%", "5M"), ("15%", "10M"), ("15%", "14M"),
    ("20%", "5M"), ("20%", "10M"), ("20%", "14M"),
    ("25%", "5M"), ("25%", "10M"), ("25%", "14M"),
    ("30%", "5M"), ("30%", "10M"), ("30%", "14M"),
    ("50%", "5M"), ("50%", "14M"), ("50%", "30M"),
    ("100%", "5M"), ("100%", "14M"), ("100%", "30M"),
]
ARM_KEY = {"baseline": "Baseline", "sambal": "SAMBAL"}


def load_configs():
    """(data budget, model size, baseline BLiMP, SAMBAL BLiMP) per cell.

    Each accuracy is the mean end-of-training BLiMP over that cell's seeds in
    the committed scaling record, rounded to the figure's one-decimal precision.
    """
    scaling = load_json(RESULTS_JSON)

    def mean_blimp(arm, budget, size):
        frac = str(float(budget.strip("%")) / 100.0)
        runs = scaling[ARM_KEY[arm]][size.lower()][frac]
        return round(statistics.mean(r["blimp"] for r in runs), 1)

    return [(d, m, mean_blimp("baseline", d, m), mean_blimp("sambal", d, m))
            for d, m in CELLS]


configs = load_configs()

budgets = sorted(set(c[0] for c in configs), key=lambda x: float(x.strip("%")))
ceiling = {d: max(c[3] for c in configs if c[0] == d) for d in budgets}
budget_color = {d: matplotlib.cm.viridis(i / (len(budgets) - 1)) for i, d in enumerate(budgets)}
size_marker = {"5M": "o", "10M": "s", "14M": "^", "30M": "D"}

xs, ys = [], []
fig, ax = plt.subplots()
for d, m, b, s in configs:
    x = b / ceiling[d] * 100.0
    y = s - b
    xs.append(x)
    ys.append(y)
    ax.scatter(x, y, color=budget_color[d], marker=size_marker[m],
               s=18, zorder=3, edgecolors="none")
xs = np.array(xs)
ys = np.array(ys)

slope, intercept = np.polyfit(xs, ys, 1)
r = float(np.corrcoef(xs, ys)[0, 1])
xl = np.array([xs.min() - 0.7, xs.max() + 0.7])
fit_color = "0.35"
ax.plot(xl, slope * xl + intercept, color=fit_color, zorder=2)
ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray", zorder=1)

ax.set_xlabel("Data-utilization: baseline accuracy as % of\n"
              "max SAMBAL accuracy at same data budget")
ax.set_ylabel("SAMBAL BLiMP advantage (pp)")
ax.set_xlim(xs.min() - 1.2, xs.max() + 1.8)
ax.set_ylim(-1.3, 3.8)

budget_handles = [Line2D([], [], marker="o", linestyle="None",
                         markerfacecolor=budget_color[d], markeredgecolor="none",
                         label=d) for d in budgets]
budget_legend = ax.legend(handles=budget_handles, title="data budget",
                          loc="upper right", ncol=2, handletextpad=0.25,
                          columnspacing=0.6, labelspacing=0.25, borderpad=0.3,
                          handlelength=1.2, fontsize=5.5, title_fontsize=6)
ax.add_artist(budget_legend)
size_handles = [Line2D([], [], marker=size_marker[sz], linestyle="None",
                       markerfacecolor=fit_color, markeredgecolor="none",
                       label=sz) for sz in ["5M", "10M", "14M", "30M"]]
fit_handle = Line2D([], [], color=fit_color,
                    label=f"linear fit ($R^2$={r*r:.2f}, $r$={r:.2f})")
ax.legend(handles=size_handles + [fit_handle], loc="lower left",
          handletextpad=0.3, labelspacing=0.25, borderpad=0.3,
          handlelength=1.4, fontsize=5.5)

plt.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / "efficiency_scatter.pdf", dpi=200)
plt.savefig(OUT / "efficiency_scatter.png", dpi=200)
print(f"wrote {OUT}/efficiency_scatter.{{pdf,png}}  (fit R^2={r*r:.2f}, r={r:.2f})")
