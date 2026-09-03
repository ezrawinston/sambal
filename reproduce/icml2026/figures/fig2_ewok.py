#!/usr/bin/env python3
"""Figure 2 — EWoK accuracy by suite (baseline vs SAMBAL).

Per-suite and overall accuracies are parsed from the committed EWoK evaluation
reports (records/ewok/{baseline,sambal}_report.txt, produced by
lm/gpt-bert/evaluation/ewok on the two arms' checkpoints). The dashed
per-suite lines mark the 95% binomial-confidence threshold above chance for
each suite's item count (recorded alongside the runs), and the dashed zero
line marks chance (50%).
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import FIGURES_OUT, load_text  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.8,
    "lines.markersize": 4.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "figure.figsize": (3.5, 2.5),
})
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

RES = "ewok"
OUT = FIGURES_OUT

DOMAINS = [
    "social-relations", "physical-relations", "agent-properties",
    "physical-interactions", "material-dynamics", "material-properties",
    "social-properties", "social-interactions", "quantitative-properties",
    "spatial-relations", "physical-dynamics",
]


def parse_report(rel):
    text = load_text(rel)
    domain_block = text.split("### DOMAIN ACCURACY")[1].split("###")[0]
    domains = dict(re.findall(r"([\w-]+): ([\d.]+)", domain_block))
    overall = float(re.search(r"### AVERAGE ACCURACY\s+([\d.]+)", text).group(1))
    return {k: float(v) for k, v in domains.items()}, overall


base_dom, base_overall = parse_report(f"{RES}/baseline_report.txt")
samb_dom, samb_overall = parse_report(f"{RES}/sambal_report.txt")

labels = [d.replace("physical", "phys.").replace("quantitative", "quant.").replace("-", " ")
          for d in DOMAINS]
baseline_vals = [base_dom[d] for d in DOMAINS]
sambal_vals = [samb_dom[d] for d in DOMAINS]
# Recorded 95% binomial-confidence thresholds against chance (50%), one per
# domain in DOMAINS order: each is the accuracy above which a binomial test at
# that domain's item count clears p<0.05. They depend on the per-domain item
# counts of the gated EWoK release, which this repo does not redistribute, so
# the values are recorded here rather than recomputed.
conf_percents = [53.50, 54.70, 52.90, 55.90, 53.50, 58.80, 57.20, 57.60, 57.00, 56.30, 61.30]

labels.append("overall")
sambal_vals.append(round(samb_overall, 1))
baseline_vals.append(round(base_overall, 1))
conf_percents.append(51.5)

plot_sambal = np.array(sambal_vals) - 50
plot_baseline = np.array(baseline_vals) - 50
plot_conf = np.array(conf_percents) - 50

fig, ax = plt.subplots(figsize=(3.5, 2.2))

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width / 2, plot_baseline, width, label="Baseline",
                color="#1f77b4", edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width / 2, plot_sambal, width, label="SAMBAL",
                color="#ff7f0e", edgecolor="black", linewidth=0.8)

for i in range(len(labels) - 1):
    rects1[i].set_alpha(0.8)
    rects2[i].set_alpha(0.8)
rects1[-1].set_alpha(1.0)
rects1[-1].set_linewidth(1)
rects2[-1].set_alpha(1.0)
rects2[-1].set_linewidth(1)

ax.axvline(x[-2] + 0.5, color="black", linestyle=":", linewidth=0.5)
for i in range(len(labels)):
    ax.hlines(y=plot_conf[i], xmin=x[i] - width, xmax=x[i] + width,
              colors="black", linestyles="dashed", linewidth=0.5)
ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
y_ticks = np.arange(-5, 16, 5)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{int(t + 50)}" for t in y_ticks])
ax.set_ylabel("Score")
ax.legend(loc="upper left")

plt.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / "ewok.pdf")
plt.savefig(OUT / "ewok.png", dpi=200)
print(f"wrote {OUT}/ewok.{{pdf,png}}")
