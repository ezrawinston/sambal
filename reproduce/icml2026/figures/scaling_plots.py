"""Plotting functions for Figures 4 and 5 (BLiMP against model size and data
fraction on the scaling grid).

Input is the parsed scaling record {arm: {size: {fraction: [{seed, syntaxgym,
blimp}, ...]}}} — the committed records/scaling/scaling_results.json, or the
output of assemble/assemble_scaling_results.py on a batch of job logs.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    # "text.usetex": True,            # Use LaTeX to render text
    "font.family": "serif",         # Use serif fonts (like Computer Modern)
    # "font.serif": ["Computer Modern Roman"],
    "font.size": 7,                # Match your document's font size (usually 10pt or 11pt)
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
        # DATA LINES & MARKERS
    "lines.linewidth": 1.0,  # Thickness of the plot lines
    "lines.markersize": 4.0,  # Size of the markers (dots, squares, etc.)

    # THE BOX AROUND THE CHART (Spines)
    "axes.linewidth": 1,  # Thickness of the 4 edges (spines)

    # TICK MARKS
    "xtick.major.width": 1,  # Thickness of X-axis major ticks
    "ytick.major.width": 1,  # Thickness of Y-axis major ticks
    "xtick.minor.width": 0.8,  # Thickness of X-axis minor ticks (if used)
    "ytick.minor.width": 0.8,  # Thickness of Y-axis minor ticks (if used)
    "figure.figsize": (3.5, 2.5)    # Default size (width, height) in inches
})


def _ensure_pdf_path(output_path: Path) -> Path:
    """Make sure grouped bar plots are saved as PDF."""
    if output_path.suffix.lower() != ".pdf":
        return output_path.with_suffix(".pdf")
    return output_path


def _format_fraction_label(frac: float) -> str:
    """Format fraction as a percentage string."""
    percent = frac * 100
    if float(int(percent)) == percent:
        return f"{int(percent)}%"
    return f"{percent:.1f}%"


def plot_line_by_size(results: Dict, metric: str, output_path: Path):
    """Create line plot with 6 lines: 3 sizes x 2 methods.

    X-axis: data fraction (0.05 to 0.3)
    Y-axis: metric performance
    Lines: one per (method, size) combination
    Points: individual seed values overlayed
    """
    methods = ["Baseline", "SAMBAL"]
    sizes = ["5m", "10m", "14m"]

    # Set y-axis minimum based on metric
    y_min = 50.0 if metric == "blimp" else 28.0

    fig, ax = plt.subplots()#figsize=(10, 6))

    # Color and style definitions
    # method_colors = {"standard": "#1f77b4", "sambal": "#ff7f0e"}
    method_colors = {"Baseline": "#1f77b4", "SAMBAL": "#ff7f0e"}
    # method_colors = {"Baseline": "#1a9850", "SAMBAL": "#e31a1c"}
    # method_colors = {"Baseline": "#66c2a5", "SAMBAL": "#d53e4f"}
    size_linestyles = {"5m": "-", "10m": "--", "14m": ":"}
    size_markers = {"5m": "o", "10m": "s", "14m": "^"}

    for method in methods:
        for size in sizes:
            if method not in results or size not in results[method]:
                continue

            size_data = results[method][size]

            # Filter to fractions 0.05 to 0.3
            fractions = sorted([f for f in size_data.keys() if 0.05 <= f <= 0.3])
            if not fractions:
                continue

            means = []
            all_points_x = []
            all_points_y = []

            for frac in fractions:
                entries = size_data[frac]
                values = [e[metric] for e in entries if e[metric] is not None]
                if values:
                    means.append(np.mean(values))
                    # Collect individual points for overlay
                    for val in values:
                        all_points_x.append(frac)
                        all_points_y.append(val)
                else:
                    means.append(np.nan)

            # Plot the mean line
            label = f"{method} {size}"
            ax.plot(fractions, means,
                   color=method_colors[method],
                   linestyle=size_linestyles[size],
                   marker=size_markers[size],
                   label=label)

            # # Overlay individual seed points (smaller, semi-transparent)
            # ax.scatter(all_points_x, all_points_y,
            #           color=method_colors[method],
            #           marker=size_markers[size],
            #           s=20,
            #           alpha=0.4,
            #           edgecolors='none')

    ax.set_xlabel("data fraction")
    ax.set_ylabel(f"{metric.upper()} accuracy (%)")
    # ax.set_title(f"{metric.upper()} Performance by Model Size and Method", fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=y_min)
    ax.set_xlim(0.04, 0.31)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_grouped_bars_by_size(results: Dict, metric: str, output_path: Path):
    """Create grouped bar plot with model sizes on x-axis and fractions within each group.

    X-axis: model sizes (5m, 14m, 30m)
    Groups: data fractions (only those available for all sizes)
    Colors: Baseline blue and Sambal orange
    Alpha: data fraction encoded via bar transparency
    Layout: standard and sambal bars adjacent within each fraction group
    """
    methods = ["Baseline", "SAMBAL"]
    sizes = ["5m", "14m", "30m"]

    # Find fractions that exist for all sizes (for both methods combined)
    fractions_per_size = {}
    for size in sizes:
        fracs = set()
        for method in methods:
            if method in results and size in results[method]:
                fracs.update(results[method][size].keys())
        fractions_per_size[size] = fracs

    # Get intersection of all fractions
    common_fractions = fractions_per_size[sizes[0]]
    for size in sizes[1:]:
        common_fractions = common_fractions & fractions_per_size[size]

    common_fractions = sorted(common_fractions)
    if not common_fractions:
        print(f"No common fractions found for sizes {sizes}")
        return

    print(f"Common fractions for {sizes}: {common_fractions}")

    # Set y-axis minimum based on metric
    y_min = 50.0 if metric == "blimp" else 28.0

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    method_colors = {"Baseline": "#1f77b4", "SAMBAL": "#ff7f0e"}
    n_sizes = len(sizes)
    n_fractions = len(common_fractions)
    n_methods = len(methods)
    bar_width = 0.8 / (n_fractions * n_methods)
    x_positions = np.arange(n_sizes)

    alpha_levels = np.linspace(0.3, 1.0, n_fractions if n_fractions > 1 else 1)
    frac_to_alpha = {frac: alpha_levels[i] for i, frac in enumerate(common_fractions)}

    label_size = plt.rcParams["axes.labelsize"]
    xtick_size = plt.rcParams["xtick.labelsize"]
    ytick_size = plt.rcParams["ytick.labelsize"]
    legend_size = plt.rcParams["legend.fontsize"] * 0.75
    annot_size = plt.rcParams["font.size"] * 0.75

    for method_idx, method in enumerate(methods):
        for frac_idx, frac in enumerate(common_fractions):
            means = []
            stds = []
            seeds_list = []

            for size in sizes:
                if method in results and size in results[method] and frac in results[method][size]:
                    entries = results[method][size][frac]
                    values = [e[metric] for e in entries if e[metric] is not None]
                    seeds = [e["seed"] for e in entries if e[metric] is not None]
                    if values:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                        seeds_list.append(seeds)
                    else:
                        means.append(0)
                        stds.append(0)
                        seeds_list.append([])
                else:
                    means.append(0)
                    stds.append(0)
                    seeds_list.append([])

            # Calculate bar position: group by method, then fraction within method
            offset = (method_idx * n_fractions + frac_idx - (n_fractions * n_methods - 1) / 2) * bar_width

            bars = ax.bar(x_positions + offset, means, bar_width,
                         yerr=stds, capsize=2,
                         color=method_colors.get(method, "#cccccc"),
                         edgecolor='black', linewidth=0.7,
                         alpha=frac_to_alpha[frac],
                         error_kw={"elinewidth": 0.2, "capthick": 0.25})

            # Add average annotations above each bar
            for j, (bar, mean_val) in enumerate(zip(bars, means)):
                if mean_val > 0:
                    ax.annotate(f"{mean_val:.1f}",
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[j] + 0.3),
                               ha='center', va='bottom', fontsize=annot_size, rotation=90)

    ax.set_xlabel("model size", fontsize=label_size)
    y_label = "BLiMP accuracy (%)" if metric == "blimp" else f"{metric.upper()} Accuracy (%)"
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sizes, fontsize=xtick_size*0.8)
    ax.set_ylim(bottom=55)
    ax.tick_params(axis='y', labelsize=ytick_size)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Create custom legend
    from matplotlib.patches import Patch
    method_handles = [
        Patch(facecolor=method_colors[m], edgecolor='none', label=m)
        for m in methods
    ]
    fraction_handles = [
        Patch(facecolor='gray', alpha=frac_to_alpha[frac], label=_format_fraction_label(frac), edgecolor='none')
        for frac in common_fractions
    ]

    method_legend = ax.legend(handles=method_handles, loc='upper left',
                              fontsize=legend_size, title_fontsize=legend_size)
    ax.add_artist(method_legend)
    frac_ncol = len(common_fractions) if len(common_fractions) <= 3 else 3
    extra_handle = Line2D([], [], linestyle="None")
    legend_handles = [extra_handle] + fraction_handles
    legend_labels = ["data    "] + [_format_fraction_label(frac) for frac in common_fractions]
    ax.legend(handles=legend_handles, labels=legend_labels,
              loc='upper right', fontsize=legend_size,
              ncol=frac_ncol + 1, bbox_to_anchor=(0.99, 0.99),
              borderaxespad=0.0, handletextpad=0.2, columnspacing=0.6)

    plt.tight_layout()
    save_path = _ensure_pdf_path(output_path)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Saved {save_path}")


def plot_grouped_bars_small_sizes(results: Dict, metric: str, output_path: Path):
    """Create grouped bar plot with small model sizes (5m, 10m, 14m) and fractions 0.05-0.3.

    X-axis: model sizes (5m, 10m, 14m)
    Groups: data fractions (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    Colors: Baseline blue and Sambal orange
    Alpha: data fraction encoded via bar transparency
    Layout: standard and sambal bars grouped by method within each size
    """
    methods = ["Baseline", "SAMBAL"]
    sizes = ["5m", "10m", "14m"]
    target_fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Filter to fractions that exist for all sizes
    common_fractions = []
    for frac in target_fractions:
        has_data = True
        for size in sizes:
            size_has_frac = False
            for method in methods:
                if method in results and size in results[method] and frac in results[method][size]:
                    size_has_frac = True
                    break
            if not size_has_frac:
                has_data = False
                break
        if has_data:
            common_fractions.append(frac)

    if not common_fractions:
        print(f"No common fractions found for sizes {sizes} in range 0.05-0.3")
        return

    print(f"Common fractions for {sizes}: {common_fractions}")

    # Set y-axis minimum based on metric
    y_min = 50.0 if metric == "blimp" else 28.0

    fig, ax = plt.subplots(figsize=(7, 3.5))

    method_colors = {"Baseline": "#1f77b4", "SAMBAL": "#ff7f0e"}
    n_sizes = len(sizes)
    n_fractions = len(common_fractions)
    n_methods = len(methods)
    bar_width = 0.8 / (n_fractions * n_methods)
    x_positions = np.arange(n_sizes)

    alpha_levels = np.linspace(0.3, 1.0, n_fractions if n_fractions > 1 else 1)
    frac_to_alpha = {frac: alpha_levels[i] for i, frac in enumerate(common_fractions)}

    label_size = plt.rcParams["axes.labelsize"]
    xtick_size = plt.rcParams["xtick.labelsize"]
    ytick_size = plt.rcParams["ytick.labelsize"]
    legend_size = plt.rcParams["legend.fontsize"] * 0.75
    annot_size = plt.rcParams["font.size"] * 0.75

    for method_idx, method in enumerate(methods):
        for frac_idx, frac in enumerate(common_fractions):
            means = []
            stds = []

            for size in sizes:
                if method in results and size in results[method] and frac in results[method][size]:
                    entries = results[method][size][frac]
                    values = [e[metric] for e in entries if e[metric] is not None]
                    if values:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            # Calculate bar position: group by method, then fraction within method
            offset = (method_idx * n_fractions + frac_idx - (n_fractions * n_methods - 1) / 2) * bar_width

            bars = ax.bar(x_positions + offset, means, bar_width,
                          yerr=stds, capsize=2,
                          color=method_colors.get(method, "#cccccc"),
                          edgecolor='black', linewidth=0.7,
                          alpha=frac_to_alpha[frac],
                          error_kw={"elinewidth": 0.2, "capthick": 0.25})

            # Add average annotations above each bar
            for j, (bar, mean_val) in enumerate(zip(bars, means)):
                if mean_val > 0:
                    ax.annotate(f"{mean_val:.1f}",
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[j] + 0.3),
                               ha='center', va='bottom', fontsize=annot_size, rotation=90)

    ax.set_xlabel("model size", fontsize=label_size)
    y_label = "BLiMP accuracy (%)" if metric == "blimp" else f"{metric.upper()} Accuracy (%)"
    ax.set_ylabel(y_label, fontsize=label_size)
    # Title intentionally omitted for cleaner layout
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sizes, fontsize=xtick_size)
    ax.set_ylim(bottom=y_min)
    ax.tick_params(axis='y', labelsize=ytick_size)
    ax.grid(True, alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    method_handles = [
        Patch(facecolor=method_colors[m], edgecolor='none', label=m)
        for m in methods
    ]
    fraction_handles = [
        Patch(facecolor='gray', alpha=frac_to_alpha[frac], label=_format_fraction_label(frac), edgecolor='none')
        for frac in common_fractions
    ]

    method_legend = ax.legend(handles=method_handles, loc='upper left',
                              fontsize=legend_size, title_fontsize=legend_size)
    ax.add_artist(method_legend)
    frac_ncol = len(common_fractions) if len(common_fractions) <= 3 else 3
    shift_inches = 0.75
    fig_width = fig.get_size_inches()[0]
    shift_frac = min(max(shift_inches / fig_width, 0.0), 1.0)
    bbox = (1 - shift_frac, 0.99)
    ax.legend(handles=fraction_handles, loc='upper right',
              fontsize=legend_size, title="data fraction", title_fontsize=legend_size,
              ncol=frac_ncol, bbox_to_anchor=bbox, borderaxespad=0.0, handletextpad=0.2)

    plt.tight_layout()
    save_path = _ensure_pdf_path(output_path)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Saved {save_path}")
