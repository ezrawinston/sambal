import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True); ap.add_argument("--labelA", required=True)
    ap.add_argument("--B", required=True); ap.add_argument("--labelB", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--ratio", action="store_true", help="Plot UPOS/Lex ratio instead of raw values")
    ap.add_argument("--diff", action="store_true", help="Plot UPOS-Lex difference instead of raw values")
    args = ap.parse_args()

    if args.ratio and args.diff:
        ap.error("Cannot use --ratio and --diff together")

    A = json.loads(Path(args.A).read_text())
    B = json.loads(Path(args.B).read_text())

    layers = list(range(1,13))
    # assert layers == B["layers"], "Layer lists differ; re‑run with same --layers"

    plt.figure()

    if args.ratio:
        # Compute ratios: infonce_upos / infonce_lex for each layer
        ratio_A = [u / l if l != 0 else float('nan') for u, l in zip(A["infonce_upos"], A["infonce_lex"])]
        ratio_B = [u / l if l != 0 else float('nan') for u, l in zip(B["infonce_upos"], B["infonce_lex"])]

        plt.plot(layers, ratio_A, marker="o", label=args.labelA)
        plt.plot(layers, ratio_B, marker="o", label=args.labelB)
        plt.xlabel("layer")
        plt.ylabel("UPOS/Lex ratio")
        # plt.title("Per‑layer InfoNCE UPOS/Lex ratio comparison")
        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    elif args.diff:
        # Compute differences: infonce_upos - infonce_lex for each layer
        diff_A = [u - l for u, l in zip(A["infonce_upos"], A["infonce_lex"])][::-1]
        diff_B = [u - l for u, l in zip(B["infonce_upos"], B["infonce_lex"])][::-1]

        plt.plot(layers, diff_A, marker="o", label=args.labelA)
        plt.plot(layers, diff_B, marker="o", label=args.labelB)
        plt.xlabel("layer")
        plt.ylabel("UPOS − Lex difference")
        plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    else:
        plt.plot(layers, A["infonce_upos"], marker="o", label=f"UPOS — {args.labelA}")
        plt.plot(layers, B["infonce_upos"], marker="o", label=f"UPOS — {args.labelB}")
        plt.plot(layers, A["infonce_lex"], marker="o", label=f"Lex — {args.labelA}")
        plt.plot(layers, B["infonce_lex"], marker="o", label=f"Lex — {args.labelB}")
        plt.xlabel("Layer index (−1=last)")
        plt.ylabel("Score (higher=better)")
        plt.title("Per‑layer InfoNCE comparison")

    plt.legend(); plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png); plt.close()

if __name__ == "__main__":
    main()
