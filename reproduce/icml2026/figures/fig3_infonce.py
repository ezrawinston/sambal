#!/usr/bin/env python3
"""Figure 3 — per-layer InfoNCE alignment difference (UPOS − Lex).

Runs the comparison plotter (evals/infonce/compare_infonce.py --diff)
on the committed per-arm InfoNCE outputs
(records/infonce/{gptbert,sambal}_infonce_hidden.json, produced by
evals/infonce/run_infonce.py).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import FIGURES_OUT, record_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = FIGURES_OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "evals/infonce/compare_infonce.py"),
        "--A", str(record_path("infonce/gptbert_infonce_hidden.json")),
        "--labelA", "Baseline",
        "--B", str(record_path("infonce/sambal_infonce_hidden.json")),
        "--labelB", "SAMBAL",
        "--diff",
        "--out_png", str(OUT / "infonce_compare_hidden_diff.png"),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT}/infonce_compare_hidden_diff.png")


if __name__ == "__main__":
    main()
