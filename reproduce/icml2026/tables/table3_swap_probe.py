#!/usr/bin/env python3
"""Table 3 — swap probe: syntax vs semantic-plausibility preferences.

Computes accuracies and mean margins for both axes and both models from the
committed probe output (records/swap_probe/paired_probe.json), prints the
table, and writes a CSV to reproduce/icml2026/out/tables/.

The probe emits `knowledge_margin` and `grammar_margin` under every flag
combination, but what those two numbers *mean* is set by the run's flags:
`--counterbalance_order` decides whether ΔK is one clause order or the mean of
both, and `--grammar_mode` decides which sentence sets ΔG compares. A probe
json produced under other flags is therefore structurally valid and silently
yields a different table. The shared `records_io.check_swap_probe_protocol`
rejects any record not produced under the expected protocol.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, check_swap_probe_protocol, load_json  # noqa: E402

RESULTS = "swap_probe/paired_probe.json"
OUT = TABLES_OUT

def stats(rows):
    k = [float(r["knowledge_margin"]) for r in rows if r.get("knowledge_margin") is not None]
    g = [float(r["grammar_margin"]) for r in rows if r.get("grammar_margin") is not None]
    return {
        "syntax_acc": 100.0 * sum(x > 0 for x in g) / len(g),
        "syntax_margin": sum(g) / len(g),
        "semantic_acc": 100.0 * sum(x > 0 for x in k) / len(k),
        "semantic_margin": sum(k) / len(k),
        "n_syntax": len(g),
        "n_semantic": len(k),
    }


def main():
    data = load_json(RESULTS)
    check_swap_probe_protocol(data, RESULTS)
    dp = data["dual_pair"]
    table = [
        ("baseline (Long)", stats(dp["normal"]["rows"])),
        ("SAMBAL (Long)", stats(dp["sambal"]["rows"])),
    ]

    print(f"{'Model':<18} {'syn Acc%':>9} {'dG':>7} {'sem Acc%':>9} {'dK':>7}   (n_syn, n_sem)")
    for name, s in table:
        print(f"{name:<18} {s['syntax_acc']:>9.1f} {s['syntax_margin']:>7.2f} "
              f"{s['semantic_acc']:>9.1f} {s['semantic_margin']:>7.2f}   "
              f"({s['n_syntax']}, {s['n_semantic']})")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table3_swap_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "syntax_acc", "syntax_margin_dG",
                    "semantic_acc", "semantic_margin_dK", "n_syntax", "n_semantic"])
        for name, s in table:
            w.writerow([name, f"{s['syntax_acc']:.1f}", f"{s['syntax_margin']:.2f}",
                        f"{s['semantic_acc']:.1f}", f"{s['semantic_margin']:.2f}",
                        s["n_syntax"], s["n_semantic"]])
    print(f"\nwrote {OUT}/table3_swap_probe.csv")


if __name__ == "__main__":
    main()
