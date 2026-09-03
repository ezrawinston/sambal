#!/usr/bin/env python3
"""Table 8 — SNLI probe results (accuracy %).

Reads the committed probe metrics (records/snli/*_metrics.json) for the full
test set, and recomputes SNLI-hard accuracy from the committed per-item
predictions (records/snli/*_predictions_test_labels.tsv) restricted to the
official SNLI hard-subset pair ids (records/snli/test_hard_pair_ids.txt).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json, load_text, record_path  # noqa: E402

RES = "snli"
OUT = TABLES_OUT


def hard_accuracy(arm, hard_ids):
    path = record_path(f"{RES}/{arm}_predictions_test_labels.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    tot = cor = htot = hcor = 0
    for r in rows:
        ok = r["label"] == r["pred_label"]
        tot += 1
        cor += ok
        if r["pair_id"] in hard_ids:
            htot += 1
            hcor += ok
    return 100.0 * cor / tot, 100.0 * hcor / htot, tot, htot


def main():
    hard_ids = set(load_text(f"{RES}/test_hard_pair_ids.txt").split())
    rows = []
    for arm, label in (
        ("baseline", "baseline"),
        ("sambal", "SAMBAL"),
        ("sambal_no_ctx", "SAMBAL (no ctx. samp.)"),
    ):
        metrics = load_json(f"{RES}/{arm}_metrics.json")
        full = 100.0 * metrics["test"]["accuracy"]
        full_re, hard, n, nh = hard_accuracy(arm, hard_ids)
        assert abs(full - full_re) < 0.005, (arm, full, full_re)
        rows.append((label, f"{full:.1f}", f"{hard:.1f}"))
    rows.append(("chance", "33.3", "33.3"))

    print(f"{'Model':<24} {'SNLI':>6} {'SNLI-hard':>10}")
    for name, full, hard in rows:
        print(f"{name:<24} {full:>6} {hard:>10}")
    print(f"\n(full test n={n}, hard subset n={nh})")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table8_snli.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "snli", "snli_hard"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT}/table8_snli.csv")


if __name__ == "__main__":
    main()
