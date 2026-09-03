"""Generate the parameter-sweep model configs used to pick the scaling-grid
sizes.

Searches, for each 1M..30M parameter target, over hidden size and head
layout under a fixed depth schedule, scoring candidates by exact parameter
count (measured by instantiating the model) plus shape-regularization
penalties. Writes one config per target as {NN}m.json (zero-padded).

The four grid sizes the experiments consume ship canonically under
lm/gpt-bert/configs/scaling_configs/; this script documents and regenerates
their derivation. The recorded outcome of the sweep is in the comment block
at the bottom.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "lm/gpt-bert/pretraining"))
from model_extra import Bert  # noqa: E402

BASE = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "position_bucket_size": 32,
    "vocab_size": 8192,
    "layer_norm_eps": 1.0e-5,
}

HEAD_DIMS = (32, 48, 64)


def round_to_multiple(x: float, m: int) -> int:
    return int(m * round(x / m))


def ffn_size(hidden_size: int) -> int:
    # keep a ~3.33x feed-forward ratio
    return round_to_multiple(hidden_size * (10.0 / 3.0), 64)


def depth_schedule(m_million: int) -> int:
    """
    Smooth schedule from 6 to 12 as m goes 1..30.
    - min depth is 6
    - max depth is 12
    """
    m = max(1, min(30, m_million))
    L = int(round(6 + (m - 1) * (12 - 6) / (30 - 1)))
    return max(6, min(12, L))


def preferred_head_dim(hidden_size: int) -> int:
    """
    Let head_dim depend on size:
    - tiny widths: prefer 32
    - mid: prefer 48
    - large: prefer 64
    """
    if hidden_size <= 160:
        return 32
    elif hidden_size <= 320:
        return 48
    else:
        return 64


def head_candidates(hidden_size: int):
    """
    Return candidate (num_heads, head_dim, head_penalty).
    Constraints:
      - head_dim in {32,48,64}
      - num_heads integer
      - num_heads between 2 and 12 (keeps it reasonable)
    Soft preference:
      - prefer head_dim near preferred_head_dim(hidden_size)
      - slight preference for 'common' head counts
    """
    pref = preferred_head_dim(hidden_size)
    common_heads = {2, 4, 6, 8, 12, 3}  # allow 3; penalize weird counts slightly

    cands = []
    for hd in HEAD_DIMS:
        if hidden_size % hd != 0:
            continue
        nh = hidden_size // hd
        if nh < 2 or nh > 12:
            continue

        # penalty for head_dim far from preference
        hd_pen = 50_000 * abs(hd - pref)  # scaled in "param units"

        # small penalty for unusual head counts (e.g., 5,7,9,10,11)
        nh_pen = 0 if nh in common_heads else 80_000

        cands.append((nh, hd, hd_pen + nh_pen))

    return cands


def make_cfg(hidden_size: int, num_layers: int, num_heads: int) -> dict:
    cfg = dict(BASE)
    cfg["hidden_size"] = hidden_size
    cfg["num_hidden_layers"] = num_layers
    cfg["num_attention_heads"] = num_heads
    cfg["intermediate_size"] = ffn_size(hidden_size)
    return cfg


def count_params(cfg: dict) -> int:
    """Exact parameter count: instantiate the model and count."""
    model = Bert(SimpleNamespace(**cfg))
    n_params = sum(p.numel() for p in model.parameters())
    del model
    return n_params


def pick_reasonable_cfg_for_target(
    m_million: int,
    target_params: int,
    count_params_fn,
    prev_cfg = None,
    hidden_min: int = 64,
    hidden_max: int = 512,
    hidden_step: int = 16,
):
    """
    Search over hidden_size and head choices with a fixed depth schedule.
    Prioritizes: reasonable shapes > exact param matching.
    """
    L = depth_schedule(m_million)

    best = None
    prev_h = prev_cfg["hidden_size"] if prev_cfg else None
    prev_L = prev_cfg["num_hidden_layers"] if prev_cfg else None

    for h in range(hidden_min, hidden_max + 1, hidden_step):
        # must have at least one valid head config
        hcands = head_candidates(h)
        if not hcands:
            continue

        for (nh, hd, head_pen) in hcands:
            cfg = make_cfg(h, L, nh)
            p = count_params_fn(cfg)
            err = abs(p - target_params)

            # --- shape regularization penalties ---

            # discourage huge width jumps between adjacent targets
            jump_pen = 0
            if prev_h is not None:
                # allow small changes; penalize big discontinuities
                jump = abs(h - prev_h)
                jump_pen = 2_000 * max(0, jump - 32)  # no penalty up to 32

            # keep depth progression smooth (should already be, but just in case)
            depth_pen = 0
            if prev_L is not None:
                depth_pen = 150_000 * max(0, prev_L - L)  # strongly discourage decreasing depth

            # tiny extra penalty if model is extremely narrow for its depth
            skinny_pen = 0
            if L >= 10 and h < 128:
                skinny_pen = 250_000

            # IMPORTANT: we accept being off by a few hundred k if it buys shape
            # so we keep err weight at 1x and use penalties of similar magnitude
            score = err + head_pen + jump_pen + depth_pen + skinny_pen

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "err": err,
                    "params": p,
                    "cfg": cfg,
                    "h": h,
                    "L": L,
                    "nh": nh,
                    "hd": hd,
                    "head_pen": head_pen,
                    "jump_pen": jump_pen,
                    "depth_pen": depth_pen,
                    "skinny_pen": skinny_pen,
                }

    if best is None:
        raise RuntimeError(f"No valid config found for {m_million}M with given search bounds.")

    return best


def main(out_dir=Path(__file__).resolve().parents[2] / "lm/gpt-bert/configs/scaling_configs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prev_cfg = None

    for m in range(1, 31):
        target = m * 1_000_000

        best = pick_reasonable_cfg_for_target(
            m_million=m,
            target_params=target,
            count_params_fn=count_params,
            prev_cfg=prev_cfg,
            hidden_min=64,
            hidden_max=512,
            hidden_step=16,
        )

        cfg = best["cfg"]
        (out / f"{m:02d}m.json").write_text(json.dumps(cfg, indent=2))

        print(
            f"{m:02d}M target: got {best['params']/1e6:.3f}M "
            f"(err {best['err']/1e3:.1f}k) -> "
            f"h={best['h']} L={best['L']} heads={best['nh']} (hd={best['hd']}) "
            f"ff={cfg['intermediate_size']}"
        )

        prev_cfg = cfg

    print(f"\nWrote configs to: {out.resolve()}")


if __name__ == "__main__":
    main()


# Recorded sweep outcome:
# 01M target: got 0.887M (err 113.1k) -> h=64 L=6 heads=2 (hd=32) ff=192
# 02M target: got 1.643M (err 357.4k) -> h=96 L=6 heads=3 (hd=32) ff=320
# 03M target: got 2.609M (err 390.8k) -> h=128 L=6 heads=4 (hd=32) ff=448
# 04M target: got 3.977M (err 22.9k) -> h=160 L=7 heads=5 (hd=32) ff=512
# 05M target: got 5.508M (err 508.2k) -> h=192 L=7 heads=4 (hd=48) ff=640
# 06M target: got 5.508M (err 491.8k) -> h=192 L=7 heads=4 (hd=48) ff=640
# 07M target: got 7.943M (err 943.0k) -> h=240 L=7 heads=5 (hd=48) ff=768
# 08M target: got 7.943M (err 57.0k) -> h=240 L=7 heads=5 (hd=48) ff=768
# 09M target: got 8.785M (err 214.9k) -> h=240 L=8 heads=5 (hd=48) ff=768
# 10M target: got 9.931M (err 68.6k) -> h=256 L=8 heads=8 (hd=32) ff=832
# 11M target: got 12.434M (err 1434.4k) -> h=288 L=8 heads=6 (hd=48) ff=960
# 12M target: got 12.434M (err 434.4k) -> h=288 L=8 heads=6 (hd=48) ff=960
# 13M target: got 12.434M (err 565.6k) -> h=288 L=8 heads=6 (hd=48) ff=960
# 14M target: got 13.680M (err 320.0k) -> h=288 L=9 heads=6 (hd=48) ff=960
# 15M target: got 13.680M (err 1320.0k) -> h=288 L=9 heads=6 (hd=48) ff=960
# 16M target: got 16.776M (err 776.1k) -> h=320 L=9 heads=10 (hd=32) ff=1088
# 17M target: got 16.776M (err 223.9k) -> h=320 L=9 heads=10 (hd=32) ff=1088
# 18M target: got 18.334M (err 334.2k) -> h=320 L=10 heads=10 (hd=32) ff=1088
# 19M target: got 18.334M (err 665.8k) -> h=320 L=10 heads=10 (hd=32) ff=1088
# 20M target: got 20.170M (err 169.8k) -> h=336 L=10 heads=7 (hd=48) ff=1152
# 21M target: got 20.170M (err 830.2k) -> h=336 L=10 heads=7 (hd=48) ff=1152
# 22M target: got 21.417M (err 582.9k) -> h=352 L=10 heads=11 (hd=32) ff=1152
# 23M target: got 23.255M (err 254.9k) -> h=352 L=11 heads=11 (hd=32) ff=1152
# 24M target: got 23.255M (err 745.1k) -> h=352 L=11 heads=11 (hd=32) ff=1152
# 25M target: got 27.678M (err 2678.4k) -> h=384 L=11 heads=6 (hd=64) ff=1280
# 26M target: got 27.678M (err 1678.4k) -> h=384 L=11 heads=6 (hd=64) ff=1280
# 27M target: got 27.678M (err 678.4k) -> h=384 L=11 heads=6 (hd=64) ff=1280
# 28M target: got 29.892M (err 1892.2k) -> h=384 L=12 heads=6 (hd=64) ff=1280
# 29M target: got 29.892M (err 892.2k) -> h=384 L=12 heads=6 (hd=64) ff=1280
# 30M target: got 29.892M (err 107.8k) -> h=384 L=12 heads=6 (hd=64) ff=1280
