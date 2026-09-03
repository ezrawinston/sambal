"""Modal variant of the corpus-ablation run: one container per shard, writing
shard outputs to a Modal volume.

    modal run modal_sambal_corpus.py --num-shards 24

Requires a Modal account; the repo is baked into the image, the source corpus
is read from the volume (upload data/babycosmofine/train.jsonl to it first), and
shard outputs land on the same volume.
"""
import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3]

app = modal.App("sambal-corpus")
vol = modal.Volume.from_name("sambal-corpus", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(str(REPO), remote_path="/repo", copy=True)
    .run_commands(
        "pip install -e /repo",
        "python -m spacy download en_core_web_trf",
        "python -c \"import nltk; nltk.download('verbnet')\"",
    )
)


@app.function(image=image, gpu="T4", volumes={"/vol": vol}, timeout=60 * 60 * 12)
def run_shard(shard_id: int, num_shards: int):
    cfg = {
        "profile": "icml2026",
        "args": {
            "input": "/vol/train.jsonl",
            "fmt": "jsonl",
            "out": f"/vol/sambal/shard_{shard_id:02d}.jsonl",
            "resume": True,
            # This container augments only its own contiguous slice of the
            # input rows; the shards partition the corpus and concatenate in
            # shard order.
            "shard": shard_id,
            "num_shards": num_shards,
        },
        "config": {"seed": shard_id},
    }
    cfg_path = f"/tmp/shard_{shard_id}.json"
    json.dump(cfg, open(cfg_path, "w"))

    import subprocess
    subprocess.run(
        ["python", "-m", "sambal.augment", "--config", cfg_path],
        check=True,
    )
    vol.commit()


@app.local_entrypoint()
def main(num_shards: int = 24):
    list(run_shard.starmap((i, num_shards) for i in range(num_shards)))
