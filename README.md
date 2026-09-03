# sambal

Code for **[Learning syntax without semantics: Disentangled tiny language models](https://openreview.net/forum?id=p7HVrmZwWB)**  
Ezra Winston and Zico Kolter, ICML 2026

Tiny LMs trained on grammatical nonsense match standard pretraining on syntax
without learning meaning — yielding more efficient and controllable models.

## Usage

**Reproduce the paper:** [`reproduce/icml2026/README.md`](reproduce/icml2026/README.md)

**Released models and data** (Hugging Face):

- [`ezrawinston/gptbert-babycosmofine`](https://huggingface.co/ezrawinston/gptbert-babycosmofine) — baseline LMs (long + short regime) and LoRA adapters
- [`ezrawinston/gptbert-sambal`](https://huggingface.co/ezrawinston/gptbert-sambal) — SAMBAL LMs and LoRA adapters
- [`ezrawinston/babycosmofine-sambal`](https://huggingface.co/datasets/ezrawinston/babycosmofine-sambal) — the ablated corpus and tokenized training bins

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10; the release is validated with 3.12.

## Layout

| dir | contents |
|---|---|
| [`sambal/`](sambal/) | ablated relexicalization pipeline |
| [`lm/`](lm/) | LM training/evaluation ([`lm/gpt-bert/VENDORED.md`](lm/gpt-bert/VENDORED.md)) |
| [`data/`](data/) | corpus fetch/build scripts and manifests |
| [`evals/`](evals/) | evaluation runners for benchmark and probe suites |
| [`reproduce/`](reproduce/) | paper reproduction: drivers, configs, results |
| [`tests/`](tests/) | test suite |

## Licensing

MIT for the code (see [`LICENSE`](LICENSE)); the tree under
[`lm/gpt-bert/`](lm/gpt-bert/) retains its
upstream MIT license. Data resources derived from third-party sources carry their own licenses
([LICENSES.md](LICENSES.md)).

## Citation

```bibtex
@inproceedings{winston2026syntax,
  title     = {Learning syntax without semantics: Disentangled tiny language models},
  author    = {Winston, Ezra and Kolter, J. Zico},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/forum?id=p7HVrmZwWB}
}
```

## Errata

Corrections to the paper's published numbers: [ERRATA.md](ERRATA.md).
