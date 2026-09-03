# tests

Test suite for the `sambal` package and the evaluators. Run with `pytest tests/`.

- [`test_core_fast_parity.py`](test_core_fast_parity.py) — document-scope core parity (`core` vs
  `core_fast`) on the fixtures under [`data/`](data/).
- [`test_core_span_freeze.py`](test_core_span_freeze.py) — span-freeze behavior of the fast core.
- [`test_drivers.py`](test_drivers.py) — driver equivalence (one-shot vs parse-then-augment).
- [`test_sharding.py`](test_sharding.py) — input-row slicing: the shard-range split, and sharded
  runs concatenating to the whole-input run.
- [`test_profile_loader.py`](test_profile_loader.py) — profile resolution
  ([`sambal/profiles/`](../sambal/profiles/)).
- [`evals/`](evals/) — identity gate for the evaluators: every scorer reproduces its frozen
  output on a tiny model ([`evals/golden/`](evals/golden/)). [`make_fixtures.py`](evals/make_fixtures.py)
  builds the model and the small inputs, [`reference.py`](evals/reference.py) runs each scorer,
  [`write_goldens.py`](evals/write_goldens.py) freezes the outputs, [`test_goldens.py`](evals/test_goldens.py)
  replays them.
