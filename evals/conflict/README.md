# Grammar–plausibility conflict benchmark

A **swap-only** minimal-pair benchmark (187 conflict items + 213 matched
controls) in which grammatical structure and lexical plausibility point at
different continuations:

- **GOOD** is grammatical but semantically implausible (role-reversed)
- **BAD** is ungrammatical but semantically plausible
- a **CONTROL** pair ensures the grammatical phenomenon is easy when
  semantics is plausible

[`conflicts.jsonl`](conflicts.jsonl) (187 rows) and
[`controls.jsonl`](controls.jsonl) (213 rows) are the item files the models
are scored on ([Guide 1.9](../../reproduce/icml2026/EVALUATION.md#19-conflict-benchmark--44-app-f));
[`pool/conflict_pool.jsonl`](pool/conflict_pool.jsonl)
(213 rows) is the candidate pool the conflict file is derived from.
[`conflict_rr_suite/`](conflict_rr_suite/) is the generator package:
[`build_suite.py`](build_suite.py) builds a pool and its controls from two
oracle models, [`derive_conflicts.py`](derive_conflicts.py) derives the
conflict set from the pool, and [`inspect_jsonl.py`](inspect_jsonl.py)
prints items from any of the files.

## Families

| family | conflict items | control items |
|---|---|---|
| `sva_role_reversal` | 60 | 67 |
| `detnum_role_reversal` | 97 | 105 |
| `passive_role_reversal` | 30 | 41 |

1. `sva_role_reversal` — subject–verb agreement conflict:
   GOOD `The cake eats the girls.` (grammatical, absurd) vs
   BAD `The girls eats the cake.` (ungrammatical, plausible);
   control `The kids read the newspaper.` vs `The kids reads the newspaper.`
2. `detnum_role_reversal` — determiner–noun number agreement with an
   asymmetric determiner pair (singular-only `a`/`each` vs number-flexible
   `some`), so only one mismatch occurs under swap:
   GOOD `A bread ate some children.` vs BAD `A children ate some bread.`;
   control `Some lions ate some pizza.` vs `Each lions ate some pizza.`
3. `passive_role_reversal` — agreement inside a passive, where the swap moves
   the number-bearing argument into subject position:
   GOOD `The teachers were read by the letter.` vs
   BAD `The letter were read by the teachers.`;
   control `The story was read by the parents.` vs
   `The story were read by the parents.`
