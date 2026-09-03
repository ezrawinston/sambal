"""Unit tests for the named-profile config expansion (sambal.profile_loader).

Pure dict-merge semantics — no engine initialization.
"""
from pathlib import Path

from sambal.profile_loader import apply_profile

PKG_DIR = Path(__file__).resolve().parent.parent / "sambal"


def test_no_profile_passthrough():
    overrides = {"args": {"input": "x"}, "config": {"seed": 1}}
    assert apply_profile(overrides) is overrides


def test_profile_expansion_and_precedence():
    merged = apply_profile({
        "profile": "icml2026",
        "args": {"input": "x"},
        "config": {"require_gpu": False, "seed": 5},
        "paths": {"npi_path": "/custom/npi.tsv"},
    })
    assert "profile" not in merged
    # profile config flows through...
    assert merged["config"]["ctx_lemma_gate"] == "freq"
    assert merged["config"]["roundtrip_ud"] is True
    # ...and the config block overrides it
    assert merged["config"]["require_gpu"] is False
    assert merged["config"]["seed"] == 5
    # args untouched
    assert merged["args"] == {"input": "x"}


def test_profile_resource_resolution():
    merged = apply_profile({"profile": "icml2026"})
    res_dir = PKG_DIR / "resources"
    assert merged["paths"]["allowed_vocab_path"] == str(res_dir / "allowed_vocab.txt")
    assert merged["paths"]["ctx_lemma_stats_path"] == str(res_dir / "lemma_stats_top_25k.pkl")
    # keywords are not treated as filenames
    assert merged["paths"]["verbnet_source"] == "nltk"


def test_paths_override_beats_profile():
    merged = apply_profile({
        "profile": "icml2026",
        "paths": {"allowed_vocab_path": "/elsewhere/vocab.txt"},
    })
    assert merged["paths"]["allowed_vocab_path"] == "/elsewhere/vocab.txt"


def test_driver_defaults_between_profile_and_config():
    # profile sets require_gpu true; driver default relaxes it
    merged = apply_profile({"profile": "icml2026"},
                           driver_defaults={"require_gpu": False})
    assert merged["config"]["require_gpu"] is False
    # explicit config beats the driver default
    merged = apply_profile({"profile": "icml2026",
                            "config": {"require_gpu": True}},
                           driver_defaults={"require_gpu": False})
    assert merged["config"]["require_gpu"] is True
    # without a profile, driver defaults do not apply
    plain = apply_profile({"config": {"seed": 1}},
                          driver_defaults={"require_gpu": False})
    assert "require_gpu" not in plain.get("config", {})
