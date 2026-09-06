"""
Configuration dataclasses for sambal.

ResourcePaths - paths to all resource files (lexicons, patterns, etc.)
Config - runtime configuration options for augmentation behavior
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class ResourcePaths:
    """Paths to all resource files used by the Augmenter."""
    verbnet_source: str = "nltk"  # "nltk" only
    noun_bucket_dir: Optional[str] = None
    function_words_path: Optional[str] = None
    npi_path: Optional[str] = None
    licensors_path: Optional[str] = None
    noun_lemmas_path: Optional[str] = None
    adj_lemmas_path: Optional[str] = None
    adv_lemmas_path: Optional[str] = None
    propn_list_path: Optional[str] = None
    given_names_path: Optional[str] = None
    fixed_mwes_path: Optional[str] = None
    mwe_patterns_jsonl_path: Optional[str] = None
    licensor_patterns_jsonl_path: Optional[str] = None  # JSONL licensor patterns (token+dep)
    # NEW (all optional; conservative behavior if omitted)
    human_nouns_path: Optional[str] = None          # lemmas that are [+human]
    allowed_vocab_path: Optional[str] = None
    allow_all_vocab_path: Optional[str] = None  # Include all tokens from this file (ignoring counts)

    # Optional: ERG/ACE-derived to-inf lexicons (JSONL)
    # Each line: {"lemma": str, "pos": "ADJ"|"VERB", "tags": [...], "licenses": [...], ...}
    toinf_adj_lexicon_jsonl: Optional[str] = None
    toinf_verb_lexicon_jsonl: Optional[str] = None

    # Optional: gender-aware replacement resources (all optional; conservative behavior if omitted)
    given_names_male_path: Optional[str] = None
    given_names_female_path: Optional[str] = None
    given_names_neutral_path: Optional[str] = None
    gendered_words_json_path: Optional[str] = None

    # Override given-name pools: if provided, these become the definitive replacement
    # pools WITHOUT vocab gating. Original given_names_*_path files are ignored for
    # substitution (but still used for recognition if provided).
    override_given_names_male_path: Optional[str] = None
    override_given_names_female_path: Optional[str] = None
    override_given_names_neutral_path: Optional[str] = None

    # Context-key lemma stats file (pickle or pickle.gz)
    # Expected format: {lemma(str): Counter({bucket_tuple: int, ...}), ...}
    ctx_lemma_stats_path: Optional[str] = None


@dataclass
class Config:
    """Runtime configuration options for augmentation behavior."""
    # --- NEW: parser backend and round-trip controls ---
    roundtrip_ud: bool = True       # reparse the augmented sentence and assert structural invariants
    roundtrip_max_tries: int = 2    # how many re-sampling attempts before falling back to original
    spacy_model: str = "en_core_web_trf"
    swap_heads_via_verbnet: bool = True  # allow verb head swaps via VN; set False to freeze all verb heads
    seed: int = 7
    replace_propn: bool = False
    debug: bool = False
    replace_pronouns: bool = True       # allow safe pronoun shuffling within case/person/number class
    adv_only_rb: bool = True             # restrict to RB (exclude WRB/wh-)
    vn_soft_prep_min_pool: int = 0        # if >0, treat adjunct preps as soft when VN pool is smaller than this
    human_to_human: bool = True
    conservative_guard: bool = True  # also protect conservative licensor fallbacks
    roundtrip_debug: bool = False  # when True (and cfg.debug True), print per-reject diffs to stdout
    debug_vn: bool = False          # log VerbNet bucket diagnostics in _verb_candidates
    verb_metrics: bool = False
    freeze_aux_hosts: bool = False
    include_proper_nouns: bool = False
    include_proper_adjs: bool = False
    include_proper_advs: bool = False
    min_vocab_freq: int = 3
    respect_human: bool = False
    reflexive_frame_check: bool = False
    respect_gender: bool = True
    # --- to-inf freeze bypass (experimental) ---
    # When True, skip freezing verbs/adjs in to-infinitive contexts when lexicons
    # are missing or have no compatible candidates. Also disables the tough-adj
    # hardcoded fallback list. Use for experimentation only.
    skip_toinf_freeze: bool = False
    # --- allowed-vocab prefiltering (optional) ---
    # When `allowed_vocab` is small, rejection-sampling (lemma → allowed_lemmas check)
    # can fail frequently and exhaust sample_max_attempts. If enabled, we apply
    # an init-time filter to the large replacement pools (nouns/adj/adv
    # and VerbNet verb index) using allowed_lemmas membership.
    prefilter_pools_by_allowed_vocab: bool = True

    # --- start-up caches ---
    # Persist the expensive start-up products (allowed lemmas, the context
    # bucket table, the filtered countability pools, the warmed per-tag pools)
    # under the resources cache directory and read them back on later starts.
    # Off: build everything in memory, read nothing, write nothing.
    startup_caches: bool = True

    # --- name-pool split (optional) ---
    # Keep full given-name pools for recognition, but optionally use a filtered
    # (allowed_lemmas-gated) pool for substitution to avoid sampling failures.
    filter_substitution_names_by_allowed_vocab: bool = True

    # --- Sentence splitting / long-text handling ---
    split_long_text_by_sentence: bool = True

    # "off" = ignore stats
    # "flat" = intersect candidates with bucket lemmas then sample uniformly
    # "freq" = intersect then sample proportional to lemma frequency within the bucket
    ctx_lemma_gate: str = "off"  # off|flat|freq

    ctx_lemma_gate_min_count: int = 1

    # After bucket selection, keep only top p fraction of candidates by cumulative frequency.
    # E.g., 0.5 keeps candidates until cumulative count >= 50% of bucket total.
    # 0.0 or 1.0 disables. Applied after bucket acceptance, respects ctx_lemma_gate_min_count as floor.
    ctx_lemma_gate_top_p: float = 0.0

    # --- Ctx-bucket backoff acceptance thresholds (optional) ---
    # When ctx-gating uses merged/backoff buckets, you may require a bucket
    # to have enough evidence before using it. Evidence is computed AFTER
    # dropping rare lemmas (< ctx_backoff_min_lemma_count) within that bucket level.
    #
    #   mode "A": require total evidence count S >= ctx_backoff_min_total_count
    #   mode "B": require number of lemmas (with count>=n) >= max(ctx_backoff_min_candidates,
    #             ctx_backoff_min_candidates_pct * n_orig)
    #   mode "off": accept the first bucket that yields a non-empty intersection.
    ctx_backoff_mode: str = "off"  # off|A|B
    ctx_backoff_min_lemma_count: int = 1  # n (per-bucket lemma min count)
    ctx_backoff_min_total_count: int = 0  # m (mode A)
    ctx_backoff_min_candidates: int = 0  # k (mode B, absolute minimum)
    ctx_backoff_min_candidates_pct: float = 0.0  # mode B: fraction of n_orig pool size (0.0-1.0)

    # Global guard for POS probing
    require_gpu: bool = True

    # --- Additional config fields ---
    spacy_pos_model: str = "en_core_web_sm"
    include_proper_humans: bool = False
    sample_max_attempts: int = 200
    light_verb_lemmas: Optional[Set[str]] = None
    freeze_toinf_selectors: bool = False
