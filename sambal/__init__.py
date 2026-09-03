"""sambal — constrained relexicalization for controlled corpus ablation.

Public API:

    from sambal import Augmenter, Config, ResourcePaths

Occurrence-scope ablation (an independent draw per token occurrence) is the
engine's ``augment`` / ``augment_parsed_doc`` path; document-scope ablation
(one consistent lemma mapping per document) is layered on top by
``sambal.core`` / ``sambal.core_fast``.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Debug infrastructure (shared across modules)
_DEBUG = False


def dbg(msg_or_fn):
    """Debug helper: prints string or executes callable if debug is enabled.

    Usage:
        dbg("message")                        # prints if debug
        dbg(lambda: expensive_operation())    # executes if debug
        dbg(lambda: print(f"x = {x}"))        # execute + print if debug
    """
    if not _DEBUG:
        return
    if callable(msg_or_fn):
        msg_or_fn()
    else:
        print(msg_or_fn)


def set_debug(enabled: bool) -> None:
    """Enable or disable debug output globally."""
    global _DEBUG
    _DEBUG = enabled


# Lazy imports so `import sambal` stays light (no spaCy at package import).
def __getattr__(name):
    if name == "Augmenter":
        from sambal.engine import Augmenter
        globals()["Augmenter"] = Augmenter
        return Augmenter
    if name in ("ResourcePaths", "Config"):
        from sambal.config import Config, ResourcePaths
        globals().update({"ResourcePaths": ResourcePaths, "Config": Config})
        return globals()[name]
    if name in ("TokenFeatures", "TOKENINFO_SCHEMA"):
        from sambal.token_features import TOKENINFO_SCHEMA, TokenFeatures
        globals().update({
            "TokenFeatures": TokenFeatures,
            "TOKENINFO_SCHEMA": TOKENINFO_SCHEMA,
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Augmenter",
    "ResourcePaths",
    "Config",
    "TokenFeatures",
    "TOKENINFO_SCHEMA",
    "dbg",
    "set_debug",
    "__version__",
]
