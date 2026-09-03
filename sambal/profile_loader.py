"""Named-profile support for driver configs.

A driver config may reference a profile by name instead of spelling out the
full engine configuration:

    {"profile": "icml2026", "args": {...}, "config": {...}, "paths": {...}}

The profile (``sambal/profiles/<name>.json``) supplies the base ``config``
block and a ``resources`` filename map resolved inside ``sambal/resources/``;
the config's own ``config``/``paths`` blocks override profile values. A
driver may additionally pass ``driver_defaults`` — config values that sit
between the profile and the user's config block (profile < driver defaults <
explicit config), for keys whose right default depends on the entry point
(e.g. the DocBin stage relaxes ``require_gpu``).
"""
from __future__ import annotations

import json
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def apply_profile(overrides: dict, driver_defaults: dict | None = None) -> dict:
    """Expand a ``profile`` key in a config-overrides dict, in place-ish.

    Returns the merged overrides; a dict without a ``profile`` key is
    returned unchanged (``driver_defaults`` apply only underneath a profile —
    a config written without a profile spells out exactly what runs).
    """
    name = overrides.get("profile")
    if not name:
        return overrides

    profile_path = _PKG_DIR / "profiles" / f"{name}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    res_dir = _PKG_DIR / "resources"
    profile_paths = {}
    for key, value in profile.get("resources", {}).items():
        if key == "verbnet_source" or not isinstance(value, str):
            profile_paths[key] = value
        else:
            profile_paths[key] = str(res_dir / value)

    merged = dict(overrides)
    merged.pop("profile")
    merged["config"] = {
        **profile.get("config", {}),
        **(driver_defaults or {}),
        **overrides.get("config", {}),
    }
    merged["paths"] = {**profile_paths, **overrides.get("paths", {})}
    return merged
