from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

OBLIQUE_VIEW_NAMES = (
    "22.5 Degree",
    "Oblique 45 Degree",
    "67.5 Degree",
    "112.5 Degree",
    "Oblique 135 Degree",
    "157.5 Degree",
)

BUNDLE_PROFILE_TO_MODEL_KEY = {
    "la": "la",
    "pa": "pa",
    "oblique": "oblique",
    "oblique_22_5": "oblique",
    "oblique_45": "oblique",
    "oblique_67_5": "oblique",
    "oblique_112_5": "oblique",
    "oblique_135": "oblique",
    "oblique_157_5": "oblique",
}

BUNDLE_PROFILE_TO_VIEWS = {
    "la": ["LA"],
    "pa": ["PA"],
    "oblique": list(OBLIQUE_VIEW_NAMES),
    "oblique_22_5": ["22.5 Degree"],
    "oblique_45": ["Oblique 45 Degree"],
    "oblique_67_5": ["67.5 Degree"],
    "oblique_112_5": ["112.5 Degree"],
    "oblique_135": ["Oblique 135 Degree"],
    "oblique_157_5": ["157.5 Degree"],
}

AVAILABLE_BUNDLE_PROFILES = tuple(BUNDLE_PROFILE_TO_MODEL_KEY.keys())


def load_checkpoint(path: str | Path, profile: str | None = None) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("bundle_format") is None:
        return checkpoint

    if profile is None:
        raise ValueError(
            "This checkpoint file is a multi-profile bundle. Please provide --profile "
            f"from: {', '.join(AVAILABLE_BUNDLE_PROFILES)}."
        )

    if profile not in BUNDLE_PROFILE_TO_MODEL_KEY:
        raise ValueError(
            f"Unknown bundle profile '{profile}'. Valid choices: {', '.join(AVAILABLE_BUNDLE_PROFILES)}."
        )

    model_key = BUNDLE_PROFILE_TO_MODEL_KEY[profile]
    model_checkpoint = deepcopy(checkpoint["models"][model_key])
    model_checkpoint["selected_views"] = list(BUNDLE_PROFILE_TO_VIEWS[profile])
    model_checkpoint["checkpoint_profile"] = profile
    model_checkpoint["checkpoint_source"] = str(path)
    return model_checkpoint
