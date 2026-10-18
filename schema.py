"""Shared fact format used by every module.

Every piece of evidence becomes a Fact: one attribute of one entity, as stated
by one source. Nothing else in the system invents its own format.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Attributes allowed for each entity type. Extraction must stay inside this list,
# and any attribute nobody has filled in is a "gap" that drives questioning.
ENTITY_ATTRIBUTES: Dict[str, List[str]] = {
    "person": [
        "upper_clothing", "lower_clothing", "headwear", "height", "build",
        "hair", "action", "position", "carrying",
    ],
    "vehicle": ["vehicle_type", "color", "position", "action"],
    "weapon": ["weapon_type", "color", "held_by"],
    "object": ["object_type", "color", "position", "state"],
    "location": ["place_type", "lighting", "time_of_day", "notable_features"],
}

# Subset of attributes we actively ask witnesses about when missing.
GAP_ATTRIBUTES: Dict[str, List[str]] = {
    "person": ["upper_clothing", "lower_clothing", "height", "build", "action", "position"],
    "vehicle": ["vehicle_type", "color", "position"],
    "weapon": ["weapon_type", "held_by"],
    "object": ["object_type", "position"],
    "location": ["place_type", "lighting", "time_of_day"],
}

# Attributes that can legitimately have several true values (a person does
# several things, moves around). Different values here are NOT a conflict.
MULTI_VALUED = {"action", "position", "notable_features"}

ROLES =["suspect", "victim", "bystander", "none"]

# The single location entity always has this id.
LOCATION_ID = "scene"


@dataclass
class Fact:
    fact_id: str
    entity: str            # canonical id, e.g. person_1, vehicle_1, scene
    entity_type: str       # person / vehicle / weapon / object / location
    role: str              # suspect / victim / bystander / none
    attribute: str         # one of ENTITY_ATTRIBUTES[entity_type]
    value: str             # short phrase, e.g. "red hoodie"
    source: str            # witness_A, witness_A_followup, cctv_frame_03
    source_type: str       # witness / followup / cctv
    viewpoint: str = ""
    quote: str = ""        # the words in the source this fact came from
    hedged: bool = False   # "I think", "maybe", ...
    base_confidence: float = 0.0
    confidence: float = 0.0  # base + corroboration, recomputed by SceneGraph
    support: List[str] = field(default_factory=list)  # origins that agree

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Fact":
        return Fact(**d)


def origin_of(source: str) -> str:
    """Independent origin of a source.

    A witness and their follow-up answers are the same origin, and all CCTV
    frames come from one camera, so neither can corroborate itself.
    """
    if source.startswith("cctv"):
        return "cctv"
    return source.replace("_followup", "")


def validate_fact_fields(entity_type: str, attribute: str) -> Optional[str]:
    """Return an error message if the type/attribute pair is not in the schema."""
    if entity_type not in ENTITY_ATTRIBUTES:
        return f"unknown entity_type '{entity_type}'"
    if attribute not in ENTITY_ATTRIBUTES[entity_type]:
        return f"attribute '{attribute}' not allowed for {entity_type}"
    return None


def schema_description() -> str:
    """Schema as text, pasted into LLM prompts."""
    lines = []
    for etype, attrs in ENTITY_ATTRIBUTES.items():
        lines.append(f"- {etype}: {', '.join(attrs)}")
    return "\n".join(lines)
