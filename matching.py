"""Text helpers: comparing values and spotting hedges / detail words."""
from __future__ import annotations

import re
from typing import Iterable, Set

COLORS = {
    "red", "blue", "green", "yellow", "black", "white", "grey", "gray", "brown",
    "orange", "purple", "pink", "maroon", "navy", "beige", "silver", "gold", "dark",
    "light",
}
# Treat these as the same colour.
COLOR_ALIASES = {"gray": "grey", "navy": "blue", "maroon": "red"}
# "dark"/"light" modify a colour; they don't make two colours disagree.
COLOR_MODIFIERS = {"dark", "light"}

# Concrete detail words. A question containing one of these that the witness
# never said themselves is leading.
DETAIL_LEXICON = COLORS | {
    "hoodie", "jacket", "shirt", "t-shirt", "tshirt", "coat", "sweater", "jeans",
    "trousers", "pants", "shorts", "skirt", "dress", "uniform", "cap", "hat",
    "helmet", "mask", "hood", "beard", "moustache", "glasses", "backpack", "bag",
    "handbag", "knife", "gun", "pistol", "rifle", "bat", "stick", "rod", "weapon",
    "car", "van", "truck", "bike", "motorcycle", "scooter", "bicycle", "auto",
    "tall", "short", "slim", "thin", "fat", "heavy", "muscular", "bald", "long",
    "blonde", "curly", "cash", "money", "phone", "wallet", "register",
}

# Garment words that say nothing about the kind of garment: "red clothes" fits "red hoodie".
GENERIC_GARMENTS = {"clothes", "clothing", "outfit", "top", "attire"}

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "with", "was",
    "were", "is", "are", "he", "she", "they", "it", "his", "her", "their", "them",
    "you", "your", "i", "me", "my", "we", "that", "this", "there", "what", "which",
    "who", "how", "where", "when", "did", "do", "does", "if", "anything", "any",
    "about", "describe", "remember", "notice", "noticed", "see", "saw", "by",
    "from", "for", "as", "be", "been", "had", "have", "has", "would", "could",
    "own", "words", "else", "part", "kind", "exactly", "person", "someone",
}

STRONG_HEDGES = [
    "i think", "maybe", "probably", "not sure", "might", "i guess", "possibly",
    "could have been", "i believe", "kind of", "sort of", "seemed", "looked like",
    "perhaps", "i'm not certain", "hard to tell",
]
MILD_HEDGES = ["pretty sure", "fairly sure", "i'm almost sure"]


def tokens(text: str) -> list:
    return re.findall(r"[a-z][a-z'\-]*", (text or "").lower())


def content_tokens(text: str) -> Set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS}


def _colors(text: str) -> Set[str]:
    out = set()
    for t in tokens(text):
        if t in COLORS and t not in COLOR_MODIFIERS:
            out.add(COLOR_ALIASES.get(t, t))
    return out


def normalize(value: str) -> str:
    return " ".join(tokens(value))


def values_agree(a: str, b: str) -> bool:
    """Do two stated values describe the same thing?

    Different colours always disagree ("red hoodie" vs "blue jacket").
    Otherwise equal, contained, or mostly-overlapping phrases agree.
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    ca, cb = _colors(na), _colors(nb)
    if ca and cb and not (ca & cb):
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = content_tokens(na), content_tokens(nb)
    if not ta or not tb:
        return False
    # Same colour and one side only says "clothes": nothing contradicts.
    if ca and cb and (ta - set(COLORS) <= GENERIC_GARMENTS or tb - set(COLORS) <= GENERIC_GARMENTS):
        return True
    overlap = len(ta & tb) / len(ta | tb)
    return overlap >= 0.5


def hedge_level(text: str) -> float:
    """0 = no hedge, 0.5 = mild ("pretty sure"), 1 = strong ("I think")."""
    t = (text or "").lower()
    if any(h in t for h in STRONG_HEDGES):
        return 1.0
    if any(h in t for h in MILD_HEDGES):
        return 0.5
    return 0.0


def vocabulary(texts: Iterable[str]) -> Set[str]:
    vocab: Set[str] = set()
    for t in texts:
        vocab |= set(tokens(t))
    return vocab
