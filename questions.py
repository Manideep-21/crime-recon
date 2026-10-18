"""Gap-driven, non-leading question generation.

1. Candidates come from the graph: gaps (nobody mentioned X) and conflicts
   (witnesses disagree on X -> ask each to re-describe X in their own words).
2. Every candidate passes through is_leading(). A question is blocked if it
   contains a concrete detail word this witness has not said themselves, or
   if it is phrased to push an answer ("Was it...", "Are you sure...").
3. For the demo we also generate "baseline" candidates the way a naive
   feedback-loop system would ("Are you sure it wasn't a blue jacket?").
   They exist only to show the filter blocking them; none reach the witness.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from graph import SceneGraph
from matching import DETAIL_LEXICON, content_tokens, tokens, vocabulary
from schema import LOCATION_ID, origin_of

TEMPLATES: Dict[str, str] = {
    "upper_clothing": "What was {ref} wearing on the upper part of their body?",
    "lower_clothing": "What was {ref} wearing on their legs?",
    "headwear": "Describe anything you noticed on the head of {ref}, if anything.",
    "height": "How would you describe the height of {ref}?",
    "build": "How would you describe the body build of {ref}?",
    "hair": "What do you remember about the hair of {ref}?",
    "action": "What did you see {ref} doing?",
    "position": "Where exactly was {ref} when you saw them?",
    "carrying": "Describe anything {ref} was holding or carrying, if anything.",
    "vehicle_type": "What kind of vehicle was {ref}?",
    "color": "What colour was {ref}?",
    "weapon_type": "Describe {ref} in your own words.",
    "held_by": "Who, if anyone, was holding {ref}?",
    "object_type": "Describe {ref} in your own words.",
    "state": "What condition was {ref} in?",
    "place_type": "What kind of place did this happen in?",
    "lighting": "How was the lighting where this happened?",
    "time_of_day": "What time of day was it?",
    "notable_features": "What else do you remember about the surroundings?",
}
RECHECK = "In your own words, please describe the {attr} of {ref} again."

ATTR_PHRASE = {
    "upper_clothing": "upper-body clothing", "lower_clothing": "lower-body clothing",
    "vehicle_type": "type", "weapon_type": "appearance", "object_type": "appearance",
    "time_of_day": "time of day", "place_type": "type of place",
}

BASE_REF = {"person": "the person", "vehicle": "the vehicle", "weapon": "the item",
            "object": "the object", "location": "the place"}

# Words our own templates use; never counted as leading.
TEMPLATE_VOCAB = vocabulary(list(TEMPLATES.values()) + [RECHECK] + list(ATTR_PHRASE.values())
                            + list(BASE_REF.values()))

PRESUPPOSING = [
    r"^\s*(was|were|did|didn't|wasn't|weren't|isn't|aren't|had|hadn't)\b",
    r"\bare you sure\b", r"\bisn't it true\b", r"\bwouldn't you say\b",
    r"\bisn't that right\b", r"\bright\?\s*$", r"\balso see\b",
]


# ------------------------------------------------------------------ filter

def witness_allowed_vocab(graph: SceneGraph, witnesses: Dict[str, dict], witness: str) -> set:
    """Every word this witness has used: statement, follow-up answers, and the
    facts extracted from them."""
    texts = [witnesses[witness]["statement"]]
    for f in graph.facts:
        if origin_of(f.source) == witness:
            texts += [f.quote, f.value]
    return vocabulary(texts)


def is_leading(text: str, allowed_vocab: set, graph: SceneGraph) -> List[str]:
    """Return the reasons a question is leading (empty list = safe)."""
    reasons = []
    graph_vocab = set()
    for f in graph.facts:
        graph_vocab |= content_tokens(f.value)
    detail_words = (DETAIL_LEXICON | graph_vocab) - TEMPLATE_VOCAB
    introduced = sorted({t for t in tokens(text) if t in detail_words and t not in allowed_vocab})
    if introduced:
        reasons.append(f"introduces detail(s) the witness never mentioned: {', '.join(introduced)}")
    for pat in PRESUPPOSING:
        if re.search(pat, text.lower()):
            reasons.append("closed / suggestive phrasing pushes a particular answer")
            break
    return reasons


# ------------------------------------------------------------------ candidates

def entity_reference(graph: SceneGraph, entity: str, entity_type: str, witness: str,
                     asked_attribute: str = "") -> Optional[str]:
    """Refer to an entity using only this witness's own words. None if they never mentioned it.

    The quote never comes from the attribute being asked about, so we don't
    remind the witness of their earlier answer to the same question.
    """
    if entity == LOCATION_ID:
        return "the place"
    own = [f for f in graph.facts if f.entity == entity and origin_of(f.source) == witness]
    if not own:
        return None
    banned = {f.quote for f in own if f.attribute == asked_attribute}
    usable = [f for f in own if f.quote and f.quote not in banned]
    usable.sort(key=lambda f: f.hedged)  # prefer quotes the witness was sure about
    quote = usable[0].quote if usable else ""
    quote = " ".join(quote.split()[:12])
    base = BASE_REF.get(entity_type, "the thing")
    return f'{base} you described ("{quote}")' if quote else base


def _q(witness, item, text, kind, witnesses) -> dict:
    return {"witness": witness, "entity": item["entity"], "entity_type": item["entity_type"],
            "role": item["role"], "attribute": item["attribute"], "text": text, "kind": kind,
            "viewpoint": witnesses[witness].get("viewpoint", "")}


def candidate_questions(graph: SceneGraph, gaps: List[dict], conflicts: List[dict],
                        witnesses: Dict[str, dict], include_baseline: bool = True) -> List[dict]:
    out = []
    for g in gaps:
        for w in witnesses:
            ref = entity_reference(graph, g["entity"], g["entity_type"], w, g["attribute"])
            if ref is None:
                continue
            out.append(_q(w, g, TEMPLATES[g["attribute"]].format(ref=ref), "gap", witnesses))

    for c in conflicts:
        if c.get("kind") == "role":
            continue
        involved = {origin_of(s) for o in c["options"] for s in o["sources"]} & set(witnesses)
        for w in sorted(involved):
            ref = entity_reference(graph, c["entity"], c["entity_type"], w, c["attribute"])
            if ref is None:
                continue
            attr = ATTR_PHRASE.get(c["attribute"], c["attribute"].replace("_", " "))
            out.append(_q(w, c, RECHECK.format(attr=attr, ref=ref), "conflict_recheck", witnesses))
            if include_baseline:
                # Naive version: feed the other witness's answer back. Must be blocked.
                for o in c["options"]:
                    if w not in {origin_of(s) for s in o["sources"]}:
                        out.append(_q(w, c, f"Are you sure it wasn't {o['value']}?",
                                      "baseline", witnesses))

    if include_baseline:
        # Naive version: ask a witness to confirm things only others reported.
        for f in graph.facts:
            if f.entity_type not in {"weapon", "object", "vehicle"}:
                continue
            for w in witnesses:
                if w in f.support:
                    continue
                item = {"entity": f.entity, "entity_type": f.entity_type,
                        "role": f.role, "attribute": f.attribute}
                out.append(_q(w, item, f"Did you also see the {f.value}?", "baseline", witnesses))

    seen, unique = set(), []
    for q in out:
        key = (q["witness"], q["text"])
        if key not in seen:
            seen.add(key)
            unique.append(q)
    for i, q in enumerate(unique, 1):
        q["id"] = f"q{i:03d}"
    return unique


REPHRASE_PROMPT = """Rewrite this interview question so it sounds natural and polite.
Do not add any new details, colours, objects or names. Keep it open-ended.
Question: {q}
Return JSON: {{"question": "..."}}"""


def generate_questions(graph: SceneGraph, gaps: List[dict], conflicts: List[dict],
                       witnesses: Dict[str, dict], llm_rephrase: bool = False,
                       include_baseline: bool = True) -> List[dict]:
    """All candidates with passed / reasons filled in."""
    questions = candidate_questions(graph, gaps, conflicts, witnesses, include_baseline)
    vocab_cache: Dict[str, set] = {}
    for q in questions:
        w = q["witness"]
        if w not in vocab_cache:
            vocab_cache[w] = witness_allowed_vocab(graph, witnesses, w)
        allowed = vocab_cache[w]
        if llm_rephrase and q["kind"] != "baseline":
            import llm
            try:
                new = llm.chat_json(REPHRASE_PROMPT.format(q=q["text"]))["question"]
                if not is_leading(new, allowed, graph):
                    q["template_text"], q["text"] = q["text"], new
            except Exception:
                pass
        q["reasons"] = is_leading(q["text"], allowed, graph)
        q["passed"] = not q["reasons"]
    return questions
