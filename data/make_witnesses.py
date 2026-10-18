"""Build one evaluation scene: answer key + 3 degraded witness statements.

No public dataset has several witnesses per incident, so we make them by
controlled memory degradation of a known-true description:
  omission       witness forgets some details
  corruption     witness remembers a detail wrongly (colour / size swapped)
  confabulation  witness "remembers" something that never happened
  vantage        a far-away witness loses fine details (hair, headwear, ...)
Every change is recorded, so evaluation knows exactly what was planted.

Step 1  caption -> answer-key facts (LLM), plus a noun per person ("man" / "woman" / "person",
        from the caption) under "nouns".  REVIEW ground_truth.json BY HAND.
Step 2  answer key -> 3 degraded fact lists (deterministic, seeded)
Step 3  each fact list -> natural first-person statement (LLM)

    python data/make_witnesses.py --scene robbery048 \
        --caption "A man in a black jacket points a gun at the cashier ..."
    # after hand-editing ground_truth.json, rebuild only steps 2-3:
    python data/make_witnesses.py --scene robbery048 --reuse-gt
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402
from config import load_config  # noqa: E402
from extraction import extract_statement, merge_entities  # noqa: E402
from matching import COLORS, values_agree  # noqa: E402
from schema import MULTI_VALUED  # noqa: E402

PROFILES = [
    {"id": "witness_A", "viewpoint": "close by, inside, a few metres away",
     "omit": 0.2, "corrupt": 0, "confab": 0, "far": False},
    {"id": "witness_B", "viewpoint": "far away, looking in from outside",
     "omit": 0.35, "corrupt": 1, "confab": 0, "far": True},
    {"id": "witness_C", "viewpoint": "brief glimpse while passing by",
     "omit": 0.5, "corrupt": 0, "confab": 1, "far": False},
]
FINE_DETAIL = {"hair", "headwear", "height", "build", "state", "color"}
SWAPS = {"tall": "short", "short": "tall", "slim": "heavy", "heavy": "slim", "thin": "stocky",
         "night": "evening", "day": "afternoon"}
COLOR_LIST = ["red", "blue", "black", "white", "grey", "green", "brown", "yellow"]
CONFAB_POOL = [
    {"entity": "weapon_9", "entity_type": "weapon", "role": "none", "attribute": "weapon_type", "value": "knife"},
    {"entity": "vehicle_9", "entity_type": "vehicle", "role": "none", "attribute": "vehicle_type", "value": "white van"},
    {"entity": "person_9", "entity_type": "person", "role": "bystander", "attribute": "upper_clothing", "value": "grey coat"},
]

STATEMENT_PROMPT = """Write a short first-person witness statement (3-6 sentences, informal, past tense)
from someone who saw this from: {viewpoint}.
These are the ONLY things the witness noticed. Mention all of them and nothing else
(no extra objects, colours, places or people):
{details}
Rules:
- Refer to people the way a witness would, by what they looked like, did or were ("a woman behind
  the counter", "the robber", "the cashier"). Never write "suspect", "victim" or "bystander".
- Keep every detail with the person it is listed under, and keep each person a man or a woman as given.
  Do not give anyone a position or action that is not listed for them.
- For details after "not sure about:", the witness is unsure: use hedging words like "I think",
  "maybe", or "I'm pretty sure" for those details only.
Return JSON: {{"statement": "..."}}"""

# Internal entity ids (person_1, weapon_9, ...) must never reach a witness or the model.
ENTITY_ID = re.compile(r"\b(?:person|weapon|vehicle|object|location|scene)_\d+\b|\bscene\b(?=\s*[:(])", re.I)
# Role labels are ours, not a witness's words.
ROLE_LABEL = re.compile(r"\b(suspects?|victims?|bystanders?)\b", re.I)
GENDER_WORDS = {"man": {"man", "men", "he", "him", "his", "guy", "boy"},
                "woman": {"woman", "women", "she", "her", "hers", "lady", "girl"}}
NOUN_WORDS = {"woman": ("woman", "lady", "girl", "female", "she", "her"),
              "man": ("man", "guy", "boy", "male", "gentleman", "he", "his")}
CONFAB_NOUNS = {"person_9": "man"}  # invented bystander; a fixed gender keeps the gender check on
ROLE_PHRASE = {
    "robbery": {"suspect": "who robbed the place", "victim": "who was being robbed (works there)"},
    "shoplifting": {"suspect": "who stole something", "victim": "who works in the shop"},
    "": {"suspect": "who committed the crime", "victim": "who the crime was against"},
}


def corrupt_value(value: str, rng: random.Random):
    words = value.split()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in COLORS:
            words[i] = rng.choice([c for c in COLOR_LIST if c != lw])
            return " ".join(words)
        if lw in SWAPS:
            words[i] = SWAPS[lw]
            return " ".join(words)
    return None


def noun_from(description: str) -> str:
    """'a woman in red' -> 'woman', 'a man in black' -> 'man', anything else -> 'person'."""
    words = set(re.findall(r"[a-z]+", description.lower()))
    hits = [noun for noun, cues in NOUN_WORDS.items() if words & set(cues)]
    return hits[0] if len(hits) == 1 else "person"


def caption_to_gt(caption: str) -> tuple:
    """Answer-key facts and {person entity: noun} from the caption."""
    raw = extract_statement(caption)
    mapping = merge_entities({"gt": raw}, use_llm=False)
    gt, nouns = [], {}
    for f in raw["facts"]:
        m = mapping[f"gt:{f['entity']}"]
        gt.append({"entity": m["entity"], "entity_type": m["entity_type"], "role": m["role"],
                   "attribute": f["attribute"], "value": f["value"]})
    for key, e in raw["entities"].items():
        m = mapping[f"gt:{key}"]
        if m["entity_type"] == "person" and m["entity"] not in nouns:
            nouns[m["entity"]] = noun_from(e["description"])
    return gt, nouns


def crime_kind(scene_id: str) -> str:
    return next((k for k in ("robbery", "shoplifting") if scene_id.lower().startswith(k)), "")


def degrade(gt: list, profile: dict, rng: random.Random) -> tuple:
    kept, log = [], []
    for f in gt:
        if profile["far"] and f["attribute"] in FINE_DETAIL:
            log.append({"type": "vantage_loss", **f})
        elif rng.random() < profile["omit"]:
            log.append({"type": "omission", **f})
        else:
            kept.append(dict(f, hedged=rng.random() < 0.25))
    # Never corrupt a multi-valued attribute: a different action/position is not a conflict,
    # so the planted conflict could never be found ("handed it to the man in black" -> "... in blue").
    candidates = [f for f in kept if f["attribute"] not in MULTI_VALUED and corrupt_value(f["value"], rng)]
    for f in rng.sample(candidates, min(profile["corrupt"], len(candidates))):
        new = corrupt_value(f["value"], rng)
        log.append({"type": "corruption", **{k: f[k] for k in ("entity", "entity_type", "role", "attribute")},
                    "true_value": f["value"], "witness_value": new})
        f["value"], f["hedged"] = new, rng.random() < 0.5
    present = {(f["entity_type"], f["attribute"], f["value"]) for f in gt}
    pool = [c for c in CONFAB_POOL if (c["entity_type"], c["attribute"], c["value"]) not in present]
    for c in rng.sample(pool, min(profile["confab"], len(pool))):
        kept.append(dict(c, hedged=rng.random() < 0.5))
        log.append({"type": "confabulation", **c})
    return kept, log


def person_nouns(facts: list, nouns: dict) -> dict:
    """{person entity: noun} for the people in these facts; every real person needs a noun."""
    out = {}
    for f in facts:
        if f["entity_type"] == "person" and f["entity"] not in out:
            noun = nouns.get(f["entity"]) or CONFAB_NOUNS.get(f["entity"])
            assert noun in ("man", "woman", "person"), \
                f"no noun for {f['entity']}: add it to \"nouns\" in ground_truth.json"
            out[f["entity"]] = noun
    return out


def entity_headings(facts: list, nouns: dict, kind: str) -> dict:
    """How each entity is introduced to the statement writer, e.g. 'a woman who works in the shop'."""
    heads, seen = {}, {}
    who = person_nouns(facts, nouns)
    for f in facts:
        e = f["entity"]
        if e in heads:
            continue
        if f["entity_type"] == "person":
            noun = who[e]
            article = "another" if seen.get(noun) else "a"
            seen[noun] = True
            phrase = ROLE_PHRASE[kind].get(f["role"], "who was just nearby" if f["role"] == "bystander" else "")
            heads[e] = f"{article} {noun} {phrase}".strip()
        elif f["entity_type"] == "location":
            heads[e] = "the place"
        else:
            heads[e] = f"{'another' if seen.get(f['entity_type']) else 'a'} {f['entity_type']}"
            seen[f["entity_type"]] = True
    return heads


def statement_prompt(viewpoint: str, facts: list, nouns: dict, kind: str = "") -> str:
    """Details grouped per entity, each entity introduced by noun + role in plain words (no ids, no labels)."""
    heads = entity_headings(facts, nouns, kind)
    blocks = []
    for e, head in heads.items():
        own = [f for f in facts if f["entity"] == e]
        sure = "; ".join(f"{f['attribute'].replace('_', ' ')} = {f['value']}" for f in own if not f.get("hedged"))
        unsure = "; ".join(f"{f['attribute'].replace('_', ' ')} = {f['value']}" for f in own if f.get("hedged"))
        blocks.append(f"- {head}: " + "; ".join(x for x in (sure, f"not sure about: {unsure}" if unsure else "") if x))
    prompt = STATEMENT_PROMPT.format(viewpoint=viewpoint, details="\n".join(blocks))
    leak = ENTITY_ID.search(prompt)
    assert leak is None, f"entity id {leak.group(0)!r} in statement prompt"
    return prompt


def statement_problem(text: str, genders: set, detail_text: str = "") -> Optional[str]:
    """Why a generated statement is unusable (None if fine).

    genders: nouns of the people the witness describes. A gender nobody has must not appear
    ("the woman" turned into "a man"), unless someone's gender is unknown or the given details
    themselves use the word ("walked to the woman's side").
    """
    if ENTITY_ID.search(text):
        return f"entity id {ENTITY_ID.search(text).group(0)!r}"
    if ROLE_LABEL.search(text):
        return f"role label {ROLE_LABEL.search(text).group(0)!r}"
    if "(unsure)" in text.lower():
        return "copied the '(unsure)' marker"
    words = set(re.findall(r"[a-z]+", text.lower())) - set(re.findall(r"[a-z]+", detail_text.lower()))
    for g, cues in GENDER_WORDS.items():
        if g not in genders and "person" not in genders and words & cues:
            return f"mentions a {g} but nobody described is a {g}"
    return None


ATTRIBUTED = {"action", "position", "upper_clothing", "lower_clothing", "headwear", "carrying"}


def attribution_problem(text: str, facts: list, extract=None) -> Optional[str]:
    """Re-read the statement with the normal extractor: no one person in it may carry details of
    two different people ("the robber ... squatted with her hands up").

    Compares who holds which detail, not the roles the extractor guesses (it may call the shop
    worker a suspect). Details two people share ("behind the counter") are skipped.
    """
    raw = (extract or extract_statement)(text)
    people = [f for f in facts if f["entity_type"] == "person" and f["attribute"] in ATTRIBUTED]
    unique = [f for f in people if not any(o["entity"] != f["entity"] and o["attribute"] == f["attribute"]
                                           and values_agree(o["value"], f["value"]) for o in people)]
    holders: Dict[str, Dict[str, str]] = defaultdict(dict)  # statement person -> {given person: one of their details}
    for f in unique:
        for g in raw["facts"]:
            if g["attribute"] == f["attribute"] and values_agree(g["value"], f["value"]):
                holders[g["entity"]].setdefault(f["entity"], f["value"])
    for details in holders.values():
        if len(details) > 1:
            return "one person in the statement has details of two people: " + " / ".join(f"'{v}'" for v in details.values())
    return None


def write_statement(viewpoint: str, facts: list, nouns: dict, kind: str = "", tries: int = 5,
                    extract=None) -> str:
    """LLM statement for these facts; retried (a little warmer each time) while it has ids, role
    labels, a gender nobody has, or a detail moved to the wrong person."""
    prompt = statement_prompt(viewpoint, facts, nouns, kind)
    genders = set(person_nouns(facts, nouns).values())
    detail_text = " ".join(str(f["value"]) for f in facts)
    problem = None
    for attempt in range(tries):
        text = llm.chat_json(prompt, temperature=0.1 + 0.2 * attempt)["statement"]
        problem = statement_problem(text, genders, detail_text) or attribution_problem(text, facts, extract)
        if problem is None:
            return text
    raise SystemExit(f"statement still unusable after {tries} tries ({problem}): {text!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="folder name under data/scenes")
    ap.add_argument("--caption", help="true description of the event (e.g. UCA caption)")
    ap.add_argument("--caption-file")
    ap.add_argument("--reuse-gt", action="store_true", help="keep the existing (hand-edited) ground_truth.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    folder = os.path.join(load_config()["paths"]["scenes_dir"], args.scene)
    os.makedirs(os.path.join(folder, "frames"), exist_ok=True)
    gt_path = os.path.join(folder, "ground_truth.json")

    if args.reuse_gt:
        gt_doc = json.load(open(gt_path))
    else:
        caption = args.caption or open(args.caption_file).read()
        facts, nouns = caption_to_gt(caption)
        gt_doc = {"scene_id": args.scene, "caption": caption, "nouns": nouns, "facts": facts}
        print(f"Answer key: {len(gt_doc['facts'])} facts, nouns {nouns}. Please review {gt_path} by hand.")
    nouns, kind = gt_doc.get("nouns", {}), crime_kind(args.scene)

    rng = random.Random(args.seed)
    witnesses, truth, planted_conflicts, planted_false = {}, {}, [], []
    for p in PROFILES:
        facts, log = degrade(gt_doc["facts"], p, rng)
        text = write_statement(p["viewpoint"], facts, nouns, kind)
        witnesses[p["id"]] = {"viewpoint": p["viewpoint"], "statement": text}
        truth[p["id"]] = {"facts_given": facts, "degradations": log}
        for d in log:
            if d["type"] == "corruption":
                planted_conflicts.append({k: d[k] for k in ("entity_type", "role", "attribute")})
            if d["type"] == "confabulation":
                planted_false.append({k: d[k] for k in ("entity_type", "role", "attribute", "value")}
                                     | {"witness": p["id"]})
        print(f"\n{p['id']} ({p['viewpoint']}):\n  {text}")

    gt_doc["planted_conflicts"] = planted_conflicts
    gt_doc["planted_false_facts"] = planted_false
    json.dump(gt_doc, open(gt_path, "w"), indent=2)
    json.dump(truth, open(os.path.join(folder, "witness_truth.json"), "w"), indent=2)

    scene_path = os.path.join(folder, "scene.json")
    scene = json.load(open(scene_path)) if os.path.exists(scene_path) else {"title": args.scene, "cctv_frames": []}
    scene["witnesses"] = witnesses
    json.dump(scene, open(scene_path, "w"), indent=2)
    print(f"\nWrote {scene_path}, {gt_path}, witness_truth.json")


if __name__ == "__main__":
    main()
