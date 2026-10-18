"""Extraction agent: witness statements + CCTV frames -> provenance-tagged facts.

Steps
1. Each source (one statement or one frame) is read on its own by Qwen2.5-VL,
   which lists the entities it mentions and the facts about them.
2. Entities from different sources are merged ("the man" from witness A and
   "the guy" from witness B -> person_1).
3. Every fact gets a base confidence from simple, explainable rules.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import llm
from config import load_config
from matching import COLOR_ALIASES, content_tokens, hedge_level, normalize, tokens, values_agree
from schema import ENTITY_ATTRIBUTES, LOCATION_ID, MULTI_VALUED, ROLES, Fact, validate_fact_fields

SYSTEM = (
    "You are a careful forensic evidence extraction assistant. You only record "
    "what the source actually states or clearly shows. You never guess, infer, "
    "or add details. You always answer with JSON only."
)

# One line per attribute, so the model knows which slot a detail belongs in.
ATTRIBUTE_GUIDE: Dict[str, Dict[str, str]] = {
    "person": {
        "upper_clothing": "shirts, t-shirts, jackets, hoodies, coats, uniforms (a hoodie is upper_clothing, not headwear)",
        "lower_clothing": "trousers, jeans, skirts, shorts, leggings (never caps, bags or shoes)",
        "headwear": "caps, hats, beanies, hoods, helmets, balaclavas",
        "height": "tall / short / average",
        "build": "slim / heavy / medium / muscular",
        "hair": "hair colour, length or style",
        "action": "what the person did; one fact per distinct action (a person can have several)",
        "position": "where the person was (behind the counter, by the door)",
        "carrying": "bags, backpacks, purses, anything held or carried that is not a weapon",
    },
    "vehicle": {
        "vehicle_type": "car, van, motorcycle, bicycle ... (only for a vehicle the witness actually describes)",
        "color": "colour of the vehicle",
        "position": "where the vehicle was",
        "action": "what the vehicle did (drove off, parked)",
    },
    "weapon": {
        "weapon_type": "knife, gun, bat ... REQUIRED for every weapon entity",
        "color": "colour of the weapon",
        "held_by": "who held it, in the witness's words (the tall man)",
    },
    "object": {
        "object_type": "what the object is (cash register, door, phone); objects people interact with",
        "color": "colour of the object",
        "position": "where the object was",
        "state": "open / broken / knocked over ...",
    },
    "location": {
        "place_type": "the whole place (convenience store, parking lot, street), never a part of it",
        "lighting": "how it was lit (brightly lit, dark, dim)",
        "time_of_day": "night / morning / afternoon / evening",
        "notable_features": "parts and fixed features of the place (snacks aisle, counter, front window)",
    },
}


def schema_guide() -> str:
    """Schema as text with a one-line guide per attribute, pasted into LLM prompts."""
    lines = []
    for etype, attrs in ENTITY_ATTRIBUTES.items():
        lines.append(f"- {etype}:")
        for attr in attrs:
            hint = ATTRIBUTE_GUIDE.get(etype, {}).get(attr, "")
            lines.append(f"    {attr}" + (f" = {hint}" if hint else ""))
    return "\n".join(lines)


STATEMENT_PROMPT = """Read this witness statement and extract every concrete detail.

Allowed entity types and their attributes (attribute = what belongs there):
{schema}

Roles: suspect, victim, bystander, none (use none for objects, weapons, vehicles and the location).
There is at most ONE location entity (entity_type "location") describing the place itself.

Return JSON exactly in this shape:
{{
  "entities": [
    {{"id": "e1", "entity_type": "person", "role": "suspect", "description": "the witness's own short words for it"}}
  ],
  "facts": [
    {{"entity": "e1", "attribute": "upper_clothing", "value": "red hoodie", "quote": "exact words from the statement", "hedged": false}}
  ]
}}

Rules:
- Only use attributes from the allowed list for that entity type.
- "value" is a short phrase (1-6 words) using the witness's own words.
- "quote" is copied exactly from the statement, including any hedge words ("I'm pretty sure he had ...").
- "hedged" is true if the witness sounded unsure about that detail ("I think", "maybe", "pretty sure").
- Hedged details are still details: extract them with "hedged": true. Never leave a detail out
  because the witness sounded unsure ("I think he wore an orange shirt" -> upper_clothing "orange shirt").
- Every weapon entity must have a weapon_type fact.
- action can have several values: extract every distinct action as its own fact
  ("walked to the counter" and "grabbed cash from the register" are two facts).
- Things people interact with (a cash register, a door, a phone) are object entities with object_type.
- If the statement mentions the kind of place, the lighting or the time of day, record each one
  as place_type / lighting / time_of_day on the single location entity.
- place_type is the whole place (a store), never a part of it (an aisle, the counter: notable_features).
- Only create a vehicle entity if a vehicle itself is described. A direction such as
  "towards the parking lot" is the person's action or position, not a vehicle.
- If the statement does not mention something, leave the fact out. Never write placeholders
  such as "not specified", "unknown", "none" or "n/a". Do not fill gaps.

Example
Statement: "It was dark in the car park. A short, heavy man in a grey beanie ran towards the exit with a bag. I'm pretty sure he had a gun."
JSON:
{{
  "entities": [
    {{"id": "e1", "entity_type": "location", "role": "none", "description": "the car park"}},
    {{"id": "e2", "entity_type": "person", "role": "suspect", "description": "a short, heavy man"}},
    {{"id": "e3", "entity_type": "weapon", "role": "none", "description": "a gun"}}
  ],
  "facts": [
    {{"entity": "e1", "attribute": "place_type", "value": "car park", "quote": "in the car park", "hedged": false}},
    {{"entity": "e1", "attribute": "lighting", "value": "dark", "quote": "It was dark", "hedged": false}},
    {{"entity": "e2", "attribute": "height", "value": "short", "quote": "A short, heavy man", "hedged": false}},
    {{"entity": "e2", "attribute": "build", "value": "heavy", "quote": "A short, heavy man", "hedged": false}},
    {{"entity": "e2", "attribute": "headwear", "value": "grey beanie", "quote": "in a grey beanie", "hedged": false}},
    {{"entity": "e2", "attribute": "carrying", "value": "bag", "quote": "with a bag", "hedged": false}},
    {{"entity": "e2", "attribute": "action", "value": "ran towards the exit", "quote": "ran towards the exit", "hedged": false}},
    {{"entity": "e3", "attribute": "weapon_type", "value": "gun", "quote": "I'm pretty sure he had a gun", "hedged": true}},
    {{"entity": "e3", "attribute": "held_by", "value": "the short, heavy man", "quote": "I'm pretty sure he had a gun", "hedged": true}}
  ]
}}

Witness statement:
\"\"\"{text}\"\"\"
"""

HEDGED_SENTENCE_PROMPT = """A witness statement was already read once. The witness sounded unsure in ONE sentence,
and unsure details are easy to miss. Extract every concrete detail from that sentence only.

Allowed entity types and their attributes (attribute = what belongs there):
{schema}

Entities already found in the statement (use these ids; "he"/"she" refers to one of them):
{entities}

If the sentence describes something not in that list, add it with id "n1", "n2", ...
Every detail is hedged: set "hedged": true. "value" is a short phrase (1-6 words) in the witness's own words.
Never write placeholders such as "not specified" or "unknown".

Return JSON exactly in this shape:
{{
  "entities": [
    {{"id": "n1", "entity_type": "weapon", "role": "none", "description": "the witness's own short words"}}
  ],
  "facts": [
    {{"entity": "e1", "attribute": "upper_clothing", "value": "orange shirt", "quote": "exact words from the sentence", "hedged": true}}
  ]
}}

Whole statement (for context only):
\"\"\"{text}\"\"\"

The sentence to extract from:
\"\"\"{sentence}\"\"\"
"""

FRAME_PROMPT = """This is a single frame from a low-resolution CCTV camera at a crime scene.
List only what is CLEARLY visible. If you are not sure about a colour or object, leave it out.

Allowed entity types and their attributes (attribute = what belongs there):
{schema}

Roles: suspect, victim, bystander, none. Only assign suspect/victim if the action makes it obvious;
otherwise use bystander for people.
There is at most ONE location entity (entity_type "location"); give it place_type, lighting and
time_of_day when they are visible. Every weapon entity must have a weapon_type fact.
Leave out anything you cannot see; never write placeholders such as "not specified" or "unknown".

Return JSON exactly in this shape:
{{
  "entities": [
    {{"id": "e1", "entity_type": "person", "role": "bystander", "description": "short visual description"}}
  ],
  "facts": [
    {{"entity": "e1", "attribute": "upper_clothing", "value": "dark jacket", "quote": "", "hedged": false}}
  ]
}}
"""

MERGE_PROMPT = """These entity mentions come from different witnesses and CCTV frames of the SAME incident.
Group mentions that refer to the same real-world thing. Only group mentions with the same entity_type.

Witnesses saw the same people from different places and describe them in different words, so
"a tall man in a red hoodie", "he" and "a slim guy in black jeans" can all be the same person.
Different or even contradicting details are NOT a reason to split: contradictions are handled later.
Roles are not shown on purpose: who robbed whom is often misread. Decide only from the
descriptions and details (clothing, gender, what the person did, where they were).
A man and a woman are never the same person.
Keep two mentions from the SAME source separate (a witness doesn't describe one person as two).

Mentions (key | type | description | details):
{mentions}

Return JSON: {{"groups": [["witness_A:e1", "witness_B:e2"], ["witness_A:e2"], ...]}}
Every mention key must appear in exactly one group.
"""

ANSWER_PROMPT = """A witness was asked: "{question}"
This question is about the attribute "{attribute}" of a {entity_type}.
The witness answered: "{answer}"

Return JSON: {{"value": "short phrase in the witness's own words, or null if they did not know or did not say", "hedged": true/false}}
"""


# ---------------------------------------------------------------- per source

# Keyword post-check for the model's most common slot mistakes (e.g. "black cap" as lower_clothing).
# Each rule only moves values out of the slots it lists; "hoodie" does not match "hood",
# "shorts" does not match "short".
CLOTHING_SLOTS = {"upper_clothing", "lower_clothing", "headwear", "carrying"}
BODY_SLOTS = {"height", "build"}
KEYWORD_ATTRIBUTES = [
    ("person", CLOTHING_SLOTS, "headwear",
     re.compile(r"\b(cap|caps|hat|hats|beanie|beanies|hood|hoods|helmet|balaclava|balaclavas|mask|masks)\b")),
    ("person", CLOTHING_SLOTS, "carrying",
     re.compile(r"\b(bag|bags|backpack|backpacks|rucksack|purse|purses|handbag|briefcase)\b")),
    ("person", BODY_SLOTS, "build",
     re.compile(r"\b(slim|thin|skinny|heavy|heavyset|stocky|muscular|medium build|average build)\b")),
    ("person", BODY_SLOTS, "height",
     re.compile(r"\b(tall|short|average height|medium height)\b")),
    ("location", {"place_type"}, "notable_features",
     re.compile(r"\b(aisle|aisles|counter|till|checkout|shelf|shelves|window|doorway)\b")),
]
WEAPON_WORDS = re.compile(r"\b(knife|knives|gun|guns|pistol|revolver|rifle|shotgun|machete)\b")
# place_type must name a kind of place; "scene" or "place where the watch was" is not one.
PLACE_WORDS = re.compile(
    r"\b(store|shop|supermarket|market|mart|mall|boutique|pharmacy|kiosk|deli|bakery|bank|atm|station|"
    r"gas|petrol|garage|restaurant|cafe|café|bar|pub|club|hotel|motel|office|building|warehouse|factory|"
    r"house|home|apartment|flat|room|lobby|hallway|corridor|elevator|stairs|street|road|alley|sidewalk|"
    r"pavement|parking|car park|lot|park|yard|square|school|hospital|clinic|church|salon|laundromat|"
    r"pawnshop|jewel\w*|showroom|dealership|bus|train|subway|metro)\b")
PLACEHOLDERS = {"not specified", "unspecified", "not mentioned", "not stated", "unknown",
                "n/a", "na", "none", "null", "unclear", "not clear", "not visible", "-",
                "nothing", "no"}
LEADING_ARTICLE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _strip_article(value: str) -> str:
    """'the store' -> 'store', so prompts never say 'a the store'."""
    return LEADING_ARTICLE.sub("", value.strip()).strip()


def _word_forms(word: str) -> set:
    """The word plus its simple-suffix stems: hoodies -> hoodie, grabbed -> grab, running -> run."""
    forms = {word}
    for suffix in ("ing", "ed", "es", "s", "d"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            forms.add(stem)
            if stem[-1] == stem[-2]:  # runn -> run, grabb -> grab
                forms.add(stem[:-1])
    return forms


def grounding_problem(value: str, text: str) -> Optional[str]:
    """Why a witness fact is not grounded in that witness's own words (None if it is).

    A fact is kept only if at least one content word of its value appears in the text.
    """
    words = content_tokens(value)
    if not words:
        return "value has no content words"
    said = set().union(*(_word_forms(t) for t in tokens(text)))
    if any(_word_forms(w) & said for w in words):
        return None
    return f"none of {sorted(words)} appear in the statement"


def _keyword_attribute(etype: str, attr: str, value: str) -> str:
    """Attribute the keyword rules say this value belongs to (unchanged if no clear rule)."""
    v = value.lower()
    if etype == "weapon" and attr != "weapon_type" and WEAPON_WORDS.fullmatch(v):
        return "weapon_type"
    hits = [target for rule_type, slots, target, pattern in KEYWORD_ATTRIBUTES
            if rule_type == etype and attr in slots and pattern.search(v)]
    if len(hits) == 1:  # "cap in his bag" or "tall, slim" is ambiguous, leave it to the model
        return hits[0]
    return attr


def _is_placeholder(value: str) -> bool:
    return value.lower().strip(" .") in PLACEHOLDERS


def _add_missing_weapon_types(entities: dict, facts: List[dict], source: str, log) -> None:
    """A weapon without weapon_type gets one from a weapon word in its quotes or description."""
    for eid, e in entities.items():
        if e["entity_type"] != "weapon":
            continue
        own = [f for f in facts if f["entity"] == eid]
        if any(f["attribute"] == "weapon_type" for f in own):
            continue
        for text, hedged in [(f["quote"], f["hedged"]) for f in own] + [(e["description"], False)]:
            m = WEAPON_WORDS.search(text.lower())
            if m:
                facts.append({"entity": eid, "attribute": "weapon_type", "value": m.group(1),
                              "quote": text, "hedged": hedged})
                log("post-check", f"{source}: {eid} weapon had no weapon_type -> added '{m.group(1)}'")
                break


def _clean_raw(raw: dict, source: str = "", log=lambda step, detail: None,
               text: Optional[str] = None) -> dict:
    """Drop anything outside the schema so later steps can trust the data.

    text: the witness's statement; facts not grounded in it are dropped (None for CCTV frames).
    """
    entities = {}
    for e in raw.get("entities", []) or []:
        etype = str(e.get("entity_type", "")).lower().strip()
        if etype not in ENTITY_ATTRIBUTES:
            continue
        role = str(e.get("role", "none")).lower().strip()
        if etype != "person" or role not in ROLES:
            role = "none"
        entities[str(e.get("id"))] = {
            "entity_type": etype, "role": role,
            "description": str(e.get("description", "")).strip(),
        }
    facts = []
    for f in raw.get("facts", []) or []:
        eid = str(f.get("entity"))
        if eid not in entities:
            continue
        attr = str(f.get("attribute", "")).lower().strip()
        value = _strip_article(str(f.get("value") or ""))
        etype = entities[eid]["entity_type"]
        if value and _is_placeholder(value):
            log("post-check", f"{source}: dropped placeholder {etype}.{attr} = '{value}'")
            continue
        problem = grounding_problem(value, text) if value and text is not None else None
        if problem:
            log("grounding", f"{source}: dropped {etype}.{attr} = '{value}' ({problem})")
            continue
        fixed = _keyword_attribute(etype, attr, value)
        if fixed != attr:
            log("post-check", f"{source}: '{value}' {etype}.{attr} -> {etype}.{fixed}")
            attr = fixed
        if etype == "location" and attr == "place_type" and value and not PLACE_WORDS.search(value.lower()):
            log("post-check", f"{source}: dropped place_type '{value}' (not a kind of place)")
            continue
        if not value or validate_fact_fields(etype, attr):
            continue
        facts.append({"entity": eid, "attribute": attr, "value": value,
                      "quote": str(f.get("quote") or ""), "hedged": bool(f.get("hedged"))})
    _add_missing_weapon_types(entities, facts, source, log)
    # A vehicle with no type or colour was only a place or direction ("towards the parking lot").
    for eid, e in list(entities.items()):
        own = {f["attribute"] for f in facts if f["entity"] == eid}
        if e["entity_type"] == "vehicle" and not own & {"vehicle_type", "color"}:
            log("post-check", f"{source}: dropped vehicle {eid} ('{e['description']}'), no vehicle described")
            del entities[eid]
            facts = [f for f in facts if f["entity"] != eid]
    return {"entities": entities, "facts": facts}


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _sentence_for(quote: str, value: str, sentences: List[str]) -> Optional[str]:
    """The statement sentence a fact came from: the one containing the quote, else the best word match."""
    q = normalize(quote)
    for s in sentences:
        if q and q in normalize(s):
            return s
    want = content_tokens(quote) or content_tokens(value)
    if not want or not sentences:
        return None
    best = max(sentences, key=lambda s: len(want & content_tokens(s)))
    return best if len(want & content_tokens(best)) / len(want) >= 0.5 else None


def _add_sentences(raw: dict, text: str) -> dict:
    """Attach the witness's own sentence to each fact; hedging is decided from it, not the model's quote."""
    sentences = _sentences(text)
    for f in raw["facts"]:
        f["sentence"] = _sentence_for(f["quote"], f["value"], sentences) or ""
        if f["sentence"]:
            f["hedged"] = hedge_level(f["sentence"]) > 0
    return raw


def _already_known(fact: dict, facts: List[dict]) -> bool:
    """Already extracted from this statement, on any entity: the first pass decided whose it is
    ("another man had a grey coat" must not become a second grey-coat person)."""
    return any(f["attribute"] == fact["attribute"] and values_agree(f["value"], fact["value"])
               for f in facts)


def _recover_hedged(clean: dict, text: str, source: str, log) -> dict:
    """Second look at each hedged sentence: the model often skips unsure details in a full statement
    ("I'm pretty sure he was wearing a plaid shirt and a hat" -> only the hat), though it extracts
    them from the sentence alone. Missing facts are added; they stay hedged (lower confidence)."""
    entities = "\n".join(f'- {eid} | {e["entity_type"]} | {e["role"]} | {e["description"]}'
                         for eid, e in clean["entities"].items()) or "(none)"
    for sentence in _sentences(text):
        if hedge_level(sentence) == 0:
            continue
        try:
            raw = llm.chat_json(HEDGED_SENTENCE_PROMPT.format(
                schema=schema_guide(), entities=entities, text=text, sentence=sentence),
                system=SYSTEM, temperature=0.0)
        except Exception as exc:  # the first pass still stands
            log("hedge pass", f"{source}: sentence skipped ({exc})")
            continue
        if not isinstance(raw, dict):
            continue
        used = {str(f.get("entity")) for f in raw.get("facts", []) or [] if isinstance(f, dict)}
        known = [{"id": eid, **e} for eid, e in clean["entities"].items() if eid in used]
        extra = [e for e in raw.get("entities", []) or []
                 if isinstance(e, dict) and str(e.get("id")) not in clean["entities"]]
        # Grounded in this sentence, not just anywhere in the statement.
        got = _clean_raw({"entities": known + extra, "facts": raw.get("facts", [])},
                         source, log, text=sentence)
        for f in got["facts"]:
            if grounding_problem(f["value"], sentence) or _already_known(f, clean["facts"]):
                continue  # (weapon_type filled in from a description is not from this sentence)
            eid = f["entity"]
            if eid not in clean["entities"]:
                fresh = f"h{len(clean['entities']) + 1}"
                while fresh in clean["entities"]:
                    fresh += "x"
                clean["entities"][fresh] = got["entities"][eid]
                for g in got["facts"]:
                    if g["entity"] == eid:
                        g["entity"] = fresh
                eid = fresh
            f["quote"] = f["quote"] or sentence
            f["hedged"] = True
            clean["facts"].append(f)
            log("hedge pass", f"{source}: added hedged {clean['entities'][eid]['entity_type']}."
                              f"{f['attribute']} = '{f['value']}' from \"{sentence}\"")
    return clean


def extract_statement(text: str, source: str = "statement", log=lambda step, detail: None) -> dict:
    raw = llm.chat_json(STATEMENT_PROMPT.format(schema=schema_guide(), text=text),
                        system=SYSTEM, temperature=0.0)
    clean = _recover_hedged(_clean_raw(raw, source, log, text=text), text, source, log)
    return _add_sentences(clean, text)


def extract_frame(image_path: str, source: str = "frame", log=lambda step, detail: None) -> dict:
    raw = llm.chat_json(FRAME_PROMPT.format(schema=schema_guide()),
                        system=SYSTEM, images=[image_path], temperature=0.0)
    return _clean_raw(raw, source, log)


# ---------------------------------------------------------------- merging

def _rule_based_groups(per_source: Dict[str, dict]) -> List[List[str]]:
    """Fallback without the LLM: every person mention starts on its own (the role-free rules in
    merge_entities join them); other entities of the same type line up by order (n-th weapon
    in each source); all locations are one place."""
    buckets: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    groups: List[List[str]] = []
    for source, raw in per_source.items():
        seen: Counter = Counter()
        for eid, e in raw["entities"].items():
            key = f"{source}:{eid}"
            if e["entity_type"] == "person":
                groups.append([key])
            elif e["entity_type"] == "location":
                buckets[("location", 0)].append(key)
            else:
                buckets[(e["entity_type"], seen[e["entity_type"]])].append(key)
                seen[e["entity_type"]] += 1
    return groups + list(buckets.values())


def _repair_groups(groups: List[List[str]], types: Dict[str, str]) -> Tuple[List[List[str]], Counter]:
    """Turn the LLM's grouping into a clean partition instead of rejecting it.

    - invented keys (not a mention) are dropped,
    - a key in several groups stays in the first one,
    - a group that mixes entity types (a person with an object) is split by type,
    - mentions the answer left out become groups of their own.
    Returns (groups, how often each repair was needed)."""
    repairs: Counter = Counter()
    seen: set = set()
    out: List[List[str]] = []
    for g in groups:
        kept = []
        for k in g:
            if k not in types:
                repairs["invented key dropped"] += 1
            elif k in seen:
                repairs["duplicate key dropped"] += 1
            else:
                kept.append(k)
                seen.add(k)
        by_type: Dict[str, List[str]] = {}
        for k in kept:
            by_type.setdefault(types[k], []).append(k)
        if len(by_type) > 1:
            repairs["mixed-type group split"] += 1
        out.extend(by_type.values())
    for k in types:
        if k not in seen:
            repairs["missing mention added alone"] += 1
            out.append([k])
    return out, repairs


def _llm_groups(per_source: Dict[str, dict], log=lambda step, detail: None) -> Optional[List[List[str]]]:
    mentions = []
    types = {}
    for source, raw in per_source.items():
        for eid, e in raw["entities"].items():
            key = f"{source}:{eid}"
            types[key] = e["entity_type"]
            details = "; ".join(f'{f["attribute"]}={f["value"]}' for f in raw["facts"] if f["entity"] == eid)
            # No role: extracted roles are often wrong and must not decide who is who.
            mentions.append(f'- {key} | {e["entity_type"]} | "{e["description"]}" | {details}')
    if not mentions:
        return []
    try:
        out = llm.chat_json(MERGE_PROMPT.format(mentions="\n".join(mentions)), system=SYSTEM,
                            temperature=0.0)
    except Exception:
        return None
    raw_groups = out.get("groups") if isinstance(out, dict) else None
    if not isinstance(raw_groups, list):
        return None
    groups = [list(map(str, g)) for g in raw_groups if isinstance(g, list) and g]
    # Repair a partial answer (rather than throwing all of it away) and say what was repaired.
    groups, repairs = _repair_groups(groups, types)
    if repairs:
        log("merge", "LLM grouping repaired: " + ", ".join(f"{n} {r}" for r, n in sorted(repairs.items())))
    else:
        log("merge", "LLM grouping used as given (no repair needed)")
    # All locations are the same place.
    loc = [k for g in groups for k in g if types[k] == "location"]
    groups = [g for g in groups if types[g[0]] != "location"]
    if loc:
        groups.append(loc)
    return groups


MALE_WORDS = {"man", "men", "guy", "guys", "boy", "male", "gentleman", "he", "him", "his", "husband",
              "father", "son", "mr", "dude"}
FEMALE_WORDS = {"woman", "women", "lady", "girl", "female", "she", "her", "hers", "wife", "mother",
                "daughter", "mrs", "ms", "miss"}


def _gender(description: str) -> Optional[str]:
    """'male' / 'female' from the mention's own words, or None if it doesn't say (or says both)."""
    words = set(tokens(description))
    male, female = bool(words & MALE_WORDS), bool(words & FEMALE_WORDS)
    return "male" if male and not female else "female" if female and not male else None


def _genders(keys: List[str], lookup: Dict[str, dict]) -> set:
    return {g for g in (_gender(lookup[k].get("description", "")) for k in keys) if g}


def _sources(keys: List[str]) -> set:
    return {k.split(":")[0] for k in keys}


def _split_inconsistent(groups: List[List[str]], lookup: Dict[str, dict], log) -> List[List[str]]:
    """Source consistency: one source never describes one person twice, and a man is not a woman.
    A person group that breaks either rule is split (first fit, in mention order)."""
    out: List[List[str]] = []
    for g in groups:
        if lookup[g[0]]["entity_type"] != "person":
            out.append(g)
            continue
        parts: List[List[str]] = []
        for k in g:
            for p in parts:
                if k.split(":")[0] not in _sources(p) and len(_genders(p + [k], lookup)) <= 1:
                    p.append(k)
                    break
            else:
                parts.append([k])
        if len(parts) > 1:
            log("merge", f"split {g} into {parts} (same source twice, or man and woman)")
        out.extend(parts)
    return out


def _can_join(a: List[str], b: List[str], lookup: Dict[str, dict]) -> bool:
    return not (_sources(a) & _sources(b)) and len(_genders(a + b, lookup)) <= 1


def _merge_lone_genders(groups: List[List[str]], lookup: Dict[str, dict], log) -> List[List[str]]:
    """If no witness describes two men, every witness's man is the same man (same for women).

    Role-free replacement of the old "each source names one suspect" rule: gender words are far
    more reliable than extracted roles. Mentions that don't give a gender are left to the other
    rules. Clashing details don't stop it: those become conflicts later.
    """
    people = [k for k in lookup if lookup[k]["entity_type"] == "person" and not k.startswith("cctv")]
    per_source = Counter((k.split(":")[0], _gender(lookup[k].get("description", ""))) for k in people)
    out: List[List[str]] = []
    joined: Dict[str, List[str]] = {}
    for g in groups:
        gs = _genders(g, lookup)
        gender = next(iter(gs)) if len(gs) == 1 else None
        lone = (gender and lookup[g[0]]["entity_type"] == "person" and not any(k.startswith("cctv") for k in g)
                and all(n == 1 for (s, x), n in per_source.items() if x == gender))
        if lone and gender in joined and _can_join(joined[gender], g, lookup):
            log("merge", f"joined {g} into {joined[gender]} (no witness describes two "
                         f"{'men' if gender == 'male' else 'women'})")
            joined[gender].extend(g)
            continue
        if lone and gender not in joined:
            joined[gender] = g
        out.append(g)
    return out


def _with_descriptions(per_source: Dict[str, dict], lookup: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Facts per mention, plus the identifying part of its description as one more detail."""
    facts_of = _facts_of(per_source)
    for k, e in lookup.items():
        core = _description_core(e.get("description", ""))
        if core and e["entity_type"] == "person":
            facts_of[k] = facts_of.get(k, []) + [{"attribute": "description", "value": core}]
    return facts_of


def _detach_cctv_people(groups: List[List[str]], lookup: Dict[str, dict]) -> List[List[str]]:
    """CCTV people are matched by _link_cctv_people only; out of the witness-side groups here."""
    is_cam = lambda k: k.startswith("cctv") and lookup[k]["entity_type"] == "person"  # noqa: E731
    rest = [g for g in ([k for k in g if not is_cam(k)] for g in groups) if g]
    return rest + [[k] for g in groups for k in g if is_cam(k)]


def _check_llm_joins(groups: List[List[str]], per_source: Dict[str, dict], lookup: Dict[str, dict],
                     log) -> List[List[str]]:
    """The LLM's person groups are suggestions: a witness mention whose details clearly agree more
    with another group (link score >= 1 and higher than with its own group) moves there."""
    facts_of = _with_descriptions(per_source, lookup)
    score = lambda a, b: _link_score(*_agreement(_facts_by_attr(a, facts_of), _facts_by_attr(b, facts_of)))  # noqa: E731
    moved = 0
    for _ in range(len(groups)):
        change = None
        for gi, g in enumerate(groups):
            if lookup[g[0]]["entity_type"] != "person" or len(g) < 2:
                continue
            for k in g:
                own = score([k], [x for x in g if x != k])
                best = max(((score([k], h), hi) for hi, h in enumerate(groups) if hi != gi
                            and lookup[h[0]]["entity_type"] == "person" and _can_join(h, [k], lookup)),
                           default=(0, None))
                if best[0] >= 1 and best[0] > own:
                    change = (gi, k, best[1], own, best[0])
                    break
            if change:
                break
        if not change:
            break
        gi, k, hi, own, new = change
        log("merge", f"LLM join overruled: {k} moved to {groups[hi]} (details agree {new:g} vs {own:g})")
        groups[hi] = groups[hi] + [k]
        groups[gi] = [x for x in groups[gi] if x != k]
        groups = [g for g in groups if g]
        moved += 1
    if moved:
        log("merge", f"LLM grouping repaired: {moved} LLM join overruled by details")
    return groups


def _link_score(agree: float, clash: int) -> float:
    """Agreeing attributes minus clashes, forgiving ONE clash: a single corrupted value
    (one witness's wrong colour) must not block a link the other attributes support.
    The clash itself is kept and found later as a conflict."""
    return agree - max(0, clash - 1) if agree else 0


DESCRIPTION_FILLER = MALE_WORDS | FEMALE_WORDS | {"person", "people", "another", "other", "who", "one"}


def _description_core(description: str) -> str:
    """The identifying part of a mention's description: 'the man behind the counter' -> 'behind counter'."""
    keep = content_tokens(description) - DESCRIPTION_FILLER
    return " ".join(t for t in tokens(description) if t in keep)


def _merge_compatible_people(groups: List[List[str]], per_source: Dict[str, dict],
                             lookup: Dict[str, dict], log) -> List[List[str]]:
    """Join witness-side person groups whose details agree: different sources, no man/woman
    clash, link score >= 1, and each is the other's unique best match. Repeats until stable.
    The witnesses' own descriptions count as one more detail ("the man behind the counter")."""
    facts_of = _with_descriptions(per_source, lookup)
    while True:
        people = [i for i, g in enumerate(groups) if lookup[g[0]]["entity_type"] == "person"
                  and not any(k.startswith("cctv") for k in g)]
        score = {}
        for i in people:
            for j in people:
                if i < j and _can_join(groups[i], groups[j], lookup):
                    s = _link_score(*_agreement(_facts_by_attr(groups[i], facts_of),
                                                _facts_by_attr(groups[j], facts_of)))
                    if s >= 1:
                        score[(i, j)] = score[(j, i)] = s

        def best(i):
            mine = sorted(((s, j) for (a, j), s in score.items() if a == i), reverse=True)
            if not mine:
                return None
            tied = [j for s, j in mine if s == mine[0][0]]
            # A tie is fine when the tied groups all agree with each other too: then they are
            # all the same person ("the man behind the counter" in three statements).
            if len(tied) == 1 or all((a, b) in score for a in tied for b in tied if a != b):
                return min(tied)
            return None

        pair = next(((i, j) for i in people for j in [best(i)] if j is not None and best(j) == i), None)
        if pair is None:
            return groups
        i, j = pair
        log("merge", f"joined {groups[j]} into {groups[i]} (details agree, score {score[pair]})")
        groups[i] = groups[i] + groups[j]
        groups = [g for n, g in enumerate(groups) if n != j]


# Colours a low-resolution camera confuses from frame to frame ("red hoodie" / "pink shirt").
COLOUR_FAMILIES = [{"red", "pink", "maroon", "orange"}, {"grey", "silver"}, {"black", "dark", "navy"},
                   {"blue", "navy"}, {"white", "cream", "beige"}, {"brown", "beige", "tan", "khaki"},
                   {"green"}, {"yellow", "gold"}, {"purple"}]
LOOKS = {"upper_clothing", "lower_clothing", "headwear", "carrying", "hair", "build", "height"}
# A pattern is a descriptor, not a colour: "patterned jacket", "checkered shirt" and "striped top" are
# compatible, and colours read off a pattern drift too much to count as a clash.
PATTERN_WORDS = {"patterned", "pattern", "checked", "checkered", "chequered", "check", "plaid", "tartan",
                 "striped", "stripes", "stripy", "floral", "printed", "print", "camo", "camouflage", "spotted"}


def _colour_families(value: str) -> set:
    words = {COLOR_ALIASES.get(t, t) for t in tokens(value)}
    return {i for i, fam in enumerate(COLOUR_FAMILIES) if words & fam}


def _patterned(value: str) -> bool:
    return bool(set(tokens(value)) & PATTERN_WORDS)


def _same_look(a: str, b: str) -> bool:
    """Lenient match for two CCTV frames: same value, same colour family, or both patterned."""
    return (values_agree(a, b) or bool(_colour_families(a) & _colour_families(b))
            or (_patterned(a) and _patterned(b)))


def _contradicts(a: str, b: str) -> bool:
    """Clearly different looks: both name a colour, and the colour families don't overlap.
    ("light-colored cap" vs "white cap" names no family on one side, so it's no contradiction;
    a patterned garment never clashes on colour.)"""
    if _patterned(a) or _patterned(b):
        return False
    fa, fb = _colour_families(a), _colour_families(b)
    return bool(fa and fb and not fa & fb)


def _facts_of(per_source: Dict[str, dict]) -> Dict[str, List[dict]]:
    return {f"{s}:{eid}": [f for f in raw["facts"] if f["entity"] == eid]
            for s, raw in per_source.items() for eid in raw["entities"]}


def _facts_by_attr(keys: List[str], facts_of: Dict[str, List[dict]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for k in keys:
        for f in facts_of.get(k, []):
            out[f["attribute"]].append(f["value"])
    return out


# Actions/positions nearly everyone in a scene does: they say little about WHO it was.
GENERIC_MOVE_WORDS = {"walked", "walk", "walks", "walking", "away", "turned", "turn", "turns", "turning", "around",
                      "left", "leave", "leaves", "leaving", "came", "come", "back", "went", "go", "out", "in",
                      "stood", "stand", "standing", "up", "down", "sat", "sit", "sitting", "talked", "talk", "talking",
                      "looked", "look", "looking", "ran", "run", "running", "moved", "move", "there", "here",
                      "nearby", "near", "close", "inside", "outside", "off", "then", "store", "shop", "scene"}
GENERIC_WEIGHT = 0.25


def _is_generic(value: str) -> bool:
    words = content_tokens(value)
    return not words or words <= GENERIC_MOVE_WORDS


def _multi_agreement(mine: List[str], theirs: List[str]) -> float:
    """Matching actions (or positions), one-to-one: 1 per specific match ("took out a black
    pistol"), GENERIC_WEIGHT per generic one ("turned around", "walked away", "left")."""
    left = list(dict.fromkeys(normalize(w) for w in theirs))
    score = 0.0
    for v in dict.fromkeys(normalize(v) for v in mine):
        w = next((w for w in left if values_agree(v, w)), None)
        if w is not None:
            left.remove(w)
            score += GENERIC_WEIGHT if _is_generic(v) or _is_generic(w) else 1.0
    return score


def _agreement(cctv: Dict[str, List[str]], witness: Dict[str, List[str]]) -> Tuple[float, int]:
    """(agreement, attributes that clash) between two people's facts. Looks count once per
    attribute, so six frames of one shirt count once; actions and positions count each matching
    specific value, generic ones low (_multi_agreement). Only a colour clash counts against
    ("white cap" vs "white hat" is the same thing in other words); two patterned garments agree."""
    agree, disagree = 0.0, 0
    for attr, values in cctv.items():
        if attr not in witness:
            continue
        if attr in ("action", "position"):
            agree += _multi_agreement(values, witness[attr])
        elif any(values_agree(v, w) or (attr in LOOKS and _patterned(v) and _patterned(w))
                 for v in values for w in witness[attr]):
            agree += 1
        elif attr not in MULTI_VALUED and all(_contradicts(v, w) for v in values for w in witness[attr]):
            disagree += 1
    return agree, disagree


def _named_role(g: List[str], lookup: Dict[str, dict]) -> bool:
    return any(lookup[k]["role"] in ("suspect", "victim") for k in g)


def _link_cctv_people(groups: List[List[str]], per_source: Dict[str, dict], lookup: Dict[str, dict],
                      log) -> List[List[str]]:
    """Give CCTV people a witness-side identity instead of leaving them as separate people.

    1. CCTV person mentions are taken out of their groups (so a witness's invented person is
       never glued to a real person on camera by the LLM), and mentions of the same person in
       different frames are merged: their looks agree, allowing colour drift and patterns.
    2. Each CCTV person joins the witness-side person whose attributes agree most: link score
       (agreeing minus clashing attributes, one clash forgiven; a man/woman mismatch counts as a
       clash) >= 1, and a unique best. Roles are soft: they never block a link and only break a tie, in favour of
       a person some source called suspect or victim. Otherwise it stays its own entity.
    """
    is_cctv = lambda k: k.startswith("cctv")  # noqa: E731
    is_person = lambda k: lookup[k]["entity_type"] == "person"  # noqa: E731
    facts_of = _facts_of(per_source)
    cctv_people = [k for g in groups for k in g if is_cctv(k) and is_person(k)]
    if not cctv_people:
        return groups
    rest = [g for g in ([k for k in g if not (is_cctv(k) and is_person(k))] for g in groups) if g]

    clusters: List[List[str]] = []
    for k in cctv_people:
        mine = _facts_by_attr([k], facts_of)
        for c in clusters:
            theirs = _facts_by_attr(c, facts_of)
            shared = [a for a in mine if a in theirs and a in LOOKS]
            agrees = [a for a in shared if any(_same_look(v, w) for v in mine[a] for w in theirs[a])]
            clash = [a for a in shared if a not in agrees
                     and all(_contradicts(v, w) for v in mine[a] for w in theirs[a])]
            # Gender isn't checked between frames: a 320x240 frame misreads it ("man in red hoodie").
            if k.split(":")[0] not in _sources(c) and agrees and not clash:
                c.append(k)
                break
        else:
            clusters.append([k])
    if len(clusters) < len(cctv_people):
        log("merge", f"CCTV: {len(cctv_people)} person mentions -> {len(clusters)} people across frames")

    targets = [g for g in rest if is_person(g[0])]
    for c in clusters:
        mine = _facts_by_attr(c, facts_of)
        # A man/woman mismatch is one more clash here, not a veto: the camera's reading of gender
        # can be wrong in a single frame, just like a colour.
        scored = sorted((((_link_score(a, d), _named_role(g, lookup)), a, d, i) for i, g in enumerate(targets)
                         for a, d0 in [_agreement(mine, _facts_by_attr(g, facts_of))]
                         for d in [d0 + (len(_genders(g + c, lookup)) > 1)]), reverse=True)
        if scored and scored[0][0][0] >= 1 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            _, a, d, i = scored[0]
            g = targets[i]
            log("merge", f"CCTV person ({', '.join(sorted({v for vs in mine.values() for v in vs}))}) "
                         f"-> witness-side person {g} (agreement {a:g}, {d} clash)")
            g.extend(c)
        else:
            rest.append(c)
    return rest


def _vote_role(group: List[str], lookup: Dict[str, dict]) -> Tuple[str, Dict[str, str]]:
    """Role of one merged person by majority vote over independent sources.

    Each witness has one vote; all CCTV frames together are one source (their majority).
    CCTV "bystander" is no vote: it is the frame prompt's default when the action isn't obvious.
    Ties: suspect/victim over bystander, then suspect before victim. Returns (role, votes)."""
    votes: Dict[str, str] = {}
    cctv = Counter(lookup[k]["role"] for k in group
                   if k.startswith("cctv") and lookup[k]["role"] in ("suspect", "victim"))
    if cctv:
        votes["cctv"] = max(cctv.items(), key=lambda rn: (rn[1], rn[0] == "suspect"))[0]
    for k in group:
        src, role = k.split(":")[0], lookup[k]["role"]
        if not src.startswith("cctv") and src not in votes and role in ("suspect", "victim", "bystander"):
            votes[src] = role
    if not votes:
        return "bystander", votes
    count = Counter(votes.values())
    order = {"suspect": 2, "victim": 1, "bystander": 0}
    return max(count, key=lambda r: (count[r], r != "bystander", order[r])), votes


def merge_entities(per_source: Dict[str, dict], use_llm: bool = True,
                   log=lambda step, detail: None) -> Dict[str, dict]:
    """Return {"source:eid": {"entity": canonical_id, "entity_type", "role", "role_votes"}}.

    Roles never decide who is who. People are merged by attributes and source consistency
    first (LLM grouping without roles, or one group per mention; then splitting groups that
    repeat a source or mix a man and a woman; joining lone men / lone women across witnesses;
    joining groups whose details agree; linking CCTV people). Only then does each merged
    person get a role, by majority vote; disagreements are kept in role_votes.
    """
    lookup = {f"{s}:{eid}": e for s, raw in per_source.items() for eid, e in raw["entities"].items()}
    groups = _llm_groups(per_source, log) if use_llm else None
    if groups is None:
        groups = _rule_based_groups(per_source)
    groups = _detach_cctv_people(groups, lookup)
    groups = _split_inconsistent(groups, lookup, log)
    if use_llm:
        groups = _check_llm_joins(groups, per_source, lookup, log)
    groups = _merge_lone_genders(groups, lookup, log)
    groups = _merge_compatible_people(groups, per_source, lookup, log)
    groups = _link_cctv_people(groups, per_source, lookup, log)

    mapping: Dict[str, dict] = {}
    counters: Counter = Counter()
    for group in groups:
        etype = lookup[group[0]]["entity_type"]
        votes: Dict[str, str] = {}
        if etype == "location":
            cid = LOCATION_ID
        else:
            counters[etype] += 1
            cid = f"{etype}_{counters[etype]}"
        if etype == "person":
            role, votes = _vote_role(group, lookup)
            if len(set(votes.values())) > 1:
                log("merge", f"{cid}: role by vote = {role}; sources disagree {votes}")
        else:
            roles = Counter(lookup[k]["role"] for k in group if lookup[k]["role"] != "none")
            role = roles.most_common(1)[0][0] if roles else "none"
        for k in group:
            mapping[k] = {"entity": cid, "entity_type": etype, "role": role, "role_votes": votes}
    return mapping


def role_votes(mapping: Dict[str, dict]) -> Dict[str, Dict[str, str]]:
    """{entity: {origin: role}} for people whose sources gave more than one role."""
    return {m["entity"]: m["role_votes"] for m in mapping.values()
            if len(set(m.get("role_votes", {}).values())) > 1}


# ---------------------------------------------------------------- confidence

def base_confidence(source_type: str, hedge: float, cfg: dict) -> float:
    c = cfg["confidence"]
    base = {"cctv": c["cctv"], "witness": c["witness"], "followup": c["followup"]}[source_type]
    return round(max(0.05, base - c["hedge_penalty"] * hedge), 3)


def build_facts(per_source: Dict[str, dict], mapping: Dict[str, dict],
                viewpoints: Dict[str, str], cfg: dict, start_id: int = 1) -> List[Fact]:
    facts: List[Fact] = []
    n = start_id
    for source, raw in per_source.items():
        stype = "cctv" if source.startswith("cctv") else "witness"
        for f in raw["facts"]:
            m = mapping[f"{source}:{f['entity']}"]
            if f.get("sentence"):
                # Statements: the witness's own sentence decides ("pretty sure" stays mild),
                # because the model's quote often cuts off the hedge.
                hedge = hedge_level(f["sentence"])
            else:
                # Frames (no sentence): hedge words in the quote; a hedge the model flagged
                # but we can't see is treated as strong.
                hedge = hedge_level(f["quote"])
                if f["hedged"] and hedge == 0:
                    hedge = 1.0
            facts.append(Fact(
                fact_id=f"f{n:03d}", entity=m["entity"], entity_type=m["entity_type"],
                role=m["role"], attribute=f["attribute"], value=f["value"],
                source=source, source_type=stype,
                viewpoint=viewpoints.get(source, "CCTV camera" if stype == "cctv" else ""),
                quote=f["quote"], hedged=hedge > 0,
                base_confidence=base_confidence(stype, hedge, cfg),
            ))
            n += 1
    return facts


def extract_scene(witnesses: Dict[str, dict], frames: List[str],
                  log=lambda step, detail: None) -> Tuple[List[Fact], Dict[str, Dict[str, str]]]:
    """witnesses: {"witness_A": {"statement": str, "viewpoint": str}}; frames: image paths.
    Returns (facts, role disagreements {entity: {origin: role}})."""
    cfg = load_config()
    per_source: Dict[str, dict] = {}
    for wid, w in witnesses.items():
        per_source[wid] = extract_statement(w["statement"], wid, log)
        log("extract", f"{wid}: {len(per_source[wid]['facts'])} facts")
    for i, path in enumerate(frames[: cfg["extraction"]["max_cctv_frames"]], 1):
        sid = f"cctv_frame_{i:02d}"
        per_source[sid] = extract_frame(path, sid, log)
        log("extract", f"{sid} ({path}): {len(per_source[sid]['facts'])} facts")

    mapping = merge_entities(per_source, use_llm=cfg["extraction"]["llm_entity_merge"], log=log)
    log("merge", f"{len(mapping)} mentions -> {len({m['entity'] for m in mapping.values()})} entities")
    viewpoints = {wid: w.get("viewpoint", "") for wid, w in witnesses.items()}
    return build_facts(per_source, mapping, viewpoints, cfg), role_votes(mapping)


# ---------------------------------------------------------------- follow-ups

def extract_answer(question: dict, answer: str, fact_id: str) -> Optional[Fact]:
    """Turn a witness's answer to one of our questions into a fact (or None)."""
    cfg = load_config()
    out = llm.chat_json(ANSWER_PROMPT.format(
        question=question["text"], attribute=question["attribute"],
        entity_type=question["entity_type"], answer=answer), system=SYSTEM, temperature=0.0)
    value = _strip_article(str(out.get("value") or ""))
    if not value or _is_placeholder(value) or grounding_problem(value, answer):
        return None
    hedge = hedge_level(answer)
    if out.get("hedged") and hedge == 0:
        hedge = 1.0
    return Fact(
        fact_id=fact_id, entity=question["entity"], entity_type=question["entity_type"],
        role=question["role"], attribute=question["attribute"], value=str(value),
        source=f"{question['witness']}_followup", source_type="followup",
        viewpoint=question.get("viewpoint", ""), quote=answer, hedged=hedge > 0,
        base_confidence=base_confidence("followup", hedge, cfg),
    )
