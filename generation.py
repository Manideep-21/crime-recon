"""Confidence-gated image generation with SDXL or SD 3.5.

Only facts chosen by conflicts.select_facts() reach the prompt. Withheld facts
are listed next to the image as "uncertain", never drawn. The prompt must also
fit the CLIP text encoder (77 tokens): facts are added by priority, and those
that don't fit are listed as "not drawn: prompt limit".
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from schema import LOCATION_ID, Fact

STYLE = "CCTV photo, wide angle, realistic"
NEGATIVE = ("text, watermark, cartoon, painting, deformed, extra people, crowd, "
            "close-up portrait, visible face, detailed face")


def _vals(attrs: Dict[str, List[str]], key: str) -> List[str]:
    return attrs.get(key, [])


def _person_phrase(a: Dict[str, List[str]]) -> str:
    adj = " ".join(_vals(a, "build")[:1] + _vals(a, "height")[:1])
    s = f"a {adj} person".replace("  ", " ")
    hair = _vals(a, "hair")
    if hair:
        s += f" with {hair[0]} hair"
    wear = _vals(a, "upper_clothing")[:1] + _vals(a, "lower_clothing")[:1] + _vals(a, "headwear")[:1]
    if wear:
        s += " wearing " + " and ".join(wear)
    if _vals(a, "carrying"):
        s += f", carrying {_vals(a, 'carrying')[0]}"
    if _vals(a, "action"):  # one image = one moment, so one action
        s += f", {_vals(a, 'action')[0]}"
    if _vals(a, "position"):
        s += f", {_vals(a, 'position')[0]}"
    return s


def _thing_phrase(etype: str, a: Dict[str, List[str]]) -> str:
    noun = (_vals(a, "vehicle_type") or _vals(a, "weapon_type") or _vals(a, "object_type")
            or [etype])[0]
    color = _vals(a, "color")
    s = f"a {color[0]} {noun}" if color else f"a {noun}"
    for key in ("state", "position", "action"):
        if _vals(a, key):
            s += f", {_vals(a, key)[0]}"
    if _vals(a, "held_by"):
        s += f", held by {_vals(a, 'held_by')[0]}"
    return s


MAX_TOKENS = 77  # CLIP text encoder window, start and end tokens included
SHORT_ACTION = 6  # words
# One image is one moment: walking, leaving or direction can't be drawn.
MOVEMENT = re.compile(
    r"\b(walk\w*|ran|runs?|running|left|leav\w*|came|comes|coming|enter\w*|exit\w*|went|goes|going|"
    r"turn\w*|towards?|away|fled|flee\w*|chas\w*|follow\w*|approach\w*|return\w*)\b", re.I)
ROLE_ORDER = {"suspect": 0, "victim": 1, "bystander": 2, "none": 3}


def approx_tokens(text: str) -> int:
    """Rough CLIP token count (words + punctuation + start/end) when no tokenizer is loaded (mock backend)."""
    return len(re.findall(r"\w+|[^\w\s]", text)) + 2


def _priority(f: Fact) -> int:
    """1 colour+garment, 2 headwear/carrying/looks/things, 3 position/place, 4 one action, 5 the rest."""
    if f.entity_type == "person":
        return {"upper_clothing": 1, "lower_clothing": 1, "headwear": 2, "carrying": 2, "hair": 2,
                "build": 2, "height": 2, "position": 3, "action": 4}.get(f.attribute, 5)
    if f.entity_type == "location":
        return {"place_type": 3, "lighting": 5, "time_of_day": 5}.get(f.attribute, 5)
    return {"weapon_type": 2, "vehicle_type": 2, "object_type": 2, "color": 2,
            "position": 3, "state": 3, "held_by": 4, "action": 4}.get(f.attribute, 5)


def _render(facts: List[Fact], people: List[str], person_facts: Dict[str, Fact]) -> Tuple[str, List[dict]]:
    """Prompt text and rendered entities for exactly these facts (+ these people, even without facts).

    person_facts: any one fact per person, for its type and role when none of its facts is selected.
    """
    by_entity: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[str, Fact] = {}
    for f in sorted(facts, key=lambda x: x.fact_id):  # keep the order witnesses told it
        by_entity[f.entity][f.attribute].append(f.value)
        meta.setdefault(f.entity, f)
    ents = list(people) + sorted(e for e in by_entity if e != LOCATION_ID and e not in people)

    parts = []
    loc = by_entity.get(LOCATION_ID)
    if loc:
        place = "a " + (_vals(loc, "place_type")[:1] or ["scene"])[0]
        if _vals(loc, "time_of_day"):
            place += f" at {_vals(loc, 'time_of_day')[0]}"
        if _vals(loc, "lighting"):
            place += f", {_vals(loc, 'lighting')[0]}"
        if _vals(loc, "notable_features"):
            place += ", with " + ", ".join(_vals(loc, "notable_features")[:2])
        parts.append(place)

    rendered = []
    for e in ents:
        f0 = meta.get(e) or person_facts[e]
        etype = f0.entity_type
        phrase = _person_phrase(by_entity[e]) if etype == "person" else _thing_phrase(etype, by_entity[e])
        parts.append(phrase)
        rendered.append({"entity": e, "entity_type": etype, "role": f0.role,
                         "attributes": {k: v for k, v in by_entity[e].items()}})
    n = sum(1 for r in rendered if r["entity_type"] == "person")
    count = f"exactly {n} {'person' if n == 1 else 'people'}" if n else "no people"
    return f"{STYLE}, {count}. " + "; ".join(parts), rendered


def _not_drawn(f: Fact, reason: str) -> dict:
    return {"fact_id": f.fact_id, "entity": f.entity, "attribute": f.attribute, "value": f.value,
            "source": f.source, "confidence": f.confidence, "reason": reason}


def build_prompt(included: List[Fact], count_tokens: Optional[Callable[[str], int]] = None,
                 max_tokens: int = MAX_TOKENS) -> Tuple[str, str, List[dict], List[dict]]:
    """Return (prompt, negative_prompt, rendered_entities, not_drawn).

    Facts are added greedily by priority (see _priority), people in role order (suspects first),
    keeping the prompt within max_tokens of the text encoder. Movement/direction actions and
    all but one short action per person are never drawn. Everything left out is in not_drawn
    with its reason; rendered_entities holds only what is really in the prompt.
    """
    count = count_tokens or approx_tokens
    facts = sorted(included, key=lambda x: x.fact_id)
    role = {f.entity: f.role for f in facts}
    people = sorted({f.entity for f in facts if f.entity_type == "person"},
                    key=lambda e: (ROLE_ORDER.get(role[e], 3), e))
    person_facts = {f.entity: f for f in facts if f.entity_type == "person"}

    not_drawn: List[dict] = []
    candidates: List[Fact] = []
    chosen_action: Dict[str, Fact] = {}
    for f in facts:
        if f.attribute in ("action", "position") and MOVEMENT.search(f.value):
            not_drawn.append(_not_drawn(f, "movement/direction"))
            continue
        if f.attribute == "action":
            if f.entity in chosen_action or len(f.value.split()) > SHORT_ACTION:
                not_drawn.append(_not_drawn(f, "one short action per person"))
                continue
            chosen_action[f.entity] = f
        candidates.append(f)

    def fits(sel: List[Fact], ppl: List[str]) -> bool:
        return count(_render(sel, ppl, person_facts)[0]) <= max_tokens

    kept_people: List[str] = []
    selected: List[Fact] = []
    for e in people:  # tier 1: each person with their colour+garment, before the next person
        clothes = [f for f in candidates if f.entity == e and _priority(f) == 1]
        # Someone described by their clothes only goes in if at least one garment fits too.
        if not fits(selected + clothes[:1], kept_people + [e]):
            continue
        kept_people.append(e)
        for f in clothes:
            if fits(selected + [f], kept_people):
                selected.append(f)
            else:
                not_drawn.append(_not_drawn(f, "prompt limit"))
    for f in candidates:  # people who don't fit at all: none of their facts are drawn
        if f.entity_type == "person" and f.entity not in kept_people:
            not_drawn.append(_not_drawn(f, "prompt limit"))
    for tier in range(1, 6):
        for f in candidates:
            if _priority(f) != tier or f in selected or (f.entity_type == "person" and (tier == 1 or f.entity not in kept_people)):
                continue
            if fits(selected + [f], kept_people):
                selected.append(f)
            else:
                not_drawn.append(_not_drawn(f, "prompt limit"))
    prompt, rendered = _render(selected, kept_people, person_facts)
    order = {f.fact_id: i for i, f in enumerate(facts)}
    return prompt, NEGATIVE, rendered, sorted(not_drawn, key=lambda d: order[d["fact_id"]])


class Generator:
    """Lazy-loads the image model on first use.

    backend 'sdxl' (generation.model), 'sd35' (generation.sd35_model, Stable Diffusion 3.5 Medium),
    or 'mock' (placeholder image, no GPU).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self._tokenizer = None

    def _model_id(self) -> str:
        if self.cfg["backend"] == "sd35":
            return self.cfg.get("sd35_model", "stabilityai/stable-diffusion-3.5-medium")
        return self.cfg["model"]

    def count_tokens(self, text: str) -> int:
        """Tokens the model's CLIP text encoder sees (start/end included); rough count for 'mock'."""
        if self.cfg["backend"] == "mock":
            return approx_tokens(text)
        if self._tokenizer is None:
            if self.pipe is not None:
                self._tokenizer = self.pipe.tokenizer
            else:  # the pipeline's own tokenizer, without loading the model
                from transformers import CLIPTokenizer

                self._tokenizer = CLIPTokenizer.from_pretrained(self._model_id(), subfolder="tokenizer")
        return len(self._tokenizer(text).input_ids)

    def _load(self):
        import torch

        if self.cfg["backend"] == "sd35":
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(self._model_id(), torch_dtype=torch.bfloat16)
        else:
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.cfg["model"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        offload = self.cfg.get("cpu_offload", True)
        if offload == "sequential":  # layer by layer: slowest, but fits in a few GB
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
            # SDXL's default tile is 1024 px, so a 1024x1024 image would never be tiled
            pipe.vae.tile_sample_min_size = 512
            pipe.vae.tile_latent_min_size = 64
        elif offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe

    def generate(self, prompt: str, negative: str, seed: int):
        if self.cfg["backend"] == "mock":
            return self._mock(prompt, seed)
        import torch

        if self.pipe is None:
            self._load()
        g = torch.Generator(device="cpu").manual_seed(seed)
        sd35 = self.cfg["backend"] == "sd35"
        return self.pipe(prompt=prompt, negative_prompt=negative,
                         num_inference_steps=self.cfg.get("sd35_steps", 40) if sd35 else self.cfg["steps"],
                         guidance_scale=self.cfg.get("sd35_guidance", 4.5) if sd35 else self.cfg["guidance_scale"],
                         width=self.cfg["width"], height=self.cfg["height"],
                         generator=g).images[0]

    def unload(self):
        """Free GPU memory (e.g. before the VLM needs it)."""
        if self.pipe is not None:
            import torch

            self.pipe = None
            torch.cuda.empty_cache()

    @staticmethod
    def _mock(prompt: str, seed: int):
        import textwrap

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (768, 768), (60 + seed % 40, 60, 70))
        d = ImageDraw.Draw(img)
        d.text((20, 20), "MOCK IMAGE (generation.backend = mock)", fill=(255, 255, 0))
        d.multiline_text((20, 60), "\n".join(textwrap.wrap(prompt, 80)), fill=(230, 230, 230))
        return img
