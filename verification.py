"""Automatic verification of a generated image against the graph.

1. Object check (OWLv2 open-vocabulary detector): every rendered entity should
   be found; anything found that the graph doesn't contain is a hallucination.
   People are found with attribute-specific queries ("person wearing red hoodie")
   and matched to person boxes one-to-one by score; more people in the image than
   were drawn counts as a hallucination.
2. Attribute check (optional, TIFA-style): yes/no questions to the VLM,
   e.g. "Is there a person wearing a red hoodie?".
3. Overlay: one box per matched entity, coloured by how well supported it is.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

LEVEL_COLORS = {
    "confirmed": (46, 160, 67),     # green: CCTV or 2+ independent sources
    "single": (230, 190, 30),       # yellow: one witness only
    "contested": (240, 130, 20),    # orange: one side of a conflict (this hypothesis)
    "unsupported": (220, 40, 40),   # red: drawn by the model, not in evidence
}

VQA_TEMPLATES = {
    "upper_clothing": "Is there a person wearing {v}?",
    "lower_clothing": "Is there a person wearing {v}?",
    "headwear": "Is there a person wearing {v}?",
    "hair": "Is there a person with {v} hair?",
    "carrying": "Is there a person carrying {v}?",
    "build": "Is there a {v} person?",
    "vehicle_type": "Is there a {v}?",
    "weapon_type": "Is there a {v}?",
    "object_type": "Is there a {v}?",
    "place_type": "Is this a {v}?",
    "time_of_day": "Does this scene look like it is {v}?",
}


# Distractor label -> words that mean that item when a witness says a person carries it.
CARRIED_ITEMS = {
    "bag": {"bag", "backpack", "rucksack", "purse", "handbag", "briefcase", "satchel"},
    "knife": {"knife", "blade"},
    "gun": {"gun", "pistol", "revolver", "rifle"},
}


def carried_by(label: str, entity: dict) -> bool:
    """Does this drawn person carry/hold an item the detector calls `label`?"""
    words = CARRIED_ITEMS.get(label, {label})
    a = entity["attributes"]
    return any(t in words for v in a.get("carrying", []) for t in v.lower().split())


def person_query(entity: dict) -> str:
    """Attribute-specific detector query for a person, e.g. 'person wearing red hoodie'."""
    a = entity["attributes"]
    for key in ("upper_clothing", "lower_clothing", "headwear"):
        if a.get(key):
            return f"person wearing {a[key][0]}"
    if a.get("carrying"):
        return f"person carrying {a['carrying'][0]}"
    return "person"


def detection_label(entity: dict) -> str:
    """What to ask the detector for, e.g. 'person', 'knife', 'black backpack'."""
    a = entity["attributes"]
    if entity["entity_type"] == "person":
        return "person"
    for key in ("vehicle_type", "weapon_type", "object_type"):
        if a.get(key):
            return a[key][0]
    return entity["entity_type"]


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: List[dict], iou_thr: float = 0.5) -> List[dict]:
    keep: List[dict] = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        if all(k["label"] != d["label"] or _iou(k["box"], d["box"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _overlap_share(item, person) -> float:
    """Share of the item box that lies inside the person box."""
    ix = max(0, min(item[2], person[2]) - max(item[0], person[0]))
    iy = max(0, min(item[3], person[3]) - max(item[1], person[1]))
    area = (item[2] - item[0]) * (item[3] - item[1])
    return ix * iy / area if area > 0 else 0.0


def _same_person(a, b) -> bool:
    """OWLv2 often boxes one person several times (whole body, torso, legs).

    Same person if the boxes overlap a lot, one sits inside the other, or they are
    stacked vertically (torso above legs) with most of their width shared.
    """
    if _iou(a, b) >= 0.5:
        return True
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    if smaller > 0 and ix * iy / smaller >= 0.6:
        return True
    narrower = min(a[2] - a[0], b[2] - b[0])
    gap = max(a[1], b[1]) - min(a[3], b[3])  # < 0 when they overlap vertically
    height = max(a[3] - a[1], b[3] - b[1])
    return narrower > 0 and ix / narrower >= 0.6 and gap <= 0.2 * height


def merge_people(dets: List[dict]) -> List[dict]:
    """One detection per person: merge part boxes into their union, keeping the best score per query.

    Repeats until stable, because a merged box can come to touch a group it didn't before
    (arm box + torso box only meet once a third box bridges them).
    """
    people = [{**d, "box": list(d["box"]), "scores": dict(d.get("scores", {}))}
              for d in sorted(dets, key=lambda x: -x["score"])]
    merged = True
    while merged:
        merged = False
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                p, d = people[i], people[j]
                if _same_person(p["box"], d["box"]):
                    p["box"] = [min(p["box"][0], d["box"][0]), min(p["box"][1], d["box"][1]),
                                max(p["box"][2], d["box"][2]), max(p["box"][3], d["box"][3])]
                    for q, sc in d["scores"].items():
                        p["scores"][q] = max(p["scores"].get(q, 0.0), sc)
                    del people[j]
                    merged = True
                    break
            if merged:
                break
    return [p for p in people if not any(_is_fragment(p["box"], q["box"]) for q in people if q is not p)]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _is_fragment(small, big) -> bool:
    """A tiny box (an arm, a hand) inside or touching a much larger person box.

    Under 10% of the big box's area: a real person further back in the aisle is usually
    bigger than that, so they are kept even when they overlap the big box's edge.
    """
    if _area(big) <= 0 or _area(small) >= 0.1 * _area(big):
        return False
    gap_x = max(big[0] - small[2], small[0] - big[2], 0)
    overlap_y = min(small[3], big[3]) - max(small[1], big[1])
    return gap_x <= 0.1 * (big[2] - big[0]) and overlap_y >= 0.5 * (small[3] - small[1])


def assign_people(queries: Dict[str, str], people: List[dict], min_score: float = 0.0) -> Dict[str, int]:
    """Match person entities to person boxes one-to-one.

    queries: entity -> its detector query. A pair needs at least min_score for the entity's
    own query (so the cashier is not given to a random bystander), and the box must score
    higher for that query than for every other drawn person's query (a box that looks more
    like "blue jacket" is never the cashier in a green uniform). Maximises the number of
    matched entities, then the summed score. Small exhaustive search (a scene has few people).
    """
    ents = list(queries)

    def score(e: str, i: int) -> float:
        s = people[i].get("scores", {})
        return s.get(queries[e], s.get("person", 0.0))

    def allowed(e: str, i: int) -> bool:
        rivals = [o for o in ents if o != e and queries[o] != queries[e]]
        return score(e, i) >= min_score and all(score(e, i) > score(o, i) for o in rivals)

    best: Tuple[Dict[str, int], float] = ({}, -1.0)

    def search(j: int, used: frozenset, cur: Dict[str, int], total: float) -> None:
        nonlocal best
        if j == len(ents):
            if (len(cur), total) > (len(best[0]), best[1]):
                best = (dict(cur), total)
            return
        for i in range(len(people)):
            if i not in used and allowed(ents[j], i):
                cur[ents[j]] = i
                search(j + 1, used | {i}, cur, total + score(ents[j], i))
                del cur[ents[j]]
        search(j + 1, used, cur, total)

    search(0, frozenset(), {}, 0.0)
    return best[0]


class Verifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = self.processor = None
        self.device = "cpu"

    def _load(self):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(self.cfg["model"])
        self.model = Owlv2ForObjectDetection.from_pretrained(self.cfg["model"]).to(self.device).eval()

    def detect(self, image, queries: List[str]) -> List[dict]:
        if self.cfg["backend"] == "none":
            return []
        import torch

        if self.model is None:
            self._load()
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Every box's score for every query (the processor's post-processing keeps only the top one).
        probs = outputs.logits[0].sigmoid()
        best, labels = probs.max(-1)
        # OWLv2 pads the image to a square, so boxes are relative to that square.
        side = max(image.size)
        dets = []
        for i in (best >= self.cfg["score_threshold"]).nonzero().flatten().tolist():
            cx, cy, w, h = (outputs.pred_boxes[0, i] * side).tolist()
            dets.append({"label": queries[int(labels[i])], "score": round(float(best[i]), 3),
                         "box": [max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                                 min(image.size[0], cx + w / 2), min(image.size[1], cy + h / 2)],
                         "scores": {q: round(float(p), 3) for q, p in zip(queries, probs[i].tolist())}})
        return nms(dets)

    def verify(self, image, rendered: List[dict], vqa: bool = False) -> dict:
        person_queries = {e["entity"]: person_query(e) for e in rendered if e["entity_type"] == "person"}
        expected = [(e["entity"], person_queries.get(e["entity"]) or detection_label(e)) for e in rendered]
        person_labels = set(person_queries.values()) | {"person"}
        queries = sorted({lbl for _, lbl in expected} | set(self.cfg["distractors"]) | {"person"})
        dets = self.detect(image, queries)

        # People: one box per person, matched to entities one-to-one by attribute-specific score.
        people = merge_people([d for d in dets if d["label"] in person_labels])
        for p in people:
            p["label"] = "person"
        assignment = assign_people(person_queries, people, self.cfg["score_threshold"])
        matched: Dict[str, dict] = {}
        for entity, i in assignment.items():
            matched[entity] = {**people[i], "query": person_queries[entity],
                               "entity_score": people[i]["scores"].get(person_queries[entity], people[i]["score"])}
        # Person-count check: any person box left over is a person nobody described.
        extra_people = [{**p, "reason": f"extra person ({len(people)} in image, {len(person_queries)} drawn)"}
                        for i, p in enumerate(people) if i not in set(assignment.values())]

        # Other entities: best box with their label.
        others = [d for d in dets if d["label"] not in person_labels]
        used = set()
        for entity, label in expected:
            if entity in person_queries:
                continue
            cands = [i for i, d in enumerate(others) if d["label"] == label and i not in used]
            if cands:
                best = max(cands, key=lambda i: others[i]["score"])
                used.add(best)
                matched[entity] = others[best]
        hallucination_min = self.cfg.get("hallucination_score", self.cfg["score_threshold"] + 0.1)
        by_id = {e["entity"]: e for e in rendered}
        carried, hallucinated = [], list(extra_people)
        for i, d in enumerate(others):
            if i in used or d["score"] < hallucination_min:
                continue
            # A bag on a person who is drawn carrying a backpack is evidence, not a hallucination.
            owner = next((e for e in person_queries if e in matched and carried_by(d["label"], by_id[e])
                          and _overlap_share(d["box"], matched[e]["box"]) >= 0.3), None)
            if owner:
                carried.append({**d, "carried_by": owner})
            else:
                hallucinated.append(d)
        missing = [e for e, _ in expected if e not in matched]

        report = {
            "expected": [{"entity": e, "label": l} for e, l in expected],
            "matched": matched, "missing": missing, "hallucinated": hallucinated, "carried": carried,
            "person_count": {"drawn": len(person_queries), "in_image": len(people)},
            "object_faithfulness": round(len(matched) / len(expected), 3) if expected else 1.0,
            "vqa": [],
        }
        if vqa and self.cfg["backend"] != "none":
            report["vqa"] = self.attribute_check(image, rendered)
            answered = [q for q in report["vqa"] if q["answer"] is not None]
            report["attribute_accuracy"] = (round(sum(q["answer"] for q in answered) / len(answered), 3)
                                            if answered else None)
        return report

    @staticmethod
    def attribute_check(image, rendered: List[dict]) -> List[dict]:
        import llm

        out = []
        for e in rendered:
            for attr, values in e["attributes"].items():
                if attr not in VQA_TEMPLATES:
                    continue
                q = VQA_TEMPLATES[attr].format(v=values[0])
                try:
                    ans = llm.yes_no(q, image)
                except Exception:
                    ans = None
                out.append({"entity": e["entity"], "attribute": attr, "question": q, "answer": ans})
        return out


def draw_overlay(image, report: dict, entity_levels: Dict[str, str], blur_faces: bool = True):
    """Boxes + labels on a copy of the image. Faces (top of each person box) are blurred."""
    from PIL import ImageDraw, ImageFilter

    img = image.copy().convert("RGB")
    person_boxes = [d["box"] for e, d in report["matched"].items() if d["label"] == "person"]
    person_boxes += [d["box"] for d in report["hallucinated"] if d["label"] == "person"]
    if blur_faces:
        for x1, y1, x2, y2 in person_boxes:
            head = (int(x1), int(y1), int(x2), int(y1 + (y2 - y1) * 0.22))
            if head[2] > head[0] and head[3] > head[1]:
                region = img.crop(head).filter(ImageFilter.GaussianBlur(12))
                img.paste(region, head)

    d = ImageDraw.Draw(img)

    def box(b, color, text):
        d.rectangle(b, outline=color, width=4)
        tx, ty = b[0] + 3, max(0, b[1] - 16)
        d.rectangle([tx - 3, ty, tx + 7 * len(text), ty + 15], fill=color)
        d.text((tx, ty + 2), text, fill=(0, 0, 0))

    for entity, det in report["matched"].items():
        level = entity_levels.get(entity, "single")
        box(det["box"], LEVEL_COLORS[level], f"{entity} [{level}]")
    for det in report["hallucinated"]:
        box(det["box"], LEVEL_COLORS["unsupported"], f"{det['label']} [NOT IN EVIDENCE]")
    return img


def summarize(report: Optional[dict]) -> str:
    if not report:
        return "not verified"
    s = (f"objects found {len(report['matched'])}/{len(report['expected'])}, "
         f"hallucinated {len(report['hallucinated'])}")
    if report.get("person_count"):
        s += f", people {report['person_count']['in_image']} in image / {report['person_count']['drawn']} drawn"
    if report.get("attribute_accuracy") is not None:
        s += f", attribute accuracy {report['attribute_accuracy']:.0%}"
    return s
