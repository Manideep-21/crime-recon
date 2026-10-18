"""Tests for the model-free logic, using the hand-made sample scene.

Run:  python tests/test_logic.py      (or: pytest tests/)
No GPU, Neo4j or LLM needed.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conflicts import build_hypotheses, find_conflicts, find_gaps, select_facts  # noqa: E402
from extraction import (_add_sentences, _clean_raw, _merge_lone_genders, _strip_article,  # noqa: E402
                        build_facts, grounding_problem)
from generation import build_prompt  # noqa: E402
from graph import SceneGraph  # noqa: E402
from schema import Fact  # noqa: E402
from verification import Verifier, draw_overlay, merge_people, person_query  # noqa: E402
from matching import values_agree  # noqa: E402
from questions import generate_questions, is_leading  # noqa: E402

SCENE = os.path.join(ROOT, "data", "scenes", "sample_robbery")
CONF = {"corroboration_bonus": 0.2}
RENDER, HYP = 0.6, 0.4


def load():
    graph = SceneGraph.load(os.path.join(SCENE, "sample_facts.json"), CONF)
    witnesses = json.load(open(os.path.join(SCENE, "scene.json")))["witnesses"]
    return graph, witnesses


def test_values_agree():
    assert values_agree("red hoodie", "bright red hoodie")
    assert not values_agree("red hoodie", "blue jacket")
    assert values_agree("ran out", "ran out the door towards the parking lot")
    assert values_agree("dark grey jacket", "grey jacket")
    assert not values_agree("black cap", "white cap")


def test_corroboration_raises_confidence():
    g, _ = load()
    store = [f for f in g.facts if f.attribute == "place_type"]
    assert all(f.confidence == 0.9 for f in store), [f.confidence for f in store]
    assert all(set(f.support) == {"witness_B", "witness_C"} for f in store)


def test_conflict_detected_and_preserved():
    g, _ = load()
    conflicts = find_conflicts(g)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert (c["entity"], c["attribute"]) == ("person_1", "upper_clothing")
    assert [o["value"] for o in c["options"]] == ["red hoodie", "blue jacket"]
    hyps = build_hypotheses(conflicts)
    assert len(hyps) == 2


def test_actions_are_not_conflicts():
    g, _ = load()
    assert not any(c["attribute"] == "action" for c in find_conflicts(g))


def test_low_confidence_detail_is_withheld():
    g, _ = load()
    conflicts = find_conflicts(g)
    h1, h2 = build_hypotheses(conflicts)
    inc, withheld, contested = select_facts(g, conflicts, h1, RENDER, HYP)
    assert "knife" in [f.value for f in withheld]
    prompt, _, rendered, _ = build_prompt(inc)
    assert "red hoodie" in prompt and "knife" not in prompt and "blue jacket" not in prompt
    assert "exactly 2 people" in prompt

    inc2, _, contested2 = select_facts(g, conflicts, h2, RENDER, HYP)
    prompt2, _, _, _ = build_prompt(inc2)
    assert "blue jacket" in prompt2 and "red hoodie" not in prompt2
    assert contested2


def test_gaps():
    g, _ = load()
    gaps = {(x["entity"], x["attribute"]) for x in find_gaps(g)}
    assert ("person_1", "position") in gaps
    assert ("person_2", "lower_clothing") in gaps
    assert ("person_1", "upper_clothing") not in gaps
    assert ("scene", "place_type") not in gaps


def test_leading_filter():
    g, w = load()
    vocab_b = set(w["witness_B"]["statement"].lower().replace(".", " ").replace(",", " ").split())
    assert is_leading("Are you sure it wasn't red hoodie?", vocab_b, g)
    assert is_leading("Was he holding a knife?", vocab_b, g)
    assert not is_leading("What did you see the person doing?", vocab_b, g)


def test_question_generation():
    g, w = load()
    qs = generate_questions(g, find_gaps(g), find_conflicts(g), w)
    passed = [q for q in qs if q["passed"]]
    blocked = [q for q in qs if not q["passed"]]
    assert passed and blocked
    # Every naive baseline question is blocked; nothing leading reaches a witness.
    assert all(not q["passed"] for q in qs if q["kind"] == "baseline")
    for q in passed:
        if q["witness"] != "witness_C":  # only C mentioned a knife
            assert "knife" not in q["text"].lower(), q["text"]
        words = set(q["text"].lower().replace("?", " ").replace('"', " ").split())
        if q["witness"] == "witness_B":
            assert "red" not in words and "hoodie" not in words, q["text"]
        if q["kind"] == "conflict_recheck":
            assert "jacket" not in words and "hoodie" not in words, q["text"]
    # Witness C never mentioned the cashier, so C is never asked about her.
    assert not any(q["witness"] == "witness_C" and q["entity"] == "person_2" for q in qs)



def test_extraction_post_check():
    raw = {"entities": [{"id": "e1", "entity_type": "person", "role": "suspect"},
                        {"id": "e2", "entity_type": "weapon", "role": "none"}],
           "facts": [{"entity": "e1", "attribute": "lower_clothing", "value": "black cap"},
                     {"entity": "e1", "attribute": "lower_clothing", "value": "black backpack"},
                     {"entity": "e1", "attribute": "upper_clothing", "value": "red hoodie"},
                     {"entity": "e1", "attribute": "lower_clothing", "value": "black jeans"},
                     {"entity": "e2", "attribute": "held_by", "value": "suspect",
                      "quote": "I'm pretty sure he had a knife in his hand", "hedged": True}]}
    logged = []
    facts = _clean_raw(raw, "witness_X", lambda step, detail: logged.append(detail))["facts"]
    got = {(f["entity"], f["attribute"], f["value"]) for f in facts}
    assert ("e1", "headwear", "black cap") in got
    assert ("e1", "carrying", "black backpack") in got
    assert ("e1", "upper_clothing", "red hoodie") in got  # "hoodie" is not "hood"
    assert ("e1", "lower_clothing", "black jeans") in got
    knife = [f for f in facts if f["attribute"] == "weapon_type"]
    assert [f["value"] for f in knife] == ["knife"] and knife[0]["hedged"]
    assert len(logged) == 3, logged



def test_extraction_post_check_fixes():
    raw = {"entities": [{"id": "e1", "entity_type": "person", "role": "suspect"},
                        {"id": "e2", "entity_type": "location", "role": "none"},
                        {"id": "e3", "entity_type": "vehicle", "role": "none", "description": "parking lot"}],
           "facts": [{"entity": "e1", "attribute": "height", "value": "slim"},
                     {"entity": "e1", "attribute": "build", "value": "tall"},
                     {"entity": "e1", "attribute": "lower_clothing", "value": "black shorts"},
                     {"entity": "e2", "attribute": "lighting", "value": "not specified"},
                     {"entity": "e2", "attribute": "time_of_day", "value": "Unknown."},
                     {"entity": "e2", "attribute": "place_type", "value": "snacks aisle"},
                     {"entity": "e3", "attribute": "position", "value": "parking lot"}]}
    out = _clean_raw(raw, "witness_X")
    got = {(f["entity"], f["attribute"], f["value"]) for f in out["facts"]}
    assert got == {("e1", "build", "slim"), ("e1", "height", "tall"), ("e1", "lower_clothing", "black shorts"),
                   ("e2", "notable_features", "snacks aisle")}, got
    assert "e3" not in out["entities"]


def test_hedge_comes_from_the_sentence():
    statement = "A slim guy in black jeans came running out. I'm pretty sure he had a knife in his hand."
    raw = {"entities": {"e1": {"entity_type": "weapon", "role": "none", "description": "a knife"},
                        "e2": {"entity_type": "person", "role": "suspect", "description": "a slim guy"}},
           "facts": [{"entity": "e1", "attribute": "weapon_type", "value": "knife", "quote": "had a knife", "hedged": False},
                     {"entity": "e2", "attribute": "build", "value": "slim", "quote": "A slim guy", "hedged": True}]}
    raw = _add_sentences(raw, statement)
    mapping = {"witness_C:e1": {"entity": "weapon_1", "entity_type": "weapon", "role": "none"},
               "witness_C:e2": {"entity": "person_1", "entity_type": "person", "role": "suspect"}}
    cfg = {"confidence": {"cctv": 0.85, "witness": 0.7, "followup": 0.65, "hedge_penalty": 0.25}}
    knife, slim = build_facts({"witness_C": raw}, mapping, {}, cfg)
    assert knife.hedged and knife.base_confidence == 0.575, knife
    assert not slim.hedged and slim.base_confidence == 0.7, slim


def test_merge_joins_lone_men_and_women_whatever_their_role():
    lookup = {"witness_A:e1": {"entity_type": "person", "role": "suspect", "description": "a man"},
              "witness_A:e2": {"entity_type": "person", "role": "victim", "description": "a woman"},
              "witness_B:e1": {"entity_type": "person", "role": "victim", "description": "the guy"},
              "witness_C:e1": {"entity_type": "person", "role": "bystander", "description": "he"}}
    groups = _merge_lone_genders([["witness_A:e1"], ["witness_A:e2"], ["witness_B:e1"], ["witness_C:e1"]], lookup,
                                 lambda step, detail: None)
    assert sorted(map(sorted, groups)) == [["witness_A:e1", "witness_B:e1", "witness_C:e1"], ["witness_A:e2"]]
    # Two men in one statement: don't guess which one C saw.
    lookup["witness_A:e3"] = {"entity_type": "person", "role": "suspect", "description": "another man"}
    groups = _merge_lone_genders([["witness_A:e1"], ["witness_A:e3"], ["witness_C:e1"]], lookup,
                                 lambda step, detail: None)
    assert len(groups) == 3


def test_grounding_check():
    statement = ("I was in the snacks aisle when a tall man in a red hoodie walked straight to the counter. "
                 "He leaned over and grabbed cash from the register.")
    raw = {"entities": [{"id": "e1", "entity_type": "person", "role": "suspect"},
                        {"id": "e2", "entity_type": "location", "role": "none"}],
           "facts": [{"entity": "e1", "attribute": "upper_clothing", "value": "red hoodies"},   # suffix
                     {"entity": "e1", "attribute": "action", "value": "grabbing cash"},        # suffix
                     {"entity": "e1", "attribute": "carrying", "value": "nothing"},            # placeholder
                     {"entity": "e1", "attribute": "headwear", "value": "No"},                 # placeholder
                     {"entity": "e1", "attribute": "position", "value": "the"},                # no content word
                     {"entity": "e2", "attribute": "lighting", "value": "brightly lit"},       # never said
                     {"entity": "e2", "attribute": "time_of_day", "value": "afternoon"}]}      # never said
    logged = []
    out = _clean_raw(raw, "witness_A", lambda step, detail: logged.append((step, detail)), text=statement)
    assert [f["value"] for f in out["facts"]] == ["red hoodies", "grabbing cash"], out["facts"]
    assert sum(step == "grounding" for step, _ in logged) == 3, logged
    assert any("afternoon" in d for _, d in logged) and any("brightly lit" in d for _, d in logged)
    # CCTV facts (no statement text) are not grounded against anything.
    cctv = _clean_raw({"entities": [{"id": "e1", "entity_type": "location", "role": "none"}],
                       "facts": [{"entity": "e1", "attribute": "lighting", "value": "dim"}]}, "cctv_frame_01")
    assert [f["value"] for f in cctv["facts"]] == ["dim"]
    assert grounding_problem("running", "He came running out.") is None
    assert grounding_problem("knife", "He came running out.")


def test_leading_articles_stripped():
    assert _strip_article("the store") == "store"
    assert _strip_article("A black cap") == "black cap"
    assert _strip_article("an old car") == "old car"
    assert _strip_article("theatre") == "theatre"
    raw = {"entities": [{"id": "e1", "entity_type": "location", "role": "none"}],
           "facts": [{"entity": "e1", "attribute": "place_type", "value": "the store"}]}
    facts = _clean_raw(raw, "witness_B", text="I could see into the store.")["facts"]
    assert [f["value"] for f in facts] == ["store"]
    prompt, _, _, _ = build_prompt([Fact(fact_id="f001", entity="scene", entity_type="location", role="none",
                                      attribute="place_type", value=facts[0]["value"], source="witness_B",
                                      source_type="witness")])
    assert "a store" in prompt and "a the" not in prompt



SUSPECT = {"entity": "person_1", "entity_type": "person", "role": "suspect",
           "attributes": {"upper_clothing": ["red hoodie"], "headwear": ["black cap"]}}
CASHIER = {"entity": "person_2", "entity_type": "person", "role": "victim",
           "attributes": {"upper_clothing": ["green uniform shirt"]}}
RED, GREEN = "person wearing red hoodie", "person wearing green uniform shirt"


def fake_verifier(dets):
    v = Verifier({"backend": "owlv2", "score_threshold": 0.2, "hallucination_score": 0.3,
                  "distractors": ["person", "knife", "bag"]})
    v.detect = lambda image, queries: [dict(d) for d in dets]
    return v


def person_det(box, red, green, person=0.3):
    scores = {RED: red, GREEN: green, "person": person}
    label = max(scores, key=scores.get)
    return {"label": label, "score": scores[label], "box": box, "scores": scores}


def test_person_query():
    assert person_query(SUSPECT) == RED
    assert person_query({"attributes": {"carrying": ["black backpack"]}}) == "person carrying black backpack"
    assert person_query({"attributes": {"action": ["ran out"]}}) == "person"


def test_people_assigned_by_attribute_score_not_order():
    # The cashier's box comes first and has the higher plain score; order-based matching would
    # give it to the suspect.
    dets = [person_det([600, 100, 800, 700], red=0.10, green=0.45, person=0.50),
            person_det([100, 100, 300, 700], red=0.40, green=0.12, person=0.35)]
    report = fake_verifier(dets).verify(None, [SUSPECT, CASHIER])
    assert report["matched"]["person_1"]["box"] == [100, 100, 300, 700]
    assert report["matched"]["person_2"]["box"] == [600, 100, 800, 700]
    assert not report["hallucinated"] and not report["missing"]


def test_extra_person_is_a_hallucination():
    dets = [person_det([100, 100, 300, 700], red=0.40, green=0.10),
            person_det([600, 100, 800, 700], red=0.10, green=0.45),
            person_det([850, 150, 1000, 700], red=0.05, green=0.05, person=0.25)]  # nobody described this one
    report = fake_verifier(dets).verify(None, [SUSPECT, CASHIER])
    assert report["person_count"] == {"drawn": 2, "in_image": 3}
    assert [d["box"] for d in report["hallucinated"]] == [[850, 150, 1000, 700]]
    assert "extra person" in report["hallucinated"][0]["reason"]


def test_part_boxes_of_one_person_are_merged():
    torso = person_det([450, 180, 600, 540], red=0.34, green=0.14)
    legs = person_det([500, 580, 600, 860], red=0.07, green=0.08, person=0.32)
    people = merge_people([torso, legs])
    assert len(people) == 1 and people[0]["box"] == [450, 180, 600, 860]
    # Real OWLv2 boxes for one SDXL person (torso, arm, legs, ...): the arm only joins once
    # the other parts have been merged. Two people in the background stay separate.
    one_person = [person_det(b, red=r, green=0.1) for b, r in [
        ([497, 173, 623, 514], 0.59), ([424, 310, 486, 492], 0.37), ([445, 309, 543, 519], 0.31),
        ([498, 312, 600, 521], 0.30), ([512, 178, 590, 313], 0.23), ([461, 529, 579, 828], 0.22)]]
    background = [person_det([669, 222, 742, 440], red=0.1, green=0.1), person_det([595, 238, 667, 438], red=0.1, green=0.1)]
    people = merge_people(one_person + background)
    assert sorted(p["box"] for p in people) == [[424, 173, 623, 828], [595, 238, 667, 438], [669, 222, 742, 440]]
    # Two people side by side stay two.
    assert len(merge_people([torso, person_det([700, 180, 850, 860], red=0.1, green=0.4)])) == 2
    # Only one person in the image but two drawn: the cashier is missing, no extra person.
    report = fake_verifier([torso, legs]).verify(None, [SUSPECT, CASHIER])
    assert list(report["matched"]) == ["person_1"] and report["missing"] == ["person_2"]
    assert not report["hallucinated"]


def test_overlay_still_blurs_and_colours():
    from PIL import Image

    img = Image.new("RGB", (400, 400), (255, 255, 255))
    for x in range(0, 400, 4):  # stripes so blurring changes pixels
        for y in range(400):
            img.putpixel((x, y), (0, 0, 0))
    dets = [person_det([50, 50, 150, 390], red=0.4, green=0.1), person_det([250, 50, 350, 390], red=0.1, green=0.4)]
    report = fake_verifier(dets).verify(img, [SUSPECT, CASHIER])
    out = draw_overlay(img, report, {"person_1": "contested", "person_2": "single"})
    assert out.getpixel((100, 70)) != img.getpixel((100, 70))  # face region blurred
    assert out.getpixel((50, 200)) == (240, 130, 20)  # contested = orange box edge
    assert out.getpixel((250, 200)) == (230, 190, 30)  # single = yellow box edge



def test_carried_bag_is_not_a_hallucination():
    suspect = {**SUSPECT, "attributes": {**SUSPECT["attributes"], "carrying": ["black backpack"]}}
    people = [person_det([100, 100, 300, 700], red=0.40, green=0.10),
              person_det([600, 100, 800, 700], red=0.10, green=0.45)]
    on_suspect = {"label": "bag", "score": 0.35, "box": [200, 300, 280, 400], "scores": {}}
    on_cashier = {"label": "bag", "score": 0.35, "box": [650, 300, 700, 400], "scores": {}}
    loose = {"label": "bag", "score": 0.35, "box": [400, 800, 450, 850], "scores": {}}
    report = fake_verifier(people + [on_suspect, on_cashier, loose]).verify(None, [suspect, CASHIER])
    assert [d["carried_by"] for d in report["carried"]] == ["person_1"]
    # The cashier carries nothing and the loose bag is on nobody: both still count.
    assert sorted(d["box"] for d in report["hallucinated"]) == [[400, 800, 450, 850], [650, 300, 700, 400]]


def test_box_must_look_most_like_its_own_entity():
    # Real H2 case: one person in a blue jacket; "green uniform shirt" scored just over the
    # threshold on him, but "blue jacket" scored higher, so he is never the cashier.
    blue, green = "person wearing blue jacket", "person wearing green uniform shirt"
    suspect = {"entity": "person_1", "entity_type": "person", "attributes": {"upper_clothing": ["blue jacket"]}}
    det = {"label": blue, "score": 0.31, "box": [520, 514, 676, 910],
           "scores": {blue: 0.31, green: 0.208, "person": 0.3}}
    report = fake_verifier([det]).verify(None, [suspect, CASHIER])
    assert list(report["matched"]) == ["person_1"] and report["missing"] == ["person_2"]
    # If the only person looks more like the cashier, the suspect is the one missing.
    det["scores"] = {blue: 0.21, green: 0.30, "person": 0.3}
    report = fake_verifier([det]).verify(None, [suspect, CASHIER])
    assert list(report["matched"]) == ["person_2"] and report["missing"] == ["person_1"]


def test_fragment_boxes_dropped():
    # Real H2 boxes: an arm fragment beside the suspect's body box.
    body = person_det([520, 514, 676, 910], red=0.4, green=0.1)
    arm = person_det([502, 613, 534, 699], red=0.25, green=0.1)
    assert [p["box"] for p in merge_people([body, arm])] == [[520, 514, 676, 910]]
    # A real person further back (11% of the area, overlapping the edge) is kept.
    near = person_det([424, 173, 623, 828], red=0.6, green=0.1)
    far = person_det([595, 238, 670, 438], red=0.1, green=0.1)
    assert len(merge_people([near, far])) == 2
    # A small box far from everyone is a distant person, not a fragment.
    assert len(merge_people([body, person_det([50, 600, 80, 680], red=0.1, green=0.1)])) == 2


def test_missing_person_regenerates_and_best_attempt_is_kept():
    import tempfile

    from PIL import Image

    from pipeline import Session

    reports = {  # seed -> (hallucinated, missing, attribute accuracy)
        42: ([], ["person_2"], 0.4),
        43: ([{"label": "bag", "score": 0.4, "box": [0, 0, 10, 10]}], [], 0.9),
        44: ([], ["person_2"], 0.5),
    }

    class FakeGenerator:
        count_tokens = staticmethod(lambda text: len(text.split()))

        def generate(self, prompt, negative, seed):
            return Image.new("RGB", (64, 64), (seed % 255, 0, 0))

    class FakeVerifier:
        def verify(self, image, rendered):
            hall, missing, _ = reports[self.seed_of(image)]
            return {"expected": [], "matched": {}, "missing": list(missing), "hallucinated": list(hall),
                    "person_count": {"drawn": 2, "in_image": 2 - len(missing)}, "vqa": []}

        @staticmethod
        def seed_of(image):  # FakeGenerator encodes the seed in the red channel
            return next(seed for seed in reports if seed % 255 == image.getpixel((0, 0))[0])

        def attribute_check(self, image, rendered):
            acc = reports[self.seed_of(image)][2]
            return [{"answer": True}] * int(acc * 10) + [{"answer": False}] * (10 - int(acc * 10))

    import copy

    s = Session("sample_robbery")
    s.store = None
    s.cfg = copy.deepcopy(s.cfg)  # independent of config.yaml: 3 attempts, VLM check on
    s.cfg["generation"]["max_retries"] = 2
    s.cfg["verification"].update(vqa_check=True, backend="owlv2")
    with tempfile.TemporaryDirectory() as tmp:
        s.out_dir = tmp
        s.graph = SceneGraph.load(os.path.join(SCENE, "sample_facts.json"), s.cfg["confidence"])
        s.analyze()
        s._generator, s._verifier = FakeGenerator(), FakeVerifier()
        s.hypotheses = s.hypotheses[:1]
        h1 = s.reconstruct()[0]
    log = h1["attempt_log"]
    assert [a["seed"] for a in log] == [42, 43, 44]  # missing person_2 on 42 -> regenerated
    # All three tie on 1 error each; seed 43 has the best attribute score, so it is kept (not the last).
    assert h1["seed"] == 43 and [a["kept"] for a in log] == [False, True, False]
    assert log[0]["missing"] == ["person_2"] and log[1]["hallucinated"] == ["bag"]



def _make_witnesses():
    sys.path.insert(0, os.path.join(ROOT, "data"))
    import make_witnesses

    return make_witnesses


def _scene_dirs():
    root = os.path.join(ROOT, "data", "scenes")
    return [os.path.join(root, d) for d in sorted(os.listdir(root))]


def test_witness_prompts_have_no_entity_ids():
    import random

    mw = _make_witnesses()
    checked = 0
    for folder in _scene_dirs():
        gt_path = os.path.join(folder, "ground_truth.json")
        gt = json.load(open(gt_path))["facts"] if os.path.exists(gt_path) else []
        if not gt or "entity" not in gt[0]:  # hand-made scenes (sample_robbery) aren't built by make_witnesses
            continue
        rng = random.Random(7)
        doc = json.load(open(gt_path))
        for profile in mw.PROFILES:  # includes the confabulated person_9 / vehicle_9 / weapon_9
            facts, _ = mw.degrade(gt, profile, rng)
            prompt = mw.statement_prompt(profile["viewpoint"], facts, doc["nouns"],
                                         mw.crime_kind(os.path.basename(folder)))
            assert not mw.ENTITY_ID.search(prompt), (folder, mw.ENTITY_ID.search(prompt).group(0))
            details = prompt.split("people):")[1].split("Rules:")[0]
            assert not mw.ROLE_LABEL.search(details), (folder, details)
            checked += 1
    assert checked >= 3


def test_witness_statements_have_no_entity_ids():
    mw = _make_witnesses()
    for folder in _scene_dirs():
        for wid, w in json.load(open(os.path.join(folder, "scene.json"))).get("witnesses", {}).items():
            leak = mw.ENTITY_ID.search(w["statement"])
            assert leak is None, f"{os.path.basename(folder)} {wid}: {leak.group(0)!r} in statement"


def test_leaky_statement_is_retried_then_refused():
    mw = _make_witnesses()
    replies = iter([{"statement": "I saw person_9 near the door."}, {"statement": "I saw a man near the door."}])
    real = mw.llm.chat_json
    no_people = lambda text: {"entities": {}, "facts": []}  # noqa: E731  (skip the LLM re-read)
    try:
        man = [ROBBERY058[2]]
        mw.llm.chat_json = lambda prompt, temperature=None: next(replies)
        assert mw.write_statement("inside", man, {"person_2": "man"}, extract=no_people) == "I saw a man near the door."
        mw.llm.chat_json = lambda prompt, temperature=None: {"statement": "weapon_1 was on the counter."}
        try:
            mw.write_statement("inside", man, {"person_2": "man"}, extract=no_people)
            raise AssertionError("leaky statement was accepted")
        except SystemExit:
            pass
    finally:
        mw.llm.chat_json = real



def test_template_function_words_are_not_leaks():
    # Regression: robbery058 reported 3 leaked questions; the only "leaked" word was "on", from the
    # template ("wearing on their legs") and also present in the answer key ("put the bag on the table").
    from evaluate import leaked_words

    gt_words = {"put", "the", "bag", "on", "table", "red", "clothes", "black", "mask", "gun", "checkout", "counter"}
    said_b = set("i saw a man in white clothes standing behind the checkout counter".split())
    for q in ['What was the person you described ("a man in white clothes") wearing on their legs?',
              'What was the person you described ("standing behind the checkout counter") wearing on the '
              'upper part of their body?',
              'What was the person you described ("standing behind the checkout counter") wearing on their legs?']:
        assert leaked_words(q, said_b, gt_words) == [], q
        assert not is_leading(q, said_b, SceneGraph("x", [], CONF)), q
    # A real leak is still counted, and the filter blocks it.
    real = "Was the man wearing a black mask?"
    assert leaked_words(real, said_b, gt_words) == ["black", "mask"]
    assert is_leading(real, said_b, SceneGraph("x", [], CONF))



ROBBERY058 = [
    {"entity": "person_1", "entity_type": "person", "role": "victim", "attribute": "upper_clothing", "value": "red clothes"},
    {"entity": "person_1", "entity_type": "person", "role": "victim", "attribute": "position", "value": "behind the counter"},
    {"entity": "person_2", "entity_type": "person", "role": "suspect", "attribute": "upper_clothing", "value": "black clothes"},
    {"entity": "person_2", "entity_type": "person", "role": "suspect", "attribute": "headwear", "value": "mask"},
    {"entity": "weapon_1", "entity_type": "weapon", "role": "none", "attribute": "weapon_type", "value": "gun"},
]


def test_statement_prompt_uses_nouns_and_plain_roles():
    mw = _make_witnesses()
    nouns = {"person_1": "woman", "person_2": "man"}
    heads = mw.entity_headings(ROBBERY058, nouns, "robbery")
    assert heads == {"person_1": "a woman who was being robbed (works there)", "person_2": "a man who robbed the place",
                     "weapon_1": "a weapon"}
    prompt = mw.statement_prompt("far away", ROBBERY058, nouns, "robbery")
    # Each person's details stay on one line, under that person.
    assert "- a woman who was being robbed (works there): upper clothing = red clothes; position = behind the counter" in prompt
    assert "- a man who robbed the place: upper clothing = black clothes; headwear = mask" in prompt
    hedged = [dict(ROBBERY058[2]), dict(ROBBERY058[3], hedged=True)]
    assert "- a man who robbed the place: upper clothing = black clothes; not sure about: headwear = mask" in \
        mw.statement_prompt("far away", hedged, nouns, "robbery")
    assert mw.entity_headings(ROBBERY058[2:3] + [dict(ROBBERY058[2], entity="person_4")], {"person_2": "man", "person_4": "man"},
                              "shoplifting") == {"person_2": "a man who stole something", "person_4": "another man who stole something"}
    try:
        mw.statement_prompt("far away", ROBBERY058, {"person_1": "woman"}, "robbery")  # person_2 has no noun
        raise AssertionError("missing noun was accepted")
    except AssertionError as e:
        assert "no noun for person_2" in str(e)


def test_statement_problems_are_caught():
    mw = _make_witnesses()
    both = {"man", "woman"}
    assert mw.statement_problem("The suspect, wearing a mask, put a bag down. She screamed.", both)
    assert mw.statement_problem("The victim, who was a man, took two steps.", both)  # shoplifting033 witness A
    assert mw.statement_problem("He was wearing a black vest (unsure).", {"man"})
    assert mw.statement_problem("A man took the watch and he left.", {"woman"})  # the woman turned into a man
    # A gender word that comes from the given details themselves is fine (shoplifting033 witness B).
    assert mw.statement_problem("He walked around the table to the woman's side.", {"man"},
                                "walked around the table to the woman's side") is None
    assert mw.statement_problem("I saw person_9 near the door.", {"man"})
    assert mw.statement_problem("A man in black pointed a gun at the cashier and she raised her hands.", both) is None
    # Someone of unknown gender (the invented bystander) may be called either.
    assert mw.statement_problem("A woman was behind the counter and a man in a grey coat stood by.", {"woman", "person"}) is None


def test_detail_told_as_the_wrong_persons_is_caught():
    mw = _make_witnesses()
    facts = [{"entity": "person_1", "entity_type": "person", "role": "victim", "attribute": "action",
              "value": "squatted with her hands above her head"},
             {"entity": "person_2", "entity_type": "person", "role": "suspect", "attribute": "headwear", "value": "mask"}]
    # robbery058 witness A: "The robber, wearing a mask, ... squatted with her hands above her head."
    misread = lambda text: {"entities": {"e1": {"role": "suspect"}, "e2": {"role": "victim"}},  # noqa: E731
                            "facts": [{"entity": "e1", "attribute": "headwear", "value": "mask"},
                                      {"entity": "e1", "attribute": "action", "value": "squatted with her hands above her head"}]}
    assert "details of two people" in mw.attribution_problem("...", facts, misread)
    fine = lambda text: {"entities": {"e1": {"role": "suspect"}, "e2": {"role": "victim"}},  # noqa: E731
                         "facts": [{"entity": "e1", "attribute": "headwear", "value": "mask"},
                                   {"entity": "e2", "attribute": "action", "value": "squatted with her hands above her head"}]}
    assert mw.attribution_problem("...", facts, fine) is None
    # Regression (shoplifting024 witness A): the extractor called the shop worker a suspect too, but
    # each person's details stayed with that person, so the statement is fine.
    shop = [{"entity": "person_1", "entity_type": "person", "role": "victim", "attribute": "action",
             "value": "turned around and put things into the cabinet"},
            {"entity": "person_2", "entity_type": "person", "role": "suspect", "attribute": "action",
             "value": "took something from the table and put it into her clothes"}]
    both_suspects = lambda text: {"entities": {"e1": {"role": "suspect"}, "e2": {"role": "suspect"}},  # noqa: E731
                                  "facts": [{"entity": "e1", "attribute": "action", "value": shop[0]["value"]},
                                            {"entity": "e2", "attribute": "action", "value": "took something from the table"}]}
    assert mw.attribution_problem("...", shop, both_suspects) is None
    # A detail both people share ("behind the counter") can't show a mix-up, so it is ignored.
    shared = [dict(shop[0], attribute="position", value="behind the counter"),
              dict(shop[1], attribute="position", value="behind the counter")]
    one = lambda text: {"entities": {"e1": {"role": "suspect"}},  # noqa: E731
                        "facts": [{"entity": "e1", "attribute": "position", "value": "behind the counter"}]}
    assert mw.attribution_problem("...", shared, one) is None


def test_corruption_changes_only_that_value():
    import random

    mw = _make_witnesses()
    profile = dict(mw.PROFILES[1], omit=0.0, far=False)  # corrupt exactly one detail, drop nothing
    nouns = {"person_1": "woman", "person_2": "man"}
    for seed in range(20):
        kept, log = mw.degrade([dict(f) for f in ROBBERY058], profile, random.Random(seed))
        changes = [(a, b) for a, b in zip(ROBBERY058, kept) if a["value"] != b["value"]]
        assert len(changes) == 1 and [d["type"] for d in log] == ["corruption"], (seed, log)
        before, after = changes[0]
        assert {k: before[k] for k in ("entity", "entity_type", "role", "attribute")} == \
               {k: after[k] for k in ("entity", "entity_type", "role", "attribute")}
        assert [(f["entity"], f["role"], f["attribute"]) for f in kept] == \
               [(f["entity"], f["role"], f["attribute"]) for f in ROBBERY058]  # no position moved, nobody re-roled
        assert mw.person_nouns(kept, nouns) == nouns  # gender untouched



def _src(*entities):
    """per_source entry: entities given as (eid, type, role, [(attribute, value), ...])."""
    return {"entities": {e: {"entity_type": t, "role": r, "description": ""} for e, t, r, _ in entities},
            "facts": [{"entity": e, "attribute": a, "value": v, "quote": "", "hedged": False}
                      for e, _, _, facts in entities for a, v in facts]}


def _people(mapping):
    """entity -> (role, sorted mention keys) for people."""
    out = {}
    for k, m in mapping.items():
        if m["entity_type"] == "person":
            out.setdefault(m["entity"], (m["role"], []))[1].append(k)
    return {e: (r, sorted(ks)) for e, (r, ks) in out.items()}


def test_cctv_person_inherits_the_witness_role():
    from extraction import merge_entities

    # robbery044: the victim seen on camera; witness C invented a bystander in a grey coat.
    per_source = {
        "witness_A": _src(("e1", "person", "suspect", [("headwear", "black hat"), ("upper_clothing", "black vest")]),
                          ("e2", "person", "victim", [("headwear", "white hat"), ("action", "raised his hands"),
                                                      ("position", "behind the counter")])),
        "witness_C": _src(("e1", "person", "suspect", [("headwear", "black hat")]),
                          ("e2", "person", "bystander", [("upper_clothing", "grey coat")])),
        "cctv_frame_01": _src(("e1", "person", "bystander", [("upper_clothing", "grey outfit"), ("headwear", "white cap")])),
        "cctv_frame_06": _src(("e1", "person", "bystander", [("upper_clothing", "gray shirt"), ("headwear", "white cap"),
                                                             ("action", "standing with hands raised"),
                                                             ("position", "behind the counter")])),
    }
    logged = []
    people = _people(merge_entities(per_source, use_llm=False, log=lambda s, d: logged.append(d)))
    victim = [ks for r, ks in people.values() if r == "victim"]
    assert victim == [["cctv_frame_01:e1", "cctv_frame_06:e1", "witness_A:e2"]], people
    # The invented bystander is NOT glued to the real person on camera.
    assert ("bystander", ["witness_C:e2"]) in people.values()
    assert len(people) == 3 and any("-> witness-side person ['witness_A:e2']" in d for d in logged)
    # Same case without the shared position: "white cap" vs "white hat" is no clash, so the agreeing
    # action alone still links the camera person to the victim.
    per_source["cctv_frame_06"]["facts"] = [f for f in per_source["cctv_frame_06"]["facts"] if f["attribute"] != "position"]
    per_source["witness_A"]["facts"] = [f for f in per_source["witness_A"]["facts"] if f["attribute"] != "position"]
    people = _people(merge_entities(per_source, use_llm=False))
    assert [ks for r, ks in people.values() if r == "victim"] == [["cctv_frame_01:e1", "cctv_frame_06:e1", "witness_A:e2"]]


def test_cctv_frames_merge_despite_colour_drift():
    from extraction import merge_entities

    # robbery058: the cashier's red top read as red hoodie / pink shirt / red shirt.
    per_source = {
        "witness_A": _src(("e1", "person", "victim", [("upper_clothing", "red clothes"), ("position", "behind the counter")]),
                          ("e2", "person", "suspect", [("upper_clothing", "black clothes"), ("headwear", "mask")])),
        "cctv_frame_01": _src(("e1", "person", "bystander", [("upper_clothing", "red hoodie")])),
        "cctv_frame_04": _src(("e1", "person", "bystander", [("upper_clothing", "pink shirt"), ("lower_clothing", "dark pants")])),
        "cctv_frame_05": _src(("e1", "person", "bystander", [("upper_clothing", "red shirt")])),
    }
    people = _people(merge_entities(per_source, use_llm=False))
    assert len(people) == 2, people
    assert sorted(people.values())[1] == ("victim", ["cctv_frame_01:e1", "cctv_frame_04:e1", "cctv_frame_05:e1",
                                                     "witness_A:e1"])
    # robbery044 frame 4: "light-colored cap" names no colour family, so it doesn't contradict "white cap".
    frames = {"cctv_frame_03": _src(("e1", "person", "bystander", [("upper_clothing", "light grey jacket"), ("headwear", "white cap")])),
              "cctv_frame_04": _src(("e1", "person", "bystander", [("upper_clothing", "gray shirt"), ("headwear", "light-colored cap")]))}
    assert len(_people(merge_entities(frames, use_llm=False))) == 1
    # A clear colour clash keeps them apart even when another look agrees.
    frames["cctv_frame_04"] = _src(("e1", "person", "bystander", [("upper_clothing", "gray shirt"), ("headwear", "red cap")]))
    assert len(_people(merge_entities(frames, use_llm=False))) == 2


def test_cctv_linking_needs_a_clear_match():
    from extraction import merge_entities

    witnesses = {"witness_A": _src(("e1", "person", "suspect", [("upper_clothing", "blue shirt")]),
                                   ("e2", "person", "victim", [("upper_clothing", "blue shirt")]))}
    # Agrees with both equally: stays its own person.
    tie = dict(witnesses, cctv_frame_01=_src(("e1", "person", "bystander", [("upper_clothing", "blue shirt")])))
    assert len(_people(merge_entities(tie, use_llm=False))) == 3
    # Two people in the same frame never merge with each other, even if they look alike.
    two = {"cctv_frame_01": _src(("e1", "person", "bystander", [("upper_clothing", "black jacket")]),
                                 ("e2", "person", "bystander", [("upper_clothing", "black jacket")]))}
    assert len(_people(merge_entities(two, use_llm=False))) == 2
    # Conflicting looks win over one agreeing detail: no link.
    other = {"witness_A": _src(("e1", "person", "suspect", [("upper_clothing", "red hoodie"), ("headwear", "black cap"),
                                                            ("position", "by the door")])),
             "cctv_frame_01": _src(("e1", "person", "bystander", [("upper_clothing", "green coat"), ("headwear", "white hat"),
                                                                  ("position", "by the door")]))}
    assert len(_people(merge_entities(other, use_llm=False))) == 2


def test_mask_is_headwear_and_place_type_must_be_a_place():
    raw = {"entities": [{"id": "e1", "entity_type": "person", "role": "suspect"},
                        {"id": "e2", "entity_type": "location", "role": "none"}],
           "facts": [{"entity": "e1", "attribute": "upper_clothing", "value": "mask"},
                     {"entity": "e1", "attribute": "upper_clothing", "value": "balaclava"},
                     {"entity": "e2", "attribute": "place_type", "value": "scene"},
                     {"entity": "e2", "attribute": "place_type", "value": "place where the watch was"},
                     {"entity": "e2", "attribute": "place_type", "value": "convenience store"},
                     {"entity": "e2", "attribute": "place_type", "value": "parking lot"}]}
    got = [(f["attribute"], f["value"]) for f in _clean_raw(raw, "witness_X")["facts"]]
    assert got == [("headwear", "mask"), ("headwear", "balaclava"),
                   ("place_type", "convenience store"), ("place_type", "parking lot")], got
    assert values_agree("red clothes", "red hoodie") and not values_agree("red clothes", "blue jacket")



def test_corruption_never_targets_multi_valued_attributes():
    import random

    from schema import MULTI_VALUED

    mw = _make_witnesses()
    profile = dict(mw.PROFILES[1], omit=0.0, far=False)
    actions_only = [{"entity": "person_3", "entity_type": "person", "role": "victim", "attribute": "action",
                     "value": "handed it to the man in black"},
                    {"entity": "person_3", "entity_type": "person", "role": "victim", "attribute": "position",
                     "value": "by the white door"},
                    {"entity": "scene", "entity_type": "location", "role": "none", "attribute": "notable_features",
                     "value": "red counter"}]
    for seed in range(30):
        kept, log = mw.degrade([dict(f) for f in actions_only], profile, random.Random(seed))
        assert not [d for d in log if d["type"] == "corruption"], (seed, log)
        assert [f["value"] for f in kept] == [f["value"] for f in actions_only]
    # With a colour in a single-valued attribute too, that one is what gets corrupted.
    mixed = actions_only + [{"entity": "person_3", "entity_type": "person", "role": "victim",
                             "attribute": "upper_clothing", "value": "red shirt"}]
    for seed in range(30):
        _, log = mw.degrade([dict(f) for f in mixed], profile, random.Random(seed))
        assert [d["attribute"] for d in log if d["type"] == "corruption"] == ["upper_clothing"], (seed, log)
        assert all(d["attribute"] not in MULTI_VALUED for d in log if d["type"] == "corruption")



def _fact(i, entity, etype, role, attribute, value):
    return Fact(fact_id=f"f{i:03d}", entity=entity, entity_type=etype, role=role, attribute=attribute,
                value=value, source="witness_A", source_type="witness", confidence=0.7)


ROBBERY_FACTS = [
    _fact(1, "person_1", "person", "suspect", "action", "walked into the checkout counter"),
    _fact(2, "person_1", "person", "suspect", "upper_clothing", "black coat"),
    _fact(3, "person_1", "person", "suspect", "headwear", "white bandana"),
    _fact(4, "person_1", "person", "suspect", "action", "pointed a gun at the cashier"),
    _fact(5, "person_1", "person", "suspect", "action", "threatened the man"),
    _fact(6, "person_2", "person", "victim", "upper_clothing", "red shirt"),
    _fact(7, "person_2", "person", "victim", "position", "behind the counter"),
    _fact(8, "person_2", "person", "victim", "action", "tried to run into the house"),
    _fact(9, "person_2", "person", "victim", "action", "raised his hands"),
    _fact(10, "scene", "location", "none", "place_type", "convenience store"),
    _fact(11, "scene", "location", "none", "lighting", "brightly lit"),
]


def fake_tokens(text):  # one token per word or punctuation mark, plus start/end
    import re

    return len(re.findall(r"\w+|[^\w\s]", text)) + 2


def test_prompt_fits_the_token_limit_by_priority():
    full, _, rendered, not_drawn = build_prompt(ROBBERY_FACTS, fake_tokens, max_tokens=77)
    assert fake_tokens(full) <= 77
    reasons = {d["value"]: d["reason"] for d in not_drawn}
    # Movement/direction is never drawn; one short action per person.
    assert reasons["walked into the checkout counter"] == "movement/direction"
    assert reasons["tried to run into the house"] == "movement/direction"
    assert reasons["threatened the man"] == "one short action per person"
    assert "pointed a gun at the cashier" in full and "raised his hands" in full
    # Movement in a position is skipped too; "into" alone is not movement.
    extra = ROBBERY_FACTS + [_fact(12, "person_2", "person", "victim", "position", "walked out of the cashier"),
                             _fact(13, "person_3", "person", "bystander", "action", "looked up into the distance")]
    _, _, _, nd = build_prompt(extra, fake_tokens)
    reasons = {d["value"]: d["reason"] for d in nd}
    assert reasons["walked out of the cashier"] == "movement/direction"
    assert "looked up into the distance" not in reasons
    assert "prompt limit" not in reasons.values()  # everything else fits in 77

    # Tight limit: clothing first, then headwear, then position/place, then the action.
    tight, _, rendered, not_drawn = build_prompt(ROBBERY_FACTS, fake_tokens, max_tokens=40)
    assert fake_tokens(tight) <= 40, (fake_tokens(tight), tight)
    assert "black coat" in tight and "red shirt" in tight and "exactly 2 people" in tight
    cut = [d["value"] for d in not_drawn if d["reason"] == "prompt limit"]
    assert "pointed a gun at the cashier" in cut and "brightly lit" in cut
    kept = {v for r in rendered for vs in r["attributes"].values() for v in vs}
    assert not kept & set(cut)  # verification only checks what is really in the prompt
    assert all(d["value"] not in tight for d in not_drawn)


def test_people_who_dont_fit_are_listed_not_drawn():
    crowd = ROBBERY_FACTS[:3] + [_fact(20 + i, f"person_{3 + i}", "person", "bystander", "upper_clothing", "grey coat")
                                 for i in range(5)]
    prompt, _, rendered, not_drawn = build_prompt(crowd, fake_tokens, max_tokens=30)
    people = [r["entity"] for r in rendered if r["entity_type"] == "person"]
    assert people[0] == "person_1" and "black coat" in prompt  # the suspect and his coat go in first
    assert f"exactly {len(people)} " in prompt and fake_tokens(prompt) <= 30
    assert prompt.count("grey coat") == len(people) - 1  # no bare "a person" crowding out clothing
    left_out = {d["entity"] for d in not_drawn if d["reason"] == "prompt limit" and d["attribute"] == "upper_clothing"}
    assert left_out and not left_out & set(people)  # a person who doesn't fit is left out entirely


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)


def _desc_src(*entities):
    """Like _src, with a description: (eid, type, role, description, [(attribute, value), ...])."""
    return {"entities": {e: {"entity_type": t, "role": r, "description": d} for e, t, r, d, _ in entities},
            "facts": [{"entity": e, "attribute": a, "value": v, "quote": "", "hedged": False}
                      for e, _, _, _, facts in entities for a, v in facts]}


def test_role_never_blocks_merging_and_is_voted():
    from extraction import merge_entities, role_votes

    # Same woman, three different roles; the man is called suspect by two witnesses, victim by one.
    per_source = {
        "witness_A": _desc_src(("e1", "person", "suspect", "a woman", [("action", "picked up the white paper")]),
                               ("e2", "person", "victim", "a man in a black coat", [("upper_clothing", "black coat")])),
        "witness_B": _desc_src(("e1", "person", "victim", "the woman", [("action", "picked up the white paper")]),
                               ("e2", "person", "suspect", "the man", [("upper_clothing", "black coat")])),
        "witness_C": _desc_src(("e1", "person", "bystander", "a lady", []),
                               ("e2", "person", "suspect", "a guy", [])),
    }
    mapping = merge_entities(per_source, use_llm=False)
    people = _people(mapping)
    assert len(people) == 2, people
    roles = {tuple(ks): r for r, ks in people.values()}
    assert roles[("witness_A:e2", "witness_B:e2", "witness_C:e2")] == "suspect"  # 2 of 3 votes
    # The woman: suspect / victim / bystander, one vote each: tie goes to a named role, suspect first.
    assert roles[("witness_A:e1", "witness_B:e1", "witness_C:e1")] == "suspect"
    votes = role_votes(mapping)
    assert len(votes) == 2 and sorted(votes["person_1"].values()) in (
        ["bystander", "suspect", "victim"], ["suspect", "suspect", "victim"])


def test_merge_by_details_without_gender_words():
    from extraction import merge_entities

    # No gender words; the details agree (one planted colour clash is forgiven), roles disagree.
    per_source = {
        "witness_A": _desc_src(("e1", "person", "suspect", "the robber", [("headwear", "black cap"),
                                                                        ("upper_clothing", "red hoodie")]),
                               ("e2", "person", "victim", "the cashier", [("upper_clothing", "white shirt"),
                                                                         ("position", "behind the counter")])),
        "witness_B": _desc_src(("e1", "person", "victim", "the robber", [("headwear", "black cap"),
                                                                       ("upper_clothing", "blue hoodie")]),
                               ("e2", "person", "suspect", "the cashier", [("upper_clothing", "white shirt")])),
    }
    people = _people(merge_entities(per_source, use_llm=False))
    assert sorted(ks for _, ks in people.values()) == [["witness_A:e1", "witness_B:e1"], ["witness_A:e2", "witness_B:e2"]]


def test_same_source_and_man_woman_never_merge():
    from extraction import _split_inconsistent

    lookup = {"witness_A:e1": {"entity_type": "person", "role": "suspect", "description": "a man"},
              "witness_A:e2": {"entity_type": "person", "role": "suspect", "description": "someone"},
              "witness_B:e1": {"entity_type": "person", "role": "suspect", "description": "a woman"}}
    groups = _split_inconsistent([["witness_A:e1", "witness_A:e2", "witness_B:e1"]], lookup, lambda s, d: None)
    assert groups == [["witness_A:e1"], ["witness_A:e2", "witness_B:e1"]]


def test_cctv_patterns_are_compatible_and_one_clash_is_forgiven():
    from extraction import _contradicts, _same_look, merge_entities

    assert _same_look("patterned jacket", "checkered shirt") and not _contradicts("dark patterned shirt", "red shirt")
    assert _contradicts("red shirt", "blue shirt")
    # Two frames of one woman: "checkered shirt" and "patterned jacket" are one look.
    frames = {"cctv_frame_05": _desc_src(("e1", "person", "bystander", "a woman", [("upper_clothing", "patterned jacket")])),
              "cctv_frame_06": _desc_src(("e1", "person", "bystander", "a woman", [("upper_clothing", "checkered shirt"),
                                                                                  ("lower_clothing", "jeans")]))}
    assert len(_people(merge_entities(frames, use_llm=False))) == 1
    # One witness's corrupted shirt colour doesn't block the link when the cap and the bag agree.
    per_source = {
        "witness_A": _desc_src(("e1", "person", "suspect", "the man", [("upper_clothing", "blue shirt"),
                                                                     ("headwear", "white cap"),
                                                                     ("carrying", "black backpack")])),
        "cctv_frame_01": _desc_src(("e1", "person", "bystander", "a man", [("upper_clothing", "red t-shirt"),
                                                                         ("headwear", "white cap"),
                                                                         ("carrying", "black backpack")])),
    }
    assert len(_people(merge_entities(per_source, use_llm=False))) == 1
    # On camera, gender is one more forgivable clash: a single frame's "man" doesn't veto a link...
    lone = {"witness_A": _desc_src(("e1", "person", "victim", "a woman", [("upper_clothing", "red clothes")])),
            "cctv_frame_01": _desc_src(("e1", "person", "bystander", "person in a red hoodie", [("upper_clothing", "red hoodie")])),
            "cctv_frame_03": _desc_src(("e1", "person", "bystander", "man in red hoodie", [("upper_clothing", "red hoodie")]))}
    assert len(_people(merge_entities(lone, use_llm=False))) == 1
    # ...but a gender clash plus a colour clash outweigh one agreeing detail.
    per_source["witness_A"]["entities"]["e1"]["description"] = "the woman"
    per_source["witness_A"]["facts"] = [f for f in per_source["witness_A"]["facts"] if f["attribute"] != "carrying"]
    assert len(_people(merge_entities(per_source, use_llm=False))) == 2


def test_role_conflicts_are_recorded_but_do_not_branch():
    from conflicts import build_hypotheses, find_role_conflicts
    from graph import SceneGraph
    from schema import Fact

    g = SceneGraph("t", [Fact("f001", "person_1", "person", "suspect", "upper_clothing", "red hoodie",
                              "witness_A", "witness", base_confidence=0.7)])
    g.role_votes = {"person_1": {"witness_A": "suspect", "witness_B": "suspect", "witness_C": "victim"}}
    rc = find_role_conflicts(g)
    assert len(rc) == 1 and rc[0]["kind"] == "role" and rc[0]["attribute"] == "role"
    assert [o["value"] for o in rc[0]["options"]] == ["suspect", "victim"]
    assert rc[0]["options"][0]["sources"] == ["witness_A", "witness_B"]
    assert build_hypotheses(rc) == build_hypotheses([])


def test_descriptions_count_as_a_detail_when_merging():
    from extraction import _description_core, merge_entities

    assert _description_core("the man behind the counter") == "behind counter"
    assert _description_core("a man") == ""
    # robbery044: witness B describes two men; the victim's only fact is "being robbed".
    per_source = {
        "witness_A": _desc_src(("e1", "person", "suspect", "a man", [("headwear", "black hat")]),
                               ("e2", "person", "victim", "the man behind the counter", [("headwear", "white hat")])),
        "witness_B": _desc_src(("e1", "person", "suspect", "a man who robbed the place", [("headwear", "black hat")]),
                               ("e2", "person", "victim", "the man behind the counter", [("action", "being robbed")])),
    }
    people = _people(merge_entities(per_source, use_llm=False))
    assert sorted(ks for _, ks in people.values()) == [["witness_A:e1", "witness_B:e1"], ["witness_A:e2", "witness_B:e2"]]
    # Three statements, same description: a three-way tie that is one person.
    per_source["witness_C"] = _desc_src(("e1", "person", "victim", "the man behind the counter", []))
    people = _people(merge_entities(per_source, use_llm=False))
    assert ["witness_A:e2", "witness_B:e2", "witness_C:e1"] in [ks for _, ks in people.values()], people


def test_partial_llm_merge_answer_is_repaired_not_rejected():
    from extraction import _repair_groups

    types = {"witness_A:e1": "person", "witness_A:e2": "object", "witness_B:e1": "person", "witness_B:e2": "person"}
    groups, repairs = _repair_groups([["witness_A:e1", "witness_B:e1", "witness_A:e9"],   # invented key
                                      ["witness_A:e2", "witness_B:e1"],                   # duplicate
                                      ["witness_C:e1"]], types)                           # all invented
    assert groups == [["witness_A:e1", "witness_B:e1"], ["witness_A:e2"], ["witness_B:e2"]]
    assert repairs == {"invented key dropped": 2, "duplicate key dropped": 1, "missing mention added alone": 1}
    groups, repairs = _repair_groups([["witness_A:e1", "witness_A:e2"]], types)           # person + object
    assert ["witness_A:e1"] in groups and ["witness_A:e2"] in groups and repairs["mixed-type group split"] == 1


def test_specific_actions_outweigh_generic_ones():
    from extraction import _agreement, _is_generic

    assert _is_generic("walked away") and _is_generic("turned around") and _is_generic("left")
    assert not _is_generic("took out a black pistol")
    robber_a = {"action": ["took out a black pistol", "took away a bottle of Coke", "turned around to leave"]}
    robber_b = {"action": ["took out a black pistol", "took away a bottle of Coke", "took the gun back"]}
    clerk = {"action": ["turned around"]}
    assert _agreement(robber_a, robber_b)[0] == 2.0
    assert _agreement(robber_a, clerk)[0] == 0.25


def test_two_men_per_witness_merge_by_what_they_did():
    from extraction import merge_entities

    # robbery003-like: every witness describes two men; the robbers share specific actions, the
    # clerk shares only "turned around" with the robber's "turned around to leave".
    per_source = {
        "witness_A": _desc_src(("e1", "person", "suspect", "a man", [("action", "took out a black pistol"),
                                                                    ("action", "took away a bottle of Coke"),
                                                                    ("action", "turned around to leave")]),
                               ("e2", "person", "victim", "the man in black clothes", [("action", "turned around")])),
        "witness_B": _desc_src(("e1", "person", "suspect", "a man in a plaid shirt", [("action", "took out a black pistol"),
                                                                                     ("action", "took the gun back")]),
                               ("e2", "person", "victim", "the man in black", [])),
        "witness_C": _desc_src(("e1", "person", "suspect", "a man who robbed the place", [("action", "took out a black pistol"),
                                                                                         ("action", "took away a bottle of Coke")]),
                               ("e2", "person", "victim", "another man", [("action", "being robbed"), ("action", "turned around")])),
    }
    people = sorted(ks for _, ks in _people(merge_entities(per_source, use_llm=False)).values())
    assert ["witness_A:e1", "witness_B:e1", "witness_C:e1"] in people, people
    # A single shared generic action is not enough to join two people.
    assert not any("witness_A:e1" in ks and ("witness_A:e2" in ks or "witness_C:e2" in ks) for ks in people)


def test_llm_join_is_overruled_when_details_point_elsewhere():
    from extraction import _check_llm_joins

    per_source = {
        "witness_A": _desc_src(("e1", "person", "suspect", "a man", [("action", "took out a black pistol"),
                                                                    ("action", "took away a bottle of Coke")])),
        "witness_B": _desc_src(("e1", "person", "suspect", "a man in a plaid shirt", [("action", "took out a black pistol")]),
                               ("e2", "person", "victim", "the man in black", [])),
    }
    lookup = {f"{s}:{e}": v for s, r in per_source.items() for e, v in r["entities"].items()}
    # The LLM put A's robber with B's clerk (who has no details at all).
    groups = _check_llm_joins([["witness_A:e1", "witness_B:e2"], ["witness_B:e1"]], per_source, lookup, lambda s, d: None)
    assert sorted(map(sorted, groups)) == [["witness_A:e1", "witness_B:e1"], ["witness_B:e2"]]
