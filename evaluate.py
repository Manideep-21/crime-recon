"""Evaluate every scene that has a ground_truth.json and print one results table.

    python evaluate.py                 # extraction + analysis metrics
    python evaluate.py --generate      # also generate images and measure hallucinations

Metrics
- recovery:          share of answer-key facts that ended up in the graph
- drawn_precision:   share of facts that WOULD be drawn (H1) that are true
  _witness / _cctv:  the same, split by where the fact came from (witness incl. follow-ups, or CCTV
                     frame). CCTV can correctly show details the caption never mentions, so a low
                     _cctv value is not always an extraction error.
- false_withheld:    planted false memories kept out of the image
- conflicts_found:   planted conflicts detected
- questions:         safe / blocked; leaked = safe questions that still contain an
                     answer-key detail the witness never said (independent check)
- halluc_first/final hallucinated objects before / after the regenerate loop
"""
from __future__ import annotations

import argparse
import csv
import os

from config import load_config
from conflicts import select_facts
from matching import STOPWORDS, tokens, values_agree, vocabulary
from pipeline import Session, list_scenes
from schema import origin_of


def _matches(fact, gt: dict) -> bool:
    return (fact.entity_type == gt["entity_type"] and fact.attribute == gt["attribute"]
            and (fact.entity_type != "person" or fact.role == gt["role"])
            and values_agree(fact.value, gt["value"]))


def leaked_words(question: str, said: set, gt_words: set) -> list:
    """Answer-key detail words in a question that this witness never said.

    Only content words count: function words such as "on" come from the question template
    ("wearing on their legs"), not from the answer key, even when the key happens to contain them.
    """
    return [t for t in tokens(question) if t not in STOPWORDS and t in gt_words and t not in said]


def evaluate_scene(scene_id: str, generate: bool, force: bool) -> dict:
    cfg = load_config()
    s = Session(scene_id)
    s.extract(force=force)
    gt = s.scene["ground_truth"]
    facts = s.graph.facts
    row = {"scene": scene_id, "facts_in_graph": len(facts)}

    found = [g for g in gt["facts"] if any(_matches(f, g) for f in facts)]
    row["recovery"] = round(len(found) / len(gt["facts"]), 3) if gt["facts"] else None

    c = cfg["confidence"]
    included, withheld, _ = select_facts(s.graph, s.conflicts, s.hypotheses[0],
                                         c["render_threshold"], c["hypothesis_threshold"])
    true_drawn = [f for f in included if any(_matches(f, g) for g in gt["facts"])]
    row["drawn_precision"] = round(len(true_drawn) / len(included), 3) if included else None
    for kind in ("witness", "cctv"):
        part = [f for f in included if (f.source_type == "cctv") == (kind == "cctv")]
        row[f"drawn_precision_{kind}"] = (round(sum(f in true_drawn for f in part) / len(part), 3)
                                          if part else None)

    planted_false = gt.get("planted_false_facts", [])
    if planted_false:
        leaked = [p for p in planted_false if any(_matches(f, p) for f in included)]
        row["false_withheld"] = f"{len(planted_false) - len(leaked)}/{len(planted_false)}"

    planted_conf = gt.get("planted_conflicts", [])
    if planted_conf:
        hit = [p for p in planted_conf if any(
            k["entity_type"] == p["entity_type"] and k["attribute"] == p["attribute"]
            and (p["entity_type"] != "person" or k["role"] == p["role"]) for k in s.conflicts)]
        row["conflicts_found"] = f"{len(hit)}/{len(planted_conf)}"

    gt_words = vocabulary(g["value"] for g in gt["facts"]) | vocabulary(
        p["value"] for p in planted_false)
    safe = [q for q in s.questions if q["passed"]]
    leaked_q = 0
    for q in safe:
        said = vocabulary([s.witnesses[q["witness"]]["statement"]] +
                          [f.quote for f in facts if origin_of(f.source) == q["witness"]])
        if leaked_words(q["text"], said, gt_words):
            leaked_q += 1
    baseline = [q for q in s.questions if q["kind"] == "baseline"]
    row["questions_safe"] = len(safe)
    row["questions_blocked"] = len(s.questions) - len(safe)
    row["baseline_blocked"] = f"{sum(not q['passed'] for q in baseline)}/{len(baseline)}"
    row["leading_leaked"] = leaked_q

    if generate:
        results = s.reconstruct()
        row["halluc_first"] = sum(r["first_attempt_hallucinations"] for r in results)
        row["halluc_final"] = sum(r["final_hallucinations"] for r in results)
        row["object_faithfulness"] = round(
            sum(r["report"]["object_faithfulness"] for r in results) / len(results), 3)
        accs = [r["report"].get("attribute_accuracy") for r in results
                if r["report"].get("attribute_accuracy") is not None]
        row["attribute_accuracy"] = round(sum(accs) / len(accs), 3) if accs else None
    return row


def average_row(rows: list) -> dict:
    """Mean of numeric columns, summed 'a/b' counts; scenes without a value are skipped."""
    avg = {"scene": f"average ({len(rows)} scenes)"}
    for col in dict.fromkeys(k for r in rows for k in r if k != "scene"):
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals and all(isinstance(v, str) and "/" in v for v in vals):
            num, den = (sum(int(v.split("/")[i]) for v in vals) for i in (0, 1))
            avg[col] = f"{num}/{den}"
        elif vals and all(isinstance(v, (int, float)) for v in vals):
            avg[col] = round(sum(vals) / len(vals), 3)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--scenes", nargs="*", help="default: every scene with ground_truth.json")
    args = ap.parse_args()

    root = load_config()["paths"]["scenes_dir"]
    scenes = args.scenes or [s for s in list_scenes()
                             if os.path.exists(os.path.join(root, s, "ground_truth.json"))]
    rows = [evaluate_scene(s, args.generate, args.force_extract) for s in scenes]
    if len(rows) > 1:
        rows.append(average_row(rows))

    cols = list(dict.fromkeys(k for r in rows for k in r))
    out_dir = load_config()["paths"]["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    with open(os.path.join(out_dir, "eval_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved to {out_dir}/eval_results.csv and eval_results.md")


if __name__ == "__main__":
    main()
