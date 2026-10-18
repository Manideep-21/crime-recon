"""Runs the whole system for one scene. Used by app.py, evaluate.py and the CLI.

    python pipeline.py --scene sample_robbery            # full run
    python pipeline.py --scene sample_robbery --no-generate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional

from config import load_config
from conflicts import build_hypotheses, find_conflicts, find_gaps, find_role_conflicts, select_facts
from graph import Neo4jStore, SceneGraph


class AuditLog:
    def __init__(self):
        self.entries: List[dict] = []

    def __call__(self, step: str, detail: str) -> None:
        entry = {"time": time.strftime("%H:%M:%S"), "step": step, "detail": detail}
        self.entries.append(entry)
        print(f"[{entry['time']}] {step}: {detail}")

    def to_markdown(self) -> str:
        return "\n".join(f"- `{e['time']}` **{e['step']}** — {e['detail']}" for e in self.entries)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            for e in self.entries:
                fh.write(json.dumps(e) + "\n")


def list_scenes() -> List[str]:
    root = load_config()["paths"]["scenes_dir"]
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "scene.json")))


def load_scene(scene_id: str) -> dict:
    """scene.json: {"title", "witnesses": {"witness_A": {"statement", "viewpoint"}}, "cctv_frames": [...]}"""
    folder = os.path.join(load_config()["paths"]["scenes_dir"], scene_id)
    with open(os.path.join(folder, "scene.json")) as fh:
        scene = json.load(fh)
    frames = scene.get("cctv_frames") or sorted(
        glob.glob(os.path.join(folder, "frames", "*.jpg")) + glob.glob(os.path.join(folder, "frames", "*.png")))
    scene["frames"] = [f if os.path.isabs(f) else os.path.join(folder, f) for f in frames]
    scene["scene_id"] = scene_id
    gt = os.path.join(folder, "ground_truth.json")
    scene["ground_truth"] = json.load(open(gt)) if os.path.exists(gt) else None
    return scene


def entity_levels(graph: SceneGraph, included, contested: set) -> Dict[str, str]:
    """One support level per entity for the overlay: the weakest of its drawn facts."""
    levels: Dict[str, str] = {}
    for f in included:
        if f.fact_id in contested:
            lvl = "contested"
        else:
            lvl = graph.support_level(f)
        rank = {"contested": 0, "single": 1, "confirmed": 2}
        if f.entity not in levels or rank[lvl] < rank[levels[f.entity]]:
            levels[f.entity] = lvl
    return levels


class Session:
    """State of one scene as it moves through the pipeline."""

    def __init__(self, scene_id: str):
        self.cfg = load_config()
        self.scene = load_scene(scene_id)
        self.scene_id = scene_id
        self.out_dir = os.path.join(self.cfg["paths"]["outputs_dir"], scene_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.log = AuditLog()
        self.graph: Optional[SceneGraph] = None
        self.conflicts: List[dict] = []
        self.hypotheses: List[dict] = []
        self.gaps: List[dict] = []
        self.questions: List[dict] = []
        self.asked: set = set()
        self.results: List[dict] = []
        self._generator = None
        self._verifier = None
        self.store = None
        if self.cfg["neo4j"]["enabled"]:
            try:
                n = self.cfg["neo4j"]
                self.store = Neo4jStore(n["uri"], n["user"], n["password"])
            except Exception as e:  # keep going without Neo4j
                self.log("neo4j", f"not available ({e}); continuing in-memory only")

    @property
    def witnesses(self) -> Dict[str, dict]:
        return self.scene["witnesses"]

    # ------------------------------------------------------------ steps
    def extract(self, force: bool = False) -> SceneGraph:
        from extraction import extract_scene

        cache = os.path.join(self.out_dir, "facts.json")
        if os.path.exists(cache) and not force:
            self.graph = SceneGraph.load(cache, self.cfg["confidence"])
            self.log("extract", f"loaded {len(self.graph.facts)} cached facts ({cache})")
        else:
            facts, role_votes = extract_scene(self.witnesses, self.scene["frames"], log=self.log)
            self.graph = SceneGraph(self.scene_id, facts, self.cfg["confidence"])
            self.graph.role_votes = role_votes
            self.graph.save(cache)
            self.log("extract", f"{len(facts)} facts from {len(self.witnesses)} witnesses "
                                f"+ {min(len(self.scene['frames']), self.cfg['extraction']['max_cctv_frames'])} CCTV frames")
        self.analyze()
        return self.graph

    def analyze(self) -> None:
        from questions import generate_questions

        self.conflicts = find_conflicts(self.graph) + find_role_conflicts(self.graph)
        self.hypotheses = build_hypotheses(self.conflicts)
        self.gaps = find_gaps(self.graph)
        q = self.cfg["questions"]
        self.questions = generate_questions(self.graph, self.gaps, self.conflicts, self.witnesses,
                                            llm_rephrase=q["llm_rephrase"],
                                            include_baseline=q["include_baseline_candidates"])
        blocked = [x for x in self.questions if not x["passed"]]
        n_role = sum(c.get("kind") == "role" for c in self.conflicts)
        self.log("analyze", f"{len(self.conflicts) - n_role} conflicts (+{n_role} role disagreements), "
                            f"{len(self.hypotheses)} hypotheses, "
                            f"{len(self.gaps)} gaps, {len(self.questions)} candidate questions "
                            f"({len(blocked)} blocked as leading)")
        if self.store:
            try:
                self.store.sync(self.graph, self.conflicts)
            except Exception as e:
                self.log("neo4j", f"sync failed: {e}")
        with open(os.path.join(self.out_dir, "analysis.json"), "w") as fh:
            json.dump({"conflicts": self.conflicts, "hypotheses": self.hypotheses,
                       "gaps": self.gaps, "questions": self.questions}, fh, indent=2)

    # ------------------------------------------------------------ witness loop
    def next_question(self, witness: str) -> Optional[dict]:
        """Next SAFE question for this witness. Blocked questions never come out of here."""
        for q in self.questions:
            if q["witness"] == witness and q["passed"] and q["text"] not in self.asked:
                self.asked.add(q["text"])
                self.log("question", f"asked {witness}: {q['text']}")
                return q
        return None

    def answer(self, question: dict, answer_text: str):
        from extraction import extract_answer

        fact = extract_answer(question, answer_text, self.graph.next_fact_id())
        if fact is None:
            self.log("answer", f"{question['witness']} could not answer ({answer_text!r})")
            return None
        self.graph.add([fact])
        self.graph.save(os.path.join(self.out_dir, "facts.json"))
        self.log("answer", f"{fact.source}: {fact.entity}.{fact.attribute} = '{fact.value}' "
                           f"(confidence {fact.confidence})")
        self.analyze()
        return fact

    # ------------------------------------------------------------ reconstruction
    def reconstruct(self) -> List[dict]:
        from generation import Generator, build_prompt
        from verification import Verifier, draw_overlay, summarize

        gcfg, vcfg, ccfg = self.cfg["generation"], self.cfg["verification"], self.cfg["confidence"]
        self._generator = self._generator or Generator(gcfg)
        self._verifier = self._verifier or Verifier(vcfg)
        self.results = []
        for h_idx, hyp in enumerate(self.hypotheses):
            included, withheld, contested = select_facts(
                self.graph, self.conflicts, hyp, ccfg["render_threshold"], ccfg["hypothesis_threshold"])
            prompt, negative, rendered, not_drawn = build_prompt(included, self._generator.count_tokens)
            self.log("generate", f"{hyp['id']}: {len(included) - len(not_drawn)} facts drawn, {len(withheld)} withheld, "
                                 f"{len(not_drawn)} not drawn (prompt)")
            for d in not_drawn:
                self.log("not drawn", f"{hyp['id']}: {d['entity']}.{d['attribute']} = '{d['value']}' "
                                      f"({d['source']}): {d['reason']}")
            self.log("prompt", f"{hyp['id']}: {prompt} [{self._generator.count_tokens(prompt)} tokens]")

            people = {r["entity"] for r in rendered if r["entity_type"] == "person"}
            attempts = []
            for attempt in range(gcfg["max_retries"] + 1):
                seed = gcfg["base_seed"] + 100 * h_idx + attempt
                image = self._generator.generate(prompt, negative, seed)
                report = self._verifier.verify(image, rendered)
                attempts.append({"seed": seed, "image": image, "report": report})
                self.log("verify", f"{hyp['id']} seed {seed}: {summarize(report)}")
                # A drawn person who isn't in the image is as wrong as an extra one: try again.
                problems = [d["label"] for d in report["hallucinated"]] + \
                           [f"missing {e}" for e in report["missing"] if e in people]
                if not problems:
                    break
                if attempt < gcfg["max_retries"]:
                    self.log("verify", f"{hyp['id']}: {', '.join(problems)} -> regenerate")

            # Keep the best attempt, not the last: fewest hallucinations + missing, then the best
            # attribute score (only the tied attempts get the slow VLM check).
            errors = lambda a: len(a["report"]["hallucinated"]) + len(a["report"]["missing"])  # noqa: E731
            fewest = min(errors(a) for a in attempts)
            tied = [a for a in attempts if errors(a) == fewest]
            if vcfg.get("vqa_check") and vcfg["backend"] != "none":
                for a in tied:
                    a["report"]["vqa"] = self._verifier.attribute_check(a["image"], rendered)
                    answered = [q for q in a["report"]["vqa"] if q["answer"] is not None]
                    a["report"]["attribute_accuracy"] = (
                        round(sum(q["answer"] for q in answered) / len(answered), 3) if answered else None)
                    self.log("verify", f"{hyp['id']} seed {a['seed']} attribute check: {summarize(a['report'])}")
            best = max(tied, key=lambda a: a["report"].get("attribute_accuracy") or 0.0)
            self.log("verify", f"{hyp['id']}: keeping seed {best['seed']} ({len(attempts)} attempts)")

            levels = entity_levels(self.graph, included, contested)
            overlay = draw_overlay(best["image"], best["report"], levels, vcfg.get("blur_faces", True))
            raw_path = os.path.join(self.out_dir, f"{hyp['id']}_raw.png")
            ov_path = os.path.join(self.out_dir, f"{hyp['id']}_overlay.png")
            best["image"].save(raw_path)
            overlay.save(ov_path)

            self.results.append({
                "hypothesis": hyp, "prompt": prompt, "seed": best["seed"],
                "included": [f.fact_id for f in included],
                "withheld": [{"fact_id": f.fact_id, "entity": f.entity, "attribute": f.attribute,
                              "value": f.value, "source": f.source, "confidence": f.confidence}
                             for f in withheld],
                "not_drawn": not_drawn,
                "contested": sorted(contested), "levels": levels,
                "report": {k: v for k, v in best["report"].items()},
                "first_attempt_hallucinations": len(attempts[0]["report"]["hallucinated"]),
                "final_hallucinations": len(best["report"]["hallucinated"]),
                "attempts": len(attempts), "raw_path": raw_path, "overlay_path": ov_path,
                "attempt_log": [{
                    "seed": a["seed"], "kept": a is best,
                    "hallucinated": [d.get("reason") or d["label"] for d in a["report"]["hallucinated"]],
                    "missing": a["report"]["missing"],
                    "person_count": a["report"].get("person_count"),
                    "attribute_accuracy": a["report"].get("attribute_accuracy"),
                    "matched": {e: [round(v) for v in d["box"]] for e, d in a["report"]["matched"].items()},
                } for a in attempts],
            })
        with open(os.path.join(self.out_dir, "reconstruction.json"), "w") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        self.log.save(os.path.join(self.out_dir, "audit_log.jsonl"))
        return self.results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="folder name under data/scenes")
    ap.add_argument("--force-extract", action="store_true", help="ignore cached facts")
    ap.add_argument("--no-generate", action="store_true", help="stop after analysis")
    args = ap.parse_args()

    s = Session(args.scene)
    s.extract(force=args.force_extract)
    print("\nCONFLICTS")
    for c in s.conflicts:
        print(f"  {c['entity']}.{c['attribute']}: " +
              " vs ".join(f"'{o['value']}' ({', '.join(o['sources'])}, {o['confidence']})" for o in c["options"]))
    print("\nQUESTIONS")
    for q in s.questions:
        mark = "PASS " if q["passed"] else "BLOCK"
        print(f"  [{mark}] {q['witness']}: {q['text']}" + ("" if q["passed"] else f"  <- {q['reasons'][0]}"))
    if not args.no_generate:
        for r in s.reconstruct():
            print(f"\n{r['hypothesis']['id']}: {r['overlay_path']}")
            print(f"  withheld: {[w['value'] for w in r['withheld']]}")
    s.log.save(os.path.join(s.out_dir, "audit_log.jsonl"))


if __name__ == "__main__":
    main()
