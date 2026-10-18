"""Gradio demo.

    python app.py      -> open http://localhost:7860

Investigator tab: evidence graph, conflicts, gaps, question filter log,
                  hypothesis images with provenance overlay, audit log.
Witness tab:      questions only. It never shows an image or another
                  witness's account (the contamination barrier).
"""
from __future__ import annotations

import argparse

import gradio as gr
import pandas as pd

from graph import draw_graph_png
from pipeline import Session, list_scenes
from verification import summarize

SESSIONS: dict = {}


def _session(scene_id: str) -> Session:
    if scene_id not in SESSIONS:
        raise gr.Error("Load the scene first (Investigator tab).")
    return SESSIONS[scene_id]


# ---------------------------------------------------------------- markdown views

def conflicts_md(s: Session) -> str:
    if not s.conflicts:
        return "_No conflicts between sources._"
    lines = []
    for c in s.conflicts:
        opts = " **vs** ".join(f"'{o['value']}' ({', '.join(o['sources'])}, conf {o['confidence']})"
                               for o in c["options"])
        lines.append(f"- **{c['entity']}** ({c['role']}) · {c['attribute']}: {opts}")
    lines.append("\n**Hypotheses (conflicts preserved, not merged):**")
    lines += [f"- **{h['id']}** {h['name']}: {h['description']}" for h in s.hypotheses]
    return "\n".join(lines)


def gaps_md(s: Session) -> str:
    if not s.gaps:
        return "_No gaps._"
    return "\n".join(f"- {g['entity']} ({g['role']}): **{g['attribute']}** never mentioned"
                     for g in s.gaps)


def filter_md(s: Session) -> str:
    passed = [q for q in s.questions if q["passed"]]
    blocked = [q for q in s.questions if not q["passed"]]
    out = [f"**{len(passed)} safe questions**, **{len(blocked)} blocked as leading**\n",
           "| | Witness | Kind | Question | Why blocked |", "|---|---|---|---|---|"]
    for q in blocked + passed:
        mark = "❌" if not q["passed"] else "✅"
        why = "; ".join(q["reasons"]) if q["reasons"] else ""
        out.append(f"| {mark} | {q['witness']} | {q['kind']} | {q['text']} | {why} |")
    return "\n".join(out)


def investigator_view(s: Session):
    conflict_ids = {fid for c in s.conflicts for o in c["options"] for fid in o["fact_ids"]}
    return (pd.DataFrame(s.graph.to_rows()), draw_graph_png(s.graph, conflict_ids),
            conflicts_md(s), gaps_md(s), filter_md(s), s.log.to_markdown())


# ---------------------------------------------------------------- callbacks

def on_load(scene_id: str, force: bool):
    s = Session(scene_id)
    SESSIONS[scene_id] = s
    s.extract(force=force)
    witnesses = list(s.witnesses)
    return investigator_view(s) + (gr.update(choices=witnesses, value=witnesses[0]),)


def on_refresh(scene_id: str):
    return investigator_view(_session(scene_id))


def on_reconstruct(scene_id: str):
    s = _session(scene_id)
    results = s.reconstruct()
    gallery = [(r["overlay_path"], f"{r['hypothesis']['id']}: {r['hypothesis']['name']}") for r in results]
    parts = ["**Legend:** 🟩 confirmed (CCTV or 2+ sources) · 🟨 one witness only · "
             "🟧 contested (this hypothesis's side of a conflict) · 🟥 drawn by the model, NOT in evidence\n"]
    for r in results:
        h = r["hypothesis"]
        parts.append(f"### {h['id']} — {h['name']}\n{h['description']}\n")
        parts.append(f"- Verification: {summarize(r['report'])} "
                     f"(hallucinations: {r['first_attempt_hallucinations']} on first try → "
                     f"{r['final_hallucinations']} after {r['attempts']} attempt(s))")
        if r["report"].get("missing"):
            parts.append(f"- Not found in image: {', '.join(r['report']['missing'])}")
        if r["withheld"]:
            parts.append("- **Uncertain — not drawn:** " + "; ".join(
                f"{w['entity']}.{w['attribute']} = '{w['value']}' ({w['source']}, conf {w['confidence']})"
                for w in r["withheld"]))
        if r.get("not_drawn"):
            parts.append("- **Not drawn — prompt limit:** " + "; ".join(
                f"{d['entity']}.{d['attribute']} = '{d['value']}' ({d['reason']})" for d in r["not_drawn"]))
        parts.append(f"- Prompt: `{r['prompt']}`\n")
    return gallery, "\n".join(parts), s.log.to_markdown()


def on_next_question(scene_id: str, witness: str):
    s = _session(scene_id)
    q = s.next_question(witness)
    if q is None:
        return "No more questions for this witness. Thank you.", None
    return q["text"], q


def on_answer(scene_id: str, question: dict, answer: str):
    if not question:
        return "Press 'Next question' first.", "", None
    if not answer.strip():
        return "Please type an answer (or 'I don't know').", answer, question
    s = _session(scene_id)
    s.answer(question, answer)
    return "Thank you, your answer has been recorded.", "", None


# ---------------------------------------------------------------- layout

def build_ui() -> gr.Blocks:
    scenes = list_scenes()
    with gr.Blocks(title="Crime Scene Reconstruction") as demo:
        gr.Markdown("# Provenance-aware crime scene reconstruction\n"
                    "Witness statements + CCTV → evidence graph → safe questions → "
                    "confidence-gated, verified reconstruction.")
        scene = gr.Dropdown(scenes, value=scenes[0] if scenes else None, label="Scene")

        with gr.Tab("Investigator"):
            with gr.Row():
                force = gr.Checkbox(label="Re-run extraction (ignore cache)", value=False)
                load_btn = gr.Button("1. Load scene & extract evidence", variant="primary")
                refresh_btn = gr.Button("Refresh (after witness answers)")
                recon_btn = gr.Button("2. Generate reconstruction", variant="primary")
            with gr.Tab("Evidence graph"):
                graph_img = gr.Image(label="Graph (green = confirmed, yellow = one witness, red = conflict)",
                                     type="pil")
                facts = gr.Dataframe(label="All facts with provenance", wrap=True)
            with gr.Tab("Conflicts & gaps"):
                with gr.Row():
                    conflicts = gr.Markdown()
                    gaps = gr.Markdown()
            with gr.Tab("Question filter"):
                qfilter = gr.Markdown()
            with gr.Tab("Reconstruction"):
                gallery = gr.Gallery(label="Hypotheses with provenance overlay", columns=2, height=600)
                details = gr.Markdown()
            with gr.Tab("Audit log"):
                audit = gr.Markdown()

        with gr.Tab("Witness interview"):
            gr.Markdown("> **Contamination barrier:** this view never shows generated images or "
                        "other witnesses' accounts. Only questions that passed the non-leading filter appear.")
            witness = gr.Dropdown([], label="Witness")
            current_q = gr.State(None)
            next_btn = gr.Button("Next question")
            question = gr.Textbox(label="Question", interactive=False, lines=2)
            answer = gr.Textbox(label="Witness answer", lines=3)
            submit = gr.Button("Submit answer", variant="primary")
            status = gr.Markdown()

        inv_outputs = [facts, graph_img, conflicts, gaps, qfilter, audit]
        load_btn.click(on_load, [scene, force], inv_outputs + [witness])
        refresh_btn.click(on_refresh, [scene], inv_outputs)
        recon_btn.click(on_reconstruct, [scene], [gallery, details, audit])
        next_btn.click(on_next_question, [scene, witness], [question, current_q])
        submit.click(on_answer, [scene, current_q, answer], [status, answer, current_q])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also serve a public *.gradio.live link (expires after a week)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
