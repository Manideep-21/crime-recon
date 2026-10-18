# Multi-Agent Crime Scene Reconstruction — final prototype (v6)

Witness statements + CCTV frames → provenance-tagged evidence graph (Neo4j) →
conflict & gap detection → non-leading follow-up questions → confidence-gated
SDXL reconstruction → automatic verification + provenance overlay.

B.Tech project, IIIT Kottayam. Everything runs locally on the lab GPU.

This is the final prototype, git tag `prototype-v6`. Development history, every evaluation and the known
limitations are in `REVIEW_NOTES.md`; the runs behind it are in `outputs/runs/`.

## How it works

```
witness statements ─┐
                    ├─► extraction.py ──► graph.py ──► conflicts.py ──► generation.py ──► verification.py ──► overlay + audit log
CCTV keyframes ─────┘   (Qwen2.5-VL)      (Neo4j)      │  hypotheses     (SDXL, only       (OWLv2 + VLM
                                                       │  gaps           confident facts)   yes/no checks)
                                                       ▼                        ▲
                                                  questions.py                  └── hallucination found → regenerate
                                                  (non-leading filter)
                                                       │
                                            Witness tab (never sees images)
```

| File | What it does |
|---|---|
| `schema.py` | The shared fact format and the fixed attribute list per entity type |
| `matching.py` | Compares values ("red hoodie" vs "blue jacket"), detects hedges ("I think") |
| `llm.py` | Client for the local Qwen2.5-VL server (Ollama or vLLM) |
| `extraction.py` | Statements and frames → facts (with a second pass over hedged sentences); merges entities across sources; base confidence |
| `graph.py` | In-memory scene graph + corroboration; Neo4j sync; graph picture for the UI |
| `conflicts.py` | Conflicts → hypotheses (H1/H2); gaps; picks which facts get drawn |
| `questions.py` | Gap/conflict questions + the non-leading filter |
| `generation.py` | Builds the SDXL prompt from drawn facts only; runs SDXL |
| `verification.py` | OWLv2 object check, VLM attribute check, coloured overlay, face blur |
| `pipeline.py` | Runs everything for one scene (also a CLI) |
| `app.py` | Gradio demo (Investigator + Witness tabs) |
| `evaluate.py` | Metrics over all scenes with an answer key |
| `data/extract_frames.py` | Video → keyframes |
| `data/make_witnesses.py` | Caption → answer key → 3 degraded witness statements |
| `tests/test_logic.py` | Tests for all model-free logic (no GPU needed; run with pytest) |
| `setup_hpc.sh` | No-sudo setup for a shared GPU container (see Setup) |

**Verification:** each drawn person is looked for with an attribute-specific OWLv2 query built from
their drawn facts ("person wearing red hoodie", or just "person"). Part boxes of one person (torso,
arm, legs) are merged, and boxes are matched to people one-to-one. A box only goes to a person if it
scores above the threshold and higher for that person's query than for any other drawn person's.
Person boxes left over count as hallucinations (more people in the image than were drawn), and so
does a missing drawn person; both trigger a regenerate, up to `generation.max_retries`. A bag or other
distractor is not a hallucination when a drawn person carries it and the box sits on that person.
Of all attempts, the one with the fewest hallucinations + missing entities is kept, ties going to the
best VLM attribute score. Every attempt is listed in `attempt_log` in `reconstruction.json`.

## Setup (shared A100 MIG container: no sudo, no Docker, no conda)

```bash
bash setup_hpc.sh
```

The script installs everything under `$HOME`:
1. `uv`, then a Python 3.10 venv in `.venv` (system Python 3.8 is too old for Gradio 5).
2. torch 2.6 + torchvision 0.21 from the **cu124** wheels, then `requirements.txt`. The cu121 index stops
   at torch 2.5, which current diffusers can't use (`torch has no attribute 'accelerator'`).
3. Ollama as a user install in `~/.local/ollama` (its `.tar.zst` is unpacked with Python, no `zstd`
   needed). Started with `OLLAMA_KEEP_ALIVE=0`, then `qwen2.5vl:7b` is pulled.
4. Neo4j switched off (`neo4j.enabled: false`), since there's no Docker. Everything else runs in memory.
5. The logic tests.

Not done by the script, but used on this machine:
- **ffmpeg** (for `data/extract_frames.py`): a static build in `~/.local/bin/ffmpeg`.
- **git** (no system git): installed from conda-forge with the static micromamba binary into
  `~/.local/git-env`. `~/.bashrc` adds `~/.local/git-env/bin` to `PATH`.

In a new shell, activate the environment and make sure Ollama is running:
```bash
cd ~/crime-recon
source .venv/bin/activate
export PATH=$HOME/.local/bin:$HOME/.local/git-env/bin:$PATH
curl -s localhost:11434/api/tags >/dev/null || \
  OLLAMA_KEEP_ALIVE=0 nohup ollama serve > ~/.ollama-serve.log 2>&1 &
```
Keep `OLLAMA_KEEP_ALIVE=0`: Ollama then frees the GPU after every request, so it never holds memory
while SDXL runs. The cost is a model reload per call.

**Models.** `generation.backend: sdxl` (the setting v6 was evaluated with). SDXL (~7 GB) and OWLv2
download from Hugging Face on first use. `sd35` (SD3.5 Medium) is optional; it is gated, so it needs the
licence accepted and `hf auth login`. It was compared with SDXL and not adopted (see `REVIEW_NOTES.md`).
`mock` draws a placeholder image with no GPU.

**GPU memory.** The GPU is shared and other users take memory, so check what's free before every long
run. `nvidia-smi` can't report it under MIG:
```bash
python -c "import torch; print(f'{torch.cuda.mem_get_info()[0]/2**30:.1f} GiB free')"
```
Set `generation.cpu_offload` in `config.yaml` from that:

| Free | `cpu_offload` | What it does |
|---|---|---|
| ≥ 20 GiB | `false` (current setting, v6 runs) | everything on the GPU |
| 10–20 GiB | `true` | model offload, ~8 GB |
| < 10 GiB | `sequential` | layer-by-layer offload + VAE tiling, < 4 GB; much slower |

An out-of-memory error under MIG shows up as `NVML_SUCCESS == r INTERNAL ASSERT FAILED`.
Ollama's quantised Qwen2.5-VL needs 6–8 GB; with less free it runs partly on the CPU (slower, still works).

With Docker available, `docker compose up -d` starts Neo4j (http://localhost:7474, neo4j / crimerecon123);
then set `neo4j.enabled: true`.

**vLLM instead of Ollama:** `vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --max-model-len 8192 --gpu-memory-utilization 0.45`,
then set `llm.base_url: http://localhost:8000/v1` and `llm.model: Qwen/Qwen2.5-VL-7B-Instruct-AWQ`.

## Run

With `.venv` activated and Ollama running (see Setup):

```bash
python -m pytest -q tests/                         # 52 logic tests, no GPU needed
python pipeline.py --scene sample_robbery          # CLI: prints conflicts, questions, image paths
python pipeline.py --scene sample_robbery --force-extract   # ignore cached facts, extract again
python app.py                                      # web demo at http://localhost:7860
python app.py --share                              # same, plus a public https://*.gradio.live link (72 h)
```

To keep the demo running after you log out:
```bash
OLLAMA_KEEP_ALIVE=0 nohup python app.py --share > ~/crime-recon-app.log 2>&1 &
grep -o 'https://[a-z0-9.-]*gradio.live' ~/crime-recon-app.log     # the public link
```

**Evaluation.** Name the scenes. Without `--scenes`, `evaluate.py` runs *every* scene that has a
`ground_truth.json`, which includes the 4 final held-out clips that were never evaluated:
```bash
python evaluate.py --generate --force-extract --scenes \
  robbery044 robbery058 sample_robbery shoplifting033 robbery012 shoplifting024 robbery003 shoplifting001
```
That is the v6 dev evaluation. It takes about 47 minutes for the 8 scenes with `cpu_offload: false`:
extraction, hedged-sentence pass, 2 hypotheses × up to `max_retries` SDXL attempts, and OWLv2 + VLM checks.
Without `--generate` it only extracts and scores the graph, which is much faster. The table goes to
`outputs/eval_results.md` and `.csv`; the v6 results are kept in `outputs/runs/v6_8scenes/`.

| Scenes | Status |
|---|---|
| robbery044, robbery058, sample_robbery, shoplifting033, robbery012, shoplifting024, robbery003, shoplifting001 | development (all v6 numbers) |
| robbery101, robbery005, shoplifting031, shoplifting041 | final held-out, `held_out: true`, **not evaluated** |

Results go to `outputs/<scene>/`: `facts.json` (cached extraction), `analysis.json`,
`H1_raw.png`, `H1_overlay.png`, `reconstruction.json` (with `attempt_log`), `audit_log.jsonl`.
Delete `facts.json`, use `--force-extract`, or use "Re-run extraction" to extract again.

**No GPU at hand?** Run the whole UI with a mock image and pre-made facts:
```bash
mkdir -p outputs/sample_robbery
cp data/scenes/sample_robbery/sample_facts.json outputs/sample_robbery/facts.json
CRIME_RECON_CONFIG=config.offline.yaml python app.py
```

## Demo script (for the review)

1. Investigator → pick `sample_robbery` → **Load scene**.
   Show the graph: every fact carries its source and confidence; the red nodes are the conflict.
2. **Conflicts & gaps**: red hoodie (A) vs blue jacket (B) → two hypotheses, nothing merged.
3. **Question filter**: naive questions like "Are you sure it wasn't red hoodie?" are blocked with a reason;
   only open questions remain.
4. **Witness interview** tab: ask witness_A a question, answer it. Point out: no images here.
   Back in Investigator, press **Refresh** — the new fact appears, tagged `witness_A_followup`.
5. **Generate reconstruction**: H1 and H2 side by side, colour-coded boxes, faces blurred,
   "Uncertain — not drawn: knife (witness_C, 0.575)", and the verification result.
6. **Audit log**: every step traceable.
7. Show the v6 table (`outputs/runs/v6_8scenes/eval_results.md`).

## Adding real scenes (UCF-Crime + UCA)

```bash
python data/extract_frames.py --video Robbery048_x264.mp4 --out data/scenes/robbery048/frames --every 2 --max 8
python data/make_witnesses.py --scene robbery048 --caption "<UCA caption text for this clip>"
# open data/scenes/robbery048/ground_truth.json, fix any wrong facts by hand, then:
python data/make_witnesses.py --scene robbery048 --reuse-gt
```
Record every hand edit in the `hand_edits` list of `ground_truth.json`; `--reuse-gt` keeps it.

The UCA caption is the "truth"; the script makes 3 witnesses from it with planted omissions,
a wrong colour/size, and one false memory, and records exactly what was planted.

## Tuning (config.yaml)

- `confidence.render_threshold` — raise it to draw less, lower to draw more.
- `confidence.hedge_penalty`, `corroboration_bonus` — how much "I think" hurts and agreement helps.
- `verification.score_threshold` / `hallucination_score` — detector sensitivity.
- `generation.max_retries` — regenerations when a hallucination is found.
- `questions.llm_rephrase` — let the LLM make questions sound natural (they are still filtered).

## Known limitations (be upfront in the review)

Full list with evidence: `REVIEW_NOTES.md`, section "Final prototype: v6".

- **SDXL person count:** it often draws more or fewer people than the prompt asks for. Verification
  regenerates, but can't force the count. Layout-guided generation (GLIGEN / regional prompting) is the fix.
- **Confident false memories can be drawn:** an unhedged detail from a single witness (0.70) passes the
  0.60 render threshold if nothing contradicts it. Example: shoplifting001, witness C's "white van".
- **Colour-overlap matching bug:** values sharing most words count as agreeing, so "black and white plaid
  shirt" vs "blue and white plaid shirt" is not flagged as a conflict (robbery003). Matching is word-based.
- **Results from 8 scenes only:** all are development scenes, one run each, 320×240 clips. The 4 final
  held-out clips were not evaluated.
- Extra people remain, from unlinked CCTV people and invented ones in generated statements; roles from
  statements are unreliable (voted, with disagreements recorded).
- SDXL reads only ~77 prompt tokens; very detailed scenes are cut down by priority.
- The overlay boxes come from detecting objects in the generated image, not from layout-controlled generation.
- At most 2 hypotheses. Synthetic witnesses are LLM-generated, and the answer keys are hand-checked to
  limit circularity.

## Ethics

Public research data only. Faces are blurred in all output. The system is decision support, not evidence;
it makes no identification or guilt claims.
