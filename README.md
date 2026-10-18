# Multi-Agent Crime Scene Reconstruction — Phase 2 prototype

Witness statements + CCTV frames → provenance-tagged evidence graph (Neo4j) →
conflict & gap detection → non-leading follow-up questions → confidence-gated
SDXL reconstruction → automatic verification + provenance overlay.

B.Tech project, IIIT Kottayam. Everything runs locally on the lab GPU.

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
| `extraction.py` | Statements and frames → facts; merges entities across sources; base confidence |
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
| `tests/test_logic.py` | Tests for all model-free logic (no GPU needed) |
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

## Setup (shared GPU container: no sudo, no Docker, no conda)

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
5. `tests/test_logic.py`.

In a new shell, activate the environment and make sure Ollama is running:
```bash
source .venv/bin/activate
export PATH=$HOME/.local/bin:$PATH
curl -s localhost:11434/api/tags >/dev/null || \
  OLLAMA_KEEP_ALIVE=0 nohup ollama serve > ~/.ollama-serve.log 2>&1 &
```

SDXL (~7 GB) and OWLv2 download automatically from Hugging Face on first use.

**GPU memory:** check what's free before running models (`nvidia-smi` can't report it under MIG):
```bash
python -c "import torch; print(f'{torch.cuda.mem_get_info()[0]/2**30:.1f} GiB free')"
```
Set `generation.cpu_offload` in `config.yaml` to match:
- **`false`** (all on GPU) when a lot is free.
- **`true`** (model offload, ~8 GB).
- **`sequential`** (layer-by-layer offload plus VAE tiling, < 4 GB) when under ~12 GB is free. This is the
  current setting: on the shared A100 only ~4–5 GB is free. Each 1024² image takes about 40–55 s.

Ollama's quantised Qwen2.5-VL needs 6–8 GB. When less is free, Ollama runs part of it on the CPU, which is
slower but works. `OLLAMA_KEEP_ALIVE=0` makes Ollama free the GPU after every request so it never holds
memory while SDXL runs. The cost is a model reload per call, so each attribute check takes about 1.5 min.
With Docker available, `docker compose up -d` starts Neo4j (http://localhost:7474, neo4j / crimerecon123);
then set `neo4j.enabled: true`.

**vLLM instead of Ollama:** `vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --max-model-len 8192 --gpu-memory-utilization 0.45`,
then set `llm.base_url: http://localhost:8000/v1` and `llm.model: Qwen/Qwen2.5-VL-7B-Instruct-AWQ`.

## Run

With `.venv` activated and Ollama running (see Setup):

```bash
python pipeline.py --scene sample_robbery          # CLI: prints conflicts, questions, image paths
python pipeline.py --scene sample_robbery --force-extract   # ignore cached facts, extract again
python app.py                                      # web demo at http://localhost:7860
python evaluate.py                                 # metrics table (add --generate for image metrics)
```

A full run of `sample_robbery` (extraction + 2 hypotheses × up to 3 attempts + attribute checks) takes
about 8–10 minutes on the shared GPU. With facts cached it takes about 8 minutes, since generation and
verification dominate.

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
7. Show `python evaluate.py` table.

## Adding real scenes (UCF-Crime + UCA)

```bash
python data/extract_frames.py --video Robbery048_x264.mp4 --out data/scenes/robbery048/frames --every 2 --max 8
python data/make_witnesses.py --scene robbery048 --caption "<UCA caption text for this clip>"
# open data/scenes/robbery048/ground_truth.json, fix any wrong facts by hand, then:
python data/make_witnesses.py --scene robbery048 --reuse-gt
```

The UCA caption is the "truth"; the script makes 3 witnesses from it with planted omissions,
a wrong colour/size, and one false memory, and records exactly what was planted.

## Tuning (config.yaml)

- `confidence.render_threshold` — raise it to draw less, lower to draw more.
- `confidence.hedge_penalty`, `corroboration_bonus` — how much "I think" hurts and agreement helps.
- `verification.score_threshold` / `hallucination_score` — detector sensitivity.
- `generation.max_retries` — regenerations when a hallucination is found.
- `questions.llm_rephrase` — let the LLM make questions sound natural (they are still filtered).

## Known limitations (be upfront in the review)

- Value matching is word-based: "red hoodie" vs "maroon sweatshirt" may be seen as different values.
  An embedding/LLM similarity check is planned for phase 3.
- SDXL reads only ~77 prompt tokens; very detailed scenes get truncated (suspects/victims go first).
- The overlay boxes come from detecting objects in the generated image, not from layout-controlled generation.
  Layout-guided generation (GLIGEN / regional prompting) is planned for the final phase.
- At most 2 hypotheses; Visual Genome graph-accuracy evaluation and full DSG scoring are planned for later.
- Synthetic witnesses are generated by an LLM; the answer key is human-checked to limit circularity.

## Ethics

Public research data only. Faces are blurred in all output. The system is decision support, not evidence;
it makes no identification or guilt claims.
