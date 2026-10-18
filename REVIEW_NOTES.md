# Review notes: 16 Oct 2026

## What the pipeline does
Witness statements (and CCTV frames, none for `sample_robbery` yet) are read by Qwen2.5-VL into
provenance-tagged facts: one attribute of one entity from one source, with a confidence that drops for
hedges ("I think", "pretty sure") and rises when independent witnesses agree. Facts go into a scene graph.
Contradictions become separate hypotheses (H1/H2) instead of being merged. Missing attributes become
gaps, which produce non-leading follow-up questions. Only facts above the confidence threshold are drawn
by SDXL. Each image is then checked by OWLv2 (objects and people) and by VLM yes/no questions (attributes),
and regenerated when something is wrong. The output is a colour-coded overlay with faces blurred.

## HPC setup (`setup_hpc.sh`)
Shared A100 container: no sudo, Docker, conda or unzip; system Python 3.8; other users hold ~35 GB of the 40 GB.
- `uv` installs Python 3.10 and a `.venv` (Gradio 5 needs Python ≥ 3.10).
- torch 2.6.0 + torchvision 0.21 from the **cu124** wheels (the cu121 index stops at torch 2.5, which current diffusers can't use). The driver supports CUDA 12.5.
- Ollama installed into `~/.local/ollama` (releases are `.tar.zst` and there's no `zstd`, so they're unpacked with Python `zstandard`). Started with `OLLAMA_KEEP_ALIVE=0` so it frees the GPU after each request. Model: `qwen2.5vl:7b`.
- Neo4j off (`neo4j.enabled: false`, no Docker). Everything else runs in memory.
- Only ~4.1–4.5 GiB of GPU is free, so `generation.cpu_offload: sequential` (layer-by-layer offload plus VAE tiling). About 40–55 s per 1024² image.

## Bugs fixed today
**Setup / runtime**
| Problem | Fix |
|---|---|
| `setup_hpc.sh` didn't exist | Wrote it (above) |
| `torch has no attribute 'accelerator'` | torch 2.6 cu124 instead of 2.5 cu121 |
| `cpu_offload: sequential` silently ran model offload | `generation.py` handles `sequential` |
| `no attribute 'enable_vae_tiling'` (diffusers 0.41) | `pipe.vae.enable_tiling()` |
| VAE out of memory (shows as an `NVML_SUCCESS` assert under MIG): the default 1024 px tile meant 1024² was never tiled | 512 px tiles |
| OWLv2 processor needs `scipy` | Added `scipy` (and `torchvision`) |
| OWLv2 post-processing renamed in transformers 5, and it only kept each box's top label | Boxes and per-query scores read directly from the logits |

**Extraction (`extraction.py`)**
| Problem | Fix |
|---|---|
| "black cap" / "black backpack" stored as `lower_clothing` | One-line guide per attribute in the prompts, plus keyword post-check (cap/hat/hood → headwear, bag/backpack → carrying, knife/gun → weapon_type), every remap logged |
| Location, height and build missed | Guide, a worked example, and a rule that the location gets place_type / lighting / time_of_day |
| Knife lost its hedge (model quoted "had a knife") and was drawn | Hedge decided from the witness's full sentence; hedged fact added to the example. Knife back at 0.575 and withheld |
| "not specified" stored as a value → false conflict | Prompt rule; placeholders dropped (incl. "nothing", "no", "none") |
| "slim" as height | Keyword rules slim/thin/heavy/stocky → build, tall/short → height |
| "snacks aisle" as place_type | Moved to notable_features (prompt + keyword rule) |
| Vehicle invented from "towards the parking lot" | Prompt rule; vehicle without type or colour dropped |
| One suspect split into two people | Merge prompt gets each mention's details plus "different details are not a reason to split"; backstop joins suspects/victims when every source names only one |
| Invented facts (witness A "afternoon", "brightly lit") | Grounding check: a witness/follow-up fact needs ≥ 1 content word that appears in that witness's own words (simple suffixes allowed), drops logged. CCTV facts exempt |
| Prompt said "a the store" | Leading articles stripped from values |
| Actions and the cash register missed | Prompt: every distinct action is its own fact; things people interact with are objects |
| Run-to-run variation | Temperature 0 for every extraction call |

**Verification (`verification.py`, `pipeline.py`)**
| Problem | Fix |
|---|---|
| People matched to boxes in list order (labels swapped, both boxes on bystanders) | Query per person from their drawn facts ("person wearing red hoodie", falling back to "person"); one-to-one matching by score |
| More people in the image than drawn wasn't noticed | Person-count check: leftover person boxes are hallucinations ("extra person (3 in image, 2 drawn)") |
| One person boxed several times (torso, arm, legs) | Part boxes merged until stable; tiny fragments (< 10% of the area, inside or touching a big box) dropped |
| Cashier assigned to the suspect (green-uniform score 0.208, just over threshold) | A box goes to an entity only if it scores ≥ threshold **and** higher for that entity's query than for every other drawn person's query |
| A bag on the backpack-carrying suspect counted as a hallucination | A distractor is excused when a drawn person carries that item and ≥ 30% of the item box lies inside that person's box |
| A missing person didn't trigger a retry | Missing people trigger a regenerate (within `max_retries`) |
| Best attempt chosen only by hallucination count | Fewest hallucinations + missing, then best attribute score (VLM check runs only on tied attempts). Every attempt recorded in `attempt_log` in `reconstruction.json` |

Face blur and box colours are unchanged. Tests: `tests/test_logic.py` has 23 model-free tests (was 8), all passing.

## Extraction before / after
Answer key: 14 facts. "Victim behind the counter" is in the key but no witness says it, so 13/14 is the ceiling.

| Run | Facts | Wrong attribute | Key facts missed | Conflicts (real) | Gaps | People | Knife confidence | Knife withheld |
|---|---|---|---|---|---|---|---|---|
| Original | 9 | 2 | 11/14 | 2 (1) | 11 | 2 | no weapon_type | only as "held_by suspect" |
| + attribute guide, post-check | 23 | 1 | 3/14 | 3 (1) | 11 | 3 | 0.70 | no |
| + hedge, placeholder, merge fixes | 22 | 0 | 3/14 | 2 (1) | 4 | 2 | 0.575 | yes |
| + grounding, articles, actions (final, 2 identical runs) | 22 | 0 | 2/14 | 1 (1) | 5 | 2 | 0.575 | yes |

The grounding check dropped nothing in the final runs. Replayed on the previous run's facts, it drops
exactly the three invented facts (witness A: carrying "nothing", lighting "brightly lit", time_of_day "afternoon").

## Verification before / after
The images differ between rows (prompts and seeds changed), except that rows 2 and 3 share the same first H1 image.

| Stage | Hyp. | Attempts | Hallucinations (first → kept) | Attribute score | Boxes on the right person? |
|---|---|---|---|---|---|
| Original extraction + original verifier | H1 | 1 | 0 → 0 | 33% | No: suspect labelled `person_2` |
| | H2 | 2 | 2 → 0 | 100% | Not checked; image had 3 people for a 2-person prompt, not flagged |
| Final extraction + original verifier | H1 | 1 | 0 → 0 | 67% | No: both boxes on two background people, suspect unboxed, 0 hallucinations reported |
| + attribute queries, count check, part merge | H1 | 3 | 2 → 1 (bag) | 33% | Yes (cashier absent, reported missing) |
| | H2 | 2 | 3 → 0 | 67% | No: cashier box on the suspect, suspect box on an arm fragment |
| **Final** | **H1** | **3** | **2 → 0** | **33%** | **Yes.** SDXL left out the cashier in all 3 attempts; she's reported missing, and the bag in his hand is excused as carried |
| | **H2** | **3** | **2 → 0** | **50%** | **Yes, both:** blue-jacket suspect and green-top second person; backpack excused |

## evaluate.py (`sample_robbery`, cached facts, no `--generate`)
| Facts | facts_in_graph | recovery | drawn_precision | false_withheld | conflicts_found | questions_safe | questions_blocked | baseline_blocked | leading_leaked |
|---|---|---|---|---|---|---|---|---|---|
| Original extraction | 9 | 0.214 | 0.400 | 1/1 | 1/1 | 26 | 10 | 10/10 | 0 |
| **Final extraction** | **22** | **0.857** | **0.667** | **1/1** | **1/1** | **9** | **6** | **6/6** | **0** |
| Hand-made `sample_facts.json` (reference) | 19 | 0.929 | 0.867 | 1/1 | 1/1 | 11 | 6 | 6/6 | 0 |

drawn_precision counts drawn facts found in the answer key. The key lists fewer actions and places than
the witnesses state ("walked straight to the counter", "snacks aisle"), so some correct drawn facts count
against it.

## Remaining limitations
- **SDXL often draws fewer people than asked:** the cashier was missing from all 3 H1 attempts. Retries can't fix this reliably; layout-guided generation (GLIGEN / regional prompting) is planned.
- **Still missed by extraction:** "night" (witness B) and the cash register object.
- **Attribute scores stay low (33–50%):** the VLM never confirms "green uniform shirt", "black jeans" or "black backpack".
- **The verification rules are heuristics with fixed thresholds** (fragment < 10% area, 30% overlap for carried items, OWLv2 scores near 0.2). A very small, distant person right next to a foreground person could be dropped as a fragment.
- **Only one scene has been evaluated,** and without `--generate`.
- **Runtime:** one full run takes about 8–10 min on the shared GPU. Each VLM attribute check takes ~1.5 min because Ollama reloads the model per question (`KEEP_ALIVE=0`).

## Files changed today
- New: `setup_hpc.sh`, `REVIEW_NOTES.md`
- `README.md`: Setup and Run sections for this HPC; verification paragraph
- `config.yaml`: Neo4j off, `cpu_offload: sequential`
- `requirements.txt`: `scipy`
- `generation.py`: sequential offload, 512 px VAE tiles
- `llm.py`: `chat_json(..., temperature=)`
- `extraction.py`: guide, example, post-check, grounding, sentence hedges, merge
- `verification.py`: person queries, assignment, count check, part/fragment merge, carried items, logit-based detection
- `pipeline.py`: retry on missing person, keep best attempt, `attempt_log`
- `tests/test_logic.py`: 15 new tests

Added during the real-data evaluation (17–18 Oct):
- `config.yaml`: `cpu_offload` set from free GPU memory before each run (currently `true`), `max_retries: 4`
- `data/make_witnesses.py`: no entity ids in statement prompts or statements (labels, retry, refuse)
- `evaluate.py`: drawn precision split into witness / CCTV; average row
- `tests/test_logic.py`: 3 tests for the id leak; the retry test no longer depends on `config.yaml`
- New scenes: `data/scenes/robbery044`, `robbery058`, `shoplifting033` (frames, scene.json, ground_truth.json, witness_truth.json)

Added in round 2 (18 Oct):
- `evaluate.py`: leak check counts content words only
- `data/make_witnesses.py`: nouns, per-person prompt, statement checks, no multi-valued corruptions
- `extraction.py`: CCTV linking and cross-frame merge, mask rule, place_type check, role ranking
- `matching.py`: generic garment words in `values_agree`
- `tests/test_logic.py`: now 36 tests
- New held-out scenes: `data/scenes/robbery012`, `data/scenes/shoplifting024`

Not changed: `schema.py`, `graph.py`, `conflicts.py`, `questions.py`, `app.py`.

Generated (gitignored or outside the repo): `.venv/`, `outputs/`, `~/.local/bin/uv`, `~/.local/ollama`, `~/.ollama` (model), Hugging Face cache (SDXL, OWLv2).

## Real-data evaluation: 17–18 Oct 2026

### GPU settings for this session
Free GPU memory rose to ~24 GiB, so the run used `cpu_offload: false` with Ollama loaded in GPU memory
(`OLLAMA_KEEP_ALIVE=10m`). On `sample_robbery` an image (SDXL + OWLv2) took ~4–5 s instead of ~40–43 s, and
a VLM attribute check took 2–22 s instead of ~100 s. Free memory is re-checked before every long run and
`cpu_offload` set from it (≥ 20 GiB `false`, ≥ 10 GiB `true`, else `sequential`). It was 15.8 GiB before
this evaluation, so it ran with `cpu_offload: true`. `max_retries` is now 4.

### Data
UCA captions (now at `Xuange923/Surveillance-Video-Understanding`; the `UCA-dataset` URL is a 404) for
three UCF-Crime clips. Only the three videos were fetched from the official 103 GB `UCF_Crimes.zip`, using
HTTP byte ranges (18.1 MB including the zip index, CRC checked). All three are 320×240, 30 fps. Frames are
6 per clip, `--every 2`, over the UCA time window.

| Scene | Clip | Window | People | Answer key (after hand edits) |
|---|---|---|---|---|
| `robbery044` | Robbery044_x264 | 0.9–15.0 s | 2 | 8 facts; deleted a phantom bystander |
| `robbery058` | Robbery058_x264 | 0.6–58.5 s | 2 | 15 facts; added gun, 4 victim actions, garment wording |
| `shoplifting033` | Shoplifting033_x264 | 0.9–30.0 s | 2 | 11 facts; fixed two actions, added the watch |

Robbery040 was replaced by Robbery058 (it had only one person, so no conflict could be planted). After
the edits, the shoplifting033 rebuild planted no conflict, so `conflicts_found` is empty for that scene.
Edits are recorded in each `ground_truth.json` under `hand_edits`.

Fixed first: `make_witnesses.py` passed entity ids (`person_9`) into the statement prompt and they leaked
into statements. It now uses readable labels ("the suspect", "a bystander") and retries, then refuses, any
statement that still contains an id. There are tests for the prompts, the saved statements and the retry.

### `python evaluate.py --generate` (all 4 scenes)
| scene | facts | recovery | drawn_precision | witness | CCTV | false_withheld | conflicts_found | safe / blocked Qs | leading_leaked | halluc first → final | object_faithfulness | attribute_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robbery044 | 29 | 0.875 | 0.462 | 0.750 | 0.000 | 1/1 | 1/1 | 33 / 7 | 0 | 2 → 1 | 0.834 | 0.800 |
| robbery058 | 29 | 0.867 | 0.444 | 0.571 | 0.000 | 1/1 | 0/1 | 28 / 2 | 3 | 2 → 0 | 0.286 | 0.667 |
| sample_robbery | 22 | 0.857 | 0.667 | 0.667 | – | 1/1 | 1/1 | 9 / 6 | 0 | 3 → 0 | 0.750 | 0.750 |
| shoplifting033 | 28 | 0.727 | 0.438 | 0.538 | 0.000 | 1/1 | – | 28 / 6 | 0 | 3 → 1 | 0.800 | 0.875 |
| **average** | 27.0 | **0.832** | **0.503** | **0.631** | **0.000** | **4/4** | **2/3** | 24.5 / 5.25 | 0.75 | 2.5 → 0.5 | **0.667** | **0.773** |

### What behaves differently on real CCTV
1. **CCTV people are never matched to the witnesses' suspect or victim.** The frame prompt tells the model
   to use `bystander` unless the role is obvious from one frame. The merge backstop only joins
   suspects/victims, so CCTV people stay separate "bystanders". Examples:
   - The robbery044 victim on CCTV (white cap, grey shirt, hands raised, behind the counter) became a
     third person.
   - The shoplifting033 suspect on CCTV (blue shirt, khaki shorts) became a fourth.

   This is why **CCTV drawn precision is 0.0 in every scene**. The CCTV details are mostly right (the
   white cap matches the victim's "white hat"), but they sit on the wrong role, and the evaluation
   requires the role to match. It also makes prompts ask for too many people: 3, 7 and 4, when each clip
   has 2.
2. **The same person is split across frames.** In robbery058 the victim's red top was read as "red
   hoodie", "pink shirt" and "red shirt" in different frames, giving three people.
3. **CCTV confirmed a false memory.** In robbery044, witness C's invented "bystander in a grey coat" was
   merged with the real victim seen on CCTV ("grey shirt"), because both were bystanders. So
   the planted confabulation was attached to a real person.
4. **One wrong role turns off the merge backstop.** In robbery058, witness C labelled the invented
   grey-coat man a second *suspect*. With two suspects in one source, the "each source names one suspect"
   rule doesn't apply, and the suspect ended up as three people. This is why that scene found 0/1
   conflicts: the planted "white clothes" landed on a separate person. It is also why object
   faithfulness is only 0.286, since 7 people were asked for and 2 drawn.
5. **Witnesses without a role aren't merged.** In shoplifting033, witnesses B and C described "a man"
   with no role (`none`), so they weren't joined to witness A's suspect either.
6. **CCTV frames give few facts:** 1–4 per frame at 320×240. Mostly clothing colour, almost never actions,
   weapons or the place. Robbery058's small, dark suspect produced no CCTV facts at all.
7. **Extraction slips that real statements exposed:** "mask" stored as upper_clothing (no keyword rule
   yet); place_type values "scene" and "place where the watch was" (they pass grounding because the words
   appear in the statement). Grounding correctly dropped witness A's "gun" in robbery058: the
   statement only says "the weapon".
8. **The generated witnesses are weaker than the hand-written ones.** The statement writer gets roles but
   no gender, so one statement made the shoplifting033 woman a man. Another described the robbery058
   suspect as "a man in white clothes standing behind the checkout counter".
9. **What still works:** the planted false memory is withheld in all 4 scenes (4/4), every baseline
   leading question is blocked (21/21), and with real frames hallucinations still drop from 2.5 to 0.5
   per scene after regeneration.

### Next fixes suggested (not done)
- Let CCTV people inherit a role during merging: a CCTV person who matches a witness's suspect or victim
  on clothing should be merged into them, not left as a bystander.
- Merge per-frame CCTV mentions of the same person, allowing colour drift ("red hoodie" / "red shirt").
- Run the one-suspect rule on roles after merging. Or treat role-less persons (`none`) like the role they
  are merged with.
- Add a `mask` → headwear rule, and reject place_type values that only repeat generic words ("scene",
  "place").
- Give the statement writer the caption's own description of each person ("a woman in red").

## Real-data evaluation, round 2: fixes, then a held-out check (18 Oct 2026)

**Superseded:** the first-round table and the "what behaves differently" list above. They describe the code before these fixes.

### Fixes (development scenes only)
- **Leak metric:** the 3 robbery058 "leaked" questions were false alarms. The only flagged word was the template's "on"
  ("wearing on their legs"), which the answer key also contains. `evaluate.py` now counts only content words.
  This is a regression test; the leading-question filter itself was correct.
- **Witness generator** (`make_witnesses.py`):
  - A hand-checkable `nouns` entry (man/woman/person) per person in `ground_truth.json`.
  - Details grouped per person under a plain-language role ("a woman who was being robbed").
  - Statements are rejected and retried for role labels ("suspect"/"victim"/"bystander"), ids, a copied
    "(unsure)", or a gender nobody has. They are also re-read by the extractor and rejected if one person
    holds two people's details.
- **CCTV linking** (`extraction.py`):
  - CCTV people are taken out of the merge groups and merged across frames (colour drift allowed, no
    colour-family clash).
  - Each joins the witnesses' suspect or victim when attributes agree (`values_agree`), with more agreeing
    than clashing attributes and a unique best match.
  - "Red clothes" now matches "red hoodie" (generic garment words).
  - Suspect/victim outrank the CCTV default "bystander".
  - New rules: mask/balaclava → headwear; place types that don't name a place are dropped.

### Development scenes: before → after (`evaluate.py --generate --force-extract`)
| Scene | Recovery | Drawn precision (all / witness / CCTV) | Conflicts | Leaked | Halluc. first → final | Object faithf. | Attribute acc. | People drawn (H1, actual 2) |
|---|---|---|---|---|---|---|---|---|
| robbery044 | 0.875 → 1.000 | 0.462/0.750/0.000 → 0.467/0.556/0.333 | 1/1 → 1/1 | 0 → 0 | 2→1 → 4→2 | 0.834 → 0.834 | 0.800 → 0.600 | 3 → 3 |
| robbery058 | 0.867 → 0.733 | 0.444/0.571/0.000 → 0.583/0.600/0.500 | 0/1 → 0/1 | 3 → 0 | 2→0 → 5→1 | 0.286 → 1.000 | 0.667 → 0.834 | 7 → 2 |
| sample_robbery | 0.857 → 0.857 | 0.667/0.667/– → same | 1/1 → 1/1 | 0 → 0 | 3→0 → 3→0 | 0.750 → 1.000 | 0.750 → 0.584 | 2 → 2 |
| shoplifting033 | 0.727 → 1.000 | 0.438/0.538/0.000 → 0.692/0.800/0.333 | n/a | 0 → 0 | 3→1 → 4→1 | 0.800 → 1.000 | 0.875 → 0.667 | 4 → 2 |
| **Average** | 0.832 → **0.897** | 0.503/0.631/0.000 → **0.602/0.656/0.389** | 2/3 → 2/3 | 0.75 → **0** | 2.5→0.5 → 4.0→1.0 | 0.667 → **0.959** | 0.773 → 0.671 | |

### Harness fixes made before the held-out run (test-data generator only; pipeline code frozen)
1. **No corruptions on multi-valued attributes.** Planted corruptions never target `schema.MULTI_VALUED`
   (action/position/notable_features). A different action is never treated as a conflict, so such a
   plant ("handed it to the man in black" → "… in green") could never be found. There's a test.
2. **The attribution check compares who holds which detail, not roles.** It used to compare extracted
   roles, and the extractor called shoplifting024's shop worker a suspect. That rejected a correct
   statement and made `--reuse-gt` stop. Shared details ("behind the counter") are ignored. There's a test.

### Held-out scenes (code frozen; never seen during development)
Robbery012 (25.0 s, window 1.0–24.2 s) and Shoplifting024 (40.3 s, window 2.2–40.2 s), fetched the
same way. Hand edits are in each `ground_truth.json`.

| Scene | Facts | Recovery | Drawn precision (all / witness / CCTV) | False withheld | Conflicts | Safe / blocked Qs | Leaked | Halluc. first → final | Object faithf. | Attribute acc. | People drawn (H1, actual 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robbery012 | 30 | 0.056 | 0.067 / 0.083 / 0.000 | 1/1 | 0/1 | 30 / 7 | 0 | 2 → 2 | 0.900 | 0.600 | 4 |
| shoplifting024 | 32 | 0.500 | 0.053 / 0.111 / 0.000 | 1/1 | n/a | 29 / 0 | 0 | 0 → 0 | 0.667 | 0.667 | 6 |
| **Average** | 31.0 | **0.278** | **0.060 / 0.097 / 0.000** | 2/2 | 0/1 | 29.5 / 3.5 | 0 | 1.0 → 1.0 | 0.784 | 0.633 | |

**The dev gains do not carry over.** The cause is role extraction on the generated statements:
- **robbery012:** all three witnesses' "man who was being robbed" was extracted as a **suspect**. Every
  cashier fact sits on the wrong role, and recovery and precision require the role to match.
  - The CCTV cashier (red t-shirt) can't be linked: the only witness-side shirt is B's planted "blue
    shirt", a colour clash. So the planted conflict is also missed.
  - The CCTV robber (dark jacket, helmet) stays a separate person: 4 people.
- **shoplifting024:** the woman is a suspect for witness A, a victim for B and a bystander for C, so she's
  split into 3 people.
  - The CCTV woman (checkered shirt) was linked to the **man**, through one shared detail: witness C's
    statement put the man "behind the counter".
  - No question was blocked: with no conflicts, no naive baseline questions were generated.
- **What holds on held-out:** planted false memories are withheld (2/2), and no leading question reaches a
  witness (0 leaked).

#### Held-out diagnosis (no code changes; the official numbers above stay as they are)
Traced with the frozen pipeline functions on the same inputs (temperature 0). The per-source mentions
and merge results match the evaluation run's `facts.json`, except one CCTV frame of shoplifting024: frame 5
came back as "dark patterned shirt" in the trace and "patterned jacket" in the evaluation. Vision
extraction isn't fully repeatable. Below, the evaluation run's own facts are used.

**robbery012: how 2 people became 4 drawn people** (H1 prompt "exactly 4 people")
| Drawn entity | Made of | Why it didn't merge |
|---|---|---|
| person_1 "suspect" | All three witnesses' **cashier** ("a man who was being robbed", "a man in a blue shirt", "a man") | Merged correctly, but the extractor labelled every mention `suspect` |
| person_3 "victim" | Witnesses B and C's **robber** ("another man in black") | Role inverted (`victim`); its only fact is C's misplaced position "walked out of the cashier" |
| person_5 "suspect" | CCTV frames 5–6: the **robber** (dark jacket, helmet) | No link: no agreeing attribute with any witness-side person. "Dark" and "blue" share the navy family, so no clash either; the score was 0 |
| person_6 "bystander" | CCTV frames 1–4: the **cashier** (red t-shirt) | No link: red vs witness B's planted "blue shirt" is a colour clash (−1) |

Not drawn: witness A's robber mention ("another man, wearing a black coat") got no facts at all, because
the black coat wasn't extracted; and witness C's invented grey-coat man was withheld at 0.45. **Net: the
cashier and the robber each appear twice,** once from the witnesses and once from CCTV.

**shoplifting024: how 2 people became 6** (H1 prompt "exactly 6 people")
| Drawn entity | Made of | Why |
|---|---|---|
| person_1 "suspect" | The **man** for witnesses A, B and C, **plus** CCTV frame 6 (checkered shirt, jeans, backpack: the **woman**) | Man labelled `suspect`. The CCTV woman was linked to him through one agreeing detail, "behind the counter": witness C's statement put the man there |
| person_2 "suspect" | The woman according to witness A | Same woman, but each witness gave her a different role, so the merge kept them apart |
| person_3 "victim" | The woman according to witness B | (as above) |
| person_4 "bystander" | The woman according to witness C | (as above) |
| person_5 "suspect" | CCTV frames 1–4: dark jacket, sitting with papers (probably the man) | No link: the witness-side man has no clothing to agree with |
| person_6 "bystander" | CCTV frame 5: "patterned jacket" (probably the woman) | Not merged with frames 1–4: "patterned" names no colour, so no look agrees |

**Net: the woman appears 3 times from the witnesses and up to 2 times from CCTV,** one of them glued to
the man. Both failures start from roles extracted from the statements; the CCTV rules then have nothing
reliable to join.

**Role-agnostic scores (DIAGNOSTIC ONLY, not an official metric).** Same matching as `evaluate.py`
(entity type + attribute + `values_agree`), but without the role check. Recovery uses all facts in
the graph; drawn precision uses the facts H1 would draw.

| Scene | Recovery: official | Recovery: role-agnostic (diag.) | Drawn precision: official | Drawn precision: role-agnostic (diag.) |
|---|---|---|---|---|
| robbery044 (dev) | 1.000 | 1.000 | 0.467 | 0.467 |
| robbery058 (dev) | 0.733 | 0.733 | 0.583 | 0.583 |
| sample_robbery (dev) | 0.857 | 0.857 | 0.667 | 0.667 |
| shoplifting033 (dev) | 1.000 | 1.000 | 0.692 | 0.692 |
| **robbery012 (held-out)** | **0.056** | 0.556 | **0.067** | 0.533 |
| **shoplifting024 (held-out)** | **0.500** | 0.875 | **0.053** | 0.263 |

On dev, ignoring roles changes nothing. On held-out it explains most of the gap: robbery012 recovers 10×
more facts once roles are ignored. What is still missed then:
- robbery012: everything about the robber (bandana, black coat, his three actions, "held by man in
  black") and the cashier's "red shirt". The CCTV says "red t-shirt", which `values_agree` doesn't match:
  it reads "t-shirt" as one word. Also "sat next to the cash register" (stored as a position).
- shoplifting024: only "put the book down".

shoplifting024's precision stays low even without roles: the drawn CCTV clothing (dark jacket, checkered
shirt, jeans) is correct on camera, but the caption has no clothing at all.

**robbery012: ground truth vs extracted facts** (an extracted fact counts officially only if the role matches too)

| Ground truth | Extracted (entity, role, source) | Official match |
|---|---|---|
| person_3 victim hair = 'black' | 'black hair' (person_1, suspect, witness_A) | **no: role** |
| person_3 victim action = 'sat next to the cash register' | — | no: not extracted |
| person_3 victim action = 'looked up into the distance' | 'looked up into the distance' (person_1, suspect, witness_A); 'looked up into the distance' (person_1, suspect, witness_B); 'looked up into the distance' (person_1, suspect, witness_C) | **no: role** |
| person_2 suspect headwear = 'white bandana' | — | no: not extracted |
| person_2 suspect upper_clothing = 'black coat' | — | no: not extracted |
| person_2 suspect action = 'walked into the checkout counter' | — | no: not extracted |
| person_3 victim upper_clothing = 'red shirt' | — | no: not extracted |
| person_3 victim action = 'walked to the cash register' | 'opened the cash register' (person_1, suspect, witness_B); 'walked to the cash register' (person_1, suspect, witness_A) | **no: role** |
| person_3 victim action = 'opened the cash register' | 'opened the cash register' (person_1, suspect, witness_B); 'walked to the cash register' (person_1, suspect, witness_A) | **no: role** |
| person_3 victim action = 'put the money in the cash register' | 'put the money in the cash register' (person_1, suspect, witness_A); 'put the money in the cash register' (person_1, suspect, witness_C) | **no: role** |
| person_3 victim action = 'handed it to the man in black' | 'handed it to another man in black' (person_1, suspect, witness_B); 'handed it to another man in black' (person_1, suspect, witness_C); 'handed it to the man in black' (person_1, suspect, witness_A) | **no: role** |
| person_3 victim action = 'closed the cashier drawer' | 'closed the cashier drawer' (person_1, suspect, witness_A) | **no: role** |
| person_3 victim action = 'walked out of the cashier' | 'walked out of the cashier' (person_1, suspect, witness_B) | **no: role** |
| person_3 victim action = 'tried to run into the house' | 'tried to run into the house' (person_1, suspect, witness_A); 'tried to run into the house' (person_1, suspect, witness_B) | **no: role** |
| person_2 suspect action = 'tried to intercept' | — | no: not extracted |
| person_2 suspect action = 'turned around and left in the opposite direction' | — | no: not extracted |
| weapon_1 none weapon_type = 'gun' | 'gun' (weapon_1, none, witness_A); 'gun' (weapon_1, none, witness_B) | yes |
| weapon_1 none held_by = 'man in black' | — | no: not extracted |

Extracted facts with no ground-truth match (13): weapon_1/none held_by='man who was being robbed' (witness_A); person_1/suspect upper_clothing='blue shirt' (witness_B); weapon_1/none held_by='man in blue shirt' (witness_B); person_1/suspect position='sitting next to the cash register' (witness_C); person_3/victim position='walked out of the cashier' (witness_C); person_4/bystander upper_clothing='grey coat' (witness_C); person_6/bystander upper_clothing='red t-shirt' (cctv_frame_01); person_6/bystander upper_clothing='red t-shirt' (cctv_frame_02); person_6/bystander upper_clothing='red t-shirt' (cctv_frame_03); person_6/bystander upper_clothing='red t-shirt' (cctv_frame_04); person_5/suspect upper_clothing='dark jacket' (cctv_frame_05); person_5/suspect headwear='helmet' (cctv_frame_05); person_5/suspect upper_clothing='dark jacket' (cctv_frame_06)

**shoplifting024: ground truth vs extracted facts** (an extracted fact counts officially only if the role matches too)

| Ground truth | Extracted (entity, role, source) | Official match |
|---|---|---|
| person_1 victim action = 'turned around and put things into the cabinet' | 'turned around and put things into the cabinet' (person_1, suspect, witness_A) | **no: role** |
| person_1 victim action = 'stood up and walked out' | 'stood up and walked out' (person_1, suspect, witness_A); 'stood up' (person_3, victim, witness_B) | yes |
| person_2 suspect action = 'picked up the white paper on the table and fanned herself' | 'picked up the white paper on the table' (person_2, suspect, witness_A); 'picked up the white paper on the table' (person_3, victim, witness_B); 'picked up the white paper on the table' (person_4, bystander, witness_C) | yes |
| person_2 suspect action = 'opened the book and looked at' | 'opened the book and looked at it' (person_3, victim, witness_B) | **no: role** |
| person_2 suspect action = 'stood up and looked back in the direction where the man left' | 'left' (person_1, suspect, witness_B); 'looked back in the direction where the man left' (person_3, victim, witness_B); 'stood up and looked back in the direction where the man left' (person_2, suspect, witness_A); 'stood up' (person_3, victim, witness_B) | yes |
| person_2 suspect action = 'took something from the table and put it into her clothes' | 'maybe she took something from the table and put it into her clothes' (person_2, suspect, witness_A); 'put it into her clothes' (person_3, victim, witness_B); 'took something from the table' (person_3, victim, witness_B) | yes |
| person_2 suspect action = 'put the book down' | — | no: not extracted |
| person_1 victim action = 'came back' | 'came back' (person_1, suspect, witness_C) | **no: role** |

Extracted facts with no ground-truth match (18): person_2/suspect action='fan herself' (witness_A); person_3/victim action='fan herself' (witness_B); person_1/suspect position='behind the counter' (witness_C); person_4/bystander action='fan herself' (witness_C); person_4/bystander position='behind the counter' (witness_C); person_5/suspect upper_clothing='dark jacket' (cctv_frame_01); person_5/suspect upper_clothing='dark jacket' (cctv_frame_02); person_5/suspect action='holding a red object' (cctv_frame_02); person_5/suspect position='sitting' (cctv_frame_02); person_5/suspect upper_clothing='dark jacket' (cctv_frame_03); person_5/suspect upper_clothing='dark jacket' (cctv_frame_04); person_6/bystander upper_clothing='patterned jacket' (cctv_frame_05); person_1/suspect upper_clothing='checkered shirt' (cctv_frame_06); person_1/suspect lower_clothing='jeans' (cctv_frame_06); person_1/suspect hair='dark' (cctv_frame_06); person_1/suspect action='writing' (cctv_frame_06); person_1/suspect position='behind the counter' (cctv_frame_06); person_1/suspect carrying='black backpack' (cctv_frame_06)

### Known limitations (current)
- **Roles from statements are unreliable.** "The man who was being robbed" or "the man who works in the
  shop" can come out as a suspect. Every later step (merge, CCTV linking, the role-matched metrics) depends on it.
  This is the next thing to fix: a role check against the statement's own words ("being robbed",
  "works there"), and role-free merging by description.
- **The CCTV link can rest on a single shared detail.** It needs a unique best match, but one agreeing
  position (shoplifting024) is enough to link the wrong person.
- **Generated statements still add small details** ("a woman behind the counter") that aren't in the
  answer key. On shoplifting024 that caused the wrong CCTV link above.
- **Some planted conflicts can't be found.** When only the corrupting witness mentions the attribute and
  the camera person isn't linked, nothing contradicts it (robbery058, robbery012).
- **SDXL often draws fewer or more people than asked.** Layout-guided generation is planned.
- **Still missed by extraction:** "night" in `sample_robbery`, objects like the cash register, and
  clothing given as a noun phrase ("a man in black").
- **Small sample:** 4 development scenes and 2 held-out scenes. All the real clips are 320×240.

## SDXL vs SD3.5 Medium (18 Oct 2026)
Both backends ran `evaluate.py --generate` on all 6 scenes with the same cached facts, the same
token-aware prompts (CLIP limit 77; the longest was 76 tokens), the same seeds, `max_retries: 4` and
`cpu_offload: true`. SD3.5 used 40 steps and guidance 4.5; SDXL used its 30 steps. The first SD3.5 run
hit the MIG out-of-memory assert at robbery012, because our Ollama (keep-alive 10 min) still held
6.7 GiB. That run was discarded and repeated in full with `OLLAMA_KEEP_ALIVE=0`.
Saved runs: `outputs/runs/run_sdxl6/`, `outputs/runs/run_sd35_6/`; a side-by-side H1 image of
robbery058 is in `outputs/runs/robbery058_sdxl_vs_sd35.jpg`.

"People" is the H1 count: OWLv2 people in the image / people in the prompt / people in the ground
truth. Seconds per image is the median gap between consecutive attempts (generation + OWLv2 check;
model loading and VLM attribute checks excluded).

| Scene | Halluc. first→final (SDXL / SD3.5) | Object faithfulness | Attribute accuracy | People image/prompt/actual (SDXL / SD3.5) | Sec per image |
|---|---|---|---|---|---|
| robbery044 | 4→2 / 1→1 | 0.834 / 0.834 | 0.600 / 1.000 | 3/3/2 / 2/3/2 | 8.5 / 30.0 |
| robbery058 | 5→0 / 5→2 | 1.000 / 1.000 | 0.667 / 1.000 | 2/2/2 / 2/2/2 | 8.0 / 25.5 |
| sample_robbery | 1→1 / 4→0 | 0.750 / 1.000 | 0.833 / 0.834 | 2/2/2 / 2/2/2 | 8.0 / 27.0 |
| shoplifting033 | 5→0 / 0→0 | 1.000 / 0.667 | 0.667 / 0.667 | 2/2/2 / 1/2/2 | 7.5 / 28.0 |
| robbery012 (held-out) | 1→1 / 0→0 | 0.500 / 0.700 | 0.700 / 1.000 | 2/4/2 / 2/4/2 | 8.5 / 23.0 |
| shoplifting024 (held-out) | 8→2 / 3→0 | 0.500 / 0.333 | 1.000 / 1.000 | 3/6/2 / 2/6/2 | 8.0 / 22.5 |
| **Average** | **4.0→1.0 / 2.2→0.5** | 0.764 / 0.756 | **0.744 / 0.917** | image = prompt: 4/6 / 2/6 | **8.0 / 27.0** (n = 29 / 37) |

Total attempts: SDXL 39, SD3.5 47. Attempts with the person count the prompt asked for: 18 vs 15.

**Interpretation**
- SD3.5 hallucinates about half as often (first and final) and follows the described clothing and
  colours better (attribute accuracy 0.917 vs 0.744). Its images look more like CCTV.
- Object faithfulness is about the same.
- SD3.5 gets the person count right less often, e.g. one person instead of two in shoplifting033.
  On robbery058 it drew a gun that isn't in the evidence (flagged as a hallucination).
- SD3.5 is about 3.4× slower per image; part of that is 40 vs 30 steps.
- People inflation on the held-out scenes (4 and 6 drawn for 2 actual) comes from extraction, not
  the generator; both backends inherit it.
- Small sample (6 scenes, one seed set, 0–8 hallucinations per scene): the differences are
  directional, not significant. The default stays `sdxl` for speed.

## Role fix: roles no longer decide who is who (18 Oct 2026)
A general design change, made and checked on the 4 development scenes only. Nothing was tuned on
robbery012 or shoplifting024, and their official numbers above are unchanged (not re-run).

**What changed** (`extraction.py`, plus small hooks in `graph.py`, `conflicts.py`, `pipeline.py`, `questions.py`)
- **Merge first, by attributes and source consistency.** The LLM merge prompt no longer shows roles
  ("who robbed whom is often misread"). Without the LLM, each person mention starts on its own. Then:
  1. Split any group that repeats a source or mixes a man and a woman (gender from the mention's own
     words).
  2. **Lone men / lone women:** if no witness describes two men, all witnesses' men are one man. This
     replaces the old "each source names one suspect" rule.
  3. **Details agree:** join two witness-side groups from different sources with link score ≥ 1 that are
     each other's best match. A tie is allowed when the tied groups all agree with each other. The
     witness's own description ("the man behind the counter", gender word removed) counts as one
     detail.
  4. Link CCTV people (below).
- **Then vote the role.** Each witness has one vote; all CCTV frames together are one source. A CCTV
  "bystander" is no vote, since it is the frame prompt's default. Ties go to suspect/victim over
  bystander, then suspect before victim.
- **Role disagreement is a conflict** (`kind: "role"`, stored as `role_votes` in `facts.json`, shown with
  the other conflicts and in the audit log). It is soft: it doesn't branch hypotheses, change what is
  drawn, or produce re-check questions (asking "was he the robber?" would be leading).
- **CCTV linking:**
  - every witness-side person is a candidate, not only suspects and victims;
  - roles only break ties;
  - one clash is forgiven when another attribute agrees (link score = agreeing − (clashing − 1));
  - a man/woman mismatch counts as one clash, not a veto, because one 320×240 frame of robbery058
    called the woman cashier "man in red hoodie";
  - pattern words (patterned, checked/checkered, plaid, striped, floral, …) are compatible descriptors:
    two patterned garments agree, and a patterned garment never clashes on colour.
- Tests: 44 (was 38), covering every rule above.

**Development scenes: before → after** (`evaluate.py --generate --force-extract`, SDXL). Before = the
SDXL run in `outputs/runs/run_sdxl6/` (round-2 facts); after = `outputs/`. People = person entities in the
graph; drawn = people in the H1 prompt; actual = 2 in every scene.

| Scene | Recovery | Drawn precision (witness / CCTV) | False withheld | Conflicts found | Leaked | Halluc. first→final | Object faithf. | Attribute acc. | People / drawn / in image |
|---|---|---|---|---|---|---|---|---|---|
| robbery044 | 1.000 → 1.000 | 0.467 → 0.467 (0.556 / 0.333 both) | 1/1 → 1/1 | 1/1 → 1/1 | 0 → 0 | 4→2 → 3→2 | 0.834 → 0.834 | 0.600 → 0.700 | 3/3/3 → 3/3/3 |
| robbery058 | 0.733 → 0.733 | 0.583 → 0.583 (0.6 / 0.5 both) | 1/1 → 1/1 | 0/1 → 0/1 | 0 → 0 | 5→0 → 5→0 | 1.000 → 1.000 | 0.667 → 0.667 | 3/2/2 → 4/2/2 |
| sample_robbery | 0.857 → 0.857 | 0.667 → 0.667 | 1/1 → 1/1 | 1/1 → 1/1 | 0 → 0 | 1→1 → 1→1 | 0.750 → 0.750 | 0.833 → 0.916 | 2/2/2 → 2/2/2 |
| shoplifting033 | 1.000 → 0.909 | 0.692 → 0.571 (0.8 / 0.333 → 0.8 / 0.0) | 1/1 → 1/1 | – | 0 → 0 | 5→0 → 2→1 | 1.000 → 0.750 | 0.667 → 0.667 | 2/2/2 → 3/3/2 |

Role disagreements found: 0 in all 4 dev scenes (their witnesses agree on roles).

Reading the table:
- robbery044's merge is now cleaner: 16 mentions → 5 entities, down from 7 in the first run of the new
  code. All three witnesses' "man behind the counter" and the camera person are now one victim; the
  remaining extra is witness C's invented grey-coat man, as before.
- robbery058 needed the gender softening: with a hard man/woman veto, the CCTV cashier didn't link
  and CCTV precision fell to 0.0. Recorded in the intermediate runs; fixed before the final run.
- **shoplifting033's drop is extraction drift, not the merge.** Witness A's "I think he was wearing a
  blue shirt" is no longer extracted. Without it nothing on the witness side agrees with the camera's
  blue shirt, so the camera man stays separate (3 people). The **pre-change** `extraction.py` misses it too
  when run now (checked on the same statement), so Qwen via Ollama is giving different output than in
  the earlier session.
- Image metrics (hallucinations, faithfulness) vary with the prompt and seed, so treat them as noise at
  this sample size.
- Cost of the design: without roles, two mentions that share no detail and no description only join
  through the lone-man / lone-woman rule. The old same-role rule joined them, rightly or wrongly.

## Held-out round 2: robbery003, shoplifting001 (code frozen at tag v4)
Two new UCA train-split clips, never used before: **Robbery003** (a robber in a plaid shirt with a
black pistol, and the clerk in black) and **Shoplifting001** (two men, both shoplifters, the first 6 of 8
sentences). The ground truth was hand-edited as recorded in each `ground_truth.json` → `hand_edits`.
**Unreviewed addition:** robbery003's `object_1` object_type "bottle of Coke" was added by me without the
user's approval. It is kept, but it was never reviewed; treat its effect on robbery003's numbers accordingly.
Planted: one shirt-colour corruption on a suspect (plaid black→blue; orange→brown) and a
confabulated white van. Evaluated with `evaluate.py --generate --force-extract`, SDXL, and no code changes
after v4. Saved in `outputs/runs/heldout2_v4/`.

| Scene | Recovery | Drawn precision (witness / CCTV) | False withheld | Conflicts found | Leaked | Halluc. first→final | Object faithf. | Attribute acc. | People drawn / in image / actual |
|---|---|---|---|---|---|---|---|---|---|
| robbery003 | 0.875 | 0.739 (0.81 / 0.0) | 1/1 | **0/1** | 0 | 6→5 | 0.643 | 0.750 | **6** / 6 / 2 |
| shoplifting001 | 0.864 | 0.794 (0.931 / 0.0) | 1/1 | **0/1** | 0 | 1→1 | 0.222 | 0.667 | **8** / 2 / 2 |

The role-agnostic diagnostic hardly differs: robbery003 0.875 / 0.739 (same as official);
shoplifting001 recovery 0.909, precision 0.824. **Roles were right here; the merge was not.**
Recovery and precision count facts, so duplicated people barely show in them. They show in the
person count (6 and 8 for 2), in object faithfulness, and in the planted conflicts: the corrupted
shirt colour sits on a different entity than the true one, so nothing conflicts.

#### Diagnosis (no code changes)
1. **The LLM merge answer is rejected, in every scene checked, dev included.** `_llm_groups` accepts
   only a clean partition. Qwen leaves out a key (robbery003, robbery044: `witness_B:e1`), invents keys
   (shoplifting001: `witness_A:e4`–`e7`), or groups a person with an object. The pipeline then always
   runs the rule-based merge. So on dev, the rules did all of the merging too.
2. **The rule-based merge can't separate two men who do different things.** Each witness describes two
   men, so the lone-man rule is off. The details rule counts an *attribute* as agreeing once, however
   many actions match. So the robber–robber pairs, which share 4–5 actions, score 1, the same as
   robber–clerk pairs that share one generic action ("turned around" in "turned around to leave";
   "stopped in front of the shelf", "walked away", "talked"). With these ties no pair is a unique best,
   and the clique exception doesn't apply. Result: every witness mention stays its own person.
3. **CCTV:**
   - robbery003: the clerk's "dark jacket"/"cap" has nothing to agree with on the witness side ("black
     clothes" wasn't extracted from the statements), so no link.
   - shoplifting001: the camera splits into an orange-shirt man (linked to witness A's man) and a
     "dark jacket" person; the white shirt is never seen on camera.

#### Suggested fixes (not made; for the next development round, to be checked on the dev scenes)
- **Repair the LLM merge answer instead of rejecting it:** add missing keys as their own groups, drop
  invented keys, split mixed-type groups. Then run the role-free checks on the result.
- **Count agreeing values for multi-valued attributes** (actions, positions), weighting generic actions
  ("turned around", "walked away") low. Then 4 shared actions outrank 1.
- Extract clothing given as a noun phrase ("the man in black" → upper_clothing "black clothes"), a
  long-standing miss.
