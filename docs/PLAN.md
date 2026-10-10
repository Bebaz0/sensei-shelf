# Plan: Fresh Eyes (Sensei challenge)

Source brief: `data/Fruits & Vegetables - Shelf Monitoring Solution.pdf`. Scoring and rules: `CLAUDE.md`. File shapes: `CONTRACTS.md`. Per-workstream checklists: `docs/TASKS.md`. Labelling rubric: `docs/LABELING.md`.

## Key insights
- **Validation frames are probably a little later than train.** The brief's example `6215/1751052241.jpg` is 20:24 Lisbon time; the last train frame of that camera is 18:35 and the latest train frame overall is 20:16 (6385). So a validation frame may have earlier neighbours only (nothing later). The evening shift is mild: most cameras already have frames up to 19:10-19:44. `temporal` and kNN code MUST work with a missing next frame, and the stock prior must not assume daytime. (Guess, unverified: cameras 6215 and 6443 have 12 frames instead of 13 because they gave frames to validation. Ask the mentors.)
- Train frames run from 06:12 to 20:16 Lisbon (05:12 to 19:16 UTC). Every camera starts at 06:12-06:14; most end between 19:10 and 19:44, 6385 at 20:16, 6215 at 18:35 and 6443 at 18:07. Opening time is unknown, but the first two frames (about 06:13 and 07:20) are pre-opening: shelves empty, restocking in progress, a night blind over 6383 and 6386 at 06:12.
- **The planogram and the shelf disagree, stably.** On 6215, areas 7 and 9 expect oranges but hold bananas from 08:33 to the last frame; at 06:14 and 07:23 areas 3 and 5 hold oranges where bananas are expected. Compliance varies by area and stays stable over time, so labelled frames of the same area predict it well (kNN `p_wrong`, temporal prior). Check each camera's contact sheet in `out/qa/sheets/` before trusting any frame as a reference.
- Compliance scoring is symmetric (+1/-2): answer "not compliant" only when calibrated P(wrong) > 0.5. Stock scoring is asymmetric: use the expected-value argmax rule from `CLAUDE.md`.
- **Compliance has few positives.** The brief's example has ~6.5% wrong areas, so ~130 wrong-product labels in 2,000 (our own share is unknown until we label; stable mismatches like 6215_7 may raise it). Do not train a full stacker for compliance. Use the VLM `matches_expected` answer plus simple Platt scaling (one logistic fit) and the 0.5 threshold. The stacker is for stock only.
- **Where the points are.** A constant "always compliant" answer already gets most compliance points; perfect compliance adds only ~60 points on 289 areas. Stock level is the main lever, so labelling, crops, VLM prompts and the asymmetric decision matter most.
- Missing area costs -5, so every area of every image is always output, with per-area fallback.
- Fine-grained look-alikes (apple varieties, loose vs bagged potatoes): ask the VLM a closed-set comparative question (target vs reference crop, candidates = products on that camera). Never an open "what is this?".
- "Earliest frame = full" is risky (pre-opening). Use a labelled-full frame as the stock reference. The earliest frame is not a compliance reference either (pre-opening shelves hold other products): use a frame of that area labelled compliant.
- Product names are Portuguese. Build `data/products_en.json` once (reviewed by a person). Names ending in KG are loose items (unit = one fruit); names with a weight (500 G, EMB. 3KG) are packages (unit = one package).

## Pipeline, per area (maps onto the existing `pipeline/` stages)
1. `common` loads planogram and images.
2. `crop` makes two crops per area, saved as PNG (never JPG, to keep pixels stable for features and the VLM cache): a warped quad (256x256) with a validity mask (5 quads extend off-image), and an axis-aligned context crop padded 25% with the quad outline drawn (pad with border fill when the quad is off-image). It also writes a QA overlay per frame (all quads labelled) and a per-camera contact sheet of each area over time.
3. Evidence from four sources:
   a. `pixel`: fill ratio vs the area's empty/background estimate, HSV histogram distance to reference, edge density, visible fraction;
   b. `pixel/neighbours.py`: kNN, and a logistic-regression probe on frozen DINOv2/CLIP embeddings, over labelled crops of the same area (excluding the same frame); keep whichever has the better LOFO;
   c. `pixel/neighbours.py`: temporal prior from the previous and, when it exists, the next frame of the same area, plus a soft change-vs-previous-frame feature;
   d. `vlm`: Gemini on every area, Claude as second model on every area (cheap at ~2,000 areas, all cached).
4. `decide` v1: hand-tuned weighted average of the VLM, embedding-probe and kNN stock probabilities (transparent and robust to the evening shift). Compliance uses Platt-scaled `matches_expected`. A learned stacker (sklearn logistic regression, leave-one-frame-out CV) is an experiment, adopted only with a clear LOFO gain.
5. `decide` rule: EV argmax for stock plus a `STOCK_BIAS` knob that leans lower when our labels drift from Sensei's; threshold 0.5 for compliance.
6. `decide` writes `out/submission.json` (official format) and `out/results.json` (dashboard).
7. Experiment at H6: one VLM call per image with numbered quads (the model sees neighbour swaps) vs per-area calls; ship whichever has the better LOFO. Per-area is the default because stock reading needs the zoomed crop.

## Validation run flow
1. `IMAGES=<val dir> scripts/run_all.sh` -> `crop --images <val dir> --append`: validation crops are added to the train crops; `out/run_frames.json` lists only the validation frames.
2. `pixel`, `vlm`, `decide` process only the run frames. kNN, probe, temporal and reference crops use the train crops plus `labels/dev.json` (committed), so they exist at validation time.
3. `decide` takes areas from `planogram.json` and frames from `run_frames.json` (fallback: list the `--images` folder), so every area of every validation frame is output even if a stage failed.
4. `out/submission.json` contains the validation frames only. Scoring is skipped (no labels).
5. LOFO / train runs: `scripts/run_all.sh` with no `IMAGES`; `run_frames.json` = train frames.

## Where each piece lives
| Piece | File |
|---|---|
| planogram + image loading, packaged-vs-loose rule | `pipeline/common/{planogram,images}.py` |
| warp, context crop, visible fraction, QA overlays | `pipeline/crop/crop.py` |
| classical features | `pipeline/pixel/features.py` |
| kNN + temporal prior (new) | `pipeline/pixel/neighbours.py` |
| Gemini + Claude clients, cache, thread pool, `--no-api` | `pipeline/vlm/ask.py` |
| decision rule (v1 weighted average + EV), fallback, optional stacker experiment | `pipeline/decide/decide.py` (+ `fuse.py`) |
| scorer (brief example table must give 320) | `pipeline/eval/score.py` |
| label merge + labeller agreement | `pipeline/eval/merge_labels.py` (new): `labels/<name>.json` -> `labels/dev.json` |
| labelling tool (owned by D) | `pipeline/eval/label_tool.py` (keyboard-driven; rubric in `docs/LABELING.md`) |
| dashboard data sync | `scripts/sync_dashboard.sh [N]` writes a small JPG demo run (N frames per camera) into `dashboard/public/data/` |
| run control | `scripts/run_all.sh` (`IMAGES`, `NO_API`, cleans stale outputs, scores only train runs with labels) |
| tuning knobs (thresholds, crop size, STOCK_BIAS, EMPTY_IS_COMPLIANT) | `pipeline/common/config.py` |
| tests | `tests/` (pytest) |

## Timeline (H0 = coding start)
- [ ] **H0-1** Everyone spends 15 min looking at images (`out/frames_by_time/`; lighting at 06:00, typical empty/half/full, occlusion). Ask the mentor questions. Note API keys and rate limits.
- [ ] **H2** scorer passes the brief's example (= 320); decision rule tests pass; `products_en.json` done; label tool skeleton; `CONTRACTS.md` filled in (done).
- [ ] **H3 (M1, baseline v0)** `run_all.sh` writes a valid `submission.json` for all frames using the fallback (needs only the planogram and the frame list: no crops, no API). Tag `submit-safe`.
- [ ] **H3-4 calibration** everyone labels the same 30 areas (`docs/LABELING.md`), compare, fix the rubric.
- [ ] **H4-6 labelling sprint, stratified first** label ~600 areas: 2-3 frames per camera spread across the day (include the last frames of the day). Extend toward ~2,000 only if the learning curve says more labels help. Budget 5-8 s per area, not 3.
- [ ] **H6 (M2, v1)** Gemini + pixel + kNN + temporal -> stacker -> decide. First LOFO score. Dashboard on real `results.json`. Re-tag `submit-safe` if better than v0.
- [ ] **H6-12** prompt iterations (V), Claude ensemble, `STOCK_BIAS` sweep, ablations, robustness (incl. evening-like frames: evaluate separately on the last 2 frames per camera), dashboard views. Rule: only merge changes that raise LOFO.
- [ ] **H12 (M3)** freeze `v2`. Sleep in shifts; one person always on call.
- [ ] **H12-18** worst-error fixes; timeline view; run-out estimate (stretch); fresh-clone rehearsal on a copied folder with flattened or renamed files, with `NO_API=1` and with API.
- [ ] **H18 (M4)** model freeze, final `submit-safe` tag, README runbook of 3 commands, committed demo run in `dashboard/public/data/`.
- [ ] **H18-22** UI polish, pitch deck, demo script, ablation chart, "how we got ground truth without labels" story.
- [ ] **H22-24** full rehearsal (pitch + live pipeline run), buffer.
- [ ] Remind the user to launch their demo/pitch skill (saved in Claude memory; skill name still to be confirmed).

## Mentor questions (H0)
1. Is an empty area `planogram_compliant` true or false? (biggest point swing; one flag in `config.py`, default true until answered)
2. Validation folder layout (`<cam>/<ts>.jpg` or flat)? When and how are the validation images handed over, and what is the submission deadline?
3. Are partly visible areas judged only on their visible part?
4. Where is the line between half and full? One example of each?
5. Submission feedback and limits?
6. Do validation frames come from the same day, and from late evening?
7. Stable mismatch: 6215_7 and 6215_9 expect oranges but hold bananas from 08:33 to closing. Is that annotated `planogram_compliant: false`?
8. Night blind over 6383 and 6386 at 06:12: what is the label of a covered area, and can validation frames have the blind down?

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Evening distribution shift in validation | handle missing next frame; evaluate on the last frames of the day separately; lean `STOCK_BIAS` lower; ask mentors |
| Our half/full labels differ from Sensei's | written rubric, 30-area calibration set, `STOCK_BIAS` (lower is cheaper), labeller agreement check, mentor examples |
| API outage or no network at validation | disk cache; `NO_API=1` path (pixel + kNN + temporal) scored and tagged on its own; retries; use whichever of Gemini/Claude is up |
| Unexpected validation layout or unknown cameras | recursive discovery in `common/images.py`, camera from thumbnail when flat, per-area fallback |
| VLM hallucinates on look-alikes | closed candidate set, comparative prompt with reference, calibration, 0.5 threshold on calibrated prob |
| Reference crop shows the wrong product (pre-opening frames, stable mismatches) | reference = a frame of that area labelled compliant, never the earliest frame; no reference when none is labelled |
| Overfitting to a few hundred labels from one day | frame-grouped CV, few stacker features, accept changes only with clear LOFO gain |
| Merge conflicts across 5 sessions | directory ownership, one label file per labeller, C owns contracts, small commits + rebase |
| Live demo breaks | dashboard reads committed static runs in `dashboard/public/data/`; record backup video at H20 |

Deliberately skipped: training a detector, YOLO, SAM, a database, Docker, auth, a separate backend server (the dashboard reads synced files). Add one only if error analysis shows a gap the current signals cannot close.

## Verification
- Scorer: brief example table (270/19/230/35/24) gives exactly 320; missing = -5; high = -3; low = -2 (`tests/test_score.py`).
- Decision rule: pairwise EV thresholds; a uniform prior gives `half`; p_wrong 0.49 compliant, 0.51 not.
- Crop: a quad with negative corners gives a valid mask with the right visible fraction.
- Contract check: `submission.json` has every area of every image and only the allowed keys.
- LOFO score on our labels is the team leaderboard; log every run in `docs/DECISIONS.md`.
