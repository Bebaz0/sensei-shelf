# Plan: Fresh Eyes (Sensei challenge)

Source brief: `data/Fruits & Vegetables - Shelf Monitoring Solution.pdf`. Scoring and rules: `CLAUDE.md`. File shapes: `CONTRACTS.md` (proposed schemas in `docs/contracts-proposal.md`). Per-workstream checklists: `docs/TASKS.md`.

## Key insights
- Validation frames come from the same day as train (the brief's example `6215/1751052241.jpg` is after the last train frame of that camera; cameras 6215 and 6443 have 12 frames instead of 13). Same-area neighbours in time are legitimate signals, and the camera is fixed, so kNN over labelled crops of the same area is strong.
- Compliance scoring is symmetric (+1/-2): answer "not compliant" only when calibrated P(wrong) > 0.5. Stock scoring is asymmetric: use the expected-value argmax rule from `CLAUDE.md`. It needs calibrated probabilities, hence a stacker trained on our labels.
- Missing area costs -5, so every area of every image is always output, with per-area fallback.
- Fine-grained look-alikes (apple varieties, loose vs bagged potatoes): ask the VLM a closed-set comparative question (target vs reference crop, candidates = products on that camera). Never an open "what is this?".
- "Earliest frame = full" is risky (first frame is ~05:12 UTC, before opening). Use a labelled-full frame as the stock reference; use the earliest frame only as the compliance reference.
- Product names are Portuguese. Build `data/products_en.json` once (reviewed by a person) and put the English gloss in prompts. Names ending in KG are loose items (unit = one fruit); names with a weight (500 G, EMB. 3KG) are packages (unit = one package).

## Pipeline, per area (maps onto the existing `pipeline/` stages)
1. `common` loads planogram and images.
2. `crop` makes two crops per area: a warped quad (256x256) with a validity mask (5 quads extend off-image), and an axis-aligned context crop padded 25% with the quad outline drawn.
3. Evidence from four sources:
   a. `pixel`: fill ratio vs the area's empty/background estimate, HSV histogram distance to reference, edge density, visible fraction;
   b. `pixel/neighbours.py`: kNN over labelled crops of the same area (excluding the same frame);
   c. `pixel/neighbours.py`: temporal prior from previous/next frames of the same area;
   d. `vlm`: Gemini on every area, Claude as second model on every area (cheap at ~2,000 areas, all cached).
4. `decide` stacker (sklearn multinomial logistic regression, leave-one-frame-out CV) gives calibrated p_stock and p_wrong.
5. `decide` rule: EV argmax for stock plus a `STOCK_BIAS` knob that leans lower when our labels drift from Sensei's; threshold 0.5 for compliance.
6. `decide` writes `out/submission.json` (official format) and `out/results.json` (dashboard).
7. Stretch: one VLM call per image with numbered quads so the model sees neighbour swaps; its output becomes an extra stacker feature.

## Where each piece lives
| Piece | File |
|---|---|
| planogram + image loading, packaged-vs-loose rule | `pipeline/common/{planogram,images}.py` |
| warp, context crop, visible fraction | `pipeline/crop/crop.py` |
| classical features | `pipeline/pixel/features.py` |
| kNN + temporal prior (new) | `pipeline/pixel/neighbours.py` |
| Gemini + Claude clients, cache, thread pool | `pipeline/vlm/ask.py` |
| stacker + decision rule + fallback | `pipeline/decide/decide.py` (+ `fuse.py`) |
| scorer (brief example table must give 320) | `pipeline/eval/score.py` |
| labelling tool | `pipeline/eval/label_tool.py` (keyboard-driven; 1/2/3 = empty/half/full, X = wrong product, U = unsure) |
| staff dashboard | `dashboard/` |
| tuning knobs (thresholds, crop size, STOCK_BIAS, EMPTY_IS_COMPLIANT) | `pipeline/common/config.py` |

## Timeline (H0 = coding start)
- [ ] **H0-1** Everyone spends 15 min looking at images (lighting at 05:12, typical empty/half/full, occlusion). Ask the mentor questions. Note API keys and rate limits.
- [ ] **H2** scorer passes the brief's example (= 320); decide rule tests pass; `products_en.json` done; label tool skeleton.
- [ ] **H3 (M1, baseline v0)** `run_all.sh` writes a valid `submission.json` for all frames using the fallback. Tag `submit-safe`.
- [ ] **H3-5 labelling sprint** ~2,000 areas at ~3 s each, 25-40 min per person, assigned by camera; 2 cameras labelled twice for agreement.
- [ ] **H5** labels merged into `labels/`; class distribution known; fallback updated from it.
- [ ] **H6 (M2, v1)** Gemini + pixel + kNN + temporal -> stacker -> decide. First LOFO score. Dashboard on real `results.json`. Re-tag `submit-safe` if better than v0.
- [ ] **H6-12** prompt iterations (V), Claude ensemble, `STOCK_BIAS` sweep, ablations, robustness, dashboard views. Rule: only merge changes that raise LOFO.
- [ ] **H12 (M3)** freeze `v2`. Sleep in shifts; one person always on call.
- [ ] **H12-18** worst-error fixes; timeline view; run-out estimate (stretch); fresh-clone rehearsal on a copied folder with flattened or renamed files, with and without API.
- [ ] **H18 (M4)** model freeze, final `submit-safe` tag, README runbook of 3 commands.
- [ ] **H18-22** UI polish, pitch deck, demo script, ablation chart, "how we got ground truth without labels" story.
- [ ] **H22-24** full rehearsal (pitch + live pipeline run), buffer.
- [ ] Remind the user to launch their demo/pitch skill (saved in Claude memory; skill name still to be confirmed).

## Mentor questions (H0)
1. Is an empty area `planogram_compliant` true or false? (biggest point swing; one flag in `config.py`, default true until answered)
2. Validation folder layout (`<cam>/<ts>.jpg` or flat)? Can we read train frames at inference time?
3. Are partly visible areas judged only on their visible part?
4. Where is the line between half and full? One example of each?
5. Submission feedback and limits?

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Our half/full labels differ from Sensei's | `STOCK_BIAS` (lower is cheaper), labeller agreement check, mentor examples |
| API outage or no network at validation | disk cache; a no-API path (pixel + kNN + temporal) scored and tagged on its own; retries; use whichever of Gemini/Claude is up |
| Unexpected validation layout or unknown cameras | recursive discovery in `common/images.py`, camera from thumbnail when flat, per-area fallback |
| VLM hallucinates on look-alikes | closed candidate set, comparative prompt with reference, stacker calibration, 0.5 threshold on calibrated prob |
| Overfitting to ~2k labels from one day | frame-grouped CV, few stacker features, accept changes only with clear LOFO gain |
| Merge conflicts across 5 sessions | directory ownership, one label file per labeller, C owns contracts, small commits + rebase |
| Live demo breaks | dashboard reads committed static runs; record backup video at H20 |

Deliberately skipped: training a detector, YOLO, SAM, a database, Docker, auth, a separate backend server (the dashboard reads `out/` files). Add one only if error analysis shows a gap the current signals cannot close.

## Verification
- Scorer: brief example table (270/19/230/35/24) gives exactly 320; missing = -5; high = -3; low = -2.
- Decision rule: pairwise EV thresholds; a uniform prior gives `half`; p_wrong 0.49 compliant, 0.51 not.
- Crop: a quad with negative corners gives a valid mask with the right visible fraction.
- Contract check: `submission.json` has every area of every image and only the allowed keys.
- LOFO score on our labels is the team leaderboard; log every run in `docs/DECISIONS.md`.
