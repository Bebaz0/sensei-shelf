# Workstream checklists

Read `CLAUDE.md` and `docs/PLAN.md` first. Edit only the files your workstream owns.

## A: crop + common
Owns `pipeline/crop/`, `pipeline/common/`, `pipeline/eval/label_tool.py`, `scripts/`.
- [ ] `planogram.py`, `images.py` implemented; recursive discovery skipping `__MACOSX` and `._*`; camera from folder, else nearest thumbnail (H2)
- [ ] `crop.py`: warp to 256x256 with validity mask for negative coordinates, plus 25% context crop with outline; write `out/crops_index.json` (H2-3)
- [ ] `label_tool.py` live by H3: keys 1/2/3 = empty/half/full, X = wrong product (picker of the camera's products), U = unsure, arrows = next area; show context crop, expected product (PT + EN), the same area's earliest-frame crop (H3)
- [ ] Support the labelling sprint: assignment by camera, progress counter, two cameras double-labelled (H3-5)
- [ ] `run_all.sh` takes an `IMAGES` override for validation runs; fresh-clone rehearsal (H12-18)

## B: pixel + neighbours
Owns `pipeline/pixel/`.
- [ ] `features.py`: fill ratio vs area background estimate, HSV histogram distance to reference, edge density, visible fraction (H4)
- [ ] `neighbours.py`: kNN over labelled crops of the same area (exclude same frame) and temporal prior from the nearest earlier/later frames (H6)
- [ ] Optional: DINOv2 embeddings for kNN, only if classical kNN is weak at H10
- [ ] Leakage check: no feature may use the frame being predicted

## V: vlm
Owns `pipeline/vlm/`, `data/products_en.json`.
- [ ] `data/products_en.json`: 94 products, PT -> EN name, loose vs package, short visual hint; reviewed by a person (H2)
- [ ] Gemini client: structured JSON output, temperature 0, sha256 disk cache in `out/cache/vlm/`, thread pool (16), retry with backoff (H3)
- [ ] Prompt v1: closed-set comparative (context crop + warp crop, reference crop, candidates = products on that camera); test on ~30 labelled areas (H5)
- [ ] Full cached run over all frames (H6); Claude client with the same interface (H7)
- [ ] Iterate prompts from error lists; bump `PROMPT_VERSION` on every change
- [ ] Stretch: one call per image with numbered quads as an extra feature
- [ ] Watch rate limits and spend; report to the team

## C: decide + score
Owns `pipeline/decide/`, `pipeline/eval/score.py`, `labels/`, `CONTRACTS.md`, `docs/`.
- [ ] Ask the mentors the open questions; record answers in `docs/DECISIONS.md` (H0)
- [ ] `score.py`: brief table gives exactly 320; missing = -5 (H2)
- [ ] `decide.py` prior-only baseline writing a valid `submission.json` for all frames with the fallback (H3, v0); tag `submit-safe`
- [ ] EV-argmax stock rule, 0.5 compliance threshold, `STOCK_BIAS`, `EMPTY_IS_COMPLIANT` in `config.py` (H3)
- [ ] Fill the TODO sections of `CONTRACTS.md` (start from `docs/contracts-proposal.md`) before others build on them (H2)
- [ ] Stacker (`fuse.py`): multinomial logistic regression, leave-one-frame-out CV, calibrated p_stock and p_wrong; `STOCK_BIAS` sweep; ablation table for the pitch (H8)
- [ ] Merge labels, labeller agreement, update fallback from the real distribution (H5)
- [ ] Final `submit-safe` tag at model freeze (H18)

## D: dashboard + pitch
Owns `dashboard/`.
- [ ] Build against mock `results.json` in `mocks/` using the proposal in `docs/contracts-proposal.md` (H2)
- [ ] Shift-start task list sorted empty > wrong product > half: crop thumbnail, location (camera/gondola), action verb ("Restock bananas", "Move melons back"), done checkbox, "verify" chip on low-confidence items (H6 on real data)
- [ ] Camera view: full frame with coloured quad overlays (green ok, amber half, red empty, purple wrong product); click an area for crop, reason, source votes
- [ ] Timeline scrubber per camera with per-area stock sparkline; stretch: "runs empty at ~HH:MM" estimate from depletion slope
- [ ] Per-product totals (sum stock across areas of the same product)
- [ ] Mobile-first, accessible (UI/UX breaks ties); final audit pass before the pitch
- [ ] Pitch assets from H18; commit static demo runs so the demo works offline; backup screen recording at H20
- [ ] At the end, remind the user to launch their demo/pitch skill
