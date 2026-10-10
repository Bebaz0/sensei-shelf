# Workstream checklists

Read `CLAUDE.md` and `docs/PLAN.md` first. Edit only the files your workstream owns.

## A: crop + common
Owns `pipeline/crop/`, `pipeline/common/`, `scripts/`.
- [x] `planogram.py`, `images.py` implemented; recursive discovery skipping `__MACOSX` and `._*`; camera from folder, else nearest thumbnail (H2)
- [x] `crop.py`: PNG crops only; warp to 256x256 with validity mask for negative coordinates; context crop padded 25% with outline and border fill when off-image; `--append` merges into `out/crops_index.json` without deleting train entries; write `out/run_frames.json` (H2-3)
- [x] QA outputs from `crop.py`: an overlay of all labelled quads per frame, and a per-camera contact sheet of each area over time (H3)
- [x] `labels/calibration.txt` (A owns this file): the 30 shared calibration areas, spread over cameras and times of day (H3)
- [ ] `scripts/run_all.sh` (`IMAGES`, `NO_API`, stale-file cleanup, conditional scoring) and `scripts/sync_dashboard.sh` (small JPG demo) are written; keep them working as the stages land
- [ ] Fresh-clone rehearsal on a copied folder with flattened or renamed files (H12-18)

## B: pixel + neighbours + eval
Owns `pipeline/pixel/`, `pipeline/eval/score.py`, `pipeline/eval/merge_labels.py`, `tests/`.
- [ ] `score.py`: brief table gives exactly 320; missing = -5; scores only labelled frames and areas (ignores unlabelled predictions); `tests/test_score.py` passes (H2, critical for C's LOFO)
- [ ] `merge_labels.py`: merge `labels/<name>.json` into `labels/dev.json`, `--agreement` report between labellers (H3)
- [ ] `pixel` stages read `out/run_frames.json` and support `--all` (H4)
- [ ] `features.py`: fill ratio vs area background estimate, HSV histogram distance to reference, edge density, visible fraction (H4)
- [ ] `neighbours.py`: kNN over labelled crops of the same area (exclude same frame) and temporal prior from the previous frame and, if it exists, the next frame (H6)
- [ ] Evaluate separately on the last 2 frames per camera to mimic the evening shift (H8)
- [ ] Embedding probe: frozen DINOv2 or CLIP embeddings of the crops plus a plain logistic regression on our human labels for stock level; compare with kNN and keep the better one (H8)
- [ ] Temporal differencing feature: change of the crop vs the previous frame of the same area (soft feature, no hard monotone rule) (H8)
- [ ] Leakage check: no feature may use the frame being predicted

## V: vlm
Owns `pipeline/vlm/`, `data/products_en.json`.
- [ ] `data/products_en.json`: 94 products, PT -> EN name, loose vs package, short visual hint; reviewed by a person (H2)
- [x] Gemini client: structured JSON output, temperature 0, sha256 disk cache in `out/cache/vlm/`, thread pool (16), retry with backoff; pin the model id in `docs/DECISIONS.md` (H3)
- [x] `--no-api` flag on `ask.py`: skip the call and write `null` per provider (H3)
- [ ] Prompt v1: closed-set comparative (context crop + warp crop, reference crop from a frame of that area labelled compliant, never the earliest frame; candidates = products on that camera); test on ~30 labelled areas (H5)
- [x] `ask.py` reads `out/run_frames.json` and supports `--all` (H3)
- [ ] Full cached run over all frames (H6); Claude client with the same interface (H7)
- [ ] Iterate prompts from error lists; bump `PROMPT_VERSION` on every change
- [ ] Head-to-head at H6: one call per image with numbered quads vs per-area calls on the proxy labels; ship whichever has the better LOFO (per-area is the default)
- [ ] Track API spend against the soft budget in `docs/DECISIONS.md`; report rate limits

## C: decide + contracts
Owns `pipeline/decide/`, `CONTRACTS.md`, `docs/`, mentor questions, `submit-safe` tag.
- [ ] Ask the mentors the open questions; record answers in `docs/DECISIONS.md` (H0)
- [x] `decide.py` prior-only baseline writing a valid `submission.json` for all frames with the fallback (H3, v0): areas from `planogram.json`, frames from `out/run_frames.json` or the `--images` folder; tag `submit-safe`
- [x] EV-argmax stock rule, 0.5 compliance threshold, `STOCK_BIAS` and `EMPTY_IS_COMPLIANT` (already in `config.py`), fitted parameters in the committed `pipeline/decide/params.json` (H3)
- [x] Compliance: Platt scaling of the VLM `matches_expected` answer (H6)
- [x] v1 stock probabilities = hand-tuned weighted average of VLM, embedding probe and kNN, then the EV rule (H6)
- [x] Experiment: learned stacker (`fuse.py`, multinomial logistic regression, leave-one-frame-out CV); adopt only with a clear LOFO gain over v1; `STOCK_BIAS` sweep; ablation table for the pitch (H8). Code done; run `python -m pipeline.decide.lofo --ablation` and `python -m pipeline.decide.fuse` once labels and `vlm.json` exist
- [ ] With real labels: `python -m pipeline.decide.lofo --fit` (writes the committed `pipeline/decide/params.json`), log the LOFO row in `docs/DECISIONS.md`
- [ ] Update fallback from the real label distribution (H5); keep `CONTRACTS.md` and `docs/DECISIONS.md` current
- [ ] Final `submit-safe` tag at model freeze (H18)

## D: dashboard + pitch
Owns `dashboard/`, `pipeline/eval/label_tool.py`.
- [ ] `label_tool.py` live by H3 (D is free until real data arrives): keys 1/2/3 = empty/half/full, X = wrong product (picker of the camera's products), U = unsure, arrows = next area; draw the area quad on the full frame (no dependency on crops) plus a zoom, show expected product (PT + EN) and the same area's earliest-frame view; writes `labels/<name>.json` following `docs/LABELING.md` (H3)
- [ ] Build against mock `results.json` in `mocks/` following `CONTRACTS.md` (H2); data is fetched from `dashboard/public/data/` (see `scripts/sync_dashboard.sh`)
- [ ] Shift-start task list sorted empty > wrong product > half: crop thumbnail, location (camera/gondola), action verb ("Restock bananas", "Move melons back"), done checkbox, "verify" chip on low-confidence items (H6 on real data)
- [ ] Camera view: full frame with coloured quad overlays (green ok, amber half, red empty, purple wrong product); click an area for crop, reason, source votes
- [ ] Timeline scrubber per camera with per-area stock sparkline; stretch: "runs empty at ~HH:MM" estimate from depletion slope
- [ ] Per-product totals (sum stock across areas of the same product)
- [ ] Mobile-first, accessible (UI/UX breaks ties); final audit pass before the pitch
- [ ] Run `scripts/sync_dashboard.sh` and commit the small demo (a few MB, JPG thumbnails) to `dashboard/public/data/` at H18 so the demo works offline; backup screen recording at H20
- [ ] At the end, remind the user to launch their demo/pitch skill
