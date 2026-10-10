# sensei-shelf: Fresh Eyes (Sensei challenge, JunctionX Lisbon 2026)

Goal: WIN. 5 engineers, each driving a Claude Code session. Read this file, then `docs/PLAN.md`, `CONTRACTS.md` and your own section of `docs/TASKS.md` before doing anything.

## The challenge in one paragraph
14 fixed ceiling cameras over fruit and vegetable gondolas (1280x720). `data/planogram.json` maps 152 perspective quads ("areas") to 94 products. For every area of an image we predict `planogram_compliant` (bool) and `stock_level` (empty/half/full). The hidden validation set is 26 images, 289 areas, annotated by Sensei. We have no labels, only 181 train/test frames (12-14 per camera, one day). A staff UI breaks ties.

## Scoring (per area, max 2)
- Compliance: correct +1, wrong -2 (symmetric, so answer "not compliant" only if P(wrong) > 0.5).
- Stock: correct +1, predicted LOWER than truth -2, predicted HIGHER than truth -3.
- Area missing from output: -2 and -3. ALWAYS output every area.
- Stock decision = argmax expected value over calibrated p_e, p_h, p_f:
  EV(empty) = p_e - 2(p_h+p_f); EV(half) = p_h - 3p_e - 2p_f; EV(full) = p_f - 3(p_e+p_h).

## Layout (pipeline stages talk only through files in `out/`, see CONTRACTS.md)
`crop` -> `pixel` + `vlm` -> `decide` -> `eval` / `dashboard`. Paths live in `pipeline/common/config.py`. Setup and run commands are in `README.md`.

## Workstreams and ownership (edit ONLY your files)
| WS | Owns |
|---|---|
| A crop + common | `pipeline/crop/`, `pipeline/common/`, `pipeline/eval/label_tool.py`, `scripts/` |
| B pixel + neighbours | `pipeline/pixel/` (features, kNN over labelled crops, temporal prior) |
| V vlm | `pipeline/vlm/`, `data/products_en.json` |
| C decide + score | `pipeline/decide/` (stacker, decision rule), `pipeline/eval/score.py`, `labels/`, `CONTRACTS.md`, `docs/` |
| D dashboard + pitch | `dashboard/` |

Cross-cutting change in a file you do not own? Add a one-line request to `docs/DECISIONS.md` or ask the owner. `CONTRACTS.md` changes go through C (PR touching it + tell the consumer).

## Hard rules
1. NEVER hand-label or inspect-to-tune on the hidden validation images. The pipeline must be fully automatic.
2. ALWAYS output every planogram area for every image; per-area try/except falls back to a safe default.
3. Cache every API call on disk (sha256 of provider, model, prompt version, image bytes, text) under `out/cache/`.
4. Small commits, `git pull --rebase` before push, never push a broken `scripts/run_all.sh`.
5. Only merge model changes that raise the leave-one-frame-out (LOFO) score; log each run in `docs/DECISIONS.md`.
6. kNN and temporal features must exclude the frame being predicted (no leakage).
7. No new dependency without a `docs/DECISIONS.md` entry. Plain functions and dicts, no speculative abstractions.
8. Secrets live in `.env` (never committed).
9. Git: one-line conventional commit messages (`feat: ...`, `fix: ...`), no body, no AI attribution.
10. No em dashes in docs or code comments.

## Open mentor questions (answers go in `docs/DECISIONS.md`)
Is an empty area compliant? Validation folder layout? Partly visible areas? Half vs full boundary? Submission feedback and limits?

## Safe submission
Tag `submit-safe` marks the last known-good commit. To produce a submission: check it out, then `python -m pipeline.crop.crop --images <validation dir>` followed by the rest of `scripts/run_all.sh`; the file to hand in is `out/submission.json`.
