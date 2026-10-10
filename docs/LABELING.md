# Labelling rubric

Labels teach the stacker and measure our score, so all five labellers must apply the same rules. Update this file (and tell everyone) when a rule changes. Mentor answers override anything here.

## Stock level (per area, judge only the visible part of the quad)
| Label | Meaning |
|---|---|
| `empty` | no product, or only a few stray items; mostly bare shelf or tray |
| `half` | clearly stocked but with visible gaps: roughly 30% to 70% of the area covered or piled |
| `full` | covered or piled across nearly the whole area, no meaningful gap |

Rules of thumb:
- Judge the area, not the camera image. A big pile in a neighbouring area does not count.
- Loose produce: look at coverage and pile height. Packaged items (names with a weight): count rows of packs against the area size.
- Occluded by a person: pick the level for the visible part. If you cannot tell, press `U` (unsure) and move on.
- When torn between two levels, pick the LOWER one (a high guess costs more in the score, so lower labels are the safe habit).
- Partly off-image quad: judge the visible part only (pending mentor answer).

## Compliance
- `planogram_compliant = true` if the product shown is the expected product in the planogram.
- If a different product (or a mix with a different product) fills most of the area, mark wrong product (`X`) and pick the observed product from the camera's list.
- An empty area: follow `EMPTY_IS_COMPLIANT` in `pipeline/common/config.py` (default true until the mentors answer).
- Same variety family but different variety (e.g. Gala vs Fuji): if you cannot tell from the crop, mark compliant and press `U`.

## Process
1. Everyone labels the same 30 calibration areas first (list in `labels/calibration.txt`, camera + frame + area, spread over all cameras and times of day).
2. Compare with `python -m pipeline.eval.merge_labels --agreement`. Discuss every disagreement and edit this rubric.
3. Then label your assigned cameras. Each person writes only `labels/<name>.json`; `merge_labels` builds `labels/dev.json`.
4. Start with a stratified ~600 areas: 2 to 3 frames per camera spread over the day, always including the last frames of the day. Extend only if the learning curve says more labels help.
5. Never label the hidden validation images.
