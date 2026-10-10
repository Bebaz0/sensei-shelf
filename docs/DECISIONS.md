# Decisions (append-only)

Format: `YYYY-MM-DD HH:MM | who | decision | why | score delta`

2026-10-10 | team | Challenge = Sensei (fruit and vegetable shelf monitoring) | ambitious, real data, objective score, demo-able UI. Ranking was Sensei > Albatroz > QuantumTrace > NOS | n/a
2026-10-10 | team | Stack = Python pipeline (file contracts in `out/`) + React dashboard in `dashboard/` | scaffold already in repo; no separate backend server, dashboard reads `out/` files | n/a
2026-10-10 | team | Vision models = Gemini and Claude ensemble on every area | cheap at ~2,000 areas, stacker learns how much to trust each | n/a
2026-10-10 | claude | Compliance threshold = 0.5 on calibrated P(wrong) | scoring is symmetric (+1/-2 both ways) | n/a
2026-10-10 | claude | Stock decision = expected-value argmax over calibrated probabilities | scoring is asymmetric (low -2, high -3) | n/a
2026-10-10 | claude | VLM prompt is closed-set and comparative (target vs reference, candidates = products on that camera) | open-set naming hallucinates on apple varieties and loose vs bagged items | n/a
2026-10-10 | claude | Stock reference = labelled-full frame; earliest frame only for compliance | first frame is ~05:12 UTC, may be pre-opening | n/a
2026-10-10 | claude | Skipped: trained detector, YOLO, SAM, database, Docker, auth, backend server | 181 unlabelled frames, 24h budget; revisit only if error analysis shows a gap | n/a
2026-10-10 | claude | `EMPTY_IS_COMPLIANT` defaults to true until mentors answer | prior majority; biggest single point swing | pending
2026-10-10 | claude | Compliance = Platt scaling of the VLM `matches_expected` answer, no full stacker | only ~6.5% wrong areas, ~130 positives in 2,000 labels, a stacker would overfit | n/a
2026-10-10 | claude | Stock stacker only for stock level; stock is where the points are (perfect compliance adds ~60 of 578) | prioritisation | n/a
2026-10-10 | claude | Validation frames are probably late evening (example 6215/1751052241.jpg = 20:24 Lisbon, last train frame 18:35); temporal and kNN features must work without a next frame | avoid a train/validation distribution shift surprise | pending mentor answer
2026-10-10 | claude | Label a stratified ~600 areas first (incl. last frames of the day), 5-8 s per area, shared rubric in docs/LABELING.md and a 30-area calibration set | 2,000 at 3 s was optimistic | n/a
2026-10-10 | claude | Score and label merge moved from C to B; C keeps decide, contracts, docs and mentors | C was overloaded and on the critical path | n/a
2026-10-10 | claude | Dashboard reads `dashboard/public/data/` filled by `scripts/sync_dashboard.sh` | Vite serves only public/ and out/ is gitignored | n/a
2026-10-10 | claude | Pre-approved deps: scikit-learn, google-genai, anthropic, python-dotenv, pytest (added to requirements.txt) | avoid per-dependency friction | n/a
2026-10-10 | claude | Models: Claude = claude-sonnet-5-5; Gemini id to be pinned by V on the first call (check AI Studio) and recorded here. Soft API budget: well under 20 USD with caching, confirm after the first full run | cost control | n/a
2026-10-10 | claude | v1 stock decision = hand-tuned weighted average of VLM, embedding probe and kNN probabilities, then EV rule; learned stacker and per-image numbered-quad VLM call are experiments adopted only with a clear LOFO gain | transparent, robust to the evening shift, quick to debug | n/a
2026-10-10 | claude | Validation runs append to the train crops (`crop --append`) and stages process only `out/run_frames.json`; `decide` takes frames and areas from `crops_index.json` | kNN, temporal and reference signals need train crops at validation time; every area must still be output | n/a
2026-10-10 | claude | `run_all.sh` cleans generated JSON per run, skips scoring on validation runs or empty labels, `NO_API=1` skips the VLM | avoid stale files and a non-zero exit after a good submission | n/a
2026-10-10 | claude | label_tool.py moves from A to D; dashboard demo = JPG thumbnails of a few frames per camera (a few MB), never full-size PNGs | A was the bottleneck; 4,000 PNG crops would be ~400 MB | n/a
2026-10-10 | claude | Embedding probe (frozen DINOv2/CLIP + logistic regression on human labels) promoted from optional to a B task; fine-tuning on VLM pseudo-labels rejected | 181 frames and noisy pseudo-labels, time sink | n/a
2026-10-10 | claude | `decide` takes areas from planogram.json and frames from run_frames.json (fallback: list the --images folder), not from crops_index.json | a crashed crop stage must not cost -5 per area | n/a
2026-10-10 | claude | Anything fitted on labels (probe, Platt, v1 weights, stacker) is scored with frame-grouped CV | otherwise LOFO is inflated and picks the wrong method | n/a
