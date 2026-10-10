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
