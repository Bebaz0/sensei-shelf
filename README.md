# TrustyShelf

JunctionX Lisbon 2026, Sensei challenge "Fresh eyes on fresh produce".

Ceiling cameras already watch the fruit and vegetable gondolas. For every image, TrustyShelf checks each planogram area and answers two questions: **is the right product there**, and **is it empty, half or full**. Staff get a prioritised list of what to restock or fix instead of walking the aisles to find out.

- [Quick start](#quick-start)
- [Run it on a new dataset](#run-it-on-a-new-dataset)
- [See the results](#see-the-results)
- [How it works](#how-it-works)
- [Numbers and honest limits](#numbers-and-honest-limits)
- [Team](#team)

## Quick start

Requires Python 3.10 or newer and, for the dashboard, Node 20 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put your key after GEMINI_API_KEY=
```

The data is not in git. Unzip the organisers' dataset so that you have:

```
data/
├── planogram.json                          # 14 cameras, 152 areas
└── Frutas e Legumes (train & test)/
    └── <camera_id>/<unix_ts>.jpg
```

Then run everything on that data and print the score against our labels:

```bash
scripts/run_all.sh
```

No Gemini key, or no network? `NO_API=1 scripts/run_all.sh` runs the same pipeline without the vision-language model. It is less accurate on the hard areas but always produces a complete answer.

## Run it on a new dataset

This is the path for the hidden validation images. It takes one command.

1. Put the new images in any folder. Either layout works:
   - one sub-folder per camera, `<camera_id>/<anything>.jpg`, or
   - one flat folder of images. The camera is then recognised from the picture itself, so renamed files are fine.
2. Run:

   ```bash
   source .venv/bin/activate
   IMAGES="/path/to/new/images" scripts/run_all.sh
   ```

3. Collect the answer: **`out/submission.json`**. It contains only the new images, with every planogram area of each one:

   ```json
   {
     "6215": {
       "1751052241.jpg": {
         "6215_1": { "planogram_compliant": true, "stock_level": "full", "stock_value": 1.0 }
       }
     }
   }
   ```

Good to know:

- **Keep the original training images in `data/`.** The new images are compared with them, area by area. On a fresh clone the script builds what it needs from them first.
- **Time.** Allow a few minutes in total. The Gemini step makes one call per image that has uncertain areas, at most 15 calls a minute on a free key, so 26 images need about two minutes of API time.
- **A different planogram.** Replace `data/planogram.json` with a file in the same format before running. Areas of cameras we have never seen fall back to the vision-language model and a safe default.
- **Nothing is ever skipped.** Every area of every image is written, even when an image is unreadable or the API fails.
- **File names.** Unix-timestamp names (as in the provided data) let the pipeline use the time of day. Other names still work.

## See the results

**Dashboard** (two terminals, from the repo root):

```bash
uvicorn api.main:app --reload            # terminal 1, http://localhost:8000
cd dashboard && npm install && npm run dev   # terminal 2, http://localhost:5173
```

It always shows the latest run: store overview, per-camera view with colour-coded areas, the restock task list, and each shelf through the day. Re-run the pipeline and refresh the page.

**Score against labels** (only for images we have labels for):

```bash
python -m pipeline.eval.score --pred out/submission.json --gt labels/dev.json
```

**Pitch slides:** open `slides/index.html` in a browser. Arrow keys move, `N` shows speaker notes, `F` is fullscreen.

## How it works

```
images ──> crop ──> pixel features ──> decide ──> uncertain areas ──> Gemini ──> decide ──> submission.json
            │        look-alikes         │                                          │
            │        stock model         └── expected-score rule                    └── results.json (dashboard)
            └── one flattened crop per planogram area
```

1. **Crop.** Each planogram quad is warped to a flat 256x256 crop, plus a wider context crop.
2. **Look at the pixels.** Fill, colour and texture features. Because the cameras never move, each crop is also compared with labelled crops of the *same area* at other times of day, and a small gradient-boosting model reads stock level from lighting-normalised features.
3. **Decide once.** The sources are averaged into probabilities for empty, half and full.
4. **Ask only when unsure** (off by default, `USE_VLM=1`: it gained +0.3% for ~11 min of API calls). Areas where the probabilities are not decisive are sent to Gemini (3.1 Flash-Lite), which sees the crop and its surroundings and chooses from that camera's product list.
5. **Decide again**, by expected score. The challenge punishes overstating stock (-3) more than understating it (-2), so the system answers "full" only when it is confident enough to be worth the risk. "Wrong product" is answered only when it is more likely than not.

Every API answer is cached on disk, so re-running costs nothing.

### Run one stage at a time

Each stage reads the previous stage's file from `out/` and can be run alone (`--help` lists the flags):

```bash
python -m pipeline.crop.crop
python -m pipeline.pixel.features
python -m pipeline.decide.decide --queue out/vlm_queue.txt
python -m pipeline.vlm.ask --areas out/vlm_queue.txt
python -m pipeline.decide.decide
python -m pipeline.eval.score --pred out/submission.json --gt labels/dev.json
./scripts/run_all.sh            # all of the above, in order (VLM off)
IMAGES=<validation dir> ./scripts/run_all.sh   # submission: out/submission.json, validation frames only
USE_VLM=1 ./scripts/run_all.sh  # also ask Gemini about the low-confidence areas (needs GEMINI_API_KEY)
WORKERS=1 ./scripts/run_all.sh  # serial, if the parallel stages ever misbehave (same output)
```

### Repository map

| Path | What it is |
|---|---|
| `pipeline/crop/` | planogram quads to crops, camera recognition for renamed files |
| `pipeline/pixel/` | pixel features, same-area look-alikes, stock model |
| `pipeline/vlm/` | Gemini client, prompt, disk cache |
| `pipeline/decide/` | fusion, expected-score rule, forward-in-time evaluation |
| `pipeline/eval/` | scorer, labelling tool, label merge |
| `api/`, `dashboard/` | read-only API over the results and the staff dashboard |
| `labels/` | our hand-made labels (the challenge provides none) |
| `docs/` | plan, labelling guide and a log of every decision with its measured effect |
| `scripts/run_all.sh` | the whole pipeline in one command |

### Inspection API

| Endpoint | What it returns |
|---|---|
| `GET /api/cameras` | planogram cameras and their products |
| `GET /api/score` | score of the last run against the labels |
| `GET /api/stats` | run totals: stock, compliance, low-confidence areas |
| `GET /api/area/{cam}/{filename}/{area_id}` | everything known about one area |
| `GET /health` | `{status: "ok"}` |

## Numbers and honest limits

- **The data:** 14 cameras, 152 areas, 94 products, 181 images from one day (06:12 to 20:16).
- **Labels:** the challenge provides none, so we labelled 1,947 areas ourselves with a written guide (`docs/LABELING.md`).
- **Score on the labelled day:** 3,425 of 3,894 points (88.0%) with `scripts/run_all.sh` (VLM off).
- **A fairer estimate:** predicting each image from *earlier* images only, as on unseen evening images: 3,099 of 3,894 (79.6%), and 488 of 608 (80.3%) on the last two images of each camera. We expect the hidden set (578 points) to land near this, about 460.
- **Wrong products:** predicting forward in time, 71 of 100 labelled wrong-product areas are flagged, and every one of them names the product our labels give (taken from the same area's earlier labelled images, no VLM).
- **The VLM is optional:** asking Gemini about the uncertain areas added +0.3% (inside noise) for about 11 minutes of API calls, so it is off by default. A full run takes about a minute.
- **Where the points go:** most remaining errors are a half-empty area called full, and evening images, when shelves run down, are the hardest.
- **Label noise is real:** two of our labellers disagreed on about 12% of stock levels, mostly half against full. No model can score above what the labels agree on.
- **The shelf does not always follow the planogram.** Some areas hold a product that is not in the planogram's list at all (for example bagged lemons in a kiwi area). We can flag those as wrong, but not name them.
- **We never looked at the hidden images.** The pipeline is fully automatic from images to `submission.json`.

Every experiment, including the ones that failed, is logged with its measured effect in `docs/DECISIONS.md`.

## Team

| Name |
|---|
| _Afonso Lima_ |
| _André Santos_ |
| _Diogo Santos_ |
| _Pedro Tomás Teixeira_ |
| _Rafael Pinho e Silva_ |

Built in 24 hours at JunctionX Lisbon 2026.
