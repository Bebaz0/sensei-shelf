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

The pipeline runs fully locally: no API key and no network are needed. A vision-language model (Gemini) is an optional extra for the uncertain areas, off by default because it did not pay for itself (see [How it works](#how-it-works)). To try it, `cp .env.example .env`, put your key after `GEMINI_API_KEY=` and run `USE_VLM=1 scripts/run_all.sh`.

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
- **Time.** Allow a few minutes in total, all of it local. On a fresh clone most of that is building the crops and the stock model from the training images; the new images themselves take seconds.
- **Check before you hand it in.** `python -m pipeline.eval.check_submission --images "/path/to/new/images"` confirms every area of every image is present, with only the official keys and values.
- **A different planogram.** Replace `data/planogram.json` with a file in the same format before running. Areas of cameras we have never seen have no labelled history to compare with, so only the pixel stock model reads them and they are answered as compliant.
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
images ──> crop ──> three local sources ──> decide ──> submission.json
            │        kNN   (same area, other frames)    │
            │        temporal (previous frame)          └── results.json (dashboard)
            │        stock model (gradient boosting)
            └── one flattened crop per planogram area
```

The cameras never move, so an area always shows the same shelf from the same angle. Everything below uses that: each crop is judged against labelled crops of the *same area* at other times of day. The default run is fully local.

1. **Crop.** Each planogram quad is warped to a flat 256x256 crop, plus a wider context crop with the quad outlined.
2. **Three local sources** give probabilities for empty, half and full:
   - **kNN.** Labelled crops of the same area vote, weighted by how similar they look (HSV histogram) and how close their time of day is. The same neighbours also vote on "wrong product", with their own tighter kernel so two wrong-looking frames are not outvoted by ten compliant ones.
   - **Temporal.** The previous labelled frame of the area. It is only a tie-breaker, used when the visual sources are not decisive.
   - **Stock model.** A small gradient-boosting classifier on lighting-normalised colour and edge features, taken relative to that area's own median over all frames. It also sees the previous stock and the kNN probabilities, and is never trained on the frame it predicts.
3. **Decide.** The sources are averaged (weights fitted by grid search on held-out frames) and the stock level is the one with the best expected score. The challenge punishes overstating stock (-3) more than understating it (-2), so a small bias leans every answer lower. "Wrong product" is answered only when it is more likely than not. An empty area counts as compliant (mentor answer).
4. **Name the problem.** For a non-compliant area, the product held by the labelled wrong-product neighbours is reported so staff know what to move back. This does not change the submission.
5. **Optional vision-language model** (`USE_VLM=1`, off by default). Areas where the probabilities are not decisive can be sent to Gemini (3.1 Flash-Lite), which sees the crop and chooses from that camera's product list. On our labels it gained +0.3% (inside the noise of a frame bootstrap) for about 11 minutes of API time, so it is not part of the default run. Its answers are cached on disk.

### Run one stage at a time

Each stage reads the previous stage's file from `out/` and can be run alone (`--help` lists the flags):

```bash
python -m pipeline.crop.crop
python -m pipeline.pixel.features
python -m pipeline.decide.decide
python -m pipeline.eval.score --pred out/submission.json --gt labels/dev.json
./scripts/run_all.sh            # all of the above, in order (VLM off)
IMAGES=<validation dir> ./scripts/run_all.sh   # submission: out/submission.json, validation frames only
USE_VLM=1 ./scripts/run_all.sh  # also ask Gemini about the low-confidence areas (needs GEMINI_API_KEY)
WORKERS=1 ./scripts/run_all.sh  # serial, if the parallel stages ever misbehave (same output)
python -m pipeline.decide.lofo --forward       # held-out evaluation, see "Numbers" below
python -m pipeline.decide.lofo --fit           # refit pipeline/decide/params.json
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
- **Held-out score:** every image is predicted with the parameters fitted on all the *other* images (leave-one-frame-out). Result: **3,442 of 3,894 points (88.4%)**. Compliance 1,887 of 1,947 (96.9%), stock 1,555 of 1,947 (79.9%). The stock level is exactly right on 1,836 of 1,947 areas (94.3%); the points lost are in the wrong ones.
- **A fairer estimate for unseen evening images:** predicting each image from *earlier* images only (forward in time): **3,099 of 3,894 points (79.6%)**, and 488 of 608 (80.3%) on the last two images of each camera. We expect the hidden set to land near this, not near the held-out figure above. Re-measure with `python -m pipeline.decide.lofo --forward`.
- **Wrong products:** predicting forward in time, 71 of 100 labelled wrong-product areas are flagged, and every one of them names the product our labels give (taken from the same area's earlier labelled images, no VLM).
- **The VLM is optional:** asking Gemini about the uncertain areas added +0.3% (inside noise) for about 11 minutes of API calls, so it is off by default. A full run takes about a minute.
- **Where the points go:** 111 of 1,947 stock levels are wrong. 59 are too high (3 points each, 177 in total) and 52 are too low (2 points each, 104 in total). 86 of the 111 are half against full (44 half called full, 42 full called half). Evening images, when shelves run down, are the hardest: after 19:00 Lisbon we score 210 of 276 points (76.1%), and only 75 of 138 stock points.
- **Label noise is real:** two of our labellers disagreed on about 12% of stock levels, mostly half against full. No model can score above what the labels agree on.
- **The shelf does not always follow the planogram.** Some areas hold a product that is not in the planogram's list at all (for example bagged lemons in a kiwi area). We flag those as wrong, and name the product only when that area has held it in a labelled frame before.
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
