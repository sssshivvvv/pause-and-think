# pause-and-think dataset: download & regeneration

We do **not** redistribute the source video clips (they are derived from
EPIC-KITCHENS, Assembly101, and Ego4D, each with its own license). Instead this
folder ships **mapping files** that tell you, for every clip referenced by the
training / benchmark JSONs, exactly which **raw source video** it came from and
the **wall-clock window (in seconds)** to cut. After you download the raw
datasets and the conversation/annotation JSONs, you regenerate the clips locally
with [`regenerate_clips.py`](./regenerate_clips.py).

```
dataset/
├── mappings/
│   ├── EK_training_data_mapping.json
│   ├── EK_benchmark_data_mapping.json
│   ├── Ego4d_training_data_mapping.json
│   ├── Ego4d_benchmark_data_mapping.json
│   ├── Assembly_training_data_mapping.json
│   └── Assembly_benchmark_data_mapping.json
├── regenerate_clips.py
└── README.md   (this file)
```

## Step 1 — Get the conversation / annotation files

These contain the `messages` (with `<thinking>` supervision), questions, ground
truth, and the `videos` / `video` paths each sample expects.

- **Training set** (`all_training_data_alpaca_thinking.json`) and the
  **train/val scene & goal splits**: *released on Hugging Face* (see the
  **Training / evaluation / benchmark data** link in the top-level
  [`README.md`](../README.md)).
- **Benchmark manifest**: already in this repo at
  [`benchmarking/benchmark_300_files.json`](../benchmarking/benchmark_300_files.json).

The `videos` / `video` strings in those files are **relative paths** of the form
`videos/…`. They are not files you need to download — they are where the
regenerated clips should live. The regeneration script writes each clip to
`<out-root>/videos/…`, so point your trainer / benchmark scripts at the same
`--out-root` (e.g. run them from that directory) and the paths resolve.

## Step 2 — Download the raw source videos

| Dataset | Where to download | What you need locally |
|--------|-------------------|------------------------|
| **EPIC-KITCHENS** | https://epic-kitchens.github.io/ — use the official [download scripts](https://github.com/epic-kitchens/epic-kitchens-download-scripts) | The full-scale RGB videos, e.g. `P32_05.MP4` (referenced by `raw_parent_video`) |
| **Ego4D** | https://ego4d-data.org/ — request access, then `ego4d --output_directory <dir> --datasets full_scale --version v2` | The **v2 `full_scale`** UUID videos, e.g. `a6b81d4e-…​.mp4` (referenced by `raw_parent_uuid`) |
| **Assembly101** | https://assembly-101.github.io/ — follow their access + download instructions | Per-recording RGB views `…/<recording>/C10118_rgb.mp4` (referenced by `raw_session_recording` + `raw_session_view`) |

## Step 3 — Regenerate the clips

Timestamps in the mappings are **seconds = raw-video wall-clock time** and are
**fps-invariant**, so they apply directly to the raw videos. The released clips
were re-encoded after downsampling, so the script re-encodes at the matching
target frame rate:

| Dataset | Target clip fps | Raw native fps |
|--------|------------------|----------------|
| EPIC-KITCHENS | **15** | ~59.94 |
| Ego4D | **15** | 30 (UUID full_scale) |
| Assembly101 | **30** | 60 (`C10118_rgb`) |

To map a second to a frame index at a given fps: `frame = round(time_sec * fps)`.

```bash
# EPIC-KITCHENS (training)
python regenerate_clips.py \
  --mapping mappings/EK_training_data_mapping.json \
  --ek-root  /path/to/EPIC-KITCHENS \
  --out-root /path/to/data_root

# Ego4D (benchmark)
python regenerate_clips.py \
  --mapping mappings/Ego4d_benchmark_data_mapping.json \
  --ego4d-root /path/to/ego4d/v2/full_scale \
  --out-root   /path/to/data_root

# Assembly101 (training)
python regenerate_clips.py \
  --mapping mappings/Assembly_training_data_mapping.json \
  --assembly-root /path/to/Assembly101 \
  --out-root      /path/to/data_root
```

Each clip is written to `<out-root>/videos/…` (the exact path used in the
JSONs). Run the same command for each of the six mapping files (training +
benchmark for all three datasets), pointing every dataset at its raw root and a
**shared** `--out-root`, then run training / benchmark inference from that
`--out-root` so the `videos/…` paths resolve. Useful flags:

- `--dry-run` — print the ffmpeg commands without cutting anything.
- `--limit N` — process only the first N clips (quick sanity check).
- `--fps F` — override the target fps.
- `--overwrite` — re-cut clips that already exist.

Requirements: Python 3.8+ and **ffmpeg** on your `PATH`.

### What gets cut for each split

- **Benchmark** clips are the *whole* released clip — the script uses
  `clip_window_in_parent` (EK/Ego4d) or `clip_window_in_raw_session` (Assembly).
- **Training** clips are the `question_video.mp4` prefix shown to the model
  before it answers — the script cuts from the clip/tile start up to the
  `question_point` (EK uses `shown_video_window_in_parent`).

## Mapping schema reference

All `*_sec` fields are seconds on the raw parent timeline; `*_ts` are
`HH:MM:SS.ss` for convenience only. No internal storage paths are present —
only public identifiers and the dataset-relative `clip_path`.

### EPIC-KITCHENS (`ek_mapping`)

- `clip_path`, `parent_video_name`, `raw_parent_video`, `high_level_goal`,
  `phase_key`, `clip_id`
- `phase_window_in_parent`: `{start, end}` (phase extent on the parent)
- `clip_window_in_parent`: clip extent on the parent (seconds + `*_ts`)
- `action_span_in_parent` *(benchmark)*: annotated-action subset of the clip
- `clip_actions`: action list from the source timestamps
- `question_point`: where the question is asked (`in_parent_sec` / `*_ts`)
- `shown_video_window_in_parent` *(training)*: exact extent of the
  `question_video` on the parent (`start_sec`, `end_sec`, `*_ts`)

### Ego4D (`ego4d_mapping`)

- `clip_path`, `parent_video_int`, `raw_parent_uuid`, `clip_index`,
  `tile_seconds`
- `clip_window_in_parent`: fixed 10 s tile on the parent (seconds + `*_ts`)
- `question_point`: `measured_duration_sec`, `clip_relative_*`, `in_parent_*`

> The integer `parent_video_int` clip is the first ~40 s of the raw UUID video
> (downsampled for release) and shares **origin t = 0** with it, so
> `clip_window_in_parent` is valid on both the integer clip and the raw UUID
> timeline.

### Assembly101 (`assembly_mapping`)

- `clip_path`, `session_id`, `phase`, `high_level_goal`, `percentage`,
  `clip_variant`
- `raw_session_recording`, `raw_session_view` (`C10118_rgb.mp4`)
- `clip_window_in_raw_session`: clip extent on the raw RGB view (seconds + `*_ts`)
- `question_point`: `measured_question_duration_sec`, `in_raw_session_sec`, `*_ts`
- `query_actions`: `[{action_name: {start_sec, end_sec, start_ts, end_ts}}, …]`
  (converted from 30 fps annotation frames; `seconds = frame / 30`)
- `response_actions`: same structure, present only when non-empty

> Assembly timestamps are **absolute wall-clock seconds** on the 60 fps
> `C10118_rgb` view, derived from Assembly101's 30 fps coarse/fine annotations.
