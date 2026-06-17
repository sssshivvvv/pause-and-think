#!/usr/bin/env python3
"""
Regenerate the pause-and-think clip files from the raw source videos.

The released training / benchmark JSONs reference video clips by path (the
`videos` / `video` fields). We do not redistribute those clips; instead, the
mapping files in `dataset/mappings/` tell you, for every clip, exactly which
raw source video it came from and the wall-clock window (in seconds) to cut.

This script reads one mapping file, locates each raw parent video under the
root you provide for that dataset, and cuts the clip with ffmpeg at the target
frame rate the released clips were encoded at (EK/Ego4d = 15 fps,
Assembly101 = 30 fps). Timestamps are fps-invariant, so the seconds in the
mapping apply directly to the raw video regardless of its native fps.

Each clip is written to <out-root>/<clip_path>, where clip_path is the exact
`videos/...` string used in the training / benchmark JSONs. Point your trainer
or the benchmark inference scripts at the same <out-root> (e.g. run them from
that directory) and the `videos/...` paths resolve.

Usage
-----
    python regenerate_clips.py \
        --mapping mappings/EK_training_data_mapping.json \
        --ek-root  /path/to/EPIC-KITCHENS \
        --out-root /path/to/data_root

    python regenerate_clips.py \
        --mapping mappings/Ego4d_benchmark_data_mapping.json \
        --ego4d-root /path/to/ego4d/v2/full_scale \
        --out-root   /path/to/data_root

    python regenerate_clips.py \
        --mapping mappings/Assembly_training_data_mapping.json \
        --assembly-root /path/to/Assembly101 \
        --out-root      /path/to/data_root

Add --dry-run to print the ffmpeg commands without running them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Frame rate the released clips were encoded at, per dataset.
TARGET_FPS = {"EK": 15, "Ego4d": 15, "Assembly": 30}

# By default we write each clip at exactly the path used in the JSONs
# (e.g. videos/...), rooted at --out-root. Set --strip-prefix to drop a leading
# component if you want a flatter output layout.
DEFAULT_PREFIX = ""


def detect_dataset(mapping_path: Path, rows: list) -> str:
    name = mapping_path.name
    for ds in ("EK", "Ego4d", "Assembly"):
        if name.startswith(ds):
            return ds
    sample = rows[0] if rows else {}
    if "ek_mapping" in sample:
        return "EK"
    if "ego4d_mapping" in sample:
        return "Ego4d"
    if "assembly_mapping" in sample:
        return "Assembly"
    raise SystemExit(f"Cannot infer dataset from {mapping_path}")


def iter_clip_entries(rows: list, key: str):
    """Yield every per-clip mapping dict regardless of training/benchmark layout."""
    for row in rows:
        m = row.get(key)
        if m is None:
            continue
        if isinstance(m, dict):  # EK benchmark stores a single dict
            yield m
        else:
            for item in m:
                if isinstance(item, dict):
                    yield item


def out_path(clip_path: str, out_root: Path, strip_prefix: str) -> Path:
    rel = clip_path
    if strip_prefix and strip_prefix in rel:
        rel = rel.split(strip_prefix, 1)[1]
    return out_root / rel


def ek_raw_candidates(ek_root: Path, raw_parent_video: str) -> list[Path]:
    # EPIC-KITCHENS layouts vary; try the flat name and the official PXX/ split.
    stem = raw_parent_video
    participant = stem.split("_")[0]
    return [
        ek_root / stem,
        ek_root / participant / stem,
        ek_root / participant / "videos" / stem,
    ]


def ego4d_raw_candidates(ego4d_root: Path, uuid_mp4: str) -> list[Path]:
    return [
        ego4d_root / uuid_mp4,
        ego4d_root / "v2" / "full_scale" / uuid_mp4,
        ego4d_root / "full_scale" / uuid_mp4,
    ]


def assembly_raw_candidates(assembly_root: Path, recording: str, view: str) -> list[Path]:
    return [
        assembly_root / recording / view,
        assembly_root / recording / "C10118_rgb" / view,
    ]


def first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.is_file():
            return c
    return None


def window_for(dataset: str, is_train: bool, m: dict) -> tuple[float, float] | None:
    """Return (start_sec, end_sec) on the raw parent for this clip."""
    if dataset == "EK":
        if is_train:
            w = m.get("shown_video_window_in_parent")
            if w and w.get("start_sec") is not None and w.get("end_sec") is not None:
                return float(w["start_sec"]), float(w["end_sec"])
            cw = m.get("clip_window_in_parent") or {}
            qp = m.get("question_point") or {}
            if cw.get("start_sec") is not None and qp.get("in_parent_sec") is not None:
                return float(cw["start_sec"]), float(qp["in_parent_sec"])
            return None
        cw = m.get("clip_window_in_parent") or {}
        if cw.get("start_sec") is not None and cw.get("end_sec") is not None:
            return float(cw["start_sec"]), float(cw["end_sec"])
        return None

    if dataset == "Ego4d":
        cw = m.get("clip_window_in_parent") or {}
        if cw.get("start_sec") is None:
            return None
        if is_train:
            qp = m.get("question_point") or {}
            end = qp.get("in_parent_sec")
            if end is None:
                end = cw.get("end_sec")
            return float(cw["start_sec"]), float(end)
        return float(cw["start_sec"]), float(cw["end_sec"])

    # Assembly
    cw = m.get("clip_window_in_raw_session") or {}
    if cw.get("start_sec") is None:
        return None
    if is_train:
        qp = m.get("question_point") or {}
        end = qp.get("in_raw_session_sec")
        if end is None:
            end = cw.get("end_sec")
        return float(cw["start_sec"]), float(end)
    return float(cw["start_sec"]), float(cw["end_sec"])


def raw_for(dataset: str, m: dict, args) -> Path | None:
    if dataset == "EK":
        if not args.ek_root:
            raise SystemExit("--ek-root is required for EK mappings")
        cands = ek_raw_candidates(Path(args.ek_root), m["raw_parent_video"])
    elif dataset == "Ego4d":
        if not args.ego4d_root:
            raise SystemExit("--ego4d-root is required for Ego4d mappings")
        cands = ego4d_raw_candidates(Path(args.ego4d_root), m["raw_parent_uuid"])
    else:
        if not args.assembly_root:
            raise SystemExit("--assembly-root is required for Assembly mappings")
        cands = assembly_raw_candidates(Path(args.assembly_root), m["raw_session_recording"], m["raw_session_view"])
    found = first_existing(cands)
    # In dry-run we just want to preview the command, so fall back to the
    # canonical first candidate even when the raw file is not present locally.
    if found is None and args.dry_run:
        return cands[0]
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True, help="Path to a *_data_mapping.json file")
    ap.add_argument("--out-root", required=True, help="Directory to write regenerated clips into")
    ap.add_argument("--ek-root", help="Root of EPIC-KITCHENS raw videos (e.g. P32/P32_05.mp4 live under here)")
    ap.add_argument("--ego4d-root", help="Root containing Ego4d v2 full_scale UUID mp4s")
    ap.add_argument("--assembly-root", help="Root of Assembly101 recordings (<recording>/C10118_rgb.mp4)")
    ap.add_argument("--fps", type=int, default=None, help="Override target fps (default: EK/Ego4d=15, Assembly=30)")
    ap.add_argument("--strip-prefix", default=DEFAULT_PREFIX, help="Optional leading path component to drop from clip_path (default: keep full path, e.g. videos/...)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N clips (for testing)")
    ap.add_argument("--overwrite", action="store_true", help="Re-cut clips that already exist")
    ap.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without running them")
    args = ap.parse_args()

    mapping_path = Path(args.mapping)
    rows = json.loads(mapping_path.read_text())
    dataset = detect_dataset(mapping_path, rows)
    is_train = "training" in mapping_path.name
    fps = args.fps if args.fps is not None else TARGET_FPS[dataset]
    key = {"EK": "ek_mapping", "Ego4d": "ego4d_mapping", "Assembly": "assembly_mapping"}[dataset]
    out_root = Path(args.out_root)

    print(f"Dataset={dataset} split={'training' if is_train else 'benchmark'} target_fps={fps}")

    done = skipped = missing_raw = bad_window = failed = 0
    for m in iter_clip_entries(rows, key):
        if args.limit is not None and done + skipped + missing_raw + bad_window + failed >= args.limit:
            break
        clip_path = m.get("clip_path")
        if not clip_path:
            continue

        win = window_for(dataset, is_train, m)
        if win is None or win[1] is None or win[1] <= win[0]:
            bad_window += 1
            continue
        start, end = win

        raw = raw_for(dataset, m, args)
        if raw is None:
            missing_raw += 1
            print(f"[missing raw] {clip_path}")
            continue

        dst = out_path(clip_path, out_root, args.strip_prefix)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(raw),
            "-r", str(fps),
            "-an",
            str(dst),
        ]
        if args.dry_run:
            print(" ".join(cmd))
            done += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, check=True)
            done += 1
        except subprocess.CalledProcessError:
            failed += 1
            print(f"[ffmpeg failed] {clip_path}")

    print(
        f"Done: cut={done} skipped_existing={skipped} missing_raw={missing_raw} "
        f"bad_window={bad_window} ffmpeg_failed={failed}"
    )
    if missing_raw:
        print("Some raw videos were not found under the provided root(s); see [missing raw] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
