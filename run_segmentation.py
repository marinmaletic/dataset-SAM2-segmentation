import os
import json
import cv2
import argparse
import numpy as np
import time
from pathlib import Path
from tkinter import Tk, filedialog

# === HARD-CODED SAM2 CHECKPOINT ===
CKPT_PATH = "/root/dataset-SAM2-segmentation/sam2/checkpoints/sam2.1_hiera_large.pt"  # <-- adjust as needed
CFG_PATH_DEFAULT = "configs/sam2.1/sam2.1_hiera_l.yaml"   

from video_utils import iter_video_frames
from sam_utils import SAM2Runner


# ---- COCO RLE helpers ----
def rle_encode_uncompressed(mask: np.ndarray):
    """COCO-style uncompressed RLE"""
    m = np.asfortranarray(mask.astype(np.uint8))
    H, W = m.shape
    r = m.reshape(-1, order="F")
    counts, run_val, run_len = [], 0, 0
    for v in r:
        if v == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = int(v)
            run_len = 1
    counts.append(run_len)
    return {"counts": counts, "size": [int(H), int(W)]}

def rle_encode_coco(mask: np.ndarray):
    """Compressed RLE via pycocotools if available; else uncompressed."""
    try:
        from pycocotools import mask as maskUtils
        m = np.asfortranarray(mask.astype(np.uint8))
        rle = maskUtils.encode(m[:, :, None])[0]
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        return rle_encode_uncompressed(mask)



def process_one_pair(video_path: Path, json_path: Path, out_json_path: Path,
                     sam_cfg: str, category_filter: int | None, seconds_limit: int):
    """Open video+json, run SAM2 on all detections, add 'segmentation' to each ann, write new JSON."""
    # Load the JSON to be saved and map id -> ann
    with open(json_path, "r") as f:
        coco = json.load(f)
    ann_by_id = {ann["id"]: ann for ann in coco["annotations"]}

    # Init SAM2 once
    sam = SAM2Runner(cfg_path=sam_cfg, ckpt_path=CKPT_PATH, device="cuda")

    # Frame limit from seconds (if any)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    max_frames = int(round(seconds_limit * fps)) if seconds_limit > 0 else None

    print(f"Video: {video_path.name} | Frames: {total_frames} | FPS: {fps:.2f}")
    if max_frames:
        print(f"Processing will stop after {max_frames} frames (due to --seconds={seconds_limit})")
    print("Starting segmentation...")

    processed_frames = 0
    updated_anns = 0
    last_print_time = time.time()

    for frame_idx, frame_bgr, anns in iter_video_frames(str(video_path), str(json_path),
                                                        category_filter=category_filter):
        if max_frames is not None and frame_idx >= max_frames:
            break

        boxes = [a["bbox"] for a in anns]
        if boxes:
            masks = sam.segment_boxes(frame_bgr, boxes)  # one mask per box, same order
            for ann_tmp, mask in zip(anns, masks):
                coco_ann = ann_by_id.get(ann_tmp["id"])
                if coco_ann is None:
                    continue
                coco_ann["segmentation"] = rle_encode_coco(mask)
                updated_anns += 1

        processed_frames += 1
        if processed_frames % 500 == 0:
            now = time.time()
            elapsed = now - last_print_time
            last_print_time = now
            print(f"  Processed {processed_frames}/{max_frames or total_frames} frames "
                  f"({updated_anns} annotations segmented) | {elapsed:.2f} seconds elapsed")

    # Ensure output dir exists and save
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f" Wrote: {out_json_path}  (Total annotations updated: {updated_anns})")
    return True


# ---- GUI pickers ----
def pick_folder(title: str) -> Path:
    root = Tk(); root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)


def main():
    ap = argparse.ArgumentParser(description="Batch add SAM2 segmentations into COCO JSONs via GUI folder pickers.")
    ap.add_argument("--cfg", default=CFG_PATH_DEFAULT, help="SAM2 config path inside package")
    ap.add_argument("--cat", type=int, default=None, help="only segment this category_id (e.g. 2). None=all")
    ap.add_argument("--seconds", type=int, default=0, help="limit per-video processing in seconds (0=full video)")
    ap.add_argument("--video_exts", default=".mp4,.mkv,.avi,.mov,.mpg,.mpeg",
                    help="comma-separated video extensions to match")
    args = ap.parse_args()

    print("Select the videos folder…")
    videos_dir = pick_folder("Select videos folder")

    print("Select the labels (JSON) folder…")
    labels_dir = pick_folder("Select labels (JSON) folder")

    # Output directory under common root
    common_root = Path(os.path.commonpath([videos_dir.resolve(), labels_dir.resolve()]))
    out_dir = common_root / "labels_with_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output JSONs will be saved to: {out_dir}")

    exts = {e.strip().lower() for e in args.video_exts.split(",") if e.strip()}
    n_total = 0
    n_done = 0

    # Map stem -> json path for quick lookup
    json_by_stem = {p.stem: p for p in labels_dir.glob("*.json")}

    for vid in videos_dir.iterdir():
        if not vid.is_file() or vid.suffix.lower() not in exts:
            continue

        stem = vid.stem
        json_path = json_by_stem.get(stem)
        n_total += 1

        if json_path is None:
            print(f"  No matching JSON for video: {vid.name} (expected {stem}.json in {labels_dir})")
            continue

        # Save as "<stem>_segmented.json"
        out_json = out_dir / f"{stem}_segmented.json"

        # Skip if already segmented
        if out_json.exists():
            print(f"⏩ Skipping {vid.name} — segmented JSON already exists: {out_json.name}")
            continue

        print(f"\n=== Processing: {vid.name}  +  {json_path.name} ===")
        ok = process_one_pair(
            video_path=vid,
            json_path=json_path,
            out_json_path=out_json,
            sam_cfg=args.cfg,
            category_filter=args.cat,
            seconds_limit=args.seconds,
        )
        if ok:
            n_done += 1

    print(f"\nDone. Processed {n_done}/{n_total} matching video/json pairs.")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
