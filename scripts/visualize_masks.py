import json
import time
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog, simpledialog

# Popup pickers
def pick_video_file() -> Path:
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=(
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.mpg *.mpeg"),
            ("All files", "*.*"),
        ),
    )
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)

def pick_json_file() -> Path:
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select the segmented JSON",
        filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
    )
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)

def ask_category_filter():
    """Ask for category filter (None = all)."""
    root = Tk(); root.withdraw()
    ans = simpledialog.askstring(
        "Category filter",
        "Enter category_id to visualize (integer), or leave empty for ALL:",
        parent=root,
    )
    root.destroy()
    if ans is None or ans.strip() == "":
        return None
    try:
        return int(ans.strip())
    except ValueError:
        print("Invalid input, showing ALL categories.")
        return None

def get_screen_size():
    root = Tk(); root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

# ---------- COCO helpers ----------
def read_coco_grouped(json_path: Path):
    with open(json_path, "r") as f:
        coco = json.load(f)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    images_sorted = sorted(images, key=lambda im: im["file_name"])
    anns_per_img = {}
    for ann in annotations:
        anns_per_img.setdefault(ann["image_id"], []).append(ann)
    return images_sorted, anns_per_img

def stable_id(ann):
    tid = ann.get("attributes", {}).get("track_id", None)
    return f"track_{tid}" if tid is not None else f"ann_{ann['id']}"

def color_for_sid(sid):
    h = abs(hash(sid)) % 360
    hsv = np.uint8([[[h // 2, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

# RLE decoding
def rle_decode(seg):
    """Decode COCO-style RLE segmentation to binary mask."""
    try:
        from pycocotools import mask as maskUtils
    except ImportError:
        maskUtils = None

    # Multipart: list of RLE dicts
    if isinstance(seg, list):
        if maskUtils is None:
            raise RuntimeError("Multipart compressed RLE requires pycocotools. Install with: pip install pycocotools")
        m = maskUtils.decode(seg)  # (H,W,N)
        if m.ndim == 3:
            m = (m.max(axis=2) > 0).astype(np.uint8)  # union of parts
        elif m.ndim == 2:
            m = (m > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected decoded shape for list RLE: {m.shape}")
        return np.ascontiguousarray(m.astype(np.uint8))

    # Single RLE dict
    if not isinstance(seg, dict):
        raise TypeError(f"Unsupported segmentation type: {type(seg)}")

    H, W = seg["size"]
    counts = seg["counts"]

    # Compressed RLE (string)
    if isinstance(counts, str):
        if maskUtils is None:
            raise RuntimeError("Compressed RLE requires pycocotools. Install with: pip install pycocotools")
        rle = {"size": [int(H), int(W)], "counts": counts.encode("ascii")}
        m = maskUtils.decode(rle)  # (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[:, :, 0]
        elif m.ndim != 2:
            raise ValueError(f"Unexpected decoded shape for compressed RLE: {m.shape}")
        return np.ascontiguousarray((m > 0).astype(np.uint8))

    # Uncompressed RLE (counts list of runs)
    if isinstance(counts, list):
        runs = counts
        total = int(H) * int(W)
        flat = np.zeros(total, dtype=np.uint8)
        val = 0
        idx = 0
        for run_len in runs:
            rl = int(run_len)
            if rl > 0 and val == 1:
                flat[idx:idx + rl] = 1
            idx += rl
            val ^= 1
        if idx < total:
            flat = np.pad(flat, (0, total - idx), constant_values=0)
        m = flat.reshape((int(H), int(W)), order="F")
        return np.ascontiguousarray(m.astype(np.uint8))

    raise TypeError(f"Unsupported RLE 'counts' type: {type(counts)}")

# visualization
def overlay_masks(frame_bgr, masks, colors, alpha=0.45):
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    for m, color in zip(masks, colors):
        if m is None:
            continue
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            continue
        color_img = np.zeros((H, W, 3), dtype=np.uint8)
        color_img[:] = color
        out = np.where(m[..., None] == 1,
                       (alpha * color_img + (1 - alpha) * out).astype(np.uint8),
                       out)
    return out

def draw_boxes(frame_bgr, anns, colors=None):
    vis = frame_bgr.copy()
    for i, ann in enumerate(anns):
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        color = (0, 255, 0) if colors is None else colors[i]
        lbl = stable_id(ann).replace("track_", "t=").replace("ann_", "a=")
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y0 - 4)
        cv2.rectangle(vis, (x0, y_text - th - 4), (x0 + tw + 6, y_text), color, -1)
        cv2.putText(vis, lbl, (x0 + 3, y_text - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis

# main playback
def main():
    video_path = pick_video_file()
    stem = video_path.stem

    candidate = (video_path.parent.parent / "labels_with_segmentation" / f"{stem}_segmented.json").resolve()
    if candidate.exists():
        json_path = candidate
    else:
        candidate2 = (video_path.parent / f"{stem}_segmented.json").resolve()
        json_path = candidate2 if candidate2.exists() else pick_json_file()

    category_filter = ask_category_filter()
    if category_filter is None:
        print("Showing ALL categories.")
    else:
        print(f"Showing ONLY category_id = {category_filter}")

    print(f"Video: {video_path}")
    print(f"JSON : {json_path}")

    images_sorted, anns_per_img = read_coco_grouped(json_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    frame_period = 1.0 / fps

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {W}x{H}, FPS: {fps:.3f}, Frames: {total}")

    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)
    scale = min(max_w / max(W, 1), max_h / max(H, 1), 2.0)
    disp_w, disp_h = max(320, int(W * scale)), max(240, int(H * scale))

    win = "Mask Visualization (space=pause, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)

    mask_cache = {}
    paused = False
    frame_idx = 0
    start_time = time.time()
    pause_accum = 0.0
    pause_started = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break

            anns = []
            if frame_idx < len(images_sorted):
                img_id = images_sorted[frame_idx]["id"]
                anns = anns_per_img.get(img_id, [])

            if category_filter is not None:
                anns = [a for a in anns if a.get("category_id") == category_filter]

            colors = [color_for_sid(stable_id(a)) for a in anns]
            masks = []
            for a in anns:
                rle = a.get("segmentation")
                if not rle:
                    masks.append(None)
                    continue
                ann_id = a["id"]
                if ann_id not in mask_cache:
                    try:
                        mask_cache[ann_id] = rle_decode(rle)
                    except Exception as e:
                        print(f"Warn: failed to decode ann {ann_id}: {e}")
                        mask_cache[ann_id] = None
                masks.append(mask_cache[ann_id])

            vis = overlay_masks(frame, masks, colors, alpha=0.45)
            vis = draw_boxes(vis, anns, colors=colors)
            cv2.imshow(win, vis)

            target_t = start_time + pause_accum + frame_idx * frame_period
            now = time.time()
            sleep_for = target_t - now
            if sleep_for > 0:
                time.sleep(sleep_for)

            frame_idx += 1

        k = cv2.waitKey(1 if not paused else 30) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            if not paused:
                paused = True
                pause_started = time.time()
            else:
                paused = False
                if pause_started is not None:
                    pause_accum += time.time() - pause_started
                    pause_started = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
