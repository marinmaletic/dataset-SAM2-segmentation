import cv2
import json
from pathlib import Path
from collections import defaultdict

def read_coco_json(json_path):
    """
    Loads COCO-style detections JSON and groups annotations by image_id.
    Returns:
        images_sorted: list of image dicts sorted by file_name
        anns_per_img: dict mapping image_id -> list of annotations
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    images_sorted = sorted(images, key=lambda im: im["file_name"])
    anns_per_img = defaultdict(list)
    for ann in annotations:
        anns_per_img[ann["image_id"]].append(ann)

    return images_sorted, anns_per_img


def iter_video_frames(video_path, json_path=None, category_filter=None):
    """
    Opens video and COCO JSON and yields frames one by one.
    Yields:
        (frame_idx, frame, anns) where:
            frame_idx (int): index of the frame
            frame (np.ndarray): BGR image from cv2
            anns (list): filtered annotations for this frame (empty if no JSON)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    images_sorted, anns_per_img = ([], {})  # defaults
    if json_path:
        images_sorted, anns_per_img = read_coco_json(json_path)

    # Normalize category_filter to list
    if category_filter is not None:
        if isinstance(category_filter, int):
            category_filter = [category_filter]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        anns = []
        if json_path and frame_idx < len(images_sorted):
            img_id = images_sorted[frame_idx]["id"]
            anns = anns_per_img.get(img_id, [])

            # Apply category_id filtering
            if category_filter is not None:
                anns = [a for a in anns if a.get("category_id") in category_filter]

        yield frame_idx, frame, anns
        frame_idx += 1

    cap.release()
