
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _boxes_xyxy_from_xywh(boxes_xywh):
    if not boxes_xywh:
        return np.empty((0, 4), dtype=np.float32)
    b = np.asarray(boxes_xywh, dtype=np.float32)
    b[:, 2] = b[:, 0] + b[:, 2]  # x1
    b[:, 3] = b[:, 1] + b[:, 3]  # y1
    return b[:, [0, 1, 2, 3]]


def _normalize_masks_single_per_box(masks):
    """
    Normalize to a list of 2D uint8 masks (H, W), one per box.
    """
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:                # (H, W)
            return [(masks > 0).astype(np.uint8)]
        if masks.ndim == 3:                # (N, H, W)
            return [(m > 0).astype(np.uint8) for m in masks]
        if masks.ndim == 4:                # (N, 1, H, W)
            if masks.shape[1] == 1:
                return [(m[0] > 0).astype(np.uint8) for m in masks]
            raise ValueError(f"Unexpected mask shape {masks.shape}")
        raise ValueError(f"Unexpected ndarray ndim {masks.ndim}")

    if isinstance(masks, (list, tuple)):
        out = []
        for m in masks:
            ma = np.asarray(m).squeeze()
            out.append((ma > 0).astype(np.uint8))
        return out

    # Fallback single
    ma = np.asarray(masks).squeeze()
    return [(ma > 0).astype(np.uint8)]


class SAM2Runner:
    """
    Initialize once; call .segment_boxes(frame_bgr, boxes_xywh) per frame.
    Returns one mask per input box.
    """

    def __init__(self, cfg_path: str, ckpt_path: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model = build_sam2(cfg_path, ckpt_path, device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def segment_boxes(self, frame_bgr: np.ndarray, boxes_xywh):
        """
        Returns:
            list of (H, W) uint8 masks, one per box (same order).
        """
        if not boxes_xywh:
            return []

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img_rgb)

        boxes_xyxy = _boxes_xyxy_from_xywh(boxes_xywh)

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, _scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes_xyxy,
                        multimask_output=False,   # single best per box
                    )
            else:
                masks, _scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_xyxy,
                    multimask_output=False,      
                )

        best_masks = _normalize_masks_single_per_box(masks)

        if len(best_masks) != len(boxes_xywh):
            if len(boxes_xywh) == 1 and len(best_masks) > 1:
                best_masks = [best_masks[0]]
            else:
                raise RuntimeError(f"Expected {len(boxes_xywh)} masks, got {len(best_masks)}")

        return best_masks
