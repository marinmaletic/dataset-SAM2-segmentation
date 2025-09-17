# dataset-SAM2-segmentation
This repository contains the code needed for augmenting an existing object detection dataset by adding SAM2 segmentation mask information.

# SAM2 Video Dataset Segmentator

This project automates **panoptic dataset generation** from videos using [Meta’s Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2).

It:
- Loads **videos** and **COCO-format detection JSONs** (with bounding boxes).
- Runs **SAM2** on each bounding box to produce segmentation masks for the selected category ID.
- Writes the resulting **COCO RLE segmentation** directly into the JSON annotations, preserving the original detections and tracking IDs.
- Saves updated JSONs into a new `labels_with_segmentation/` folder.

This is especially useful for cases where object detection.datasets are abundant, but segmentation ones are hard to find.

---

## Features

- **Automatic matching** of videos ↔ JSONs by filename.
- **GUI folder pickers** to choose video & labels folders.
- **Video duration limit** (`--seconds`) for quick experiments.
- **Object category** (`--cat`) for specifying exact category ID to be segmented.
- **Resume segmentation** skips videos that were already segmented.
- **Progress printing** with per-500-frame info.
- Uses **pycocotools compressed RLE** if available, otherwise falls back to uncompressed.

---

## Installation

### 1. Clone this repository

```bash
git clone --recursive https://github.com/marinmaletic/dataset-SAM2-segmentation.git
```

### 2. Install SAM2
```bash
cd dataset-SAM2-segmentation/sam2
pip install -e .
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Inside `run_segmentation.py` configure the SAM2 checkpoints path if needed (currently configured as this repo is installed in root folder).

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Prepare data
Organize files like this:
```bash
dataset_root/
├── videos/
│   ├── video_01.mp4
│   ├── video_02.mp4
│   └── ...
└── labels/
    ├── video_01.json
    ├── video_02.json
    └── ...
```

Filenames must match (e.g. video_01.mp4 ↔ video_01.json).

### 5. Run the segmentator
```bash
python run_segmentation.py --cat 2 --seconds 0
```

```bash
--cat 2 → only segment objects with category_id = 2

--seconds 0 → process entire video (use e.g. --seconds 60 to limit to first 60 seconds)
```

You will:

1. Select the videos folder from a popup dialog.

2. Select the labels (JSON) folder.

3. The script will process each pair and write new JSONs into:
```bash
<common_root>/labels_with_segmentation/<video_stem>_segmented.json
```

### Resume later

If you stop the script mid-way, re-run it.
Already-processed videos (with _segmented.json present) will be skipped automatically.