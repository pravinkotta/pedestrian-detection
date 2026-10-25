"""Dataset + collate, reading the *Silver* COCO annotations.

Promoted from the notebook's ``StreetSceneDataset`` (cell 25), with two
deliberate changes:

1. **Reads COCO JSON instead of ``meta.json``.**  The notebook read raw
   ``[x1, y1, x2, y2]`` boxes straight from meta.json. Here we read the
   Silver output so training, evaluation, CI and Evidently all share one
   canonical format.

2. **Converts ``xywh -> xyxy``.**  torchvision's Faster R-CNN expects
   ``[x1, y1, x2, y2]``; COCO stores ``[x, y, w, h]``. Getting this wrong
   produces a model that trains without error and learns garbage — there is
   no crash, only silently collapsed boxes. ``test_dataset`` guards it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def xywh_to_xyxy(boxes):
    """COCO [x, y, w, h] -> torchvision [x1, y1, x2, y2]."""
    out = boxes.clone()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


class CocoDetectionDataset(Dataset):
    """COCO-formatted detection dataset.

    Parameters
    ----------
    ann_file : path to a COCO ``instances`` JSON (Silver or Gold output)
    image_dir : folder containing the images named in ``file_name``
    transforms : optional joint transform applied to the image only —
        torchvision detection models take boxes in *original image
        coordinates*, so any resize must also scale ``target['boxes']``.
        The default ``ToTensor`` needs no such handling.
    """

    def __init__(self, ann_file: Path, image_dir: Path, transforms=None):
        self.ann_file = Path(ann_file)
        self.image_dir = Path(image_dir)
        self.transforms = transforms or T.ToTensor()

        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"{self.ann_file} not found — run the data pipeline first:\n"
                "  python -m src.data.make_bronze\n"
                "  python -m src.data.convert_to_coco"
            )

        coco = json.loads(self.ann_file.read_text(encoding="utf-8"))
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}
        self.classes = [self.categories[k] for k in sorted(self.categories)]

        # index annotations by image so __getitem__ is O(1) not O(N)
        anns_by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue          # crowd regions are excluded from training
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.images = [
            im for im in coco["images"] if im["id"] in anns_by_image
        ]
        self.anns_by_image = anns_by_image
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.images)

    @property
    def num_classes(self) -> int:
        """Classes *including* background (COCO ids are 1..N)."""
        return len(self.categories) + 1

    def __getitem__(self, idx: int):
        rec = self.images[idx]
        path = self.image_dir / rec["file_name"]

        with Image.open(path) as im:
            img = im.convert("RGB")

        anns = self.anns_by_image[rec["id"]]
        boxes = torch.as_tensor(
            [a["bbox"] for a in anns], dtype=torch.float32
        ).reshape(-1, 4)
        labels = torch.as_tensor(
            [a["category_id"] for a in anns], dtype=torch.int64
        )
        boxes = xywh_to_xyxy(boxes)

        # clip defensively — a box extending past the edge breaks RoIAlign
        boxes[:, 0::2].clamp_(0, rec["width"])
        boxes[:, 1::2].clamp_(0, rec["height"])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        return self.transforms(img), target


def collate(batch):
    """Detection targets vary in count, so we zip rather than stack."""
    return tuple(zip(*batch))


def make_loader(ann_file: Path, image_dir: Path, batch_size: int,
                shuffle: bool, num_workers: int = 0):
    from torch.utils.data import DataLoader

    ds = CocoDetectionDataset(ann_file, image_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        # Windows: num_workers > 0 needs a __main__ guard or it deadlocks
        persistent_workers=num_workers > 0,
    )
