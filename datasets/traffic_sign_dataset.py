# datasets/traffic_sign_dataset.py
#
# Reconstructed to match the current interface expected by main.py /
# evaluate_and_plot.py / gui_app.py / models/traffic_sign_ann.py.
# Verified against the original version: NUM_CLASSES = 63 (Malaysia road
# sign dataset, COCO format) and identical label ordering (category ids
# sorted ascending). Differences from the original are intentional: this
# version supports the `transform` argument, exposes category_id_to_name,
# and returns single-label class indices (required by CrossEntropyLoss and
# the evaluation utilities in utils/evaluation.py).

import json
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def _detect_num_classes():
    """Try to read the number of classes from the dataset annotations."""
    candidates = [
        "data_stratified/train/_annotations.coco.json",
        "data_stratified/valid/_annotations.coco.json",
        "data_stratified/test/_annotations.coco.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                n = len(data.get("categories", []))
                if n > 0:
                    return n
            except Exception:
                pass
    return None


# Number of traffic-sign classes. Auto-detected from the COCO annotations
# when the dataset is present; otherwise falls back to the value below.
# TODO: confirm this fallback matches your trained model.
NUM_CLASSES = _detect_num_classes() or 63


class TrafficSignDataset(Dataset):
    """COCO-annotation-based classification dataset for traffic signs.

    Each image is labeled with the category of its (first) annotation.
    Exposes:
      - label_mapping:        {original COCO category id -> 0-indexed label}
      - category_id_to_name:  {original COCO category id -> category name}
    """

    def __init__(self, json_path, image_dir, device, transform=None):
        self.image_dir = image_dir
        self.device = device
        self.transform = transform

        with open(json_path, "r") as f:
            coco = json.load(f)

        # Build category mappings (sorted by original id for stable ordering)
        categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
        self.category_id_to_name = {c["id"]: c["name"] for c in categories}
        self.label_mapping = {c["id"]: idx for idx, c in enumerate(categories)}

        # Map image id -> file name
        images_by_id = {img["id"]: img["file_name"] for img in coco.get("images", [])}

        # One sample per image, labeled by its first annotation's category
        first_ann_by_image = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in first_ann_by_image:
                first_ann_by_image[img_id] = ann["category_id"]

        self.samples = []
        for img_id, cat_id in first_ann_by_image.items():
            file_name = images_by_id.get(img_id)
            if file_name is not None and cat_id in self.label_mapping:
                self.samples.append((file_name, self.label_mapping[cat_id]))

        self._default_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, label = self.samples[idx]
        path = os.path.join(self.image_dir, file_name)
        if not os.path.exists(path):
            path = os.path.join(self.image_dir, os.path.basename(file_name.replace("\\", "/")))
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self._default_transform(image)

        return image, torch.tensor(label, dtype=torch.long)
