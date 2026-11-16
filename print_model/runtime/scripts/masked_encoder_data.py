"""Ultrasound data for Stage 1 masked autoencoder training."""
import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from print_model.data_paths import study_input
from print_model.preprocessing import build_transforms


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest(data_root):
    table = pd.read_excel(data_root / "stage1_ids.xlsx", usecols=["ID"], dtype={"ID": str})
    identifiers = table["ID"].str.strip().tolist()
    if any(pd.isna(identifier) or not identifier for identifier in identifiers):
        raise ValueError("Stage 1 nodule IDs must be nonempty")
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Supply a nonempty list of unique Stage 1 nodule IDs")

    image_dir = study_input(data_root, "stage1_ultrasound")
    paths = []
    for identifier in identifiers:
        views = sorted(image_dir.glob(f"{identifier}_*.jpg"))
        if not views:
            views = sorted(image_dir.glob(f"{identifier}*.jpg"))
        if not views:
            raise RuntimeError(f"No images found for {identifier}")
        paths.extend(views)

    expected = os.environ.get("THYROID_STAGEB_IMAGE_COUNT")
    if expected is not None and len(paths) != int(expected):
        raise RuntimeError(f"Stage 1 image count mismatch: {len(paths)} != {expected}")
    return len(identifiers), paths


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = build_transforms(use_grayscale=True, training=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("L"))


def encoder_state(model):
    prefixes = ("cls_token", "pos_embed", "patch_embed.", "blocks.", "norm.")
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if key.startswith(prefixes)}
