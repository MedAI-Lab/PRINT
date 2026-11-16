#!/usr/bin/env python3
"""Patient bag data for PRINT."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from print_model.data_paths import study_input
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT_RAW = os.environ.get("THYROID_DATA_ROOT")
DATA_ROOT = (
    (Path(_DATA_ROOT_RAW) if Path(_DATA_ROOT_RAW).is_absolute() else ROOT / _DATA_ROOT_RAW)
    if _DATA_ROOT_RAW else ROOT / "data"
)
from print_model.preprocessing import build_transforms


class PatientBag(Dataset):
    def __init__(self, frame, targets, training):
        self.frame = frame.reset_index(drop=True); self.targets = targets
        self.tf = build_transforms(use_grayscale=True, training=training)
        self.image_dir = study_input(DATA_ROOT, "stage2_ultrasound")
        self.counts = []
        for pid in self.frame.ID.astype(str): self.counts.append(len(sorted(self.image_dir.glob(f"{pid}_*.jpg"))))
        if not all(self.counts): raise RuntimeError("Stage 2 manifest contains patient without image")
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]; pid = str(row.ID).strip(); paths = sorted(self.image_dir.glob(f"{pid}_*.jpg"))
        images = [self.tf(Image.open(p).convert("L")) for p in paths]
        return images, self.targets[pid], int(row.Label), pid


def collate(batch):
    b = len(batch); nv = max(len(x[0]) for x in batch); shape = batch[0][0][0].shape
    images = torch.zeros((b, nv, *shape)); mask = torch.zeros((b, nv), dtype=torch.bool)
    for i, (views, _, _, _) in enumerate(batch):
        images[i, :len(views)] = torch.stack(views); mask[i, :len(views)] = True
    return images, mask, torch.stack([x[1] for x in batch]), torch.tensor([x[2] for x in batch]), [x[3] for x in batch]


def corr(a,b,rank=False):
    if np.std(a)<1e-8 or np.std(b)<1e-8:return 0.
    v=spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1]
    return float(v) if np.isfinite(v) else 0.


def metrics(pred,true):
    pc=[corr(pred[:,i],true[:,i]) for i in range(pred.shape[1])]
    sp=[corr(pred[:,i],true[:,i],True) for i in range(pred.shape[1])]; prof=[corr(pred[i],true[i]) for i in range(len(pred))]
    vr=np.var(pred,0)/(np.var(true,0)+1e-8)
    return {
      "factor_pearson_mean":float(np.mean(pc)),"factor_pearson_median":float(np.median(pc)),
      "factor_spearman_mean":float(np.mean(sp)),"factor_spearman_median":float(np.median(sp)),
      "profile_pcc_mean":float(np.mean(prof)),"profile_pcc_median":float(np.median(prof)),
      "mse_z":float(np.mean((pred-true)**2)),"mae_z":float(np.mean(abs(pred-true))),
      "variance_ratio_mean":float(np.mean(vr)),"variance_ratio_min":float(np.min(vr))}
