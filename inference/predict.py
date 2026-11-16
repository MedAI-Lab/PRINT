"""Predict PRINT risk from one or more preselected ultrasound views of one nodule."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from print_model.network import predict_fold, load_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--vpr-only", action="store_true", help="Output the 13 VPRs for each fold")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("images", nargs="+", type=Path, help="All views must belong to the same nodule")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA inference requires BF16 support")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    images = load_views(args.images).to(args.device)
    manifest = args.manifest or args.checkpoints / "folds.json"
    specs = json.loads(manifest.read_text())
    if not args.vpr_only and any("head" not in spec for spec in specs):
        parser.error("Risk prediction requires Stage 3 diagnostic parameters. Supply the manifest from training/fold_manifest.py.")
    names = specs[0]["geometry"]["v14_geometry"]["factor_names"]
    if [s["fold"] for s in specs] != list(range(10)) or any(s["geometry"]["v14_geometry"]["factor_names"] != names for s in specs):
        raise ValueError("Invalid fold manifest")
    folds = [predict_fold(spec, args.checkpoints, images, vpr_only=args.vpr_only) for spec in specs]
    result = dict(views=len(args.images), factor_order=names, folds=folds)
    if not args.vpr_only:
        result["risk"] = float(np.mean([f["risk"] for f in folds]))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
