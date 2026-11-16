"""Export the Stage 1 masked autoencoder and Stage 2 VPR reconstruction models."""
import argparse
import hashlib
import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from fold_manifest import describe


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", "--experiments", dest="checkpoints", required=True, type=Path,
                        help="Stage 2 output directory containing fold0 through fold9")
    for key in ("mae", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temporary:
        bundle = Path(temporary) / "weights"
        bundle.mkdir()
        export(args, bundle)
        bundle.rename(args.out)


def export(args, bundle):
    mae = torch.load(args.mae, map_location="cpu", mmap=True, weights_only=True)
    state = mae["model"]
    if mae["epoch"] != 150 or not {"mask_token", "decoder_pred.weight", "patch_embed.proj.weight"} <= state.keys():
        raise ValueError("Expected a full epoch-150 Stage 1 MAE with encoder and pixel decoder")
    torch.save({"model": state, "epoch": 150}, bundle / "full_mae_checkpoint.pt")
    del mae, state
    specs = [describe(i, args.checkpoints / f"fold{i}", args.checkpoints) for i in range(10)]
    for spec in specs:
        source = args.checkpoints / spec["checkpoint"]
        if digest(source) != spec["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch for fold {spec['fold']}")
        model = torch.load(source, map_location="cpu", mmap=True, weights_only=False)["model"]
        state = {k: v for k, v in model.items() if k != "mask_token" and not k.startswith("decoder_")}
        path = bundle / f"vpr_fold{spec['fold']}.safetensors"
        save_file(state, path)
        spec.update(checkpoint=path.name, sha256=digest(path))
        del state, model
    (bundle / "folds.json").write_text(json.dumps(specs, indent=2) + "\n")
    files = sorted(bundle.iterdir())
    (bundle / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))


if __name__ == "__main__":
    main()
