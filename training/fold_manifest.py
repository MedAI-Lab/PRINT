"""Describe fitted fold artifacts relative to a checkpoint root."""
import argparse
import hashlib
import json
from pathlib import Path


def describe(fold, directory, root):
    checkpoint = directory / "best_joint_checkpoint.pt"
    config = json.loads((Path(__file__).parent / "vpr_config.json").read_text())["architecture"]
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    geometry = json.loads((directory / "geometry.json").read_text())
    if geometry["outer_fold"] != fold:
        raise ValueError("Mismatched checkpoint geometry fold")
    marker = directory / "agate.json"
    if json.loads(marker.read_text()) != "no-beta":
        raise ValueError("Expected a PRINT VPR inference module trained without beta")
    return dict(agate="no-beta", fold=fold, checkpoint=str(checkpoint.resolve().relative_to(root.resolve())),
                sha256=digest, beta=None,
                geometry=geometry, config=config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Directory containing fold0 through fold9")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vpr-manifest", type=Path, help="Stage 2 manifest in the checkpoint root")
    a = p.parse_args()
    released = None
    if a.vpr_manifest:
        released = json.loads(a.vpr_manifest.read_text())
        if a.vpr_manifest.resolve().parent != a.root.resolve() or [s["fold"] for s in released] != list(range(10)):
            p.error("Supply a ten-fold Stage 2 manifest located in --root")
    specs = []
    for i in range(10):
        folder = a.root / f"fold{i}"
        spec = released[i] if released is not None else describe(i, folder, a.root)
        head = json.loads((folder / "head.json").read_text())
        if head["fold"] != i:
            raise ValueError("Diagnostic head fold mismatch")
        spec["head"] = head["head"]
        specs.append(spec)
    with a.out.open("x") as f:
        json.dump(specs, f, indent=2)

if __name__ == "__main__":
    main()
