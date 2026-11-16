"""Extract fold-specific VPRs for a development or evaluation manifest."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from print_model.network import load_fold, load_views, factor_batch

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--trained-fold", type=Path, help="Use a newly trained fold directory")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", type=Path)
    source.add_argument("--workbook", type=Path)
    for name in ("images", "checkpoints", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(10))
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    a = p.parse_args()
    from stage_tables import diagnostic_records
    rows = diagnostic_records(a.workbook, a.images) if a.workbook else json.loads(a.records.read_text())
    ids = np.array([r["record_id"] for r in rows], dtype=str)
    if not len(rows) or len(set(ids)) != len(ids):
        raise ValueError("Expected unique record IDs")
    if a.trained_fold:
        from fold_manifest import describe
        spec = describe(a.fold, a.trained_fold, a.checkpoints)
    else:
        manifest = a.manifest or a.checkpoints / "folds.json"
        spec, = [s for s in json.loads(manifest.read_text()) if s["fold"] == a.fold]
    model = load_fold(spec, a.checkpoints, a.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    values = []
    for start in range(0, len(rows), 16):
        chunk = rows[start:start + 16]
        views = [load_views([a.images / path for path in r["image_paths"]])[0] for r in chunk]
        batch = pad_sequence(views, batch_first=True).to(a.device)
        counts = torch.tensor([len(v) for v in views], device=a.device)
        mask = torch.arange(batch.shape[1], device=a.device)[None] < counts[:, None]
        values.append(factor_batch(model, batch, mask))
    with a.out.open("xb") as f:
        np.savez_compressed(f, record_id=ids, Label=[r["Label"] for r in rows],
                            factor_fold=a.fold, factor_z=np.concatenate(values))

if __name__ == "__main__":
    main()
