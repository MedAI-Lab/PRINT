"""Combine the ten held-out predictions on the original RPR scale."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nested_cv import outer_folds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--workbook", type=Path)
    source.add_argument("--splits", type=Path)
    a = p.parse_args()
    from stage_tables import cohort, paired_folds
    folds = paired_folds(cohort(a.workbook, "paired", "ID")) if a.workbook else outer_folds(a.splits)
    factors = json.loads((Path(__file__).resolve().parents[1] / "print_model/factor_order.json").read_text())
    frames = []
    for fold in folds:
        i = fold["fold"]
        frame = pd.read_csv(a.root / f"fold{i}/oof.csv", dtype={"ID": str})
        if frame.ID.tolist() != fold["validation_patient_ids"] or not frame.fold.eq(i).all():
            raise ValueError(f"OOF membership mismatch for fold {i}")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    scores = {}
    for f in factors:
        values = table[["pred_" + f, "true_" + f]]
        if not np.isfinite(values).all().all() or (values.nunique() < 2).any():
            raise ValueError(f"Invalid pooled OOF values for {f}")
        scores[f] = float(values.iloc[:, 0].corr(values.iloc[:, 1]))
    table.to_csv(a.out, index=False, mode="x")
    print(json.dumps(dict(n=len(table), per_factor=scores, mean_pearson=float(np.mean(list(scores.values())))), indent=2))


if __name__ == "__main__":
    main()
