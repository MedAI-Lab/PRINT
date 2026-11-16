"""Fit the PRINT diagnostic module for one development fold."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def fit(features, labels, penalty):
    mean = features.mean(0)
    sd = features.std(0)
    sd[sd < 1e-8] = 1
    x = (features - mean) / sd

    def objective(w):
        logits = np.einsum("ij,j->i", x, w[:-1]) + w[-1]
        loss = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-abs(logits)))
        grad = (expit(logits) - labels) / len(labels)
        return (loss.mean() + penalty * np.dot(w[:-1], w[:-1]) / 2,
                np.r_[np.einsum("ij,i->j", x, grad) + penalty * w[:-1], grad.sum()])

    solution = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                        options={"maxiter": 500, "ftol": 1e-12})
    if not solution.success:
        raise RuntimeError(solution.message)
    return dict(mean=mean.tolist(), sd=sd.tolist(), coef=solution.x[:-1].tolist(),
                intercept=float(solution.x[-1]), lam=penalty)


def probability(head, x):
    return expit(np.einsum("ij,j->i", (x - head["mean"]) / head["sd"], head["coef"]) + head["intercept"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--workbook", type=Path)
    source.add_argument("--splits", type=Path)
    for name in ("cache", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(10))
    args = p.parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        x = data["factor_z"].astype(float)
        y = data["Label"].astype(float)
        ids = data["record_id"].astype(str).tolist()
        if int(data["factor_fold"]) != args.fold:
            raise ValueError("Feature fold mismatch")
    if args.workbook:
        from stage_tables import cohort, diagnostic_split
        records = cohort(args.workbook, "nodules", "record_id")
        if set(ids) != set(records.index) or not np.array_equal(y, records.loc[ids, "Label"]):
            raise ValueError("Cached IDs and labels must match the diagnostic workbook")
        split = diagnostic_split(records, args.fold)
    else:
        split, = [s for s in json.loads(args.splits.read_text())["folds"] if s["fold"] == args.fold]
    lookup = {value: index for index, value in enumerate(ids)}
    tr, va = ([lookup[i] for i in split[key]] for key in ("train_record_ids", "validation_record_ids"))
    if sorted(tr + va) != list(range(len(ids))) or len(lookup) != len(ids) or x.shape != (len(ids), 13):
        raise ValueError("Invalid split or features")
    ranked = []
    for penalty in (0.001, 0.01, 0.1):
        head = fit(x[tr], y[tr], penalty)
        auc = roc_auc_score(y[va], probability(head, x[va]))
        ranked.append((auc, penalty, head))
    best_auc, _, best = max(ranked, key=lambda item: (item[0], item[1]))
    with args.out.open("x") as f:
        json.dump(dict(fold=args.fold, head=best, validation_auc=best_auc), f, indent=2)

if __name__ == "__main__":
    main()
