"""Select the Stage 2 training epoch by nested cross-validation."""
import json
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def outer_folds(path):
    folds = json.loads(path.read_text())["folds"]
    if sorted(f["fold"] for f in folds) != list(range(10)):
        raise ValueError("Expected ten distinct outer folds")
    all_ids = [pid for f in folds for pid in f["validation_patient_ids"]]
    cohort = set(all_ids)
    if len(cohort) != len(all_ids):
        raise ValueError("Repeated outer holdout ID")
    for f in folds:
        train = f["train_patient_ids"]
        if len(train) != len(set(train)) or set(train) != cohort - set(f["validation_patient_ids"]):
            raise ValueError("Outer training IDs must be the complement of the holdout")
    return sorted(folds, key=lambda f: f["fold"])


def run(cli, train):
    from print_model.data_paths import study_input

    from stage_tables import cohort, paired_folds
    if getattr(cli, "workbook", None):
        cli.paired = cohort(cli.workbook, "paired", "ID")
        outer = paired_folds(cli.paired)[cli.fold]
    else:
        outer = outer_folds(cli.splits)[cli.fold]
    if cli.steps:
        train(cli, outer)
        return
    if cli.out.exists():
        raise FileExistsError(cli.out)
    settings = json.loads(Path(__file__).with_name("vpr_config.json").read_text())["parameters"]
    if cli.epochs < settings["selection_min_epoch"]:
        raise ValueError("Epoch budget is below the first eligible selection epoch")
    ids = np.asarray(outer["train_patient_ids"])
    clinical = cli.paired if hasattr(cli, "paired") else pd.read_excel(study_input(cli.data, "stage2_clinical.xlsx"), dtype={"ID": str}).set_index("ID", verify_integrity=True)
    labels = clinical.loc[ids, "Label"]
    if set(labels) != {0, 1} or labels.value_counts().min() < 5:
        raise ValueError("Five stratified inner folds require at least five cases per class")
    partitions = StratifiedKFold(5, shuffle=True, random_state=settings["seed"] + cli.fold)
    histories = []
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cli.out.parent) as scratch:
        for i, (fit, val) in enumerate(partitions.split(ids, labels)):
            split = dict(train_patient_ids=ids[fit].tolist(), validation_patient_ids=ids[val].tolist())
            inner = copy(cli)
            inner.out = Path(scratch) / str(i)
            train(inner, split, inner=True)
            rows = [json.loads(line) for line in (inner.out / "metrics.jsonl").read_text().splitlines()]
            histories.append({r["epoch"]: r["factor_pearson_mean"] for r in rows if r["checkpoint_eligible"]})
    common = sorted(set.intersection(*(set(h) for h in histories)))
    scores = {e: float(np.mean([h[e] for h in histories])) for e in common}
    if not scores or not np.isfinite(list(scores.values())).all():
        raise ValueError("No valid epoch shared by all five inner validation folds")
    epoch = max(common, key=scores.get)
    final = copy(cli)
    final.epochs = epoch
    train(final, outer)
    (cli.out / "selected.json").write_text(json.dumps(dict(epoch=epoch, split="inner_validation",
        metric="factor_pearson_mean", scores_by_epoch=scores, inner_fold_scores=histories), indent=2))
