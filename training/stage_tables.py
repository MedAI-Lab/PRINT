"""Read the paired and diagnostic cohort workbooks."""
from pathlib import Path

import numpy as np
import pandas as pd


def cohort(path, sheet, key):
    frame = pd.read_excel(path, sheet_name=sheet, dtype={key: str}, keep_default_na=False)
    ids = frame[key].str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{sheet} requires unique, nonempty {key} values")
    frame[key] = ids
    for column, allowed in (("Label", range(2)), ("fold", range(10))):
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.isin(allowed).all():
            raise ValueError(f"Invalid {column} in {sheet}")
        frame[column] = values.astype(int)
    if set(frame.fold) != set(range(10)):
        raise ValueError(f"{sheet} must contain all ten folds")
    return frame.set_index(key, drop=False)


def paired_folds(frame):
    ordered = frame.sort_index()
    return [dict(fold=i, train_patient_ids=ordered.index[ordered.fold != i].tolist(),
                 validation_patient_ids=ordered.index[ordered.fold == i].tolist()) for i in range(10)]


def paired_targets(frame, names, csv):
    if csv is not None:
        target = pd.read_csv(csv, dtype={"ID": str}, keep_default_na=False).set_index("ID", verify_integrity=True)
        if set(target.index) != set(frame.index):
            raise ValueError("RPR target IDs must match the paired workbook")
    else:
        target = frame
    missing = set(names) - set(target.columns)
    if missing:
        raise ValueError(f"Missing RPR columns: {', '.join(sorted(missing))}")
    values = target.loc[frame.index, names].apply(pd.to_numeric)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Supply finite values for all 13 RPRs or pass --targets")
    return values


def diagnostic_records(path, images):
    nodules = cohort(path, "nodules", "record_id")
    views = pd.read_excel(path, sheet_name="views", dtype={"record_id": str, "image_path": str},
                          keep_default_na=False)
    views["record_id"] = views.record_id.str.strip()
    order = pd.to_numeric(views.view_order)
    if not (order.ge(1) & order.mod(1).eq(0)).all():
        raise ValueError("view_order must contain positive integers")
    views["view_order"] = order.astype(int)
    if views.duplicated(["record_id", "view_order"]).any() or set(views.record_id) != set(nodules.index):
        raise ValueError("Each nodule needs views with distinct view_order values")
    groups = views.sort_values("view_order").groupby("record_id")
    rows = []
    for key, row in nodules.iterrows():
        paths = groups.get_group(key).image_path.tolist()
        for name in paths:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not (images / relative).is_file():
                raise ValueError(f"Expected an existing relative image path for {key}: {name}")
        rows.append(dict(record_id=key, Label=int(row.Label), image_paths=paths))
    return rows


def diagnostic_split(records, fold):
    return dict(train_record_ids=sorted(records.index[records.fold != fold]),
                validation_record_ids=records.index[records.fold == fold].tolist())
