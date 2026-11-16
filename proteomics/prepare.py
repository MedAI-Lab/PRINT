"""Select discovery samples and differentially expressed proteins from supplier log2 data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import ttest_ind

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workbook", type=Path)
    source.add_argument("--labels", type=Path)
    for name in ("matrix", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.workbook:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
        from stage_tables import cohort
        labels = cohort(args.workbook, "paired", "ID").Label.sort_index()
    else:
        labels = pd.read_csv(args.labels).set_index("ID", verify_integrity=True).Label.sort_index()
    if set(labels) != {0, 1}:
        raise ValueError("Expected binary pathology labels")
    book = openpyxl.load_workbook(args.matrix, read_only=True, data_only=True)
    try:
        sheet = book.active
        sheet.reset_dimensions()
        rows = sheet.values
        header = next(rows)
        proteins = pd.DataFrame(rows, columns=header).dropna(how="all").set_index(header[0], verify_integrity=True)
    finally:
        book.close()
    primary = [c for c in proteins if not str(c).endswith("_rep")]
    mapping = {str(c).replace("_", "-"): c for c in primary}
    if len(mapping) != len(primary):
        raise ValueError("Ambiguous primary sample names after ID normalization")
    data = proteins[[mapping[i] for i in labels.index]].T.astype(float)
    data.index = labels.index
    if not np.isfinite(data.values).all():
        raise ValueError("Expected finite supplier-imputed values")
    data = data.loc[:, data.std() >= 1e-8]
    malignant, benign = (data.loc[labels == value].to_numpy() for value in (1, 0))
    p = ttest_ind(malignant, benign, axis=0, equal_var=False).pvalue
    fc = np.log2(np.exp2(malignant).mean(0) / np.exp2(benign).mean(0))
    selected = (p < 0.05) & (abs(fc) > np.log2(1.5))
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / "differential.csv"
    data.loc[:, selected].to_csv(output)
    def digest(path):
        with path.open("rb") as fp:
            return hashlib.file_digest(fp, "sha256").hexdigest()
    manifest = dict(supplier_sha256=digest(args.matrix), differential_features=int(selected.sum()),
                    hashes={output.name: digest(output)})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
