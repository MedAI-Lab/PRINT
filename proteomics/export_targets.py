"""Export the ordered, fitted patient-by-factor matrix."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fit", type=Path, help="PLIER result directory")
parser.add_argument("output", type=Path)
args = parser.parse_args()
names = json.loads((Path(__file__).resolve().parents[1] / "print_model/factor_order.json").read_text())
dictionary = pd.read_csv(args.fit / "factor_dictionary.csv")
named = dictionary.loc[~dictionary.PLIER_name.str.fullmatch(r"LV\s*\d+"), "factor"].tolist()
if named != names:
    raise ValueError("Fitted factor identities differ from the PRINT reference")
values = pd.read_csv(args.fit / "B_all.csv", dtype={"ID": str}, keep_default_na=False).set_index("ID", verify_integrity=True)[names]
if not np.isfinite(values).all().all():
    raise ValueError("Nonfinite RPR")
values.to_csv(args.output)
