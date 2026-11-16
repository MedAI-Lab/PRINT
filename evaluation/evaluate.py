"""Evaluate one row per nodule at a development-set threshold."""
import argparse
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("predictions", help="CSV with ID, label and probability")
parser.add_argument("--threshold", required=True, type=float)
args = parser.parse_args()
data = pd.read_csv(args.predictions, dtype={"ID": str})
if (data.ID.isna().any() or not data.ID.is_unique or set(data.label) != {0, 1}
        or not data.probability.between(0, 1).all() or not 0 <= args.threshold <= 1):
    raise ValueError("Expected unique IDs, both binary labels, valid probabilities and threshold")
tn, fp, fn, tp = map(int, confusion_matrix(data.label, data.probability >= args.threshold).ravel())
print(json.dumps(dict(n=len(data), threshold=args.threshold, auroc=roc_auc_score(data.label, data.probability),
                      tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=tp/(tp+fn), specificity=tn/(tn+fp)), indent=2))
