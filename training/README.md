# Train PRINT

Run the commands in Bash from the repository root in the [Linux GPU environment](../README.md#quick-start). Set the shell variables below to your input paths. Use a new directory for `RUN`. For example, set `DATA="/path/to/data"` in Bash, replacing the example path with your own. In the commands below, `"$DATA"` expands to that path, and `$fold` takes the values 0 through 9 in the training loops.

| Variable | Contents |
|---|---|
| `DATA` | Ultrasound images and clinical spreadsheets described below |
| `URFM_INIT` | Official URFM checkpoint pretrained on multi-organ ultrasound images |
| `PAIRED` | Completed [Stage 2 workbook](../templates/stage2_paired.xlsx) |
| `DIAGNOSTIC` | Completed [Stage 3 workbook](../templates/stage3_diagnostic.xlsx) |
| `IMAGES` | Root directory for the image paths in the Stage 3 workbook |
| `RUN` | Output directory for the ten-fold ensemble |

## Stage 1 · Adapt URFM to ultrasound images of follicular thyroid neoplasms

Under `DATA`, provide `stage1_ids.xlsx` and the `stage1_ultrasound/` image directory. List one unique nodule ID per row as text in the `ID` column, in dataset order. Stage 1 uses masked image reconstruction without diagnostic labels or protein targets. The epoch-150 checkpoint from the 300-epoch learning-rate schedule initializes each VPR model.

```sh
python training/adapt_encoder.py --data "$DATA" --init "$URFM_INIT" --out "$RUN/mae"
export MAE="$RUN/mae/full_mae_checkpoint.pt"
```

If you already have the full domain-adapted MAE checkpoint, set `MAE` to its path and start below.

## Stage 2 · Train the PRINT VPR reconstruction model

Put JPEG views named `ID_view.jpg` under `$DATA/stage2_ultrasound/`. In the `paired` sheet, enter one nodule per row with its text `ID`, binary pathology `Label` (0 benign, 1 malignant), and outer holdout `fold` (0–9). Enter the 13 molecular targets in the named columns. If starting from proteomic measurements, leave those columns blank, use the same workbook in the [protein target workflow](../proteomics/README.md), and add `--targets work/rpr.csv` to the training command. External targets are matched by ID.

Within each of the ten outer training partitions, PRINT constructs five malignancy-stratified inner folds and fits target scaling using the training data for each fold. Starting at epoch 70, it selects the eligible epoch with the highest Pearson correlation averaged equally over the 13 representations and five inner folds. Ties retain the earlier epoch. The outer model starts from the same MAE checkpoint and trains on the entire outer training partition for the selected number of epochs.

```sh
for fold in {0..9}; do
  python training/train_vpr.py --data "$DATA" --workbook "$PAIRED" \
    --mae "$MAE" --fold "$fold" --out "$RUN/fold$fold" || exit 1
done
```

Each fold writes a checkpoint, the selected epoch and held-out predictions. To collect the predictions on a common RPR scale and compute pooled per-representation correlations, run:

```sh
python training/collect_oof.py --root "$RUN" --workbook "$PAIRED" --out "$RUN/oof.csv"
```

## Stage 3 · Fit the PRINT diagnostic module

In the `nodules` sheet, enter one row per nodule with its text `record_id`, binary `Label`, and diagnostic holdout `fold` (0–9). In `views`, enter one row per image with its `record_id`, positive integer `view_order` (unique within the nodule), and `image_path` relative to `IMAGES`. Keep nodule rows in dataset order. Each diagnostic validation fold needs both labels for AUROC.

For each fold, freeze the fitted VPR module and use it to extract features from the development images. The diagnostic partition is defined separately from the paired-cohort partition. In each diagnostic fold, the program fits standardization and L2-regularized logistic regression on the other nine folds. It selects the penalty (0.001, 0.01 or 0.1) by validation AUROC, choosing the larger penalty for ties.

```sh
for fold in {0..9}; do
  python training/extract_features.py --fold "$fold" --workbook "$DIAGNOSTIC" --images "$IMAGES" \
    --checkpoints "$RUN" --trained-fold "$RUN/fold$fold" --out "$RUN/features$fold.npz" || exit 1
  python training/train_diagnostic_head.py --fold "$fold" --cache "$RUN/features$fold.npz" \
    --workbook "$DIAGNOSTIC" --out "$RUN/fold$fold/head.json" || exit 1
done
python training/fold_manifest.py --root "$RUN" --out "$RUN/folds.json"
```

The manifest pairs each VPR checkpoint with its diagnostic parameters. Use `RUN` as the checkpoint directory for [inference](../inference/README.md). The [weight export guide](../weights/README.md) describes how to package Stages 1 and 2.

### Fit Stage 3 from the Stage 2 weight package

Set `RUN` to the directory containing the downloaded `folds.json` and ten Stage 2 checkpoints. Run the following commands with your diagnostic workbook and images:

```sh
for fold in {0..9}; do
  mkdir -p "$RUN/fold$fold"
  python training/extract_features.py --fold "$fold" --workbook "$DIAGNOSTIC" --images "$IMAGES" \
    --checkpoints "$RUN" --out "$RUN/features$fold.npz" || exit 1
  python training/train_diagnostic_head.py --fold "$fold" --cache "$RUN/features$fold.npz" \
    --workbook "$DIAGNOSTIC" --out "$RUN/fold$fold/head.json" || exit 1
done
python training/fold_manifest.py --root "$RUN" --vpr-manifest "$RUN/folds.json" --out "$RUN/diagnostic_folds.json"
```

Pass `--checkpoints "$RUN" --manifest "$RUN/diagnostic_folds.json"` to the inference command.
