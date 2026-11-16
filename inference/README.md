# Extract VPRs and predict malignancy risk

Run the commands from the repository root after [installing PRINT](../README.md#quick-start). For VPR extraction alone, run `python inference/predict.py --vpr-only --checkpoints /path/to/stage2 view1.jpg view2.jpg` with the ten published Stage 2 checkpoints and `folds.json`. The JSON contains `views`, `factor_order` and a VPR vector for each fold.

For malignancy risk, set `PRINT_WEIGHTS` to the Stage 3 training output directory containing `folds.json`. The manifest identifies the ten checkpoints, their SHA-256 digests and their paired diagnostic parameters.

```sh
export PRINT_WEIGHTS=/path/to/weights
python inference/predict.py --device cuda --checkpoints "$PRINT_WEIGHTS" view1.jpg view2.jpg
```

## Images

Supply selected, cropped views of **one nodule** per invocation, in acquisition or dataset order. JPEG and PNG are supported. PRINT converts each view to grayscale, resizes it to 224 × 224 pixels, expands it to three channels and standardizes each channel using its spatial mean and sample standard deviation. GPU inference uses BF16 autocast.

## Output

The command writes one JSON object per nodule to standard output.

| Mode | Top-level fields | Fields in each entry of `folds` |
|---|---|---|
| VPR extraction (`--vpr-only`) | `views`, `factor_order`, `folds` | `fold`, `vpr` |
| Malignancy risk prediction (default) | `views`, `factor_order`, `folds`, `risk` | `fold`, `vpr`, `risk` |

`views` is the number of input views. Each entry in `folds` identifies the fold (0–9) and contains its 13 standardized VPRs in `factor_order`. Each VPR is averaged over predictions from the original and horizontally flipped views.

In risk prediction mode, each fold’s `risk` is the malignancy probability from its paired diagnostic model. The top-level `risk` is the arithmetic mean of the ten fold probabilities.

Keep the fold index with each VPR vector because standardization is fold-specific. To use a manifest in another location, supply `--manifest /path/to/folds.json`; checkpoint paths are resolved relative to `--checkpoints`.
