# PRINT model weights

Download the Stage 1 MAE and Stage 2 VPR models from [Pretrained weights](../README.md#pretrained-weights).

| File | Contents |
|---|---|
| `full_mae_checkpoint.pt` | Complete Stage 1 masked autoencoder saved at epoch 150 after adapting the ultrasound-pretrained URFM encoder to follicular thyroid neoplasms. Includes the image encoder, mask token and pixel decoder. Use this file as `--mae` for Stage 2 training. |
| `vpr_fold0.safetensors` through `vpr_fold9.safetensors` | Fold-specific parameters for reconstructing the 13 VPRs from ultrasound views. |
| `folds.json` | Fold identifiers, checkpoint paths and SHA-256 digests, architecture configuration and target geometry. |
| `SHA256SUMS` | SHA-256 digests for the manifest and model files. |

## Export the weights

From the repository root, set `MAE` to the full Stage 1 checkpoint and `RUN` to the Stage 2 [training output](../training/README.md) containing `fold0/` through `fold9/`. Choose a new output directory.

```sh
export PRINT_WEIGHTS=/path/to/exported_weights
python weights/export.py --checkpoints "$RUN" --mae "$MAE" --out "$PRINT_WEIGHTS"
(cd "$PRINT_WEIGHTS" && sha256sum -c SHA256SUMS)
```

To extract fold-specific VPRs for one nodule, run `python inference/predict.py --vpr-only --checkpoints "$PRINT_WEIGHTS" view1.jpg view2.jpg`. To estimate malignancy risk, fit the Stage 3 diagnostic models and build their ensemble manifest as described in the [training guide](../training/README.md).

## License

The Stage 1 MAE and Stage 2 VPR reconstruction weights are licensed under the [MIT License](https://huggingface.co/Jimmyzhe/PRINT/blob/c7c500b8371f5537b1f08340c32eb51a9bb90ef2/LICENSE). See the [weight provenance notice](https://huggingface.co/Jimmyzhe/PRINT/blob/c7c500b8371f5537b1f08340c32eb51a9bb90ef2/NOTICE.md) for URFM attribution.
