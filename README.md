# PRINT

**PRINT (Protein Representation INference from Thyroid Ultrasound)** uses routine ultrasound to assess malignancy risk in follicular thyroid neoplasms. It learns from ultrasound images paired with tissue proteomics: protein abundance patterns linked to biological pathways provide 13 molecular targets for training.

For a new nodule, the **PRINT VPR inference module** estimates these patterns from ultrasound views alone and returns 13 virtual protein representations (VPRs). The **PRINT diagnostic module** uses this molecular profile to estimate the probability of malignancy.

- [Inference only](#inference-only) — extract VPRs or predict risk with a fitted diagnostic module.
- [Training](#training) — prepare cohort data and train one or more stages.

## Pretrained weights

To produce the released Stage 1 checkpoint, we initialized the encoder with the official URFM weights (pretrained on multi-organ ultrasound images) and continued self-supervised training with a masked autoencoder on 4,666 ultrasound images from 1,616 thyroid nodules (follicular adenomas and follicular carcinomas) in discovery cohort B.

| Stage | Weights | Download |
|---|---|---|
| Stage 1 · Adaptation to follicular thyroid neoplasms | Full epoch-150 masked autoencoder, including the encoder, mask token and pixel decoder (`full_mae_checkpoint.pt`). | [Download Stage 1 MAE](https://huggingface.co/Jimmyzhe/PRINT/resolve/c7c500b8371f5537b1f08340c32eb51a9bb90ef2/stage1/full_mae_checkpoint.pt) |
| Stage 2 · VPR reconstruction | Ten fold-specific VPR reconstruction models (`vpr_fold0.safetensors` through `vpr_fold9.safetensors`) and their ensemble manifest (`folds.json`). | [Download Stage 2 files](https://huggingface.co/Jimmyzhe/PRINT/tree/c7c500b8371f5537b1f08340c32eb51a9bb90ef2/stage2) |

The [weight file guide](weights/README.md) describes the files included in each download. To predict malignancy probabilities, fit the Stage 3 diagnostic parameters using the [training guide](training/README.md).

## Quick start

Run the commands below from the repository root. The reference environment for training and inference uses Linux x86-64, Python 3.12.3, PyTorch 2.8.0 with CUDA 12.8, and an NVIDIA RTX 5090 with a compatible driver. CUDA inference requires a GPU with BF16 support.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128"
```

## Inference only

### Stage 1 · Reuse the thyroid-adapted URFM backbone

Stage 1 adapts the URFM visual backbone to ultrasound images of follicular thyroid neoplasms through self-supervised training. This stage has no standalone inference step. Researchers can use the encoder from the [Stage 1 MAE checkpoint](#pretrained-weights) to initialize and fine-tune models for their own ultrasound studies.

### Stage 2 · Extract VPRs from ultrasound images

Download the Stage 2 files into one directory. Supply the selected ultrasound views of one nodule in order:

```sh
python inference/predict.py --vpr-only --checkpoints /path/to/stage2 view1.jpg view2.jpg
```

The command returns 13 standardized VPRs per fold. It requires no labels, cohort workbook or diagnostic parameters.

### Stages 2 and 3 · Predict nodule malignancy risk

After fitting Stage 3, set `PRINT_WEIGHTS` to the training directory containing the diagnostic parameters in `folds.json` and the ten Stage 2 checkpoints it references.

```sh
export PRINT_WEIGHTS=/path/to/weights
python inference/predict.py --device cuda --checkpoints "$PRINT_WEIGHTS" /path/to/nodule/view1.jpg /path/to/nodule/view2.jpg
```

When starting from the Stage 2 weight package, pass `--manifest "$PRINT_WEIGHTS/diagnostic_folds.json"` to use the manifest generated in Stage 3. Supply the paths to the selected views of one nodule. The [inference guide](inference/README.md) describes preprocessing and the JSON output.

To evaluate predictions on a labeled cohort, supply a CSV with columns `ID`, `label` and `probability` to `evaluation/evaluate.py` and specify the development-set threshold with `--threshold`.

## Training

### Data preparation

| Starting point | Tables to prepare | Other inputs |
|---|---|---|
| Stage 1 encoder adaptation | `stage1_ids.xlsx` (one text `ID` per nodule) | Stage 1 ultrasound images and URFM initialization |
| Stage 2 with existing RPR targets | [Paired workbook](templates/stage2_paired.xlsx) with the 13 target columns filled | Paired ultrasound images and a full Stage 1 MAE checkpoint |
| Stage 2 from proteomic measurements | Paired workbook with ID, label and fold; protein abundance matrix | Paired images, Stage 1 MAE and Reactome GMT |
| Stage 3 diagnostic fitting | [Diagnostic workbook](templates/stage3_diagnostic.xlsx) with nodule and view sheets | Development images and ten trained or downloaded Stage 2 models |
| Full workflow from proteomic measurements | Stage 1 ID workbook, paired workbook, diagnostic workbook and protein abundance matrix | Images for each stage, URFM initialization and Reactome GMT |

### Run training

For a full run, follow [protein target construction](proteomics/README.md) and the [three training stages](training/README.md). Run the commands in sequence. Stage 2 reads the generated target CSV directly, Stage 3 reads the Stage 2 checkpoints, and risk prediction reads the fitted diagnostic manifest. With existing RPR targets in the paired workbook, skip protein target construction.

Protein target construction uses R 4.4.3.

The bundled [URFM source](print_model/runtime/third_party/URFM/LICENSE) and [PLIER package](proteomics/vendor/COPYING) include their respective licenses.

## License

Original PRINT source code is licensed under the [Apache License, Version 2.0](LICENSE). Third-party components retain their respective licenses. The Stage 1 MAE and Stage 2 VPR reconstruction weights are licensed under the [MIT License](https://huggingface.co/Jimmyzhe/PRINT/blob/c7c500b8371f5537b1f08340c32eb51a9bb90ef2/LICENSE). See the [weight provenance notice](https://huggingface.co/Jimmyzhe/PRINT/blob/c7c500b8371f5537b1f08340c32eb51a9bb90ef2/NOTICE.md) for URFM attribution.

## Declaration

PRINT is currently intended for research use only. Any clinical use must comply with the applicable laws and regulations of the country or region where the model is used.
