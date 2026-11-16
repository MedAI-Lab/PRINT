# Construct reference protein representations

Provide these inputs:

- A log2-scale protein abundance workbook with missing values imputed. Put protein IDs in the first column and sample IDs in the column headers. Sample IDs are matched after replacing underscores with hyphens. Columns ending in `_rep` are excluded.
- A completed [Stage 2 workbook](../templates/stage2_paired.xlsx) with nodule IDs, binary labels and fold assignments.
- A Reactome pathway file in GMT format.

Under R 4.4.3, run `renv::restore()` from the `proteomics/` directory to install the versions in [renv.lock](renv.lock). Then run the following commands from the repository root. Set `PAIRED` to your Stage 2 workbook and use a fresh work directory.

```sh
mkdir work
python proteomics/prepare.py --matrix protein_Samplematrix_imputeNA.xlsx --workbook "$PAIRED" --out work/input
cp ReactomePathways.gmt work/input/
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript proteomics/fit_plier.R work
python proteomics/export_targets.py work/result work/rpr.csv
```

Export checks that the fitted factor identities match the PRINT reference.

The output `work/rpr.csv` contains one row per nodule and the 13 [reference protein representations](../print_model/factor_order.json). Pass it to Stage 2 with `--targets work/rpr.csv`; targets are matched by ID. These targets are fitted on the full paired cohort before image-model cross-validation.
