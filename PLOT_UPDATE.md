# Paper-style five-fold loss and accuracy figure

This update adds plotting code only. Model architecture, loss, configuration,
cached features, optimizer, sampling, checkpoints and training identity are
unchanged. It can be installed after the current server study completes and
generates figures directly from its saved `training_history.csv` files. No
retraining or GPU allocation is required by the standalone command.

## Generate after updating the server code

```bash
cd /data1/ansh/Diwakar/November_Segmentation_research/DINO_CoReSeg
.venv/bin/python -m src.paper_cv_plots --results-dir outputs/dino_coreseg_gpu_100_cv5 --max-epochs 100
```

The default writes both a percentage version and a 0–1 version:

```text
outputs/dino_coreseg_gpu_100_cv5/figures/
  04_cv_loss_accuracy_percent.png
  04_cv_loss_accuracy_percent.pdf
  04_cv_loss_accuracy_percent.svg
  04_cv_loss_accuracy_percent.tiff
  05_cv_loss_accuracy_0_1.png
  05_cv_loss_accuracy_0_1.pdf
  05_cv_loss_accuracy_0_1.svg
  05_cv_loss_accuracy_0_1.tiff
  cv_loss_accuracy_caption.txt
  cv_loss_accuracy_manifest.json
```

Each figure has two side-by-side panels with individual fold colours, solid
training curves, dashed validation curves, grid lines, two-column legends and
panel captions matching the requested layout. PNG/TIFF exports use 600 DPI;
PDF/SVG retain vector lines and text. TIFF uses LZW compression. The original
separate loss/accuracy/ROC figures remain available.

The updated standard CV plot routine also creates these figures at its existing
plot checkpoints. Partial runs use 300 DPI to reduce overhead; after all fold
evaluations complete, exports use 600 DPI. Loading this code into an already
running Python process is not automatic; use the standalone command later.

## What the curves mean

- Input columns: `epoch`, `train_loss_0_1`, `val_loss_0_1`, `train_accuracy`,
  `val_accuracy`. No model inference is performed.
- Normalized loss is the existing predictive monitor `L/(1+L)`; the percentage
  figure multiplies it by 100. This is not a percentage error or the composite
  optimization objective. Raw optimization losses remain in the input CSVs.
- Accuracy is macro per-image pixel accuracy on the 350 × 350 monitoring grid.
  The percentage version multiplies the saved fraction by 100.
- Validation curves use each fold's inner validation set. They are not the
  outer-held-out scores, final-test scores or lesion-detection sensitivity.
- Both axes display the full 0–100 or 0–1 range. No smoothing, forced curve
  shapes, per-fold normalization, interpolation or extrapolation is applied.
- Missing folds/epochs are not generated. A partial study produces only its
  recorded curves. `--max-epochs 100` sets the horizontal axis extent; it does
  not fill unfinished epochs. The JSON manifest records available epochs.

Options: `--scale percent`, `--scale unit`, `--scale both` (default), `--dpi 600`,
`--folds 5`. Omit `--max-epochs` to fit the x-axis to available measurements.
The command exits with an explanation if there are no histories, required
columns are missing, epochs are invalid, or measurements are nonfinite/outside
the valid fraction range.

## Local validation

All **10 dedicated plotting tests passed** in a temporary Python 3.14 environment
with Matplotlib 3.11.2. The server's pinned environment was not reinstalled or
executed locally. Training identity source files and the scientific YAML were
verified byte-for-byte against the previous committed revision.

The dedicated plotting tests use temporary fixtures, not research measurements.
They verify exact plotted values, ten fold curves, solid/dashed labels, partial
histories, full axis limits, invalid data handling, source preservation, all
export formats and command-line execution without importing PyTorch/CUDA.
No new scientific results or real-data preview are generated locally.
