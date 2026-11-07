# DINO-CoReSeg V2 — independent 448-resolution study

This is a separate project. It never imports code or environments from V1.
V1 code, caches, splits, checkpoints, predictions, histories and metrics are read-only.
Both projects access the same external dataset folders; datasets are not copied.

## Folder layout and paths

```text
/data1/ansh/Diwakar/November_Segmentation_research/
  DINO_CoReSeg/                completed V1, unchanged
  DINO_CoReSeg_V2/             this independent project
  dataset-1/
  dataset-2/Dataset/
  dataset-3/PolypDataset/
```

On the Mac, V1 is currently named `DINO_CoReSeg_GPU_100E`; V2 is beside it.
Do not rename V1. The server configuration defaults to `v1_root: ../DINO_CoReSeg`.
Override paths with `CORESEG_DATA_ROOT`, `CORESEG_V1_ROOT`, `--data-root`, or
`--v1-root`. Dataset root means the parent containing all three dataset folders.

V2 has its own `.venv`, `src/`, `configs/`, `vendor/`, `cache_v2/`, and `outputs/`.
Copied `common.py`, `runtime.py`, `metrics.py` and grouping helpers are local to V2.
`v1_source_snapshot/` records the pre-V2 local source hashes and source copies;
it is a reference snapshot, not evidence of the server's actual version.

## Preserve and inspect V1 before any V2 training

The server `prepare` stage reads actual V1 train/validation/test manifests,
exclusions, image identities, duplicate links, selected checkpoints, histories
and all five completed CV checkpoint markers. It refuses missing or mismatched
frozen checkpoints. It records whether `.git` exists, but source identity uses
SHA-256 hashes and the implementation identity saved in the checkpoint.

A V2-owned preservation manifest hashes every scientific file under the two V1
output directories, the original manifests/splits, source/config files and
available backbone identity/weights. Source copies and the selected validation
record are stored under `reports/v1_preservation/` inside the V2 experiment.
These checks may take time on large V1 prediction folders. Successful full
execution verifies the same V1 manifest again. No V1 scientific file is edited.

Actual source-collection use is saved as `reports/dataset_usage_report.csv`:

- Dataset-1 Kvasir-SEG and Sessile are provenance/subgroup references. They do
  not create extra training copies of images already represented in dataset-3.
  Additional linked-membership columns distinguish indirect exposure from
  direct use of a physical source collection. Source-image pHash comparisons,
  file hashes and Sessile membership are reported separately.
- Dataset-2 verifies 432 supplied-positive-mask pairs and 7,749 healthy frames
  before exclusions. Healthy targets are derived empty masks. Annotated
  overlays are excluded. Actual retained counts come from the frozen manifests.
- Dataset-3 verifies 1,450 unaugmented development pairs and the five official
  test collections (100 Kvasir, 62 ClinicDB, 380 ColonDB, 60 CVC-300, 196 ETIS).
  DA/OLDTR collections are counted from actual files and excluded initially.
- Patient IDs, duplicate components, file/decoded/normalized identities and
  detected image links are checked for development/test overlap. This verifies
  detected identities; it cannot establish the absence of every possible
  unknown patient or near-duplicate relationship in public datasets.

The server's actual frozen usage report is not available locally: SSH
noninteractive authentication was rejected. Run `prepare` on the server to
produce it. No substitute usage counts have been invented.

## Controlled experiment sequence

Run experiments one at a time, freezing architecture choices using development
validation. Do not rank variants using their official test scores.

| Config | Experiment ID | Change relative to preceding stage | Trainable parameters |
|---|---|---|---:|
| `v2_resolution_448.yaml` | `dino_coreseg_v2_448` | V1 decoder at input 448 / RGB 224; no new decoder modules/loss/schedule | 85,287 |
| `v2_rgb_path.yaml` | `dino_coreseg_v2_rgb_path` | Add a small RGB context path | 92,320 |
| `v2_boundary_refinement.yaml` | `dino_coreseg_v2_boundary_refinement` | Add coarse-to-fine residual refinement; boundary loss still zero-weighted | 106,594 |
| `v2_boundary_loss.yaml` | `dino_coreseg_v2_boundary_loss` | Add boundary-loss coefficient 0.1 to the same multiscale architecture | 106,594 |
| `v2_multilayer_dino.yaml` | `dino_coreseg_v2_multilayer` | Add normalized DINO layers 2,5,8,11 (zero-based), with lightweight learnable fusion | 180,902 |

The extra RGB-only and boundary-loss-only configs prevent introducing all
changes at once. The required three principal configs are also present.
No variants run automatically in sequence; the researcher chooses the next
stage using development validation and available resources.

All runs use the same frozen DINOv2-S/14 (22,056,576 frozen parameters), seed 42,
AdamW, initial LR 0.0003, weight decay 0.0001, 100 maximum epochs, no augmentation,
threshold 0.5 and V1's feasible-first normal-FPR checkpoint selection.
Early stopping is off initially; enable it only under a new experiment ID.
Cosine scheduling and minimum LR are configurable, but the initial run uses
constant LR to retain experimental control. LR is never GPU/batch-scaled.

The primary dimensions are derived from configuration: `448 / 14 = 32` patch
tokens along each axis, 384 channels, projection 64, eight prototypes and RGB
refinement 224. Model/loss code does not assume a 25-grid or 350-pixel input.
A configured **selection grid of 350** preserves V1's threshold/FPR monitoring
protocol; it is independent of V2's 448-pixel model output. Native final-test
metrics and clearly prefixed standardized-grid metrics are also separate.

Dataset-3 development images are already 352 × 352. Upsampling them to 448
increases token density but cannot recover image detail absent from those
source files. The resolution experiment does not silently replace them with
higher-resolution dataset-1 images.

## Architecture and mathematically specified losses

`resolution`: normalized final DINO patch map → 64-channel projection → initial
head. Cosine dissimilarity from eight training-normal prototypes enters signed
counter-evidence fusion (CEF). A lightweight RGB path and residual lesion
refinement (RPR) operate at configured RGB resolution. Predictions are bilinearly
restored to configured model resolution. At V1 dimensions, the resolution-only
model has identical state keys, parameters and forward behavior to the preserved
V1 decoder; this is tested.

Later variants add a half-resolution RGB context path, then coarse/fine residual
corrections. No component-size filtering removes candidate lesions. Multilayer
fusion projects intermediate normalized maps into the small shared latent space;
prototypes still use the final 384-channel DINO map. The official documented
`get_intermediate_layers(..., reshape=True, norm=True)` API is used. CLS/register
tokens are not segmentation features. The main transformer stays frozen.

Mask patch occupancy is exact nonoverlapping 14 × 14 average pooling. Background
eligibility additionally excludes a two-pixel-dilated lesion halo. Grid/feature
shapes must agree. Prototype vectors are unit-normalized; their training IDs,
resolution, feature identity and seed are recorded. V1 banks cannot be reused.

The optimization objective retains category-balanced positive BCE + soft Dice,
healthy BCE, contrast weight 0.1 and auxiliary weight 0.3. Small-positive weights
are clamped to [1,3] using the training-only area median. Boundary weight is zero
for the resolution/RGB/refinement controls.
V2 computes loss reductions in FP32 under mixed precision for numerical stability;
the baseline objective algebra is preserved, but AMP arithmetic is not bitwise
identical to V1's mixed-precision reductions.

For the separate boundary-loss experiment:

```text
edge(Y) = dilate_3(Y) - erode_3(Y), with zero-padded target boundaries
w_i = 1 + beta * max(1 - d_infinity(i, edge(Y)) / (R + 1), 0)
L_boundary = mean_over_positive_images [sum_i w_i * BCE(logit_i,Y_i) / sum_i w_i]
L = L_V1 + lambda_boundary * L_boundary
```

Square dilations implement the truncated Chebyshev-distance weights exactly.
Defaults: R=4 model pixels, beta=2, lambda=0.1 only in the boundary-loss variant.
All-healthy batches contribute differentiable zero boundary loss. This has no
undefined empty-mask division and uses targets only from the current development
batch during training/validation monitoring. No test targets fit any loss.

## GPU selection, installation and execution

Use **only GPU(s) allocated to you**. V2 does not automatically select all RTX
cards. Single GPU is the default; V1's DataParallel broadcast crashed on this
server, so initial V2 should retain a single allocated RTX 3090. Multiple GPUs
require an explicit mask plus `gpu.all_visible_devices: true` in a separately
identified configuration; verify communication and batch semantics first.

After uploading the V2 folder beside V1 and confirming the shared datasets:

```bash
cd /data1/ansh/Diwakar/November_Segmentation_research/DINO_CoReSeg_V2
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh install
./run_v2.sh test
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh check
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh prepare
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh smoke
```

Upload from the Mac with this single-line command (datasets are outside V2 and
are not transferred):

```bash
rsync -avP --exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='cache_v2/' --exclude='outputs/' --exclude='.mplcache/' --exclude='*.zip' /Users/prabhavkumar/Diwakar/November_Segmentation_research_/DINO_CoReSeg_V2/ ansh@192.168.51.15:/data1/ansh/Diwakar/November_Segmentation_research/DINO_CoReSeg_V2/
```

Alternatively, `scripts/build_bundle.py` creates `DINO_CoReSeg_V2.zip` inside
this project. Extract it beside V1. It excludes environments, datasets, weights,
caches and results; installation fetches the official weight file and checks
its SHA-256. `python scripts/verify_bundle.py` verifies the extracted source
manifest before installation. The Mac folder already contains a local copy of
the verified weights, so rsync can transfer those without downloading again.

The smoke/full pipeline first runs the unit suite in a CPU-only subprocess.
The smoke stage then extracts eight real development samples, checks the actual
448/32-grid decoder gradients via one training epoch, validates prototype fitting
and checkpoint restoration, and logs measured GPU memory/hardware. It is an
integration check, not a research result. Inspection/shape tests locally do not
establish CUDA compatibility; run these stages on the server first.

Only when the allocation is available and those checks pass:

```bash
# Audit/preserve, smoke, full development extraction, 100-epoch holdout,
# frozen final-test extraction/evaluation, latency, paper figures and report:
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh start
./run_v2.sh status
tail -f outputs/dino_coreseg_v2_448/logs/console.log

# Graceful stop, then after the process exits:
./run_v2.sh stop
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh resume
```

`nohup` survives closing SSH, not server restarts or scheduler allocation expiry.
Use scheduler jobs or `foreground` in tmux when required by server policy.
No full run was launched automatically on the shared server.

Independent stages and later controls:

```bash
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh extract
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh train
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh evaluate
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh smoke --config configs/v2_rgb_path.yaml
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh start --config configs/v2_rgb_path.yaml
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh resume --config configs/v2_rgb_path.yaml
```

The default study is **one holdout experiment**, not five-fold CV. Optional
V2 CV is explicit and expensive (five additional models):

```bash
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh start-cv
.venv/bin/python -m src.plot_paper --experiment dino_coreseg_v2_448_cv5
```

Holdout comparisons import the exact V1 frozen splits. CV imports the exact five
V1 fold manifests, verifies coverage/group separation, and copies them into V2.
It does not reconstruct membership from a seed or a different scikit-learn version.
Use `resume-cv` for a detached CV restart and `status` to check its launcher.

Automatic physical batch probing measures every selected GPU and keeps 30%
headroom plus optimizer-state reserve. Effective batch is microbatch × gradient
accumulation, recorded explicitly. Default accumulation is one. If accumulation
is enabled, contrastive pairs and positive-area normalization are microbatch
local; it is not mathematically identical to one larger physical batch. Changing
batch policy or GPU count requires a new experiment ID, not silent resume.

## Resume, logs and retention

`last.pt` is the durable optimizer-boundary commit: it includes model, optimizer,
scheduler, scaler, RNG, cursor, running loss sums, history, selected model, exact
configuration, source, cache and GPU identities. A stop during accumulation finishes
that optimizer group before committing. Hard kills can lose work since the most
recent commit (10 optimizer steps / 60 seconds by default). Monitor interruptions
repeat monitoring rather than reapplying committed SGD. Resume rejects changed
scientific/source/device/batch identity. Best checkpoint recovery uses the selected
model stored in the committed `last.pt`, avoiding partial multi-file updates.

Every epoch records raw optimization/monitor loss components, positive Dice/IoU,
recall, small-polyp Dice/recall, empty predictions, component lesion recall, normal
FPR, positive-image pixel accuracy, LR, epoch time, GPU peak allocation and selection
score/constraint. Predictive monitor segmentation loss is balanced across positive
and healthy categories over the complete partition. Monitor total combines that
loss with the actually computed batch-averaged auxiliary/contrastive terms and
the positive-image boundary mean. Optimization components separately retain the
actual sampled-batch objective. Missing metrics use JSON null / blank CSV fields. Raw epoch CSV
and JSONL histories are atomic and canonical; no missing epochs are synthesized.

Permanent layout:

```text
outputs/dino_coreseg_v2_448/
  config/       resolved paths, frozen splits, exact config, hardware, selection
  logs/         CSV/JSONL histories, extraction, progress, console/pipeline logs
  checkpoints/  prototypes, selected best, resumable last, FROZEN marker
  predictions/  full native binary masks + per-image metadata; selected probabilities
  evaluation/   resumable evaluation, per-image/per-dataset CSVs, ROC histograms
  figures/      final paper figures, captions and selection CSV
  reports/      V1 preservation, dataset use, checks, memory probes, one report
  smoke/        clearly labelled real-data integration artifacts
```

A counterfactual early-stopping analysis replays the validation-only rule in
`reports/early_stopping_analysis.json` without changing the actual run.

No per-epoch scientific checkpoint copies are kept by default. Optional
`checkpoint.keep_epoch_every` enables explicit archival intervals. Identical
feature preparations share one V2-only cache namespace; multilayer maps require
a different namespace. Temporary files are not valid committed artifacts.
An optional cleanup report lists V2-only candidates but **never deletes** files:

```bash
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh cleanup-dry-run
```

Explicit approval is required before any future deletion. V1, shared datasets,
shared caches, best/last checkpoints and active files are protected.

## Evaluation and paper figures

The frozen selected model is evaluated once on the six external partitions.
Report each dataset separately. Native Dice, IoU, precision, recall, F2, boundary
F1/IoU, HD95/ASSD and empty-prediction rate are saved per image. Standard-grid
metrics are prefixed `standard_`. Empty native predictions retain the explicit
image-diagonal surface-distance penalty; a valid-surface-only table and individual
failure list prevent interpreting this penalty as genuine boundary displacement.
Small-polyp strata use training-only native-area tertiles. Detection recall is
one-to-one matching of 8-connected GT/predicted components at IoU ≥0.1; it is not
pixel recall and does not constitute a clinical detection criterion.

Normal metrics include any-foreground frame FPR, component count, pixel area,
frame/pixel specificity and configured component gates. Patient-cluster bootstrap
intervals are provided for FAR where grouping is available. ROC uses all native
pixels with 65,536 score bins. It is quantized pixel ROC, not patient/image AUC.
Timing separates cached decoder, live backbone and image-to-native-mask inference.
The full end-to-end timer includes preprocessing and transfers, excluding model
loading; the first measured image is retained to match the V1 timer scope.

V1's previous test results motivated these hypotheses. Freeze V2 architecture
choices on validation before final evaluation, and disclose reuse of benchmark
sets when reporting the comparison. It is not a new prospective clinical test.

Plot without CUDA or retraining:

```bash
.venv/bin/python -m src.plot_paper --experiment dino_coreseg_v2_448
```

`training_curves.*` shows raw segmentation loss and positive-image pixel accuracy
(%). `training_curves_dice.*` shows raw segmentation loss and positive-image Dice.
One holdout run shows only two real curves. CV shows subtle per-fold curves,
prominent train/validation means and light SD bands on shared recorded epochs.
No missing epochs are filled, no raw loss is forced to [0,1], and no loss is
mislabelled as a percentage. PDF/SVG use vector text/lines; PNG uses 300 DPI.

The final `qualitative_segmentation.*` has Kvasir, ClinicDB, ColonDB, ETIS,
FAR-polyp and FAR-normal rows with original / GT / prediction / TP-FP-FN overlay.
Positive examples use predeclared median Dice except ETIS (lower quartile), and
normal uses median identifier. Masks/images stay in native coordinates and aspect
ratios. Selection CSV records IDs, checkpoint hashes, Dice/IoU and reasons. Normal
GT is derived empty and normal Dice/IoU remain missing. Original clinical images
are never false-colored; only the final error-overlay column uses TP/FP/FN colors.
Probability retention re-runs the same original inference batches for selected
samples and verifies exact agreement with committed binary masks.

Corrected/verified V1 reporting is written only inside V2:

```bash
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh report-v1
.venv/bin/python -m src.plot_paper --experiment dino_coreseg_gpu_100 --v1-root ../DINO_CoReSeg --data-root ..
```

The selected frozen epoch is matched explicitly to history. `best.pt` may already
have a history ending at that epoch, so the review does not presume V1 was wrong.
Historical V1 CSVs and predictions are never overwritten. Cross-grid validation/
external-test differences are labelled descriptive, not causal domain-gap effects.

## Technical references

- Official DINOv2 source/API: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
- Official DINOv2 source revision and license are in `vendor/dinov2/`.
- Scheduler/optimizer resume: https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html
- CUDA 12.6 preserves V100 and RTX support for the pinned PyTorch release:
  https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290

See `VALIDATION_V2.md` for actual validation performed and remaining server checks.
No V2 accuracy or generalization improvement is claimed before measured results.
