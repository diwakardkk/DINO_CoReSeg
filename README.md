# DINO-CoReSeg — V1 results and independent V2 study

V1 completed five-fold cross-validation and a separate holdout run, with **100 epochs per model**. V2 is an independent, higher-resolution study in [`DINO_CoReSeg_V2/`](DINO_CoReSeg_V2/); its performance results are not yet available.

## Completed V1 results

The table below transcribes the completed server's `holdout/metrics/final_metrics.csv` supplied by the project owner. Positive-row values are **means across images at native resolution**, not pooled pixel ratios. There are **2,004 final-test images: 848 polyp-positive and 1,156 healthy**. The selected holdout checkpoint is epoch **60**, with validation Dice **0.8326059** and validation healthy-frame FPR **0.0811496**. These validation figures come from the owner's checkpoint report.

| Final-test partition | Target | Images | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | F2 ↑ | HD95 ↓ (px) | ASSD ↓ (px) | Healthy-frame FPR ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kvasir | Polyp | 100 | 0.8617 | 0.7925 | 0.8949 | 0.8557 | 0.8557 | 73.22 | 36.95 | — |
| CVC-ClinicDB | Polyp | 62 | 0.8615 | 0.7886 | 0.8780 | 0.8669 | 0.8622 | 22.62 | 8.61 | — |
| CVC-300 | Polyp | 60 | 0.7994 | 0.6935 | 0.9210 | 0.7533 | 0.7639 | 24.21 | 9.70 | — |
| CVC-ColonDB | Polyp | 380 | 0.6503 | 0.5597 | 0.7663 | 0.6246 | 0.6300 | 134.55 | 106.98 | — |
| ETIS-LaribPolypDB | Polyp | 196 | 0.5943 | 0.5217 | 0.6464 | 0.5984 | 0.5928 | 406.70 | 333.50 | — |
| FAR | Polyp | 50 | 0.6616 | 0.5956 | 0.7195 | 0.6635 | 0.6610 | 162.97 | 132.22 | — |
| FAR | Healthy | 1,156 | — | — | — | — | — | — | — | 0.06574 (6.57%) |

Healthy-frame FPR means any predicted foreground on a healthy frame at the saved threshold; normal-frame specificity is **0.93426 (93.43%)**. Normal-image Dice/IoU are intentionally undefined here. Native HD95/ASSD depend on image dimensions and include V1's empty-prediction penalty; they are not physical distances in millimetres. V1 missed the entire lesion mask on 11.84% of ColonDB positives, 17.86% of ETIS positives and 14.00% of FAR positives. High pixel accuracy alone should not be interpreted as high lesion recall.

Across the five held-out CV folds, positive Dice was **0.81392 ± 0.01809**, IoU **0.73742 ± 0.01934**, native-pixel AUC **0.98078 ± 0.00905**, and healthy-frame FPR **0.09254 ± 0.01136** (fold mean ± standard deviation, not confidence intervals). Pooled held-out native-pixel AUC was **0.98024**. These development CV results are separate from the final-test table and are not patient-level AUC.

The reported completion markers confirm all five folds and the 2,004-image holdout evaluation finished. Rounded table entries and full-precision machine-readable transcriptions are in [`results/v1/`](results/v1/). Raw checkpoints, private per-image records, predictions and datasets are **not uploaded to GitHub**. These summaries are transcribed from owner-supplied server output; the completed server artifacts were not independently downloaded in this update.

## Independent V2 code

See [`DINO_CoReSeg_V2/README_V2.md`](DINO_CoReSeg_V2/README_V2.md) for installation, separate experiment configurations, shared dataset paths, resume, native-resolution evaluation, ROC and paper figures. V2 keeps all generated caches, checkpoints, logs and results inside its own folder. V1 scientific code, configurations and results remain unchanged.

The first V2 experiment uses frozen DINOv2-S/14 with **448 × 448 input, a 32 × 32 patch grid, 384 feature channels, 64 projection channels, eight prototypes and RGB refinement at 224 × 224**. Resolution-only, RGB context, multiscale refinement, boundary loss and multilayer DINO changes have separate versioned configurations. The default is a 100-epoch holdout run; five-fold CV is a separate explicit command.

Validation status: **36 CPU tests passed on the server**, a CUDA kernel check passed, and the real-data CUDA smoke stage reported `smoke_complete`. The owner subsequently launched V2; the latest supplied status was `preserving_v1`. **No completed V2 training or performance result is claimed.**

For the current server, keep the deployed sibling layout:

```text
November_Segmentation_research/
  DINO_CoReSeg/       completed V1
  DINO_CoReSeg_V2/    independent V2
  dataset-1/
  dataset-2/
  dataset-3/
```

The GitHub checkout contains V1 source at its root and V2 source in a subfolder. When running V2 directly from that nested checkout, explicitly set the paths to the completed V1 folder and the shared dataset parent:

```bash
cd DINO_CoReSeg_V2
export CORESEG_V1_ROOT=/data1/ansh/Diwakar/November_Segmentation_research/DINO_CoReSeg
export CORESEG_DATA_ROOT=/data1/ansh/Diwakar/November_Segmentation_research
# Example GPU index only; use the allocation assigned to you.
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh install
./run_v2.sh test
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh check
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh prepare
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh smoke
# Start only when no matching run is already alive:
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh start
./run_v2.sh status
# Resume after interruption:
CUDA_VISIBLE_DEVICES=5 ./run_v2.sh resume
```

Plot saved V2 histories without CUDA or retraining:

```bash
.venv/bin/python -m src.plot_paper --experiment dino_coreseg_v2_448
```

V2 reads the actual frozen V1 manifests and provenance to preserve the comparison; it does not import V1 source, environments or feature caches. Do not replace or restart the owner's currently running server copy just to update this repository.

---

## Original V1 GPU package documentation


For the side-by-side five-fold training/validation loss and accuracy figure,
see [PLOT_UPDATE.md](PLOT_UPDATE.md). It includes a standalone command that
generates the figure from saved epoch CSVs without retraining or CUDA.

Upload this entire folder and the three dataset folders to your NVIDIA SSH server. It does **not** import the existing `research/` project, its virtual environment, CPU checkpoints or CPU feature cache. Official DINOv2 source and source revision are included; the installer downloads and verifies the official pretrained weights. Dataset inventories and hashes are generated locally on your server. Dataset images are not duplicated into this code bundle.

**Default full workflow:** audit and reconcile duplicates → create patient/group-safe splits → real CUDA smoke test → batched frozen-DINO extraction → **five CV models, 100 epochs each** → **one separate holdout model, 100 epochs** → six final-test partitions → metrics, tables, segmentation/localization figures and report. This means **600 model-epochs in total**, not 100 epochs divided across folds. `early_stopping: false` makes all models run the requested full 100 epochs; checkpoint selection still uses validation only. No baseline or ablation models are included.

## RTX 3090 / V100 server setup

The supplied server snapshot has **six RTX 3090 24-GB cards** (physical indices **0, 1, 3, 5, 6, 7**) and **two V100S 32-GB cards** (**2, 4**). All eight already had running workloads; utilization in that screenshot is not this experiment's utilization. Use only the GPU allocation available to your job. GPU memory is separate per device, not a combined 208-GB pool.

For the six matching RTX 3090 cards, use the new launcher:

```sh
cd ~/Diwakar/November_Segmentation_research/DINO_CoReSeg
./run_3090.sh install
./run_3090.sh check
./run_3090.sh start
./run_3090.sh status
# After an interruption:
./run_3090.sh resume
```

Without an existing `CUDA_VISIBLE_DEVICES`, this launcher selects RTX 3090 cards only, puts the one with the most free memory first for gathered outputs, and saves their UUIDs/order in `outputs/launcher/rtx3090_selection.json`. That choice is reused for installation, check, start and resume. It does not change clocks, power limits, or any existing processes. The mask supplied by you or your scheduler is preserved, and scheduler jobs without an explicit mask must provide their allocation. `status` and `stop` need no selection.

If only a subset is assigned to you, specify it consistently, for example:

```sh
# Example only: choose the indices actually allocated to your job.
export CUDA_VISIBLE_DEVICES=5,6,7
./run_3090.sh install
./run_3090.sh check
./run_3090.sh start
```

To deliberately include both GPU types when **all eight are allocated to you**:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./install_server.sh
./run_server.sh check
./run_server.sh start
```

The installer defaults to **PyTorch 2.11.0 / torchvision 0.26.0 with cu126** so both RTX 3090 and V100 work. The driver's displayed CUDA 13.0 is its supported maximum, not the required wheel version. PyTorch 2.11 cu128/cu130 wheels do not support Volta/V100; keep cu126 for mixed/V100 use. See the [official architecture matrix](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290) and [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/595/minor-version-compatibility.html).

Mixed RTX/V100 DataParallel uses FP16 AMP; the RTX-only group automatically uses native BF16 when supported. Equal batch scattering across unlike cards can be slower; compare actual measured throughput before assuming that adding V100s improves speed. This package keeps the existing DataParallel training and global contrastive loss; it does not replace them with DDP or alter the research objective.

Batch probes check peak usage and free memory on **every selected GPU**, retain the configured 30% headroom, and reserve additional memory for optimizer states. The global training batch is capped at `64 × selected GPUs` and extraction at `32 × selected GPUs`, and reduced on OOM/insufficient headroom. These are caps, not guaranteed batch sizes or benchmark-optimal choices. Probe decisions are saved in `outputs/<experiment>/reports/batch_probe_*.json`. The primary GPU still holds gathered tensors and the global loss. Pinned, persistent data loaders choose up to eight workers per loader based on CPU affinity and the three concurrently retained loaders. The RTX launcher enables expandable allocator segments unless overridden.

The 85,287-parameter decoder, cached feature I/O, prototype fitting, audits and native CPU evaluation can limit GPU utilization. Neither 100% utilization nor faster training on six/eight cards is guaranteed. Automatic batch sizing maximizes within the configured memory cap, not measured speed; different global batches also change optimization steps per epoch. The full study remains five folds × 100 epochs plus a separate 100-epoch holdout model, with the same per-epoch results, checkpoint recovery, metrics and figures. This source revision requires a fresh experiment if you already trained with a previous revision; old checkpoints reject changed implementation/device identities rather than silently continuing a changed study.

## 1. Upload layout

Clone the code repository on your server (datasets are **not** stored in Git):

```sh
git clone https://github.com/diwakardkk/DINO_CoReSeg.git DINO_CoReSeg_GPU_100E
cd DINO_CoReSeg_GPU_100E
python3 scripts/verify_bundle.py
```

The repository includes the official DINOv2 source and a SHA-256-verified weight downloader. It excludes research dataset images, masks, archives and dataset-derived manifests. The pipeline discovers pairs and generates private audit records from your server datasets. Keep your datasets on the server separately using the layout below, or set `CORESEG_DATA_ROOT` to their existing parent directory. Results, checkpoints, feature caches and virtual environments are ignored by Git.

```text
/your/server/work/
├── DINO_CoReSeg_GPU_100E/       this complete code folder
│   ├── configs/gpu_100.yaml
│   ├── src/  tests/  scripts/
│   ├── vendor/dinov2/          official source, license and SOURCE_COMMIT.txt
│   ├── vendor/dinov2_vits14_pretrain.pth   downloaded during installation
│   ├── install_server.sh
│   └── run_server.sh
├── dataset-1/
│   ├── kvasir-seg.zip
│   └── kvasir-sessile.zip
├── dataset-2/
│   └── Dataset.zip            OR extracted Dataset/
└── dataset-3/
    └── PolypDataset.zip       OR extracted PolypDataset/
```

The data parent defaults to `..` relative to this code folder. To use data elsewhere, set `CORESEG_DATA_ROOT=/absolute/path/to/data-parent` or pass `--data-root /absolute/path/to/data-parent` to the launcher. This path must contain `dataset-1`, `dataset-2`, and `dataset-3`, not the contents of one individual dataset. Archive-only dataset-2/dataset-3 uploads are securely extracted once into staging folders and atomically renamed; original ZIP files are preserved. Existing extracted folders are not overwritten. DA/OLDTR remain excluded from training.

Example upload from your Mac (replace the SSH destination):

```sh
rsync -avP DINO_CoReSeg_GPU_100E/ USER@SERVER:/your/server/work/DINO_CoReSeg_GPU_100E/
rsync -avP dataset-1/ USER@SERVER:/your/server/work/dataset-1/
rsync -avP dataset-2/Dataset.zip USER@SERVER:/your/server/work/dataset-2/
rsync -avP dataset-3/PolypDataset.zip USER@SERVER:/your/server/work/dataset-3/
```

Create destination dataset directories first if needed. Do not upload the Mac `.venv`, CPU caches or CPU checkpoints. Allow **at least 60–100 GB free server storage** for archives/extraction, caches, all epoch checkpoints, native-resolution probability maps, predictions and figures; exact usage depends on compression and datasets. A local SSD/NVMe is preferable to slow network storage.

## 2. Install on the server

Use Linux with NVIDIA drivers and Python **3.11 or 3.12** where possible:

```sh
cd /your/server/work/DINO_CoReSeg_GPU_100E
chmod +x install_server.sh run_server.sh
./install_server.sh
./run_server.sh check
```

The installer defaults to the official **PyTorch 2.11.0 / torchvision 0.26.0 CUDA 12.6 wheels**, then installs pinned scientific packages. A compatible NVIDIA driver/GPU allocation is required; the code fails rather than silently training on CPU. Installation needs internet once for dependencies and official pretrained weights. Existing weights at `vendor/dinov2_vits14_pretrain.pth` are verified and reused. After installation, model loading is offline. To prepare weights separately, run `python3 scripts/download_weights.py`, or copy the verified local checkpoint into that path.

If your administrator requires a different compatible PyTorch CUDA build, use the official installer guidance and record the resulting versions:

```sh
CUDA_TAG=cu126 TORCH_VERSION=2.11.0 TORCHVISION_VERSION=0.26.0 ./install_server.sh
```

Do not guess a CUDA wheel tag from the toolkit alone: check driver compatibility and PyTorch's available wheel/version pairs. See https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/. `requirements.server.lock.txt` and each execution's environment report record actual installed versions. The pinned Linux CUDA environment must be validated on the server; local CPU tests are not evidence that CUDA was exercised.

## 3. Start, disconnect, inspect, stop and resume

```sh
# Full 5-fold CV + separate holdout training/evaluation, all at 100 epochs:
./run_server.sh start

# It is detached with nohup, so closing SSH does not send it a hangup.
tail -f outputs/launcher/console.log

# Reconnect later:
./run_server.sh status
nvidia-smi

# Graceful stop; saves a safe current batch position:
./run_server.sh stop

# Wait until the old process has exited, then resume automatically:
./run_server.sh resume
```

For foreground execution inside `tmux` or a scheduler allocation:

```sh
./run_server.sh foreground
```

If the server is managed by Slurm/PBS, run inside an allocated GPU job and obey its process/time limits. `nohup` cannot keep a job alive after a scheduler allocation expires or the server shuts down. Checkpoints provide recovery in those cases. A lock prevents concurrent writers in the same package directory; the PID/status helper does not substitute for scheduler monitoring.

Data in a different location:

```sh
CORESEG_DATA_ROOT=/data/your-datasets ./run_server.sh start
# Or:
./run_server.sh start --data-root /data/your-datasets
```

NVIDIA device selection:

```sh
# Default: all visible GPUs are used through DataParallel.
CUDA_VISIBLE_DEVICES=0,1 ./run_server.sh start
# One allocated GPU:
CUDA_VISIBLE_DEVICES=0 ./run_server.sh start
```

Preserve the same visible GPU layout, configuration and precision policy when resuming. Batch size is probed once and recorded; resumption uses that saved batch size. Do not edit finalized manifests or model code in a running experiment. To change the scientific protocol, start a new version/output directory.

## 4. Optional modes

```sh
./run_server.sh start --mode cv         # only five-fold CV, 100 epochs/fold
./run_server.sh start --mode holdout    # one 100-epoch model + final six test partitions
./run_server.sh start --mode evaluate  # evaluate an already completed/frozen holdout model
./run_server.sh foreground --mode plots
./run_server.sh foreground --mode audit
```

The default mode is `all`. All modes automatically reuse completed valid work. Testing remains blocked until the corresponding model reaches its stopping condition and is frozen. The GPU smoke run is a separate one-epoch integration test on eight development samples; it is never reported as the scientific model.

## 5. GPU execution and scientific choices

- DINOv2-S/14 is frozen and evaluated on CUDA. Batched inputs are 350 × 350 with official ImageNet normalization; 625 spatial patch tokens produce 384 × 25 × 25 features. CLS/register tokens are excluded and shapes are checked. Cache storage is float16; decoder inputs return to float32 under mixed precision.
- Projection, eight 384D normal prototypes, residual CEF, and RGB/DINO residual RPR are preserved. No architecture is replaced. MiniBatchKMeans remains CPU-based to retain the original clustering algorithm; I/O, audit, SciPy surface metrics and plotting also use CPU.
- Automatic BF16 where supported, otherwise FP16 with GradScaler; pinned host memory, asynchronous transfers, spawned persistent data workers, TF32 and cuDNN benchmarking are configured. Batched extraction/training probe memory with headroom and use all visible devices by default. The default cap is 64 training samples and 32 extraction samples **per GPU**; adjust before starting if needed. The selected effective batch is logged. Learning rate is **not silently scaled** with batch size.
- The decoder is deliberately small and frozen-feature training can be I/O-bound. No promise of continuous 100% GPU utilization is made, and no artificial busy work is added. Multi-GPU DataParallel can have overhead for this small model. A single fast GPU can be more efficient; select devices explicitly if that matters.
- Throughput defaults change numerical precision/batch size relative to the CPU experiment. This is a **new experiment** with fresh model weights, not continuation of the three CPU epochs. Seeds are saved; TF32/AMP/cuDNN benchmarking mean bitwise equality across hardware is not promised.

## 6. What is saved if it stops?

For **every completed epoch**: full model + optimizer + scaler + RNG checkpoint, separate JSON metrics, and an atomically updated history CSV. All 100 epoch checkpoints are retained. `best.pt` follows the predeclared validation Dice/FPR selection rule; `last.pt` is the commit record for resumption. Accepted best checkpoints can be recovered from retained epoch files if a process died between separate file replacements.

Within an epoch: default checkpoint every **10 optimizer steps** or **60 seconds**, whichever is reached first, plus the last training batch. Step logs are flushed individually. A graceful SIGTERM/SIGINT saves at a safe batch boundary. Recovery restores optimizer/scaler/RNG and a deterministic batch cursor, avoiding a restart from epoch one. Interrupted validation is repeated without replaying committed optimizer steps. Cache files and per-image evaluation results are saved independently and reused.

**SIGKILL, power loss or disk failure cannot guarantee zero lost work.** Steps since the last durable checkpoint can be lost (normally at most nine optimizer steps); replayed step log entries can repeat global step IDs, so use committed checkpoints/history as authoritative. Set `checkpoint.every_steps: 1` before a new study if every-step persistence matters more than throughput. Checkpoint files are atomically replaced with file fsync, but filesystem/hardware durability still applies.

## 7. Evaluation, plots and tables

- Five outer CV folds with grouped inner validation for epoch selection. All patient/duplicate-connected groups stay together. Prototypes are rebuilt from each fold's training subset only. Original FAR and official benchmark final tests remain excluded from CV.
- One separate holdout model evaluated independently on FAR patients, Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300 and ETIS. No final test threshold/model tuning. CVC-300/CVC-ColonDB held-out image correlation is documented.
- Native-resolution per-image Dice, IoU, precision, recall, specificity, F2, boundary F1/IoU, HD95, ASSD, MAE, pixel accuracy and raw confusion counts. Surface failures receive explicit image-diagonal pixel penalties with counts; none are omitted.
- Normal-frame ungated/gated FPR, frame/pixel specificity, FP area/fraction and 8-connected component counts. Published FAR 100-pixel gate is at native resolution; separately labeled 256-grid gates are exploratory. Healthy masks are explicitly derived empty targets.
- Training-derived lesion-size strata and patient-cluster bootstrap intervals. Quantized streaming pixel ROC/AUC (65,536 bins) counts all evaluated pixels without holding billions of scores in RAM.
- Requested loss/accuracy/ROC plots: predictive loss shown as **L/(1+L)** on 0–1 axes, raw losses retained; macro image pixel accuracy on 0–1 axes; genuine held-out ROC curves and AUC per fold/mean. Accuracy is background-sensitive and is not image classification accuracy. Optimization still uses segmentation + 0.10 contrastive + 0.30 auxiliary.
- Original | Ground truth | Predicted mask | Predicted localization, plus error overlays, successes, low-Dice cases, small cases, healthy FPs and counter-evidence maps. Masks/probabilities are saved at native dimensions. PNG 300 DPI plus PDF/SVG figures use white backgrounds.
- Source/size performance, raw training components, runtime/peak-memory records, separate decoder and end-to-end latency, CSV/Markdown/LaTeX tables, source/version/weight identities and a Markdown experiment report. Optional saliency metrics are omitted rather than using unverified implementations.

Figures update periodically (default every five epochs, plus first/final epoch) and after evaluation/interruption. **Metrics and checkpoints still save every epoch.** Partial plots reflect actual available histories; no missing fold/epoch curves are invented.

## 8. Outputs to collect

```text
outputs/dino_coreseg_gpu_100/
├── logs/pipeline.log                     complete stdout/stderr
├── smoke/                               integration artifacts only
├── holdout/
│   ├── checkpoints/epoch_001.pt … epoch_100.pt
│   ├── epochs/epoch_001.json … epoch_100.json
│   ├── best.pt / last.pt / FROZEN.json
│   ├── logs/steps.jsonl / epochs.jsonl
│   ├── training_history.csv / progress.json
│   ├── predictions/<test-source>/        masks, probabilities, metadata
│   ├── metrics/                         per-image/source/size/ROC/CI/runtime
│   └── figures/                         segmentation, localization, plots
├── tables/                              CSV, Markdown, LaTeX
├── figures/architecture.{png,pdf,svg}
├── reports/EXPERIMENT_SUMMARY.md
└── pipeline_status.json
outputs/dino_coreseg_gpu_100_cv5/
├── splits/                              immutable grouped fold membership
├── fold_01/ … fold_05/                   same checkpoint/log/result structure
├── figures/01_train_validation_loss.*
├── figures/02_train_validation_accuracy.*
├── figures/03_roc_auc.*
├── fold_metrics.csv / cv_summary.json
└── status.json
cache/                                  resumable CUDA DINO feature cache
manifests/ / splits/                     audit and original holdout manifests
requirements.server.lock.txt            exact installed server environment
```

Check `pipeline_status.json` and the per-fold evaluation status before claiming completion. A checkpoint file or a plot alone does not establish a finished study. Keep the entire output directories and configuration/version files for research reproducibility.

## Local validation limits and sources

The package is checked locally for syntax, architecture/metric behavior, safe extraction, atomic checkpointing, and step/epoch resumption on CPU test fixtures. **CUDA execution and the pinned Linux environment must be checked on the NVIDIA server.** No GPU training results are included or fabricated. The launcher runs a real CUDA smoke test before full work.

Official technical references: https://docs.pytorch.org/docs/stable/notes/amp_examples.html and https://pytorch.org/get-started/previous-versions/. Official DINO source is bundled with its original license; official pretrained weights are downloaded during installation and their configured SHA-256 is verified. Dataset licenses/clinical provenance are not established solely by local file counts. `CPU_METHODS_REFERENCE.md` preserves the original scientific design; GPU-specific changes and current settings are defined here and in `configs/gpu_100.yaml`.

## Upload checksum verification

The upload ZIP excludes local virtual environments, caches, tests' temporary data and run outputs. After extracting it on the server, optionally verify its static files before installation:

```sh
python3 scripts/verify_bundle.py
```

`LOCAL_VALIDATION.md` records exactly what was and was not checked locally. If you intentionally edit source/configuration files, the original bundle checksum will differ; record that change and start a new scientific configuration rather than altering a running study.

Dataset archive hashes are established on the first server execution and checked on resume. They are not compared against a published reference archive hash. Expected collection counts, pair matching and image/mask checks remain enforced.
