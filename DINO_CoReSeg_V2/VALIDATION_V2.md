# V2 validation record — 7 November 2026

## Completed locally

- **36 tests passed**, six expected optional-xFormers warnings; 12.16 seconds.
- AST parsing passed for 24 project Python files. Both shell launchers passed
  `bash -n`.
- All 31 original V1 source files covered by the pre-change snapshot still match
  their SHA-256 hashes. V1's Git working tree is clean. No V1 file was modified.
- Actual official, frozen DINOv2-S/14 weights were loaded for CPU feature tests:
  final maps `[1,1,384,32,32]`; four-layer maps `[1,4,384,32,32]` at input 448.
  The backbone has 22,056,576 frozen parameters.
- All five decoder variants passed shape and gradient tests. Trainable counts:
  resolution 85,287; RGB context 92,320; multiscale refinement 106,594;
  boundary-loss variant 106,594; multilayer 180,902.
- Tested exact patch-mask occupancy, small/thin/healthy targets, prototype
  normalization and training membership, old-bank rejection, patient/image
  leakage detection, exact preserved CV membership and dataset-3 usage checks.
- Tested interruption/resume of optimizer, scheduler, accumulation cursor and
  evaluation aggregation; selected-checkpoint preservation; matching the selected
  epoch rather than assuming the last epoch; and V1 read-only reporting.
- Tested CSV-only plotting, unequal fold lengths and six-row qualitative
  alignment using explicitly constructed test fixtures. These fixtures are
  **not experimental results** and are not delivered as paper figures.

Machine-readable details are in `VALIDATION_V2.json`. The local test environment
was macOS ARM64, Python 3.14.7, CPU PyTorch 2.15.0; it is deliberately external
to V2 and is not shipped. It differs from the pinned Linux CUDA installation
(Python 3.10–3.13, PyTorch 2.11.0/torchvision 0.26.0, CUDA 12.6 wheels).
The launcher reruns the CPU suite in the installed server environment before
the real-data CUDA smoke test.

## Still required on the server

Noninteractive SSH access to `ansh@192.168.51.15` was rejected with
`Permission denied (publickey,password)`. No server files were modified and
no V2 GPU job was launched.

Consequently, actual frozen server dataset usage and implementation identity
have not been independently read here; GPU compatibility, CUDA gradients,
GPU memory, throughput and full-data training/evaluation have not been measured.
There is no V2 accuracy result or evidence of improved generalization yet.
Do not interpret passing CPU tests as validation of those measurements.

On an allocated GPU, run `install`, `test`, `check`, `prepare`, and `smoke` as
documented in `README_V2.md`. `prepare` reads and hashes the real completed V1
artifacts without writing to V1, copies exact split membership into V2, audits
the shared data and writes the actual dataset usage report. `smoke` measures
memory on real development samples. Only after these pass should `start`
launch the 100-epoch holdout experiment. Five-fold CV is a separate explicit job.

The full run saves histories, resumable/selected checkpoints, native masks,
per-image and per-dataset metrics, surface-failure tables, ROC histograms,
timings, selected probabilities, vector paper figures and a report inside V2.
The final preservation check compares the original V1 artifact hashes again.
