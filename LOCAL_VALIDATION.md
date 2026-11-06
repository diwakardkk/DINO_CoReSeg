# Local package validation

This is a code/package check, not a GPU experiment result.

- **27 tests passed** on the available macOS CPU environment, including model shape/finite gradients, one-bit masks, safe archive extraction, immutable manifests, nested patient/duplicate-group isolation, metric edge cases, and atomic checkpoint replacement.
- A deliberate mid-epoch interruption on CPU test fixtures resumed at the saved batch position and produced the same final model tensors as uninterrupted execution. Completed-epoch checkpoints and JSON files were retained.
- Native-resolution predictions, saved probability/mask agreement, per-image/source metrics, ROC, resuming evaluation and generating research tables/reports were exercised on CPU test fixtures in temporary directories.
- The official pretrained DINO source/weights available locally (the repository installer downloads weights) loaded offline and independently, produced `[1,384,25,25]` spatial features, and retained all **22,056,576 frozen parameters**.
- Mocked multi-GPU tests cover RTX-only selection, persistent UUID/order, scheduler/user masks, BF16/FP16 fallback, secondary-GPU memory pressure, CPU affinity, launcher environment propagation and per-device CUDA preflight failures. These simulate control flow and do not measure actual GPU performance.
- Configuration requests 100 epochs, disables early stopping by default, enables five-fold CV and the separate holdout study. Shell scripts and all Python modules parse successfully.
- Local environment: Python 3.14, PyTorch 2.14 CPU build. **CUDA was unavailable and was not exercised.** The pinned Linux/PyTorch CUDA installation must be checked on the target server. The launcher requires a real CUDA smoke test before full training and refuses a silent CPU fallback.
- No 100-epoch experiment, cross-validation results, final GPU metrics or GPU performance claims are included. All actual study outputs will be generated on the server.

Raw unit-test verdict: `local_validation.txt`. The package's original CPU research folder and datasets were preserved.
