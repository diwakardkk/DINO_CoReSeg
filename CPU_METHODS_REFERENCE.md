# Predeclared first proposed-model experiment

This is an implementation specification, not evidence of novelty or performance. Only the full DINO-CoReSeg model is implemented. No baseline comparisons or ablations are performed. Seed 42 and all thresholds/coefficients are fixed before final testing in configs/coreseg_v1.yaml and splits/protocol.json.

## Data, labels and duplicate reconciliation

The authoritative uploaded audits are read before building the experiment inventory. Operational data are dataset-3's 1,450 NewTR pairs, dataset-2's 432 supplied positive pairs and 7,749 healthy originals, and the 798 named benchmark test pairs. Dataset-2 is called FAR in the experiment, following the user's terminology; matching public counts support that interpretation but local archive provenance has not been cryptographically verified. Dataset-1 Kvasir and sessile archives establish known ID overlap and are not additional training cases. DA and OLDTR are excluded entirely.

Benchmark masks are opened for header dimension validation only during audit. Their pixel labels are first loaded for cache/evaluation after the chosen checkpoint is frozen. No test-mask foreground fraction, size bin, clustering, model selection, or threshold tuning occurs before that point. Test images are permitted in an image-only leakage screen; exclusions use content similarity and do not use test outcomes.

Image identity checks use file SHA-256, native decoded-RGB SHA-256 including shape, and a 64 × 64 bilinear RGB normalized hash. A 64-bit perceptual hash uses 32 × 32 grayscale Lanczos resizing and the leading 8 × 8 orthonormal DCT coefficients, thresholded by the median excluding the DC coefficient. DC is set to zero. Candidate pairs have Hamming distance ≤4; a pigeonhole index searches all such pairs without approximate neighbor retrieval. This heuristic cannot detect all correlated frames or establish patient identity.

Patient rows and all detected duplicate/near-duplicate image links are unioned into connected components before partitioning. Development components linked to an official benchmark test image are excluded, leaving the official test collections intact. FAR patient components linked to benchmark development are conservatively excluded. Redundant normalized-identical development copies are excluded with an explicit manifest entry; contradictory positive/negative labels on normalized-identical content cause all such copies to be excluded. Remaining near-duplicate candidates are reconciled by keeping their entire connected component in a single split, rather than asserting clinical equivalence. If any detected cross-split link remains, splitting fails.

NewTR uses an approximately 80/20 training/validation split. FAR uses approximately 70/15/15 patients for training/validation/test, stratified on patient/component positivity where possible. Source labels are Kvasir only when matched to the supplied Kvasir IDs; numeric NewTR IDs retain the label `unverified_numeric_source` rather than being claimed as ClinicDB. Whole duplicate components constrain the exact ratios. A deterministic greedy assignment balances weighted deficits within positivity/source strata. Exact patient counts, exclusions and retained sample counts are saved. Manifests are immutable: regeneration with different contents fails loudly.

Healthy targets are explicitly derived all-zero masks, not supplied pixel annotations. Positive patient status is not applied to healthy frames. Clinical/procedure metadata are never model inputs. No validation or test patches enter prototype fitting.

## Frozen features and caches

The official non-register DINOv2-S/14 backbone is loaded via its official torch.hub local-source entry point with the official checkpoint, strict state-dict matching, `.eval()`, and `requires_grad_(False)`. All computation uses CPU and configurable thread count; xFormers is disabled. Online loading via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` is also supported when `backbone.offline=false`. Local offline loading avoids any additional network requirement.

RGB is bicubically stretched to 350 × 350, converted to [0,1], then normalized with mean (0.485,0.456,0.406) and standard deviation (0.229,0.224,0.225). This uses the official normalization; square stretching without a classification crop is the specified dense-segmentation preprocessing choice. `forward_features()['x_norm_patchtokens']` must have shape B × 625 × 384; it excludes CLS/register tokens and is reshaped to B × 384 × 25 × 25. Extraction runs under `torch.inference_mode()`.

Cache files are compressed NPZ with float16 DINO features, uint8 compact RGB (3 × 100 × 100), uint8 binary target (1 × 350 × 350), and JSON metadata. Masks are converted to `L` before thresholding at 128, then nearest-neighbor resized. RGB refinement input uses bilinear resize of original RGB. Features become float32 and RGB becomes float32 [0,1] when loaded for decoder training. Cache names are SHA-256 of namespaced sample identifiers. Model checkpoint SHA, source revision, preprocessing version, split, image/mask hashes and provenance metadata are saved. Creation is atomic; an existing cache is validated and reused or rejected, never silently replaced. Interrupted temporary files can be replaced because they are not valid final cache artifacts.

No augmentation is used initially. The configurable aligned feature/RGB/mask flip option is disabled by default and is only an approximation to recomputing a transform-sensitive DINO backbone. No test augmentation is supported.

## Architecture

1. **Projection:** 384 → 64 pointwise convolution, 8-group GroupNorm, GELU.
2. **Normal prototypes:** choose equal numbers of training healthy frames and positive-background frames (up to 1,000 images per category), with at most 16 patches per image. A 14 × 14 pooling grid gives the lesion occupancy of each aligned patch. Patches must have occupancy ≤0.01 in both the target and its two-pixel dilation; this adds a conservative boundary exclusion halo. Sample the same total patch count from each category. L2-normalize raw 384D vectors, fit 8 MiniBatchKMeans centers (seed 42, 3 initializations, batch 512, max 100 iterations), and normalize centers again. These fixed vectors remain in raw DINO space, separate from trainable projection features.
3. **Counter-evidence:** for normalized feature f and prototypes p_k, use d=(1−max_k cosine(f,p_k))/2, clamped to [0,1]. This is dissimilarity, not a polyp probability. A separately supervised monotonic affine calibration softplus(a)d+b provides auxiliary lesion/background logits.
4. **Initial head:** 64 → 32 3 × 3 Conv/GroupNorm/GELU, then pointwise 32 → 1 logits.
5. **CEF:** concatenate projected features, dissimilarity and initial logits (66 channels). Use 3 × 3 blocks 66 → 32 → 16, then 16 → 1 signed residual. Fused logits are initial + tanh(γ_raw) × residual, with γ_raw=0.01 initially.
6. **RPR:** project 64 → 16 DINO features and bilinearly upsample them to 100 × 100. A two-block 3 × 3 RGB CNN produces 16 channels at 100 × 100. Concatenate the 16 DINO channels, 16 RGB channels and one fused-logit channel; use 33 → 32 → 16 convolutional blocks and a pointwise signed residual head. Refined logits are upsampled fused logits + tanh(η_raw) × residual, with η_raw=0.01. Bilinearly upsample final logits to 350 × 350 with align_corners=False.

There are no BatchNorm layers, hard negative gates, additional transformers, pretrained RGB backbones, or lesion-deletion gates. All decoder parameters are differentiable, including the bounded residual scales. “Region preservation” is an operational residual-refinement design; it has no proven topology or lesion-preservation guarantee. The raw dissimilarity map is fixed for fixed features/prototypes; its calibration and use in residual fusion are trainable. The projected-feature contrastive objective supplies gradients even though DINO/prototypes are frozen. No component benefits can be inferred from a first full-model experiment alone.

## Losses

For each positive image, use pixel-mean BCEWithLogits plus soft Dice loss, with epsilon 1e-6. Small-polyp weight is w_i=clip(median_training_positive_area / max(resized_target_area,1/350²),1,3). Normalize the weighted positive sum by the sum of positive weights. For normal images, use only pixel-mean BCE against the derived empty target. Compute positive and normal contributions separately and average the categories present; a large normal pool cannot dominate via frame count.

Patch supervised contrastive loss acts on L2-normalized 64D projection features at temperature 0.1. Foreground patches have occupancy ≥0.90, background patches ≤0.01, ambiguous occupancies are excluded. Deterministically take at most 64 patches per class per batch. Other patches of the same class are positives in the log-softmax denominator, excluding each anchor itself; anchors without another same-class patch are excluded. If only one class is available, the contrastive term is zero with a valid computational graph.

Auxiliary loss is the mean of (a) the average segmentation objective on initial, fused and 100-grid refined logits, all interpolated to 350, and (b) confident-patch calibrated-counter BCE, balanced across foreground/background patch classes. This explicitly supervises low-resolution localization and refinement without an additional decoder.

Total loss is segmentation + 0.10 contrastive + 0.30 auxiliary. Coefficients are initial predeclared choices, not optimized values. Trainable gradients are checked for presence/finite values and clipped to L2 norm 5 before AdamW.

## Training and selection

AdamW: learning rate 0.0003, weight decay 0.0001, batch size 4, maximum 60 epochs, early-stopping patience 12. Each batch contains two positives and two normals. Positive source is sampled uniformly over available source labels, then an image uniformly within source. Each epoch makes ceil(number_of_training_positives/2) batches. Normals are randomly permuted and cycled, with a new deterministic epoch seed; full normal-pool coverage per epoch is not claimed. Optimizer, model, epoch, history, RNG, prototypes/config identity and split fingerprints are checkpointed. Time-budgeted stopping happens only after a complete epoch; resume continues from the next epoch without changing the predeclared protocol.

Validate once per epoch at 350 × 350, threshold 0.5. Positive Dice/IoU and small-case Dice/recall are image macro means; normal FPR means any predicted foreground pixel. Select feasible epochs (normal FPR ≤0.10) ahead of all infeasible epochs, then maximize positive Dice, then minimize FPR. If no feasible epoch exists, minimize normal FPR and break ties by positive Dice. This strict ungated constraint may be infeasible; its violation is reported rather than adjusted. Patience counts failure to improve that complete lexicographic selection rule.

Training completion means maximum epochs or predeclared early stopping, not a runtime pause. Only then is best.pt SHA-256 frozen with the selected epoch and validation rule, enabling test extraction/evaluation. Test evaluation cannot change the frozen threshold or checkpoint/config identity.

Lesion-size bins are fixed to training-positive original-grid area tertiles: small ≤ first tertile, medium ≤ second tertile, large above. They represent relative mask area, not physical polyp diameter.

## Evaluation definitions

Report six partitions independently: held-out FAR patients plus Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300 and ETIS-LaribPolypDB. Apply sigmoid at the 350 grid, bilinearly restore probabilities to each image's original dimensions, then threshold at 0.5. Compute per-image results before macro means.

- Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN); precision=TP/(TP+FP), defined as zero for an empty prediction on a positive image; recall=TP/(TP+FN); specificity=TN/(TN+FP), undefined if no background exists; F2=5TP/(5TP+4FN+FP). MAE is pixel mean absolute probability minus target.
- Boundaries are foreground minus one binary erosion with zero outside the frame. Boundary F1 uses a fixed two-pixel Euclidean matching tolerance. Boundary IoU uses inward boundary bands with erosion radius max(1,round(0.02 × image diagonal)). Distances are pixels; no physical spacing is available.
- HD95 and ASSD use the concatenation of both directed boundary-to-boundary Euclidean distance arrays. On an empty prediction for a positive target, Dice/IoU/precision/recall/F2/boundary scores are zero; HD95/ASSD receive the image diagonal as an explicit finite penalty. `surface_failure` and `empty_prediction` counts are retained. No failed prediction is dropped from averages. Optional saliency metrics are omitted rather than using unverified implementations.
- Normal frames: ungated FP-frame rate, complementary frame specificity, pixel specificity, foreground area/fraction, and number of 8-connected components. Component-area gate settings 1,10,50,100,500 pixels report whether any component reaches the gate, separately from ungated results. Native-grid gates have native-grid units.
- Published FAR gate check: the official `configs/canonical.yaml` explicitly specifies `eval_resolution: native` and `min_area_px: 100`; 256 is the training size, not the canonical scoring grid. Therefore native-resolution `fp_frame_gate_100` is the protocol-aligned secondary gated result. Our patient split, training on healthy frames, and model differ from the paper, so this is not a full protocol replication. Separately report **exploratory 256 × 256** probability-map gate sensitivity in `grid256_*` columns. Those columns are not labeled as published-protocol results, and no 100-pixel gate is treated as scale-invariant. This corrects the initial interpretation of the prompt using primary-source code before test evaluation.
- FAR positive Dice and normal FPR receive 95% percentile intervals from 500 patient-cluster bootstrap resamples, seed 42. Each resample includes all retained frames of the selected patient clusters. Native benchmark patient IDs are unavailable, so no false patient-aware benchmark interval is claimed.
- Report source-wise and size-wise macro performance, surface failures, and descriptive selected-validation-minus-test Dice gaps; the latter are not causal estimates of domain shift because validation mixes sources.
- Separate measured decoder-only latency from full CPU inference. End-to-end measurements include disk image load, resizing/normalization, backbone, decoder and prediction restoration, but exclude one-time model load and metrics. Keep per-sample times, including the first/warm-up sample. Peak process RSS is obtained from resource.getrusage (macOS bytes; Linux KiB converted to bytes). Cache disk bytes and trainable/frozen parameter counts are reported separately.

## Sources checked

- Official source and normalization: https://github.com/facebookresearch/dinov2 and https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
- Official spatial forward output: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
- Official weights: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
- FAR paper (256-grid input and connected-component FP gate, checked 6 November 2026): https://link.springer.com/article/10.1007/s10278-026-02268-5

The local source revision and actual checkpoint SHA-256 are saved in cache/backbone_identity.json and execution reports. The official source is unmodified and retains its license. Dataset licenses, consent and acquisition provenance are not inferred solely from matching file counts.

- Official FAR scoring resolution and gate: https://github.com/FAR-POLYP-SEG/Codes/blob/main/configs/canonical.yaml and https://github.com/FAR-POLYP-SEG/Codes/blob/main/farseg/fp_eval.py

## Detected overlap among official held-out test sources

The image-only audit found 94 exact/perceptual candidate links between CVC-300 and CVC-ColonDB. They remain inside the held-out global test split, so they do not contaminate training/validation. Retain the supplied named sets, report their metrics separately, and disclose their correlation; they must not be treated as independent domain-replication evidence. Exact link reasons are saved in reports/evaluation_partition_overlap.json, reproducibly rebuilt by src.report. No mask labels were used for this finding.
