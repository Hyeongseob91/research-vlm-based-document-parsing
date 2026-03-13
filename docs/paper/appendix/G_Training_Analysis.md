# Appendix G: LoRA Training Analysis

## G.1 Purpose

This appendix documents the LoRA fine-tuning process for WigtnOCR-2B, including training curves, hyperparameter sensitivity, and qualitative output comparison across training stages. This analysis supports the distillation effectiveness claims in RQ5 and provides practical guidance for reproducing or extending the training.

## G.2 Training Data Summary

| Property | Value |
|----------|-------|
| Total pages (pre-filter) | 4,501 |
| Passed validation (score ≥ 3) | ~3,300+ |
| After bias correction (max_doc_ratio=0.25) | ___ |
| Train split (90%) | ___ |
| Val split (10%) | ___ |
| Image resolution | 200 DPI PNG |
| Format | ms-swift JSONL |

### G.2.1 Training Data Composition

| Dataset | Pre-filter | Post-filter | Post-downsample | Ratio |
|---------|-----------|-------------|-----------------|-------|
| KoGovDoc (10 docs) | 3,637 | ___ | ___ | ___% |
| ArXivPapers (39 papers) | 864 | ___ | ___ | ___% |
| **Total** | 4,501 | ___ | ___ | 100% |

### G.2.2 kogov_008 Bias Correction

| Stage | kogov_008 Pages | Ratio |
|-------|----------------|-------|
| Original | 1,929 | 53.0% |
| After validation filter | ___ | ___% |
| After downsampling (max 25%) | ___ | 25.0% |

## G.3 Training Configuration

```yaml
model: Qwen/Qwen3-VL-2B-Instruct
framework: ms-swift 4.0.1
tuner: lora (rank=8, alpha=32, all-linear)
freeze_vit: true
freeze_aligner: true
epochs: 3
learning_rate: 1e-4
batch_size: 1 per device
gradient_accumulation: 4
max_length: 4096
precision: bfloat16
deepspeed: zero2
gpus: 2x RTX PRO 6000 (98GB each)
```

## G.4 Training Curves

> **Status**: Pending — will be populated from training logs (ms-swift tensorboard output).

### G.4.1 Loss Curves

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|----------|---------------|
| 0.5 | ___ | ___ | ___ |
| 1.0 | ___ | ___ | ___ |
| 1.5 | ___ | ___ | ___ |
| 2.0 | ___ | ___ | ___ |
| 2.5 | ___ | ___ | ___ |
| 3.0 | ___ | ___ | ___ |

### G.4.2 Convergence Analysis

- **Epoch of best val loss**: ___
- **Overfitting signals**: ___
- **Early stopping**: Applied / Not applied

## G.5 Qualitative Output Comparison

> **Status**: Pending — will show same page parsed by base 2B vs LoRA 2B vs 30B teacher.

### G.5.1 Example: Academic Paper Table

**30B Teacher** (GT):
```markdown
(example output)
```

**2B Base** (pre-training):
```markdown
(example output)
```

**WigtnOCR-2B** (post-LoRA):
```markdown
(example output)
```

### G.5.2 Example: Korean Government Table

(Same format as above)

## G.6 Hyperparameter Sensitivity

> **Status**: Pending — will include ablation if resources permit.

| Parameter | Tested Values | Best | Notes |
|-----------|-------------|------|-------|
| LoRA rank | 4, 8, 16 | ___ | Higher rank = more capacity but slower |
| Learning rate | 5e-5, 1e-4, 2e-4 | ___ | |
| Epochs | 1, 2, 3, 5 | ___ | Watch for overfitting |
| freeze_vit | True, False | True | Unfreezing may help but risks catastrophic forgetting |

## G.7 Training Cost

| Resource | Value |
|----------|-------|
| Total training time | ___ hours |
| GPU hours | ___ (2 GPUs × ___ hours) |
| Peak VRAM usage | ___GB / 98GB per GPU |
| Checkpoint size (LoRA adapter) | ___MB |
| Full model merge size | ___GB |

## G.8 Reproducibility Checklist

- [ ] Random seed: 42
- [ ] ms-swift version: 4.0.1
- [ ] Transformers version: 4.57.3
- [ ] PyTorch version: 2.9.1
- [ ] CUDA version: 12.8
- [ ] Training data JSONL: SHA256 = ___
- [ ] Base model: Qwen/Qwen3-VL-2B-Instruct (commit: ___)
