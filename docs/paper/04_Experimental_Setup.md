# 4. Experimental Setup

## 4.1 Datasets

### 4.1.1 Training Datasets

Two datasets are constructed for pseudo-labeling:

| Dataset | Documents | Total Pages | Language | Source | PDF Type |
|---------|-----------|-------------|----------|--------|----------|
| **KoGovDoc** | 10 | 3,637 | Korean | Korean government publications | Digital PDF |
| **ArXivPapers** | 39 | 864 | English | arXiv preprints (cs.CL, cs.CV, cs.LG) | Digital PDF |
| **Total** | 49 | 4,501 | Bilingual | — | — |

#### KoGovDoc Dataset

10 Korean government documents (kogov_001 through kogov_011, excluding kogov_005) covering smart city plans, government reports, and policy announcements. Document sizes range from 16 to 1,929 pages, with significant size imbalance:

| Document | Pages | Ratio | Content Type |
|----------|-------|-------|-------------|
| kogov_008 | 1,929 | 53.0% | Large policy document with extensive tables |
| kogov_003 | 556 | 15.3% | Government planning report |
| kogov_004 | 340 | 9.4% | Technical specification |
| Others (7) | 812 | 22.3% | Various government documents |

**Bias Correction**: kogov_008's 53% dominance is addressed via per-document downsampling (max_ratio=25%) during training data preparation.

#### ArXiv Papers Dataset

39 English academic papers selected from arXiv with structural diversity requirements:

| Category | Definition | Count | Ratio |
|----------|-----------|-------|-------|
| table-heavy | ≥ 3 result comparison tables | 16 | 41% |
| equation-heavy | ≥ 5 display equations | 12 | 31% |
| mixed | Tables + equations + figures | 10 | 26% |
| code-block | Algorithm pseudocode | 1 | 3% |

Year distribution spans 2013-2023 (11 years). Papers use `paper.pdf` naming (vs `doc.pdf` for KoGovDoc).

### 4.1.2 Evaluation Benchmark

**OmniDocBench**: A standardized document parsing benchmark containing 1,355 pages across 9 document types with expert-annotated ground truth. Includes text, titles, tables (HTML→markdown conversion), equations (LaTeX), figures, and footnotes.

### 4.1.3 Dataset Structure

```
datasets/
├── documents/           # KoGovDoc
│   ├── kogov_001/
│   │   ├── doc.pdf
│   │   ├── gt_pages/page_0001.md ... page_NNNN.md
│   │   ├── gt.md (merged)
│   │   ├── gt_metadata.json
│   │   └── metadata.json
│   ├── ...
│   └── validation_report.json
├── papers/              # ArXivPapers
│   ├── arxiv_001/
│   │   ├── paper.pdf
│   │   ├── gt_pages/page_0001.md ... page_NNNN.md
│   │   ├── gt.md (merged)
│   │   └── gt_metadata.json
│   ├── ...
│   └── validation_report.json
├── omnidocbench/         # Evaluation benchmark
│   ├── OmniDocBench.json
│   └── images/
└── training/            # Prepared training data
    ├── train.jsonl
    ├── val.jsonl
    ├── images/
    └── data_stats.json
```

## 4.2 GT Generation Results

### 4.2.1 Generation Statistics

| Dataset | Model | Pages | Batch Size | Avg Time/Page |
|---------|-------|-------|------------|---------------|
| KoGovDoc | Qwen3-VL-30B | 3,637 | 4 | ~40s |
| ArXivPapers | Qwen3-VL-30B | 864 | 4 | ~45s |

All pages successfully generated with per-page caching for fault tolerance.

### 4.2.2 GT Validation Results

| Dataset | Sample Ratio | Sampled | Avg Score | Pass Rate | Failed |
|---------|-------------|---------|-----------|-----------|--------|
| KoGovDoc | 30% | 1,047 | 3.3/5 | 75.1% | 256 |
| ArXivPapers | 100% | 864 | 3.0/5 | 73.8% | 218 |

#### Error Type Breakdown

| Error Type | Documents | Papers | Description |
|------------|-----------|--------|-------------|
| think_tag | 392 | 333 | Thinking/CoT text contamination |
| truncation | 259 | 227 | Abrupt content endings |
| other | — | 226 | Miscellaneous quality issues |
| table_broken | 199 | 87 | Malformed table structures |
| ocr_error | — | 41 | Character recognition errors |

**Primary failure mode**: Thinking tag contamination (36% of failures in Papers). Root cause: `enable_thinking: False` in initial generation prevented proper `<think>` tag wrapping, causing untagged reasoning to leak into content. Fixed by switching to `enable_thinking: True` with `--reasoning-parser qwen3` for re-generation.

### 4.2.3 Failed Page Re-generation

474 failed pages (256 documents + 218 papers) are re-generated with corrected configuration:
- `enable_thinking: True` (proper tag generation)
- `--reasoning-parser qwen3` on vLLM (server-side separation)
- Enhanced `_clean_response()` with pattern-based fallback

## 4.3 Model Configurations

### 4.3.1 Teacher VLM (GT Generation)

```yaml
model: Qwen/Qwen3-VL-30B-A3B-Thinking
serving: vLLM v0.13.0
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.95
  max_model_len: 16384
  reasoning_parser: qwen3
temperature: 0.1
max_tokens: 8192
enable_thinking: true
```

### 4.3.2 Judge LLM (GT Validation)

```yaml
model: Qwen/Qwen3.5-122B-A10B-NVFP4
serving: vLLM (nightly)
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.80
  reasoning_parser: deepseek_r1
temperature: 0.1
max_tokens: 16384
# Note: enable_thinking NOT set — model uses deepseek_r1 parser
```

### 4.3.3 Student VLM (Fine-tuning Target)

```yaml
model: Qwen/Qwen3-VL-2B-Instruct
training_framework: ms-swift 4.0.1
  tuner_type: lora
  lora_rank: 8
  lora_alpha: 32
  target_modules: all-linear
  freeze_vit: true
  freeze_aligner: true
  deepspeed: zero2
  nproc_per_node: 2
```

### 4.3.4 Evaluation Models

For OmniDocBench evaluation, the teacher model uses the Instruct variant:

```yaml
model: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
serving: vLLM v0.13.0
  tensor_parallel_size: 2
  max_model_len: 16384
  enforce_eager: true
temperature: 0.1
max_tokens: 8192
enable_thinking: false
```

## 4.4 Hardware Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition × 2 |
| VRAM | 98GB per GPU (196GB total) |
| RAM | 128GB DDR5 |
| Storage | NVMe SSD |
| CUDA | 12.8 |
| PyTorch | 2.9.1 |
| Transformers | 4.57.3 |

GPU sharing: Teacher (30B), Judge (122B), and Student training use the same GPU pair sequentially — not concurrent.

## 4.5 Evaluation Protocol

### 4.5.1 OmniDocBench Evaluation Pipeline

```
1. VLM Inference:
   For each page image in OmniDocBench:
     pred_markdown = VLM(image, system_prompt, user_prompt)
     save to results/<model>/predictions/<image_stem>.md

2. Evaluation:
   OmniDocBenchEvaluator computes:
     - Text NED (sample_avg, page_avg, edit_whole)
     - Table TEDS / TEDS-S
     - Formula CDM F1 / ExpRate / NED
     - Reading Order NED
```

### 4.5.2 Metrics Overview

| Metric | Category | Direction | Description |
|--------|----------|-----------|-------------|
| Text NED | Text | lower=better | Normalized Edit Distance for text blocks |
| Table TEDS | Table | higher=better | Tree Edit Distance Similarity for tables |
| Table TEDS-S | Table | higher=better | Structure-only TEDS (ignores cell text) |
| Formula CDM F1 | Formula | higher=better | Visual character detection matching |
| Formula CDM ExpRate | Formula | higher=better | Exact match rate after rendering |
| Formula NED | Formula | lower=better | NED for LaTeX formula text |
| Reading Order NED | Order | lower=better | NED over element ordering sequence |

### 4.5.3 Reproducibility

- Random seed: 42 for all stochastic operations
- All configurations versioned in `configs/`
- Results stored as structured JSON
- VLM temperature: 0.1 for near-deterministic output
- Resume support: existing prediction files are skipped
- Error marker: `<!-- ERROR: ... -->` in failed prediction files
