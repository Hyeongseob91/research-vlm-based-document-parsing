# Appendix B: Full Results Tables

## B.1 Purpose

This appendix provides complete numerical results for all experiments. Main text presents aggregate results; this appendix enables detailed reproduction and analysis.

## B.2 OmniDocBench Full Results

### B.2.1 All Models — Text NED

| Model | sample_avg | page_avg | edit_whole | Direction |
|-------|:----------:|:--------:|:----------:|:---------:|
| Qwen3-VL-30B | 0.289 | 0.415 | 0.331 | lower=better |
| Qwen3-VL-2B | 0.364 | 0.376 | 0.503 | |
| **WigtnOCR-2B** | **0.288** | **0.304** | **0.293** | |
| Marker | **0.218** | **0.244** | **0.197** | |

### B.2.2 All Models — Table TEDS

| Model | TEDS | TEDS-S | Direction |
|-------|:----:|:------:|:---------:|
| Qwen3-VL-30B | 0.523 | 0.657 | higher=better |
| Qwen3-VL-2B | 0.561 | 0.667 | |
| **WigtnOCR-2B** | **0.649** | **0.732** | |
| Marker | 0.586 | 0.658 | |

### B.2.3 All Models — Formula CDM

| Model | F1 | ExpRate | NED | Direction |
|-------|:--:|:------:|:---:|:---------:|
| Qwen3-VL-30B | **0.939** | **0.692** | **0.161** | F1/ExpRate: higher, NED: lower |
| Qwen3-VL-2B | 0.865 | 0.504 | 0.220 | |
| **WigtnOCR-2B** | 0.884 | 0.600 | 0.214 | |
| Marker | 0.863 | 0.582 | 0.255 | |

### B.2.4 All Models — Reading Order & Coverage

| Model | Reading Order NED | Pages Evaluated | Skip Rate |
|-------|:-----------------:|:--------------:|:---------:|
| Qwen3-VL-30B | 0.227 | 1,280 | 5.5% |
| Qwen3-VL-2B | 0.300 | 1,100 | 18.8% |
| **WigtnOCR-2B** | **0.211** | 1,276 | 5.8% |
| Marker | **0.165** | 1,349 | 0.4% |

## B.3 Pilot Study Results (3 Documents)

### B.3.1 CER — Complete Results

| Document | Type | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|------|---------------|----------------|---------------|----------------|
| test_1 | Korean Gov (Scan) | N/A | 91.87% | N/A | 536.50% |
| test_2 | English Paper (Scan) | 99.59% | 40.80% | 120.54% | 33.06% |
| test_3 | English Paper (Digital) | 51.25% | 40.79% | 64.11% | 57.71% |

### B.3.2 Structure F1 — Precision/Recall Detail

| Document | Parser | Precision | Recall | F1 | TP | FP | FN |
|----------|--------|-----------|--------|----|----|----|-----|
| test_3 | Text-Baseline | 0.00% | 0.00% | 0.00% | 0 | 11 | 24 |
| test_3 | Image-Baseline | 0.00% | 0.00% | 0.00% | 0 | 0 | 24 |
| test_3 | Text-Advanced | 72.41% | 87.50% | 79.25% | 21 | 8 | 3 |
| test_3 | Image-Advanced | 70.00% | 87.50% | 77.78% | 21 | 9 | 3 |

## B.4 GT Validation Statistics

### B.4.1 Error Type Co-occurrence

| Error Pair | Count | Ratio |
|------------|-------|-------|
| think_tag + truncation | ___ | ___% |
| think_tag + table_broken | ___ | ___% |
| truncation + table_broken | ___ | ___% |
