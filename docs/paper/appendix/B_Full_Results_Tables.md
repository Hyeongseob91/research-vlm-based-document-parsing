# Appendix B: Full Results Tables

## B.1 Purpose

This appendix provides complete numerical results for all experiments, including per-parser breakdowns and full precision/recall analysis. Main text presents aggregate results; this appendix enables detailed reproduction and analysis.

## B.2 Pilot Study Results (3 Documents)

### B.2.1 CER — Complete Results

| Document | Type | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|------|---------------|----------------|---------------|----------------|
| test_1 | Korean Gov (Scan) | N/A | 91.87% | N/A | 536.50% |
| test_2 | English Paper (Scan) | 99.59% | 40.80% | 120.54% | 33.06% |
| test_3 | English Paper (Digital) | 51.25% | 40.79% | 64.11% | 57.71% |

### B.2.2 Structure F1 — Precision/Recall Detail

| Document | Parser | Precision | Recall | F1 | TP | FP | FN |
|----------|--------|-----------|--------|----|----|----|-----|
| test_3 | Text-Baseline | 0.00% | 0.00% | 0.00% | 0 | 11 | 24 |
| test_3 | Image-Baseline | 0.00% | 0.00% | 0.00% | 0 | 0 | 24 |
| test_3 | Text-Advanced | 72.41% | 87.50% | 79.25% | 21 | 8 | 3 |
| test_3 | Image-Advanced | 70.00% | 87.50% | 77.78% | 21 | 9 | 3 |

### B.2.3 Latency

| Document | Parser | Total Time | Stage 1 | Stage 2 |
|----------|--------|------------|---------|---------|
| test_3 | Text-Baseline | 2.31s | 2.31s | — |
| test_3 | Image-Baseline | 0.27s | 0.27s | — |
| test_3 | Text-Advanced | 42.92s | 2.28s | 40.64s |
| test_3 | Image-Advanced | 35.75s | 0.27s | 35.48s |

## B.3 OmniDocBench Full-Scale Results

> **Status**: Pending — will be populated after evaluation pipeline completion.

### B.3.1 Parsing Quality — All Parsers

| Parser | CER ↓ | WER ↓ | Structure F1 ↑ | TEDS ↑ | Latency |
|--------|-------|-------|----------------|--------|---------|
| PyMuPDF | ___ | ___ | ___ | ___ | ___s |
| RapidOCR | ___ | ___ | ___ | ___ | ___s |
| 2B base | ___ | ___ | ___ | ___ | ___s |
| WigtnOCR-2B (LoRA) | ___ | ___ | ___ | ___ | ___s |
| 30B teacher | ___ | ___ | ___ | ___ | ___s |

### B.3.2 Step 1: Chunking Quality — BC (Parser × Chunker)

| Parser | Header-based | Semantic | Fixed-512 |
|--------|-------------|----------|-----------|
| PyMuPDF | N/A | ___ | ___ |
| 2B base | ___ | ___ | ___ |
| WigtnOCR-2B | ___ | ___ | ___ |
| 30B teacher | ___ | ___ | ___ |

### B.3.3 Step 1: Chunking Quality — CS (Parser × Chunker)

| Parser | Header-based | Semantic | Fixed-512 |
|--------|-------------|----------|-----------|
| PyMuPDF | N/A | ___ | ___ |
| 2B base | ___ | ___ | ___ |
| WigtnOCR-2B | ___ | ___ | ___ |
| 30B teacher | ___ | ___ | ___ |

### B.3.4 Step 2: Retrieval Performance

| Parser | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 |
|--------|-------|-------|--------|-----|---------|
| PyMuPDF | ___ | ___ | ___ | ___ | ___ |
| 2B base | ___ | ___ | ___ | ___ | ___ |
| WigtnOCR-2B | ___ | ___ | ___ | ___ | ___ |
| 30B teacher | ___ | ___ | ___ | ___ | ___ |

## B.4 GT Validation Statistics

### B.4.1 Score Distribution

| Score | KoGovDoc (n=1,047) | ArXivPapers (n=864) |
|-------|--------------------|---------------------|
| 1 | ___ (___%) | ___ (___%) |
| 2 | ___ (___%) | ___ (___%) |
| 3 | ___ (___%) | ___ (___%) |
| 4 | ___ (___%) | ___ (___%) |
| 5 | ___ (___%) | ___ (___%) |

### B.4.2 Error Type Co-occurrence

| Error Pair | Count | Ratio |
|------------|-------|-------|
| think_tag + truncation | ___ | ___% |
| think_tag + table_broken | ___ | ___% |
| truncation + table_broken | ___ | ___% |
