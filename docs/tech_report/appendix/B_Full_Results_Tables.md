# Appendix B: Full Results Tables

## B.1 Lexical Accuracy Results

### B.1.1 Character Error Rate (CER) - Complete Results

#### Per-Document CER

| Document ID | Document Type | VLM (Qwen3-VL) | pdfplumber | Docling+RapidOCR |
|-------------|---------------|----------------|------------|------------------|
| test_1 | Korean Gov Doc | TBD% | TBD% | TBD% |
| test_2 | Receipt Image | TBD% | N/A | TBD% |
| test_3 | Academic Paper | 56.29%* | 99.62%* | N/A* |

*Preliminary results

#### CER Breakdown by Error Type

| Document | Parser | Substitutions | Deletions | Insertions | Total Chars |
|----------|--------|--------------|-----------|------------|-------------|
| test_1 | VLM | TBD | TBD | TBD | TBD |
| test_1 | pdfplumber | TBD | TBD | TBD | TBD |
| test_1 | Docling | TBD | TBD | TBD | TBD |
| test_2 | VLM | TBD | TBD | TBD | TBD |
| test_2 | Docling | TBD | TBD | TBD | TBD |
| test_3 | VLM | TBD | TBD | TBD | TBD |
| test_3 | pdfplumber | TBD | TBD | TBD | TBD |

### B.1.2 Word Error Rate (WER) - Complete Results

#### Per-Document WER

| Document ID | Document Type | Tokenizer | VLM | pdfplumber | Docling |
|-------------|---------------|-----------|-----|------------|---------|
| test_1 | Korean | MeCab | TBD% | TBD% | TBD% |
| test_2 | Korean | MeCab | TBD% | N/A | TBD% |
| test_3 | English | Whitespace | 70.85%* | 100%* | N/A* |

*Preliminary results

#### WER Breakdown by Error Type

| Document | Parser | Substitutions | Deletions | Insertions | Total Words |
|----------|--------|--------------|-----------|------------|-------------|
| test_1 | VLM | TBD | TBD | TBD | TBD |
| test_1 | pdfplumber | TBD | TBD | TBD | TBD |
| test_2 | VLM | TBD | TBD | TBD | TBD |
| test_3 | VLM | TBD | TBD | TBD | TBD |
| test_3 | pdfplumber | TBD | TBD | TBD | TBD |

## B.2 Structural Integrity Results

### B.2.1 Boundary Score (BS) - Complete Results

| Document | VLM BS | pdfplumber BS | Docling BS | GT Boundaries |
|----------|--------|---------------|------------|---------------|
| test_1 | TBD | TBD | TBD | TBD |
| test_2 | TBD | N/A | TBD | TBD |
| test_3 | TBD | TBD | N/A | TBD |
| **Mean** | TBD | TBD | TBD | - |
| **Std** | TBD | TBD | TBD | - |

### B.2.2 Chunk Score (CS) - Complete Results

| Document | VLM CS | pdfplumber CS | Docling CS |
|----------|--------|---------------|------------|
| test_1 | TBD | TBD | TBD |
| test_2 | TBD | N/A | TBD |
| test_3 | TBD | TBD | N/A |
| **Mean** | TBD | TBD | TBD |
| **Std** | TBD | TBD | TBD |

### B.2.3 Structure Element Detection

#### VLM Parser

| Document | Headers | Tables | Lists | Code | Blockquotes |
|----------|---------|--------|-------|------|-------------|
| test_1 | TBD | TBD | TBD | TBD | TBD |
| test_2 | TBD | TBD | TBD | TBD | TBD |
| test_3 | TBD | TBD | TBD | TBD | TBD |

#### Ground Truth

| Document | Headers | Tables | Lists | Code | Blockquotes |
|----------|---------|--------|-------|------|-------------|
| test_1 | TBD | TBD | TBD | TBD | TBD |
| test_2 | TBD | TBD | TBD | TBD | TBD |
| test_3 | TBD | TBD | TBD | TBD | TBD |

#### Detection Accuracy

| Element | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Headers | TBD% | TBD% | TBD |
| Tables | TBD% | TBD% | TBD |
| Lists | TBD% | TBD% | TBD |
| Code | TBD% | TBD% | TBD |

## B.3 Retrieval Performance Results

### B.3.1 Hit Rate@k - Baseline (pdfplumber/RapidOCR)

| Document | Q&A Count | HR@1 | HR@3 | HR@5 | HR@10 |
|----------|-----------|------|------|------|-------|
| test_1 | TBD | TBD% | TBD% | TBD% | TBD% |
| test_2 | TBD | TBD% | TBD% | TBD% | TBD% |
| test_3 | TBD | TBD% | TBD% | TBD% | TBD% |
| **Total** | TBD | TBD% | TBD% | TBD% | TBD% |

### B.3.2 Hit Rate@k - VLM (Qwen3-VL)

| Document | Q&A Count | HR@1 | HR@3 | HR@5 | HR@10 |
|----------|-----------|------|------|------|-------|
| test_1 | TBD | TBD% | TBD% | TBD% | TBD% |
| test_2 | TBD | TBD% | TBD% | TBD% | TBD% |
| test_3 | TBD | TBD% | TBD% | TBD% | TBD% |
| **Total** | TBD | TBD% | TBD% | TBD% | TBD% |

### B.3.3 Hit Rate Improvement (VLM - Baseline)

| Document | ΔHR@1 | ΔHR@3 | ΔHR@5 | ΔHR@10 |
|----------|-------|-------|-------|--------|
| test_1 | TBD | TBD | TBD | TBD |
| test_2 | TBD | TBD | TBD | TBD |
| test_3 | TBD | TBD | TBD | TBD |
| **Mean** | TBD | TBD | TBD | TBD |

### B.3.4 MRR Results

| Parser | test_1 | test_2 | test_3 | Mean | 95% CI |
|--------|--------|--------|--------|------|--------|
| Baseline | TBD | TBD | TBD | TBD | [TBD, TBD] |
| VLM | TBD | TBD | TBD | TBD | [TBD, TBD] |
| **Δ** | TBD | TBD | TBD | TBD | - |

### B.3.5 Hit Rate by Query Type

#### Baseline Parser

| Query Type | Count | HR@1 | HR@3 | HR@5 |
|------------|-------|------|------|------|
| Factual | TBD | TBD% | TBD% | TBD% |
| Table Lookup | TBD | TBD% | TBD% | TBD% |
| Multi-hop | TBD | TBD% | TBD% | TBD% |
| Inferential | TBD | TBD% | TBD% | TBD% |

#### VLM Parser

| Query Type | Count | HR@1 | HR@3 | HR@5 |
|------------|-------|------|------|------|
| Factual | TBD | TBD% | TBD% | TBD% |
| Table Lookup | TBD | TBD% | TBD% | TBD% |
| Multi-hop | TBD | TBD% | TBD% | TBD% |
| Inferential | TBD | TBD% | TBD% | TBD% |

## B.4 Latency Results

### B.4.1 Processing Time (seconds)

| Document | Pages | VLM | pdfplumber | Docling |
|----------|-------|-----|------------|---------|
| test_1 | TBD | TBD s | TBD s | TBD s |
| test_2 | 1 | TBD s | N/A | TBD s |
| test_3 | TBD | 15.61s* | 18.12s* | 6.85s* |

*Preliminary results

### B.4.2 Per-Page Latency

| Parser | Min | Max | Mean | Std | Median |
|--------|-----|-----|------|-----|--------|
| VLM | TBD s | TBD s | TBD s | TBD s | TBD s |
| pdfplumber | TBD s | TBD s | TBD s | TBD s | TBD s |
| Docling | TBD s | TBD s | TBD s | TBD s | TBD s |

## B.5 Ablation Study Results

### B.5.1 Prompt Variation - CER

| Prompt | test_1 | test_2 | test_3 | Mean |
|--------|--------|--------|--------|------|
| v1 (Extraction) | TBD% | TBD% | TBD% | TBD% |
| v2 (Transcription) | TBD% | TBD% | TBD% | TBD% |
| v3 (Minimal) | TBD% | TBD% | TBD% | TBD% |
| v4 (XML) | TBD% | TBD% | TBD% | TBD% |

### B.5.2 Prompt Variation - Hallucination Rate

| Prompt | test_1 | test_2 | test_3 | Mean |
|--------|--------|--------|--------|------|
| v1 (Extraction) | TBD% | TBD% | TBD% | TBD% |
| v2 (Transcription) | TBD% | TBD% | TBD% | TBD% |
| v3 (Minimal) | TBD% | TBD% | TBD% | TBD% |
| v4 (XML) | TBD% | TBD% | TBD% | TBD% |

### B.5.3 Resolution Study

| DPI | CER | WER | Latency | Image Size |
|-----|-----|-----|---------|------------|
| 72 | TBD% | TBD% | TBD s | TBD KB |
| 150 | TBD% | TBD% | TBD s | TBD KB |
| 300 | TBD% | TBD% | TBD s | TBD KB |

### B.5.4 Chunking Strategy Study

| Strategy | chunk_size | overlap | BS | CS | HR@5 |
|----------|------------|---------|----|----|------|
| Fixed | 500 | 50 | TBD | TBD | TBD% |
| Fixed | 1000 | 100 | TBD | TBD | TBD% |
| Semantic | auto | - | TBD | TBD | TBD% |
| Hierarchical | section | 0 | TBD | TBD | TBD% |

## B.6 Statistical Analysis

### B.6.1 Paired t-test Results (VLM vs Baseline)

| Metric | Mean Diff | t-statistic | p-value | Significant? |
|--------|-----------|-------------|---------|--------------|
| CER | TBD | TBD | TBD | TBD |
| WER | TBD | TBD | TBD | TBD |
| BS | TBD | TBD | TBD | TBD |
| CS | TBD | TBD | TBD | TBD |
| HR@5 | TBD | TBD | TBD | TBD |
| MRR | TBD | TBD | TBD | TBD |

*Significance threshold: α = 0.05

### B.6.2 Effect Size (Cohen's d)

| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| CER | TBD | TBD |
| WER | TBD | TBD |
| BS | TBD | TBD |
| CS | TBD | TBD |
| HR@5 | TBD | TBD |
| MRR | TBD | TBD |

*Small: 0.2, Medium: 0.5, Large: 0.8

### B.6.3 Bootstrap 95% Confidence Intervals

| Metric | Parser | Mean | 95% CI Lower | 95% CI Upper |
|--------|--------|------|--------------|--------------|
| CER | VLM | TBD% | TBD% | TBD% |
| CER | Baseline | TBD% | TBD% | TBD% |
| HR@5 | VLM | TBD% | TBD% | TBD% |
| HR@5 | Baseline | TBD% | TBD% | TBD% |

*1000 bootstrap resamples

## B.7 Raw Data Reference

All raw experimental data is stored in:
- `results/lexical/` - CER, WER calculations
- `results/structural/` - BS, CS calculations
- `results/retrieval/` - HR@k, MRR calculations
- `results/latency/` - Processing time measurements
- `results/ablation/` - Ablation study data

Data format: JSON with timestamps and configuration metadata
