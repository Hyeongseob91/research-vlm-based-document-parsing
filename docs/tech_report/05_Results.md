# 5. Results

## 5.1 Lexical Accuracy Results

### 5.1.1 Character Error Rate (CER)

<!-- TODO: Update with actual experimental results -->

| Document | VLM (Qwen3-VL) | pdfplumber | Docling+RapidOCR |
|----------|----------------|------------|------------------|
| test_1 (Korean) | TBD% | TBD% | TBD% |
| test_2 (Receipt) | TBD% | N/A | TBD% |
| test_3 (Academic) | 56.29%* | 99.62%* | N/A* |
| **Average** | TBD% | TBD% | TBD% |

*Preliminary results from initial experiments

**Key Observations**:
<!-- TODO: Fill after experiments -->
1. [Observation about VLM performance]
2. [Observation about traditional OCR limitations]
3. [Document type specific findings]

### 5.1.2 Word Error Rate (WER)

| Document | VLM (Qwen3-VL) | pdfplumber | Docling+RapidOCR |
|----------|----------------|------------|------------------|
| test_1 (Korean) | TBD% | TBD% | TBD% |
| test_2 (Receipt) | TBD% | N/A | TBD% |
| test_3 (Academic) | 70.85%* | 100%* | N/A* |
| **Average** | TBD% | TBD% | TBD% |

**Tokenization Impact**:
- Korean: MeCab morphological tokenization
- English: Whitespace tokenization

### 5.1.3 Statistical Significance

| Comparison | Metric | Difference | p-value | Cohen's d |
|------------|--------|------------|---------|-----------|
| VLM vs pdfplumber | CER | TBD | TBD | TBD |
| VLM vs pdfplumber | WER | TBD | TBD | TBD |
| VLM vs Docling | CER | TBD | TBD | TBD |
| VLM vs Docling | WER | TBD | TBD | TBD |

## 5.2 Structural Integrity Results

### 5.2.1 Boundary Score (BS)

| Document | VLM | pdfplumber | Docling |
|----------|-----|------------|---------|
| test_1 | TBD | TBD | TBD |
| test_2 | TBD | TBD | TBD |
| test_3 | TBD | TBD | TBD |
| **Average** | TBD | TBD | TBD |

**Interpretation**: Higher BS = Better semantic boundary alignment

### 5.2.2 Chunk Score (CS)

| Document | VLM | pdfplumber | Docling |
|----------|-----|------------|---------|
| test_1 | TBD | TBD | TBD |
| test_2 | TBD | TBD | TBD |
| test_3 | TBD | TBD | TBD |
| **Average** | TBD | TBD | TBD |

**Interpretation**: Higher CS = Better intra-chunk coherence

### 5.2.3 Structure Element Analysis

| Element Type | VLM Detection | OCR Detection | Delta |
|--------------|---------------|---------------|-------|
| Headers | TBD% | TBD% | TBD |
| Tables | TBD% | TBD% | TBD |
| Lists | TBD% | TBD% | TBD |
| Code Blocks | TBD% | TBD% | TBD |

## 5.3 Retrieval Performance Results

### 5.3.1 Hit Rate@k

**Baseline (pdfplumber/RapidOCR)**:

| Document | HR@1 | HR@3 | HR@5 | HR@10 |
|----------|------|------|------|-------|
| test_1 | TBD% | TBD% | TBD% | TBD% |
| test_2 | TBD% | TBD% | TBD% | TBD% |
| test_3 | TBD% | TBD% | TBD% | TBD% |
| **Average** | TBD% | TBD% | TBD% | TBD% |

**VLM (Qwen3-VL)**:

| Document | HR@1 | HR@3 | HR@5 | HR@10 |
|----------|------|------|------|-------|
| test_1 | TBD% | TBD% | TBD% | TBD% |
| test_2 | TBD% | TBD% | TBD% | TBD% |
| test_3 | TBD% | TBD% | TBD% | TBD% |
| **Average** | TBD% | TBD% | TBD% | TBD% |

**Improvement**:

| Document | ΔHR@1 | ΔHR@3 | ΔHR@5 | ΔHR@10 |
|----------|-------|-------|-------|--------|
| test_1 | TBD | TBD | TBD | TBD |
| test_2 | TBD | TBD | TBD | TBD |
| test_3 | TBD | TBD | TBD | TBD |

### 5.3.2 Mean Reciprocal Rank (MRR)

| Parser | MRR | 95% CI |
|--------|-----|--------|
| Baseline | TBD | [TBD, TBD] |
| VLM | TBD | [TBD, TBD] |
| **Improvement** | TBD | |

### 5.3.3 Query Type Analysis

| Query Type | Baseline HR@5 | VLM HR@5 | Improvement |
|------------|---------------|----------|-------------|
| Factual | TBD% | TBD% | TBD |
| Table Lookup | TBD% | TBD% | TBD |
| Multi-hop | TBD% | TBD% | TBD |
| Inferential | TBD% | TBD% | TBD |

**Key Finding**: [Expected: VLM shows largest improvement on table-related queries]

## 5.4 Latency Analysis

### 5.4.1 Processing Time

| Parser | test_1 | test_2 | test_3 | Avg/Page |
|--------|--------|--------|--------|----------|
| VLM | TBD s | TBD s | TBD s | TBD s |
| pdfplumber | TBD s | TBD s | TBD s | TBD s |
| Docling | TBD s | TBD s | TBD s | TBD s |

*Preliminary: VLM ~15s, pdfplumber ~18s, RapidOCR ~7s for test_3

### 5.4.2 Cost-Quality Trade-off

| Parser | Avg CER | Avg Latency | Quality/Cost Ratio |
|--------|---------|-------------|-------------------|
| VLM | TBD% | TBD s | TBD |
| pdfplumber | TBD% | TBD s | TBD |
| Docling | TBD% | TBD s | TBD |

## 5.5 Ablation Study Results

### 5.5.1 Prompt Variation Impact

| Prompt Version | CER | WER | Hallucination Rate |
|----------------|-----|-----|-------------------|
| v1 (extraction) | TBD% | TBD% | TBD% |
| v2 (transcription) | TBD% | TBD% | TBD% |
| v3 (minimal) | TBD% | TBD% | TBD% |
| v4 (XML) | TBD% | TBD% | TBD% |

### 5.5.2 Resolution Impact

| DPI | CER | Latency | Quality/Cost |
|-----|-----|---------|--------------|
| 72 | TBD% | TBD s | TBD |
| 150 | TBD% | TBD s | TBD |
| 300 | TBD% | TBD s | TBD |

### 5.5.3 Chunking Strategy Impact

| Strategy | BS | CS | HR@5 |
|----------|----|----|------|
| Fixed (500) | TBD | TBD | TBD% |
| Semantic | TBD | TBD | TBD% |
| Hierarchical | TBD | TBD | TBD% |

## 5.6 Summary of Key Results

### 5.6.1 Hypothesis Validation

| Research Question | Finding | Support |
|-------------------|---------|---------|
| RQ1: Lexical Fidelity | VLM achieves TBD% lower CER | TBD |
| RQ2: Structural Preservation | VLM achieves TBD higher BS | TBD |
| RQ3: Retrieval Impact | VLM improves HR@5 by TBD% | TBD |

### 5.6.2 Document Type Analysis

| Document Type | Best Parser | Rationale |
|---------------|-------------|-----------|
| Digital PDF (simple) | TBD | TBD |
| Digital PDF (complex) | TBD | TBD |
| Scanned Document | TBD | TBD |
| Multi-column | TBD | TBD |
| Table-heavy | TBD | TBD |

### 5.6.3 Statistical Summary

```
Overall Performance Improvement (VLM vs Baseline):
├── CER: -TBD% (p=TBD, d=TBD)
├── WER: -TBD% (p=TBD, d=TBD)
├── BS:  +TBD  (p=TBD, d=TBD)
├── CS:  +TBD  (p=TBD, d=TBD)
├── HR@5: +TBD% (p=TBD, d=TBD)
└── MRR: +TBD  (p=TBD, d=TBD)
```
