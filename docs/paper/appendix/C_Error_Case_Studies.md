# Appendix C: Error Case Studies

## C.1 Purpose

This appendix presents representative error cases from both the pseudo-labeling pipeline (GT generation failures) and the evaluation pipeline (parser comparison). Each case includes root cause analysis and downstream impact assessment. Cases are selected to illustrate the error taxonomy from Section 6.2.

## C.2 Error Taxonomy

| Category | Code | Severity | Pipeline Stage |
|----------|------|----------|---------------|
| Thinking Contamination | THINK_CONTAM | Critical | GT Generation |
| Content Truncation | TRUNCATION | High | GT Generation |
| Table Corruption | TABLE_CORRUPT | Medium | GT Generation / Parsing |
| Hallucination | HALLUC | Critical | Parsing |
| Header Level Error | HEADER_LEVEL | Low | Parsing |
| Reading Order Error | READ_ORDER | High | Parsing |

## C.3 GT Generation Error Cases

### Case 1: Thinking Tag Contamination (THINK_CONTAM)

**Source**: kogov_008, page 879
**Configuration**: `enable_thinking: False` (initial, pre-fix)

**Contaminated Output** (excerpt):
```markdown
Okay, let me look at this document page carefully. I see a table with government
budget allocations. Let me transcribe it...

| 구분 | 예산액 | 집행액 |
|------|--------|--------|
| 기본사업 | 5,230 | 4,891 |
```

**Root Cause**: `enable_thinking: False` suppressed `<think>` tags but model still "thought" — reasoning text leaked directly into content without tag markers, making post-processing detection difficult.

**Fix**: `enable_thinking: True` + `--reasoning-parser qwen3` → thinking goes to `reasoning_content` field, clean content in `content` field.

**Re-generated Output** (clean):
```markdown
| 구분 | 예산액 | 집행액 |
|------|--------|--------|
| 기본사업 | 5,230 | 4,891 |
```

### Case 2: Content Truncation (TRUNCATION)

**Source**: arxiv_033, page 45
**Symptom**: GT ends mid-sentence

**Truncated Output**:
```markdown
## 4.3 Experimental Results

Table 4 shows the comparison of different methods on the GLUE benchmark.
Our approach achieves state-of-the-art results on 7 out of 8 tasks, with
particularly strong performance on tasks requiring
```

**Root Cause**: `max_tokens: 8192` insufficient for complex pages with dense tables and equations.
**Judge Detection**: `completeness_signals` dimension scored 1/5 — "abrupt ending detected."

### Case 3: Table Corruption (TABLE_CORRUPT)

**Source**: kogov_003, page 112
**Symptom**: Merged cells and complex header structure lost

> Pending: will include actual GT vs corrupted output comparison from validation results.

## C.4 Parsing Error Cases

### Case 4: VLM Hallucination on Scanned Korean Document (HALLUC)

**Source**: test_1 (Korean government scan)
**Parser**: Image-Advanced (RapidOCR + VLM 2B)
**CER**: 536.50%

**Analysis**: VLM generated ~19,000 characters from a source containing ~500 visible characters. The model fabricated entire tables, policy descriptions, and budget figures not present in the original document.

**Root Cause**: Low-quality scanned image with poor contrast. VLM "filled in" expected content based on document type rather than actual image content.

**Downstream Impact**: Complete retrieval failure — all fabricated content would be indexed and returned for unrelated queries.

### Case 5: Multi-Column Reading Order (READ_ORDER)

**Source**: test_3 (two-column academic paper)
**Comparison**: PyMuPDF vs VLM

**PyMuPDF Output** (incorrect order):
```
1 Introduction
Chain-of-thought prompting is a technique... 2 Related Work
Prior work has explored various prompting methods...
```

**VLM Output** (correct order):
```markdown
## 1 Introduction

Chain-of-thought prompting is a technique...

## 2 Related Work

Prior work has explored various prompting methods...
```

**Impact on Step 1 (Chunking)**: PyMuPDF's incorrect reading order breaks semantic coherence → header-based chunking impossible (no headers) → semantic chunking also degraded due to topic mixing within chunks.

**Impact on Step 2 (Retrieval)**: Queries about "Introduction" content may retrieve mixed chunks containing both Introduction and Related Work text, lowering Hit@K precision.

### Case 6: Header Level Misassignment (HEADER_LEVEL)

**Source**: Korean government document
**Parser**: VLM 2B (base, pre-LoRA)

**Ground Truth**: `### 1.1 목적` (h3)
**VLM Output**: `## 1.1 목적` (h2)

**Impact**: Minor — header-based chunking creates slightly larger chunks but section boundaries are still detected. Structure F1 counts this as TP (correct element type, wrong level).

## C.5 Error Frequency Summary

> **Status**: Pending — will be populated from OmniDocBench full evaluation.

### C.5.1 GT Generation Errors (Post-Fix)

| Category | Count | % of Total | Mitigation |
|----------|-------|-----------|------------|
| THINK_CONTAM | ___ | ___% | enable_thinking: True + reasoning-parser |
| TRUNCATION | ___ | ___% | Re-generation with higher max_tokens |
| TABLE_CORRUPT | ___ | ___% | Manual review for critical tables |

### C.5.2 Parsing Errors by Parser

| Category | PyMuPDF | 2B base | WigtnOCR-2B | 30B teacher |
|----------|---------|---------|-------------|-------------|
| HALLUC | 0 | ___ | ___ | ___ |
| READ_ORDER | ___ | ___ | ___ | ___ |
| HEADER_LEVEL | N/A | ___ | ___ | ___ |
| TABLE_CORRUPT | ___ | ___ | ___ | ___ |
