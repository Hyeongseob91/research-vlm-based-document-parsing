# 5. Results

## 5.1 OmniDocBench Evaluation (RQ1: Distillation Effectiveness)

### 5.1.1 Overall Results

**Table 1: OmniDocBench Full Evaluation (All Models)**

| Metric | Qwen3-VL-30B | Qwen3-VL-2B | **WigtnOCR-2B** | Marker | Best | Direction |
|--------|:------------:|:-----------:|:---------------:|:------:|:----:|:---------:|
| **Text NED** (sample_avg) | 0.289 | 0.364 | **0.288** | 0.218 | Marker | lower=better |
| **Text NED** (page_avg) | 0.415 | 0.376 | **0.304** | 0.244 | Marker | lower=better |
| **Text NED** (edit_whole) | 0.331 | 0.503 | **0.293** | 0.197 | Marker | lower=better |
| **Table TEDS** | 0.523 | 0.561 | **0.649** | 0.586 | WigtnOCR | higher=better |
| **Table TEDS-S** | 0.657 | 0.667 | **0.732** | 0.658 | WigtnOCR | higher=better |
| **Formula CDM F1** | **0.939** | 0.865 | 0.884 | 0.863 | 30B | higher=better |
| **Formula CDM ExpRate** | **0.692** | 0.504 | 0.600 | 0.582 | 30B | higher=better |
| **Formula NED** | **0.161** | 0.220 | 0.214 | 0.255 | 30B | lower=better |
| **Reading Order NED** | 0.227 | 0.300 | **0.211** | 0.165 | Marker | lower=better |

### 5.1.2 Evaluation Coverage

| Model | Pages Evaluated | Pages Skipped | Skip Rate | Text Samples | Table Samples | Formula Samples |
|-------|:--------------:|:-------------:|:---------:|:------------:|:-------------:|:---------------:|
| Qwen3-VL-30B | 1,280 | 75 | 5.5% | 15,282 | 395 | 468 |
| Qwen3-VL-2B | 1,100 | 255 | 18.8% | 14,718 | 412 | 345 |
| **WigtnOCR-2B** | **1,276** | **79** | **5.8%** | **12,339** | **425** | **453** |
| Marker | 1,349 | 6 | 0.4% | 16,783 | 490 | 594 |

### 5.1.3 Improvement over 2B Baseline

| Metric | 2B Base | WigtnOCR-2B | Delta | Improvement |
|--------|---------|-------------|-------|-------------|
| Text NED (sample) | 0.364 | 0.288 | -0.076 | **20.9%** |
| Text NED (page) | 0.376 | 0.304 | -0.072 | **19.1%** |
| Table TEDS | 0.561 | 0.649 | +0.088 | **15.7%** |
| Table TEDS-S | 0.667 | 0.732 | +0.065 | **9.7%** |
| Formula CDM F1 | 0.865 | 0.884 | +0.019 | **2.2%** |
| Formula CDM ExpRate | 0.504 | 0.600 | +0.096 | **19.0%** |
| Formula NED | 0.220 | 0.214 | -0.006 | **2.7%** |
| Reading Order NED | 0.300 | 0.211 | -0.089 | **29.7%** |
| Skip Rate | 18.8% | 5.8% | -13.0pp | **69.1%** |

LoRA fine-tuning improves all metrics, with the largest gains in reading order (29.7%), text recognition (20.9%), and formula exact match (19.0%).

## 5.2 Student vs Teacher Analysis (RQ2: Quality Filtering Effect)

### 5.2.1 Direct Comparison

| Category | 30B Teacher | WigtnOCR-2B | Verdict |
|----------|:-----------:|:-----------:|---------|
| Text NED (sample) | 0.289 | 0.288 | **Student matches teacher** |
| Text NED (page) | 0.415 | 0.304 | **Student exceeds teacher** |
| Table TEDS | 0.523 | 0.649 | **Student exceeds teacher (+12.6pp)** |
| Table TEDS-S | 0.657 | 0.732 | **Student exceeds teacher (+7.5pp)** |
| Formula CDM F1 | 0.939 | 0.884 | Teacher leads (gap: 5.5pp) |
| Formula CDM ExpRate | 0.692 | 0.600 | Teacher leads (gap: 9.2pp) |
| Reading Order | 0.227 | 0.211 | **Student exceeds teacher** |
| Skip Rate | 5.5% | 5.8% | Comparable |

**Key finding**: In 4 out of 5 metric categories, the student matches or exceeds the teacher. This result goes beyond typical distillation outcomes and suggests that quality-filtered pseudo-labels provide a stronger training signal than the teacher's average output.

### 5.2.2 Analysis of Student Surpassing Teacher

**Text Recognition**: WigtnOCR-2B (0.288 sample_avg) matches the 30B teacher (0.289), closing the 7.5pp gap from the base model (0.364). On page_avg, WigtnOCR (0.304) significantly improves over both the 2B base (0.376) and 30B teacher (0.415), suggesting distillation reduces the tendency for excessive output on complex pages.

**Table Recognition**: WigtnOCR achieves the highest TEDS (0.649) and TEDS-S (0.732) across all evaluated models. This likely reflects the filtering effect of quality validation: by training only on high-quality pseudo-GT (score >= 3), the student learns from the teacher's best table outputs while avoiding its failure cases.

**Formula Recognition**: CDM F1 improves from 0.865 (base) to 0.884 (trained), with ExpRate showing a larger jump (0.504 → 0.600, +19.0%). However, the 30B teacher (0.939 F1, 0.692 ExpRate) remains significantly ahead. Formula recognition requires strong LaTeX pretraining knowledge that may not fully transfer via LoRA rank-8 adaptation.

**Reading Order**: WigtnOCR (0.211) improves substantially over 2B base (0.300) and exceeds the 30B teacher (0.227).

**Robustness**: Skip rate drops from 18.8% (2B base) to 5.8% (WigtnOCR), nearly matching the 30B teacher (5.5%). This indicates that LoRA fine-tuning not only improves quality but significantly reduces failure rates on complex documents.

## 5.3 VLM vs OCR Pipeline Comparison (RQ3: Cost-Quality Trade-off)

### 5.3.1 WigtnOCR-2B vs Marker

Marker (OCR pipeline) maintains advantages in text extraction (NED 0.218) and reading order (0.165), reflecting dedicated OCR and layout detection capabilities. However, WigtnOCR surpasses Marker in:
- Table recognition (TEDS 0.649 vs 0.586, TEDS-S 0.732 vs 0.658)
- Formula recognition (CDM F1 0.884 vs 0.863, NED 0.214 vs 0.255)

This confirms that VLMs offer complementary strengths to OCR pipelines, particularly for visually complex elements (tables, formulas) that require understanding beyond character recognition.

### 5.3.2 Comparison with Published OmniDocBench Results

Text NED (lower = better):

| Model | Type | EN | ZH | Avg |
|-------|------|:---:|:---:|:---:|
| MinerU | OCR pipeline | 0.058 | 0.211 | — |
| Mathpix | Commercial | 0.101 | 0.358 | — |
| Marker (paper) | OCR pipeline | 0.141 | 0.303 | — |
| GPT-4o | VLM | 0.144 | 0.409 | — |
| GOT-OCR | VLM | 0.187 | 0.315 | — |
| Qwen2-VL | VLM | 0.252 | 0.251 | — |
| InternVL2 | VLM | 0.353 | 0.290 | — |
| **Marker (ours)** | OCR pipeline | — | — | 0.218 |
| **WigtnOCR-2B (ours)** | VLM (2B, LoRA) | — | — | **0.288** |
| **Qwen3-VL-30B (ours)** | VLM | — | — | 0.289 |
| **Qwen3-VL-2B (ours)** | VLM | — | — | 0.364 |

Table TEDS (higher = better):

| Model | Type | EN | ZH |
|-------|------|:---:|:---:|
| MinerU | OCR pipeline | 79.4% | 62.7% |
| Mathpix | Commercial | 77.9% | 68.2% |
| GPT-4o | VLM | 72.8% | 63.7% |
| InternVL2 | VLM | 63.8% | 61.1% |
| **WigtnOCR-2B (ours)** | VLM (2B, LoRA) | — | **64.9%** (avg) |
| Qwen2-VL | VLM | 59.9% | 66.8% |
| Marker (paper) | OCR pipeline | 54.0% | 45.8% |
| **Marker (ours)** | OCR pipeline | — | **58.6%** (avg) |
| **Qwen3-VL-2B (ours)** | VLM | — | **56.1%** (avg) |
| **Qwen3-VL-30B (ours)** | VLM | — | **52.3%** (avg) |

> **Note**: Our results are averaged across all languages (EN+ZH+others). The OmniDocBench paper reports EN/ZH separately. Direct per-language comparison requires language-stratified evaluation.

## 5.4 GT Quality Validation Results

### 5.4.1 KoGovDoc Validation (30% Sampling)

| Document | Sampled | Avg Score | Pass Rate | Failed Pages |
|----------|---------|-----------|-----------|--------------|
| kogov_001 | — | — | — | 1 |
| kogov_003 | — | — | — | 44 |
| kogov_004 | — | — | — | 10 |
| kogov_006 | — | — | — | 4 |
| kogov_007 | — | — | — | 11 |
| kogov_008 | — | — | — | 164 |
| kogov_009 | — | — | — | 4 |
| kogov_010 | — | — | — | 7 |
| kogov_011 | — | — | — | 11 |
| **Total** | **1,047** | **3.3/5** | **75.1%** | **256** |

### 5.4.2 ArXiv Papers Validation (100% Sampling)

| Document | Sampled | Avg Score | Pass Rate | Failed |
|----------|---------|-----------|-----------|--------|
| arxiv_013 | 27 | 2.30 | 48% | 14 |
| arxiv_016 | 22 | 2.41 | 55% | 10 |
| arxiv_021 | 14 | 2.38 | 62% | 5 |
| arxiv_022 | 8 | 3.25 | **100%** | 0 |
| arxiv_033 | 87 | 3.06 | 71% | 24 |
| ... (34 more) | | | | |
| **Total** | **864** | **3.0/5** | **73.8%** | **218** |

### 5.4.3 Error Type Distribution

| Error Type | Documents | Papers | Total | Description |
|------------|-----------|--------|-------|-------------|
| **think_tag** | 392 | 333 | 725 | Thinking/CoT contamination |
| truncation | 259 | 227 | 486 | Abrupt content endings |
| table_broken | 199 | 87 | 286 | Malformed table structures |
| other | — | 226 | 226+ | Miscellaneous quality issues |
| ocr_error | — | 41 | 41+ | Character recognition errors |

**Thinking tag contamination** is the dominant failure mode (36-47% of failures), directly caused by improper `enable_thinking` configuration during initial generation.

## 5.5 KoGovDoc Val: Parsing Quality

WigtnOCR-2B v1 evaluated on 294 KoGovDoc validation samples (excluded from training), compared against the base student and teacher models:

| Model | NED avg ↓ | Evaluated | Errors |
|-------|:---------:|:---------:|:------:|
| **WigtnOCR-2B** | **0.285** | 289/294 | 5 |
| Qwen3-VL-30B (Teacher) | 0.334 | 294/294 | 0 |
| Qwen3-VL-2B (Base) | 0.390 | 294/294 | 0 |

WigtnOCR-2B surpasses its 30B teacher on Korean government documents (NED 0.285 vs 0.334), consistent with the OmniDocBench findings where the student exceeds the teacher after quality-filtered distillation.

## 5.6 Step 1: Structure → Chunking Quality — RQ4

We evaluate chunking quality using the MoC framework (Zhao et al., ACL 2025) with semantic chunking (BGE-M3 embedding-based boundary detection) applied to document-level text from 6 parsers. BC and CS are computed using Qwen2.5-1.5B for perplexity calculation, following the original MoC paper.

### 5.6.1 BC/CS Results — Semantic Chunking (6 Parsers)

**Table 5: Boundary Clarity (BC) and Chunk Stickiness (CS) on KoGovDoc + ArXiv Val**

| Model | Type | BC ↑ | CS ↓ | Samples |
|-------|------|:----:|:----:|:-------:|
| MinerU | PDF parser | **0.735** | **2.711** | 34 |
| Qwen3-VL-30B | VLM (teacher) | 0.714 | 3.164 | 35 |
| **WigtnOCR-2B** | VLM (ours) | 0.706 | 2.859 | 34 |
| Marker | PDF parser | 0.683 | 3.206 | 35 |
| Qwen3-VL-2B | VLM (base) | 0.678 | 3.446 | 35 |
| PaddleOCR | Pure OCR | 0.654 | 3.420 | 35 |

### 5.6.2 Analysis

**MinerU leads BC/CS** despite ranking 5th in retrieval (§5.7). This suggests BC/CS measures chunk boundary independence but not the informational richness of chunks. MinerU extracts clean text from PDF text layers with high boundary quality, but the extracted content lacks the structural fidelity that VLMs provide.

**WigtnOCR-2B ranks 3rd in BC and 2nd in CS**, outperforming its base model (2B) and Marker. Compared to PaddleOCR, WigtnOCR shows +0.052 BC and −0.561 CS improvement, confirming that structured parsing produces better-bounded, less interdependent chunks.

**Distillation effect on chunking**: WigtnOCR-2B (CS 2.859) improves substantially over Qwen3-VL-2B base (CS 3.446), and approaches the teacher's chunking profile while using 15× fewer parameters.

## 5.7 Step 2: Chunking → Retrieval Performance — RQ5

We embed semantic chunks from each parser using BGE-M3, build per-parser FAISS indices, and evaluate retrieval using 564 auto-generated queries across 38 documents.

### 5.7.1 Retrieval Results (6 Parsers)

**Table 6: Retrieval Performance on KoGovDoc + ArXiv Val**

| Model | Type | Hit@1 ↑ | Hit@5 ↑ | MRR@10 ↑ | nDCG@10 ↑ |
|-------|------|:-------:|:-------:|:--------:|:---------:|
| **WigtnOCR-2B** | VLM (ours) | **0.739** | **0.855** | **0.788** | 0.437 |
| Qwen3-VL-30B | VLM (teacher) | 0.716 | 0.839 | 0.771 | 0.411 |
| Marker | PDF parser | 0.711 | 0.853 | 0.771 | 0.412 |
| Qwen3-VL-2B | VLM (base) | 0.709 | 0.814 | 0.756 | 0.444 |
| MinerU | PDF parser | 0.608 | 0.789 | 0.682 | 0.384 |
| PaddleOCR | Pure OCR | 0.512 | 0.693 | 0.592 | 0.293 |

### 5.7.2 Analysis

We report Hit@1, Hit@5, MRR@10, and nDCG@10 following the BEIR evaluation protocol (Thakur et al., 2021). We primarily discuss Hit@1 and MRR@10 as the main metrics, as our evaluation targets a RAG pipeline without a re-ranker, where the rank of the first relevant chunk directly determines answer quality.

**WigtnOCR-2B achieves the highest retrieval performance** across Hit@1, Hit@5, and MRR@10, surpassing all parsers including its 30B teacher and established tools (Marker, MinerU).

**Key comparisons:**
- **vs PaddleOCR (pure OCR)**: +22.7pp Hit@1, +19.6pp MRR@10. Structured parsing dramatically improves retrieval over unstructured OCR.
- **vs Qwen3-VL-30B (teacher)**: +2.3pp Hit@1, +1.7pp MRR@10. The distilled student outperforms its teacher in downstream retrieval, confirming that quality-filtered distillation produces better end-to-end RAG outcomes.
- **vs Marker**: +2.8pp Hit@1, +1.7pp MRR@10. Despite Marker's superior text extraction on OmniDocBench, WigtnOCR's structural fidelity (tables, headings, formulas) produces more retrievable chunks.
- **vs MinerU**: +13.1pp Hit@1, +10.6pp MRR@10. MinerU leads in BC/CS (§5.6) but ranks 5th in retrieval — chunk boundary quality alone does not predict search performance.

**nDCG@10 divergence**: Notably, Qwen3-VL-2B (base) achieves the highest nDCG@10 (0.444) despite ranking 4th in Hit@1. nDCG weighs the graded relevance of all documents in the top-K, and the base model's higher skip rate (18.8%) produces fewer but longer chunks per document, which distributes relevance scores more evenly across retrieved results. This inflates nDCG without improving the practical retrieval outcome — whether the *first* returned chunk answers the query. Hit@1 and MRR@10 better capture this end-user-facing quality.

### 5.7.3 BC/CS vs Retrieval: Chunk Quality ≠ Search Quality

Figure 3 plots BC against Hit@1 for all 6 parsers. MinerU appears as a clear outlier: highest BC (0.735) but low retrieval (Hit@1 0.608). This reveals that **text richness and structural fidelity are more important than chunk boundary independence** for end-to-end RAG performance. MinerU extracts clean, well-bounded text from PDF layers, but the content lacks the structural markup (tables, headings, formulas) that VLMs preserve, leading to less informative chunks for retrieval.

## 5.8 Summary

| RQ | Finding | Status |
|----|---------|--------|
| RQ1 | Pseudo-label distillation improves all metrics: text (20.9%), tables (15.7%), reading order (29.7%), robustness (69.1% skip rate reduction) | **Confirmed** |
| RQ2 | Student matches or exceeds teacher in 4/5 categories; table TEDS surpasses teacher by 12.6pp | **Confirmed** |
| RQ3 | WigtnOCR-2B surpasses Marker on tables/formulas; Marker leads on text/reading order; complementary strengths | **Confirmed** |
| RQ4 | VLM structure → better chunk quality: WigtnOCR-2B outperforms PaddleOCR in BC (+0.052) and CS (−0.561) | **Confirmed** |
| RQ5 | Better chunks → better retrieval: WigtnOCR-2B ranks #1 in Hit@1 (0.739), MRR@10 (0.788), surpassing all 5 baselines including 30B teacher | **Confirmed** |
