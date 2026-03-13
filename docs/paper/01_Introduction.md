# 1. Introduction

## 1.1 Problem Statement

Retrieval-Augmented Generation (RAG) systems depend on accurate document parsing to create meaningful text chunks for semantic search. Traditional OCR methods (PyMuPDF, RapidOCR) extract text but fundamentally fail to preserve structural elements:

- **Table structures** collapse into unformatted text streams
- **Multi-column layouts** produce incorrect reading order
- **Header hierarchies** lose semantic relationships
- **Lists and enumerations** merge into continuous paragraphs

These structural failures propagate through the RAG pipeline: poor parsing → poor chunk boundaries → poor retrieval. Our preliminary experiments confirm this — baseline parsers achieve Structure F1 = 0% on all test documents, meaning zero structural elements are preserved. The central question is whether this structural loss measurably degrades retrieval performance, and if so, whether a compact VLM can address this gap.

Large Vision-Language Models (VLMs) such as Qwen3-VL-30B can directly convert document page images to well-structured markdown. However, deploying 30B+ models in production is impractical due to high inference latency (35-115s per page) and GPU memory requirements (dual 48GB+ GPUs with tensor parallelism).

## 1.2 Research Questions

This study establishes a causal chain — **parsing → chunking → retrieval** — through five research questions organized in two evaluation steps:

### Prerequisite & Core Parsing Quality

| RQ | Question | Metrics | Role |
|----|----------|---------|------|
| RQ1 | Can baseline text extraction provide sufficient quality as VLM input? | CER, WER | Prerequisite validation |
| RQ2 | Does VLM-based parsing preserve document structure significantly better than traditional OCR? | Structure F1, TEDS | Core hypothesis |

### Step 1: Does Better Structure Produce Better Chunks?

| RQ | Question | Metrics | Role |
|----|----------|---------|------|
| RQ3 | Does VLM structural preservation produce measurably better chunk quality than traditional OCR? | BC, CS (across 3 chunking strategies) | Causal link: structure → chunking |

### Step 2: Do Better Chunks Produce Better Retrieval?

| RQ | Question | Metrics | Role |
|----|----------|---------|------|
| RQ4 | Do higher-quality chunks from VLM-structured output yield better retrieval performance? | Hit@K, MRR, nDCG | Causal link: chunking → retrieval |

### Distillation

| RQ | Question | Metrics | Role |
|----|----------|---------|------|
| **RQ5** | **Can a compact VLM (2B) match large VLM (30B) quality through pseudo-label distillation?** | All above metrics | **Practical deployment** |

RQ3-RQ4 together establish the causal chain: **better parsing → better chunks → better retrieval**. RQ5 demonstrates that this quality is achievable with a production-viable model size.

## 1.3 Approach Overview

We propose a four-stage pseudo-labeling pipeline with two-step evaluation:

```
Stage 1: Pseudo GT Generation
    PDF pages → Qwen3-VL-30B (teacher) → Structured Markdown GT
    - 4,501 pages across 49 documents (Korean + English)
    - Concurrent batch processing (4 pages/batch)

Stage 2: GT Quality Validation
    GT Markdown → Qwen3.5-122B (judge LLM, text-only) → Quality Scores
    - 5-dimension scoring (structure, tables, completeness, hallucination, formatting)
    - Score 1-5 per page, threshold ≥ 3 for acceptance

Stage 3: Training Data Preparation
    Validated GT + PDF images → ms-swift JSONL format
    - Quality filtering (score ≥ 3)
    - Bias correction (document-level downsampling)
    - Train/val split (90/10)

Stage 4: LoRA Fine-tuning
    Qwen3-VL-2B-Instruct + LoRA → WigtnOCR-2B
    - ms-swift framework, DeepSpeed ZeRO-2
    - Vision encoder frozen, LLM layers only
```

### Two-Step Evaluation Design

```
Evaluation Step 1: Structure → Chunking Quality  (RQ3)
    Parser outputs → 3 Chunking strategies → BC/CS comparison
    - Header-based chunking (structure-dependent)
    - Semantic chunking (embedding-based)
    - Fixed-size chunking (baseline)
    → Establishes: SF1 ↑ correlates with BC/CS ↑

Evaluation Step 2: Chunking → Retrieval Performance  (RQ4)
    Best chunks from Step 1 → Embedding → Vector search
    - Auto-generated query set from OmniDocBench GT
    - Hit@K, MRR, nDCG metrics
    → Establishes: BC/CS ↑ correlates with Hit@K ↑
```

## 1.4 Contributions

1. **Pseudo-Labeling Pipeline**: An end-to-end framework for generating, validating, and utilizing pseudo ground truth from large VLMs to train compact models for document parsing.

2. **Quality Validation Method**: A text-based (no image) GT validation approach using a separate judge LLM that evaluates five quality dimensions, achieving reliable contamination detection (thinking-tag leakage, truncation, table corruption).

3. **Two-Step Causal Evaluation**: A methodology that separately establishes (a) structure → chunking quality and (b) chunking quality → retrieval performance, providing evidence for the full parsing-to-retrieval causal chain.

4. **Large-Scale Bilingual Dataset**: KoGovDoc (3,637 pages, Korean) and ArXivPapers (864 pages, English) with per-page quality scores and validation metadata.

5. **Practical Deployment Model**: A LoRA-adapted 2B VLM that approaches 30B quality across parsing, chunking, and retrieval metrics while maintaining production-viable inference latency.

## 1.5 Scope and Limitations

**In Scope**: Korean government documents, English academic papers, digital and scanned PDFs, tables, multi-column layouts, headers, lists, mathematical notation.

**Out of Scope**: Handwritten text, complex diagrams/charts, real-time streaming, end-to-end answer generation quality (RAGAs).

## 1.6 Paper Organization

- **Section 2**: Related work in document parsing, VLM applications, chunking strategies, and retrieval evaluation
- **Section 3**: Methodology — pseudo-labeling pipeline and two-step evaluation framework
- **Section 4**: Experimental setup — datasets, models, and configurations
- **Section 5**: Results — Step 1 (chunking quality) and Step 2 (retrieval performance)
- **Section 6**: Discussion — error analysis, implications, and limitations
- **Section 7**: Conclusion and future work
