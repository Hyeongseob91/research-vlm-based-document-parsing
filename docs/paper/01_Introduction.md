# 1. Introduction

## 1.1 Problem Statement

Retrieval-Augmented Generation (RAG) systems depend on accurate document parsing to create meaningful text chunks for semantic search. Traditional OCR methods (PyMuPDF, Tesseract) extract text but fundamentally fail to preserve structural elements:

- **Table structures** collapse into unformatted text streams
- **Multi-column layouts** produce incorrect reading order
- **Header hierarchies** lose semantic relationships
- **Lists and enumerations** merge into continuous paragraphs

These structural failures propagate through the RAG pipeline: poor parsing → poor chunk boundaries → poor retrieval. The central question is whether this structural loss measurably degrades retrieval performance, and if so, whether a compact VLM can address this gap.

Large Vision-Language Models (VLMs) such as Qwen3-VL-30B can directly convert document page images to well-structured markdown. However, deploying 30B+ models in production is impractical due to high inference latency and GPU memory requirements (dual 48GB+ GPUs with tensor parallelism).

## 1.2 Research Questions

This study addresses two themes — **distillation effectiveness** and **downstream impact** — through five research questions:

### Distillation & Parsing Quality

| RQ | Question | Benchmark | Metrics |
|----|----------|-----------|---------|
| RQ1 | Can pseudo-label distillation transfer parsing quality from 30B → 2B? | OmniDocBench | NED, TEDS, CDM, Reading Order |
| RQ2 | Does quality-filtered training enable the student to match or surpass the teacher? | OmniDocBench | Student vs Teacher comparison |
| RQ3 | What quality-cost trade-off does the distilled model offer vs OCR baselines? | OmniDocBench | vs Marker, MinerU |

### Downstream Impact (Two-Step Causal Evaluation)

| RQ | Question | Benchmark | Metrics |
|----|----------|-----------|---------|
| RQ4 | Does VLM-structured output produce better chunks than unstructured text? | KoGovDoc | BC, CS (MoC framework) |
| RQ5 | Do higher-quality chunks yield better retrieval performance? | KoGovDoc | Hit@K, MRR, nDCG |

RQ4-RQ5 together establish the causal chain: **better parsing → better chunks → better retrieval**.

## 1.3 Approach Overview

We propose a four-stage pseudo-labeling pipeline with two-benchmark evaluation:

```
Stage 1: Pseudo GT Generation
    PDF pages → Qwen3-VL-30B (teacher) → Structured Markdown GT
    - 4,501 pages across 49 documents (Korean + English)

Stage 2: GT Quality Validation
    GT Markdown → Qwen3.5-122B (judge LLM, text-only) → Quality Scores
    - 5-dimension scoring, threshold ≥ 3 for acceptance

Stage 3: Training Data Preparation
    Validated GT + PDF images → ms-swift JSONL format
    - Quality filtering + bias correction + train/val split

Stage 4: LoRA Fine-tuning
    Qwen3-VL-2B-Instruct + LoRA → WigtnOCR-2B
```

### Evaluation Design

```
Benchmark 1: OmniDocBench (international, 1,355 pages)
    → RQ1-3: Parsing quality comparison (NED/TEDS/CDM/ReadingOrder)

Benchmark 2: KoGovDoc Val (Korean, 294 samples)
    → Step 1 (RQ4): Parser output → 3 chunking strategies → BC/CS
    → Step 2 (RQ5): Best chunks → embedding → retrieval → Hit@K/MRR
    → Causal chain: structure ↑ → BC/CS ↑ → Hit@K ↑
```

## 1.4 Contributions

1. **Pseudo-Labeling Pipeline**: End-to-end framework for generating, validating, and utilizing pseudo GT from large VLMs to train compact models.

2. **Quality-Filtered Distillation**: Empirical evidence that training on curated pseudo-labels (score ≥ 3) enables the student to surpass the teacher in 4/5 evaluation categories.

3. **Two-Step Causal Evaluation**: Methodology that separately establishes (a) structure → chunking quality and (b) chunking quality → retrieval performance on KoGovDoc, providing evidence for the full parsing-to-retrieval causal chain.

4. **Quality Validation Method**: Text-based (no image) GT validation using a separate judge LLM, achieving reliable contamination detection.

5. **Large-Scale Bilingual Dataset**: KoGovDoc (3,637 pages, Korean) and ArXivPapers (864 pages, English) with per-page quality scores.

6. **Thinking Tag Contamination Analysis**: Identification and resolution of the primary failure mode in VLM pseudo-labeling (36-47% of failures).

## 1.5 Scope and Limitations

**In Scope**: Korean government documents, English academic papers, digital PDFs, tables, multi-column layouts, headers, lists, mathematical notation.

**Out of Scope**: Handwritten text, complex diagrams/charts, real-time streaming, end-to-end answer generation quality (RAGAs).

## 1.6 Paper Organization

- **Section 2**: Related work in document parsing, VLM distillation, chunking, and retrieval evaluation
- **Section 3**: Methodology — pseudo-labeling pipeline and two-step evaluation framework
- **Section 4**: Experimental setup — datasets, models, and configurations
- **Section 5**: Results — OmniDocBench (RQ1-3) and KoGovDoc chunking/retrieval (RQ4-5)
- **Section 6**: Discussion — error analysis, implications, and limitations
- **Section 7**: Conclusion and future work
