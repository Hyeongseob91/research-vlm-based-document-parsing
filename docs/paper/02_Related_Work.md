# 2. Related Work

## 2.1 Document Understanding and Layout Analysis

### 2.1.1 Traditional OCR Systems

- **Tesseract** (Smith, 2007): Open-source OCR engine with LSTM-based recognition
- **ABBYY FineReader**: Enterprise-grade OCR with document structure recognition

**Limitations**: These systems focus on character recognition and struggle with complex table structures, multi-column reading order, and semantic relationship preservation.

### 2.1.2 PDF Text Extraction Libraries

- **pdfplumber** (Singer-Vine, 2022): Precise positional text extraction with table detection
- **PyMuPDF (fitz)**: Fast PDF rendering and text extraction
- **pdfminer.six**: Layout-aware text extraction

### 2.1.3 Layout-Aware Models

- **LayoutLM** (Xu et al., 2020): Pre-trained model combining text and layout information
- **LayoutLMv3** (Huang et al., 2022): Unified text-image-layout pre-training

## 2.2 Vision-Language Models for Document Parsing

### 2.2.1 General-Purpose VLMs

- **GPT-4V** (OpenAI, 2023), **Claude 3** (Anthropic, 2024), **Gemini Pro Vision** (Google, 2024)

### 2.2.2 Document-Specialized VLMs

- **Qwen-VL** (Bai et al., 2023): Open-source VLM with strong Chinese/English support
- **Qwen2-VL** (Wang et al., 2024), **Qwen3-VL** (2025): Progressive improvements
- **Nougat** (Blecher et al., 2023): Academic paper to markdown conversion
- **Docling** (IBM, 2024): Document-specialized parsing with OCR integration

### 2.2.3 Prompt Engineering for Small VLMs

We found that 2B-parameter models require explicit, rule-based instructions:

1. **v1 (Implicit)**: Generic prompt → Structure F1 = **0%** (no markdown heading markers)
2. **v2 (Explicit)**: CRITICAL RULES + heading-level mapping → Structure F1 = **79.25%**

**Key insight**: Small VLMs need MUST/NEVER keywords and explicit number→level mapping.

## 2.3 Knowledge Distillation

- **Hinton et al. (2015)**: Original KD framework using soft labels
- **Pseudo-labeling**: Teacher generates training labels for unlabeled data

**Gap**: Most VLM distillation studies focus on general visual understanding. Few address structured document parsing with quality-filtered pseudo-labels.

## 2.4 Semantic Chunking and RAG

### 2.4.1 Chunking Strategies

| Strategy | Description | Structure Dependency |
|----------|-------------|---------------------|
| **Header-based** | Split at markdown `#` headings | **High** — requires structural markup |
| **Semantic** | Split at embedding distance peaks | **Low** — works on any text |
| **Fixed-size** | Split at token count boundaries | **None** — baseline |

**Key Insight**: VLM-generated markdown naturally provides structural cues (headings, tables) that enable header-based chunking — unavailable from traditional OCR output.

### 2.4.2 Label-Free Chunk Quality Metrics

MoC (Metrics of Chunks, Zhao et al., ACL 2025) proposes label-free metrics:
- **Boundary Clarity (BC)**: Embedding cosine distance across chunk boundaries — higher = cleaner topic separation
- **Chunk Stickiness (CS)**: Intra-chunk embedding similarity — higher = better coherence

These metrics correlate with downstream retrieval performance (BC↔ROUGE-L: r=0.88) without requiring labeled QA pairs.

### 2.4.3 Retrieval Evaluation

Standard metrics: Hit Rate@k, MRR, NDCG, Recall@k.

## 2.5 Evaluation Metrics for Document Parsing

- **NED**: Normalized Edit Distance (character-level)
- **TEDS / TEDS-S**: Tree Edit Distance Similarity for tables (structure + content / structure only)
- **CDM**: Character Detection Matching for formulas (OmniDocBench official metric)
- **OmniDocBench** (CVPR 2025): Standardized benchmark — 1,355 pages, 9 document types

## 2.6 Gap Analysis

1. **No quality-filtered distillation**: Existing pseudo-labeling approaches lack automated quality validation
2. **No causal decomposition**: Studies compare end-to-end but don't isolate structure → chunking → retrieval
3. **Lack of Korean support**: Most benchmarks focus on English/Chinese
4. **Thinking contamination**: Unreported failure mode in VLM pseudo-labeling

This study addresses these gaps with a quality-filtered distillation pipeline, two-step causal evaluation on KoGovDoc, and multi-metric evaluation on OmniDocBench.
