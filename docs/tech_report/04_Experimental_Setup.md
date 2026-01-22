# 4. Experimental Setup

## 4.1 Dataset Description

### 4.1.1 Test Documents Overview

| ID | Document Type | Language | Pages | Complexity | Source |
|----|---------------|----------|-------|------------|--------|
| test_1 | Government Announcement | Korean | 1 | Tables, Headers | Public AI Support Program |
| test_2 | Receipt Image | Korean | 1 | Structured Text | Restaurant Receipt |
| test_3 | Academic Paper | English | Multi | Multi-column, References | arXiv (CoT Prompting) |

### 4.1.2 Test Case Details

#### Test 1: Korean Government Document (test_data_1.pdf)

**Characteristics**:
- Digital PDF with embedded text
- Complex table structures
- Multiple sections with headers
- Mix of Korean text and numbers
- Official document formatting

**Ground Truth Focus**:
- Table structure preservation
- Header hierarchy
- List formatting
- Number accuracy

#### Test 2: Receipt Image (test_data_2.jpg)

**Characteristics**:
- Scanned/photographed document
- Structured layout (merchant info, items, totals)
- Mixed fonts and sizes
- Korean and numbers

**Ground Truth Focus**:
- Text extraction accuracy
- Line item preservation
- Amount accuracy

#### Test 3: Academic Paper (Chain-of-Thought-Prompting.pdf)

**Characteristics**:
- Multi-page academic PDF
- Two-column layout
- Figures and tables
- References section
- Mathematical notation

**Ground Truth Focus**:
- Reading order (column handling)
- Section boundaries
- Figure/table separation
- Reference accuracy

### 4.1.3 Ground Truth Creation

Ground truth files were manually created following these guidelines:

1. **Markdown Format**: Use standard markdown syntax
2. **Structure Preservation**: Maintain headers, lists, tables
3. **Verbatim Text**: No paraphrasing or summarization
4. **Completeness**: Include all visible text
5. **Format Consistency**: Standardized markdown styling

**Quality Assurance**:
- Manual review by document owner
- Cross-validation against original PDF
- Edge case annotation (unclear text marked)

## 4.2 Q&A Dataset Generation

### 4.2.1 Generation Strategy

Q&A pairs are generated using LLM-assisted creation:

```python
prompt = """
Based on the following document content, generate {n} question-answer pairs.

Requirements:
1. Questions should be answerable from the document
2. Include diverse question types:
   - Factual: "What is X?"
   - Table lookup: "How much is Y?"
   - Multi-hop: "Compare X and Y"
   - Inferential: "Why does X happen?"
3. Provide exact text spans as answers
4. Include difficulty ratings (easy/medium/hard)

Document:
{document_content}
"""
```

### 4.2.2 Question Types

| Type | Description | Example | Target |
|------|-------------|---------|--------|
| Factual | Direct fact retrieval | "What is the support period?" | Single chunk |
| Table Lookup | Information from tables | "What is the maximum funding?" | Table cell |
| Multi-hop | Combine multiple facts | "Compare A and B programs" | Multiple chunks |
| Inferential | Reasoning required | "Why is this program important?" | Implicit info |

### 4.2.3 Dataset Statistics

<!-- TODO: Fill after Q&A generation -->
| Document | Total Q&A | Factual | Table | Multi-hop | Inferential |
|----------|-----------|---------|-------|-----------|-------------|
| test_1 | 10-20 | TBD | TBD | TBD | TBD |
| test_2 | 10-20 | TBD | TBD | TBD | TBD |
| test_3 | 10-20 | TBD | TBD | TBD | TBD |

## 4.3 Parser Configuration

### 4.3.1 VLM Parser Settings

```yaml
vlm_parser:
  model: "Qwen3-VL-2B-Instruct"
  api_url: "http://localhost:8000/v1/chat/completions"
  temperature: 0.0  # Deterministic output
  max_tokens: 4096
  image_resolution: 300  # DPI
  timeout: 60  # seconds per page
  prompt_version: "v2"  # Transcription-focused
```

### 4.3.2 OCR Parser Settings

```yaml
ocr_parser:
  library: "pdfplumber"
  table_settings:
    vertical_strategy: "lines"
    horizontal_strategy: "lines"
    snap_tolerance: 3
  text_settings:
    x_tolerance: 3
    y_tolerance: 3
```

### 4.3.3 Docling Parser Settings

```yaml
docling_parser:
  ocr_engine: "rapidocr"
  enable_table_detection: true
  enable_layout_analysis: true
  language: "korean+english"
```

## 4.4 Chunking Configuration

### 4.4.1 Semantic Chunking Parameters

**Controlled variables** (identical for all experiments):

```yaml
chunking:
  strategy: "recursive_character"
  chunk_size: 500
  chunk_overlap: 50
  separators:
    - "\n\n"  # Paragraph break
    - "\n"    # Line break
    - ". "    # Sentence end
    - " "     # Word break
  length_function: "character_count"
```

### 4.4.2 Embedding Configuration

```yaml
embedding:
  model: "jhgan/ko-sroberta-multitask"
  dimension: 768
  normalize: true
  batch_size: 32
```

### 4.4.3 Retrieval Configuration

```yaml
retrieval:
  method: "cosine_similarity"
  top_k: [1, 3, 5, 10]
  threshold: 0.0  # No threshold filtering
```

## 4.5 Evaluation Environment

### 4.5.1 Hardware

| Component | Specification |
|-----------|--------------|
| CPU | [Specify] |
| GPU | [Specify - for VLM inference] |
| RAM | [Specify] |
| Storage | SSD |

### 4.5.2 Software

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime |
| pdfplumber | 0.11.0+ | PDF extraction |
| jiwer | 3.0.0+ | CER/WER calculation |
| konlpy | 0.6.0+ | Korean tokenization |
| sentence-transformers | Latest | Embeddings |
| langchain | Latest | Chunking |

### 4.5.3 Reproducibility

- Random seed: 42 (where applicable)
- All experiments logged with timestamps
- Configuration files versioned
- Results stored in structured JSON format

## 4.6 Experimental Protocol

### 4.6.1 Baseline Measurement

1. Parse documents with pdfplumber (digital) / RapidOCR (scanned)
2. Apply semantic chunking with fixed parameters
3. Generate embeddings for all chunks
4. Run retrieval for all Q&A pairs
5. Calculate metrics: CER, WER, Hit Rate, MRR

### 4.6.2 VLM Measurement

1. Parse documents with Qwen3-VL
2. Apply identical chunking
3. Generate embeddings (same model)
4. Run retrieval (same queries)
5. Calculate metrics

### 4.6.3 Structural Analysis

1. Compare chunking outputs between parsers
2. Calculate Boundary Score (BS)
3. Calculate Chunk Score (CS)
4. Correlate with retrieval performance

## 4.7 Ablation Study Design

### 4.7.1 Prompt Variation

| Version | Approach | Hypothesis |
|---------|----------|------------|
| v1 | "extraction expert" | Higher hallucination |
| v2 | "transcription engine" | Lower hallucination |
| v3 | Minimal instruction | Baseline behavior |
| v4 | XML structured output | Explicit structure |

### 4.7.2 Resolution Study

| DPI | Image Size | Hypothesis |
|-----|------------|------------|
| 72 | Small | Fast but lower quality |
| 150 | Medium | Balanced |
| 300 | Large | Best quality, slower |

### 4.7.3 Chunking Strategy Study

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| Fixed (500) | Fixed character count | Simple baseline |
| Semantic | Topic-based splitting | Better coherence |
| Hierarchical | Structure-aware | Preserves sections |
