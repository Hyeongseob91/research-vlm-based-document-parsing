# 3. Methodology

## 3.1 Evaluation Framework Overview

Our evaluation framework measures document parsing quality across three dimensions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │   PHASE 1   │   │   PHASE 2   │   │   PHASE 3   │          │
│   │   Lexical   │ → │  Structural │ → │  Retrieval  │          │
│   │  Accuracy   │   │  Integrity  │   │ Performance │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐             │
│   │ CER, WER  │    │  BS, CS   │    │ HR@k, MRR │             │
│   └───────────┘    └───────────┘    └───────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 Parser Implementations

### 3.2.1 VLM Parser (Qwen3-VL)

**Architecture**: Vision-Language Model with multimodal understanding

**Process**:
1. Convert PDF pages to images (300 DPI)
2. Send images to Qwen3-VL API
3. Receive structured markdown output
4. Concatenate multi-page results

**Prompt Strategy** (v2 - Transcription-focused):
```
You are a document transcription engine. Your sole purpose is to convert
the given image into markdown text format. You MUST only transcribe what
is actually visible in the image - do not add any additional information,
explanations, or content that is not present in the original document.

Rules:
1. Transcribe ALL visible text exactly as shown
2. Use markdown formatting to preserve structure (headers, lists, tables)
3. For tables, use markdown table syntax
4. Preserve the original language (Korean, English, etc.)
5. Do NOT add explanations, summaries, or interpretations
6. If text is unclear, indicate with [unclear] rather than guessing
```

**Output Format**: Markdown with preserved structure

### 3.2.2 OCR Parser (pdfplumber)

**Architecture**: Rule-based PDF text extraction

**Process**:
1. Extract text boxes with positional information
2. Apply reading order heuristics
3. Detect and extract tables separately
4. Merge text and tables

**Limitations**:
- Requires embedded text (digital PDFs only)
- Table detection depends on line rules
- No scanned document support

### 3.2.3 Docling Parser (RapidOCR)

**Architecture**: Document AI pipeline with OCR

**Process**:
1. Document layout analysis
2. Region classification (text, table, figure)
3. OCR on image regions
4. Structured output generation

**Limitations**:
- Slower processing
- May struggle with complex layouts
- Dependent on OCR accuracy

## 3.3 Evaluation Metrics

### 3.3.1 Phase 1: Lexical Accuracy

#### Character Error Rate (CER)

Measures character-level accuracy using Levenshtein edit distance:

$$CER = \frac{S + D + I}{N}$$

Where:
- $S$ = Number of character substitutions
- $D$ = Number of character deletions
- $I$ = Number of character insertions
- $N$ = Total characters in reference

**Implementation**: Using `jiwer` library with custom preprocessing

#### Word Error Rate (WER)

Measures word-level accuracy with appropriate tokenization:

$$WER = \frac{S_w + D_w + I_w}{N_w}$$

**Tokenization Options**:
| Language | Tokenizer | Rationale |
|----------|-----------|-----------|
| Korean | MeCab | Morphological analysis for agglutinative language |
| English | Whitespace | Space-separated tokens |
| Mixed | MeCab + fallback | Primary Korean, whitespace for others |

### 3.3.2 Phase 2: Structural Integrity

#### Boundary Score (BS)

Measures alignment between predicted and ground truth semantic boundaries:

$$BS = \frac{|B_{pred} \cap B_{gt}|}{|B_{gt}|}$$

Where:
- $B_{pred}$ = Set of predicted chunk boundaries
- $B_{gt}$ = Set of ground truth semantic boundaries

**Boundary Definition**: Points where semantic context shifts (section changes, topic transitions)

**Implementation**:
1. Identify structural markers (headers, blank lines, section breaks)
2. Generate boundary positions for both reference and candidate
3. Calculate intersection with tolerance window (±n characters)

#### Chunk Score (CS)

Measures semantic coherence within each chunk:

$$CS = \frac{1}{|C|} \sum_{c \in C} coherence(c)$$

Where:
- $C$ = Set of generated chunks
- $coherence(c)$ = Semantic coherence of chunk $c$ (embedding similarity variance)

**Implementation**:
1. Generate embeddings for sentences within each chunk
2. Calculate intra-chunk similarity variance
3. Lower variance = higher coherence = better score

### 3.3.3 Phase 3: Retrieval Performance

#### Hit Rate@k

Proportion of queries where the relevant chunk appears in top-k results:

$$HitRate@k = \frac{|\{q : relevant(q) \in top_k(q)\}|}{|Q|}$$

Where:
- $Q$ = Set of test queries
- $relevant(q)$ = Ground truth relevant chunk for query $q$
- $top_k(q)$ = Top-k retrieved chunks for query $q$

#### Mean Reciprocal Rank (MRR)

Average inverse rank of first relevant result:

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

Where $rank_i$ is the rank of the first relevant result for query $i$.

## 3.4 Normalization Process

To ensure fair comparison, we apply normalization before metric calculation:

### 3.4.1 Markdown Stripping

Remove markdown syntax for lexical comparison:
- Headers: `# Header` → `Header`
- Bold/Italic: `**bold**` → `bold`
- Links: `[text](url)` → `text`
- Tables: Preserve cell content, remove pipes

### 3.4.2 Whitespace Normalization

Standardize whitespace handling:
- Collapse multiple spaces to single space
- Normalize newlines (CRLF → LF)
- Trim leading/trailing whitespace

### 3.4.3 Unicode Normalization

Apply NFKC normalization for consistent character representation:
- Full-width → half-width
- Compatibility characters normalized

## 3.5 Experimental Design

### 3.5.1 A/B Comparison Framework

```
┌─────────────────────────────────────────────────────────────────┐
│  Experiment A (Baseline)                                         │
│  ─────────────────────────────────────────────────────────────  │
│  PDF → pdfplumber/RapidOCR → Chunking → Embedding → Retrieval   │
│                                                      ↓          │
│                                                  Baseline       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Experiment B (VLM)                                              │
│  ─────────────────────────────────────────────────────────────  │
│  PDF → VLM Parser → Chunking → Embedding → Retrieval            │
│                                                      ↓          │
│                                                   VLM           │
└─────────────────────────────────────────────────────────────────┘

Hypothesis: VLM > Baseline (Hit Rate, MRR)
```

### 3.5.2 Variable Control

**Fixed Variables** (identical across experiments):
- Chunking algorithm: RecursiveCharacterTextSplitter
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Embedding model: ko-sroberta-multitask
- Retrieval method: Cosine similarity

**Independent Variable**:
- Parser type: VLM, pdfplumber, Docling

**Dependent Variables**:
- CER, WER, BS, CS, Hit Rate@k, MRR

## 3.6 Statistical Analysis

### 3.6.1 Significance Testing

- **Paired t-test**: Compare VLM vs Baseline on same documents
- **Wilcoxon signed-rank**: Non-parametric alternative if normality violated

### 3.6.2 Effect Size

- **Cohen's d**: Standardized mean difference
  - Small: d = 0.2
  - Medium: d = 0.5
  - Large: d = 0.8

### 3.6.3 Confidence Intervals

- **Bootstrap 95% CI**: 1000 resamples for robust estimation
- Report: Mean ± 95% CI for all metrics
