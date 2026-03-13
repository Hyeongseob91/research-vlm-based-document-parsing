# Appendix E: Chunking Strategy Implementation Details

## E.1 Purpose

Step 1 of our evaluation (RQ3: Structure → Chunking) compares three chunking strategies to isolate the effect of structural preservation on chunk quality. This appendix documents each strategy's implementation, parameters, and behavior on structured vs unstructured input, ensuring reproducibility of BC/CS measurements.

## E.2 Strategy Specifications

### E.2.1 Header-Based Chunking

**Mechanism**: Split text at markdown heading boundaries (`# `, `## `, `### `, etc.).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Split markers | `^#{1,6}\s` (regex) | Standard markdown headings |
| Min chunk size | 100 tokens | Avoid trivially small chunks |
| Max chunk size | 2,048 tokens | Prevent oversized chunks from missing headers |
| Overflow handling | Split at paragraph boundary | When section exceeds max size |

**Structure Dependency**: **High** — requires markdown heading markup. On PyMuPDF output (no headings), this chunker produces a single chunk containing the entire document, resulting in degenerate BC/CS scores.

**Expected Behavior**:
```
VLM output (has headers):          PyMuPDF output (no headers):
┌──────────────────┐               ┌──────────────────┐
│ ## 1. Introduction│               │ 1. Introduction  │
│ Text text text...│               │ Text text text...│
├──────────────────┤  ← boundary   │ 2. Methods       │
│ ## 2. Methods    │               │ Text text text...│  ← no boundaries
│ Text text text...│               │ 3. Results       │
├──────────────────┤  ← boundary   │ Text text text...│
│ ## 3. Results    │               └──────────────────┘
│ Text text text...│                 (single chunk)
└──────────────────┘
  (3 clean chunks)
```

### E.2.2 Semantic Chunking

**Mechanism**: Split at embedding distance peaks — points where consecutive sentence embeddings are most dissimilar, indicating topic transitions.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding model | BGE-M3 | Multilingual (Korean + English) |
| Distance metric | Cosine distance | Standard for embedding comparison |
| Breakpoint threshold | Percentile-based (95th) | Adaptive to document characteristics |
| Min chunk size | 100 tokens | Avoid trivially small chunks |

**Structure Dependency**: **Low** — works on any text input regardless of formatting. Identifies topic shifts from semantic content, not markup.

**Implementation**: LangChain `SemanticChunker` with custom embedding API wrapper.

### E.2.3 Fixed-Size Chunking

**Mechanism**: Split at fixed token count boundaries with no semantic awareness.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 512 tokens | Standard RAG chunk size |
| Overlap | 50 tokens | Prevent information loss at boundaries |
| Tokenizer | tiktoken (cl100k_base) | Consistent token counting |

**Structure Dependency**: **None** — pure mechanical split. Serves as lower-bound baseline for BC/CS.

## E.3 Comparison on Same Input

> **Status**: Pending — will include actual output examples from evaluation.

**Example document**: OmniDocBench academic paper page

| Strategy | Chunks | Avg Size | BC | CS |
|----------|--------|----------|-----|-----|
| Header-based (VLM input) | ___ | ___ tokens | ___ | ___ |
| Header-based (PyMuPDF input) | 1 (degenerate) | ___ tokens | N/A | N/A |
| Semantic (VLM input) | ___ | ___ tokens | ___ | ___ |
| Semantic (PyMuPDF input) | ___ | ___ tokens | ___ | ___ |
| Fixed-512 (VLM input) | ___ | 512 tokens | ___ | ___ |
| Fixed-512 (PyMuPDF input) | ___ | 512 tokens | ___ | ___ |

## E.4 Key Comparison Logic

The critical experiment is:

```
Header-based(VLM output)  vs  Semantic(PyMuPDF output)
         ↑                              ↑
  Structure-dependent            Structure-independent
  chunking on structured         chunking on unstructured
  input                          input
```

If Header(VLM) > Semantic(PyMuPDF) in BC/CS, this proves that **explicit structural markup enables better chunking than what embedding-based methods can achieve on unstructured text** — validating that VLM structural preservation has direct downstream value.

## E.5 Edge Cases

| Scenario | Header-based | Semantic | Fixed-512 |
|----------|-------------|----------|-----------|
| No headings in input | Single chunk (degenerate) | Normal operation | Normal operation |
| Very short page (<100 tokens) | Single chunk | Single chunk | Single chunk |
| Table-only page | Single chunk (no `#`) | May split mid-table | Splits mid-table |
| Code block page | Single chunk | Topic-based splits | Mechanical splits |
