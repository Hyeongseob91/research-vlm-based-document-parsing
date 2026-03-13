# 6. Discussion

## 6.1 Pseudo-Labeling Effectiveness

### 6.1.1 GT Quality vs. Training Signal

The 74-75% acceptance rate (score ≥ 3) indicates that approximately 25% of pseudo-labeled pages contain quality issues. Key question: is this sufficient for effective distillation?

We argue yes, for two reasons:
1. **Quality filtering**: Only accepted pages enter training data, removing contaminated samples
2. **Scale advantage**: Even after filtering, ~3,300+ pages remain — substantially more than manually annotated alternatives

### 6.1.2 Thinking Tag Contamination

The dominant failure mode (36-47% of failures) — thinking text leaking into GT — has a clear root cause and fix:

| Configuration | Tag Behavior | Contamination Risk |
|---|---|---|
| `enable_thinking: False` | Model suppresses tags but still "thinks" | **High** — untagged reasoning in content |
| `enable_thinking: True` (without parser) | `<think>...</think>` tags in content | Medium — tags present but mixed |
| `enable_thinking: True` + `reasoning-parser` | Thinking separated to `reasoning_content` | **Low** — clean content field |
| Fallback: `_clean_response()` | Split on `</think>`, pattern detection | Low — catches remaining cases |

This finding has practical implications beyond our pipeline: any system using Qwen3-VL for structured output must handle thinking contamination explicitly.

### 6.1.3 Text-Based Validation Viability

Using a text-only judge (122B LLM without image) successfully detects:
- Thinking tag contamination (hallucination_signals dimension)
- Truncated content (completeness_signals)
- Broken table formatting (table_quality)
- Structural inconsistencies (structure_quality)

This validates the design decision to avoid requiring a VLM judge, significantly reducing validation cost and complexity.

## 6.2 Error Analysis

### 6.2.1 VLM Error Taxonomy

| Category | Frequency | Severity | Root Cause |
|----------|-----------|----------|------------|
| THINKING_CONTAMINATION | High (36-47%) | Critical | Improper enable_thinking config |
| TRUNCATION | High (25-31%) | High | max_tokens insufficient for complex pages |
| TABLE_CORRUPTION | Medium (10-24%) | Medium | Complex merged cells, nested tables |
| HALLUCINATION | Low (test_1: 536% CER) | Critical | Unclear/low-quality input images |
| FALSE_POSITIVE_STRUCTURE | Medium (~27%) | Low | Over-detection of structural elements |

### 6.2.2 Document-Specific Challenges

**Korean Government Documents (KoGovDoc)**:
- Dense tables with merged cells and legal numbering (조/항/목)
- kogov_008 (53% of data) contains repetitive tabular formats
- Mixed Korean-English content requires careful prompt design

**English Academic Papers (ArXiv)**:
- Two-column layouts cause reading order ambiguity
- Mathematical notation (LaTeX) interleaved with text
- Figure/table captions vs. body text distinction
- Reference sections create alignment issues for CER measurement

### 6.2.3 CER Fairness Analysis

Our hypothesis testing (H₀ vs H₁) confirmed that CER ~40% reflects genuine structural differences between pandoc-derived GT and PyMuPDF extraction, not measurement artifacts. Normalization attempts consistently worsened CER by breaking edit distance alignment:

| Normalization | Expected | Actual | Verdict |
|---|---|---|---|
| Remove references | -5~15pp | +16pp (worse) | H₁ rejected |
| Remove page numbers | -2~5pp | -0.4pp | Negligible |
| Remove citations | -2~3pp | +18pp (worse) | H₁ rejected |

**Implication**: CER should be used as a prerequisite check (threshold-based), not as the primary optimization target for VLM-based parsing.

## 6.3 Two-Step Evaluation Design Rationale

### 6.3.1 Why Separate Steps?

A direct comparison of "parser A retrieves better than parser B" conflates multiple variables:
- Parsing quality differences
- Chunking strategy effects
- Embedding model behavior on different text formats

By separating into two steps, we isolate each causal link:

```
Step 1: Controls chunking strategy, varies parser
    → Isolates: Does structure quality affect chunk quality?

Step 2: Controls parser (uses best from Step 1), varies chunking
    → Isolates: Does chunk quality affect retrieval?
```

### 6.3.2 Chunking Strategy Selection

Three strategies were chosen to span the structure-dependency spectrum:

| Strategy | Why Included |
|----------|-------------|
| **Header-based** | Directly tests VLM structural output — if VLM provides no headers, this chunker produces poor results |
| **Semantic** | Structure-agnostic baseline — tests whether embedding-based splitting can compensate for lack of structure |
| **Fixed-size** | Absolute baseline — no intelligence, pure mechanical splitting |

The critical comparison is: **Header(VLM) vs Semantic(PyMuPDF)**. If header-based chunking on VLM output beats semantic chunking on unstructured text, this proves that explicit structure provides value beyond what embeddings can infer.

### 6.3.3 Query Set Design Considerations

Auto-generated queries from GT have a potential bias: they may favor structured text that matches the GT formatting. We mitigate this by:
1. Generating queries from **content**, not formatting (stripping markdown before query generation)
2. Using three query types (passage, heading, entity) to cover different retrieval patterns
3. Token overlap-based relevance judgment rather than exact match

## 6.4 Implications for RAG System Design

### 6.4.1 Hybrid Parsing Strategy

```
Document Input
     │
     ▼
┌─────────┐
│ Scanned?│──Yes──► VLM (Required) ⚠ Korean quality check
└────┬────┘
     │No
     ▼
┌──────────────┐
│ Complex      │
│ Layout?      │──Yes──► WigtnOCR-2B (LoRA) for production
│ (Tables,     │         or 30B teacher for offline
│ Multi-column)│
└──────┬───────┘
       │No
       ▼
  PyMuPDF (Fast, Sufficient)
```

### 6.4.2 Chunking Strategy Recommendation

Based on our two-step evaluation design, the recommended chunking strategy depends on parser output quality:

| Parser Output | Recommended Chunker | Rationale |
|--------------|-------------------|-----------|
| VLM (structured markdown) | Header-based | Exploits structural markup directly |
| Traditional OCR (plain text) | Semantic | No structure to exploit, embedding-based is best available |
| Mixed / Unknown | Semantic | Safe default, structure-independent |

### 6.4.3 Prompt Engineering for Small VLMs

Key findings from prompt evolution (v1→v2):
1. **Explicit rules are essential**: "MUST use #", "NEVER output without #" — 2B models need direct instructions
2. **Number→level mapping**: "1→##, 2.1→###" outperforms implicit "use appropriate headings"
3. **System/User separation**: Role definition (system) + task instructions (user) improves structure quality
4. **Temperature 0.1**: Near-deterministic with slight flexibility

## 6.5 Limitations

### 6.5.1 Dataset Limitations
- **Pilot study size**: RQ1-RQ2 results based on 3 documents (preliminary)
- **Language coverage**: Korean + English only; no CJK generalization
- **Document diversity**: Government docs and academic papers; no financial reports, medical records, etc.
- **GT quality**: Pseudo-labels from 30B model, not expert annotations

### 6.5.2 Methodological Limitations
- **Single teacher model**: Results specific to Qwen3-VL-30B; other teachers may produce different quality
- **Single judge model**: Qwen3.5-122B validation scores not cross-validated with human judgment
- **No image-based validation**: Text-only judge cannot detect visual fidelity issues (misread characters, layout errors)
- **Auto-generated queries**: Retrieval evaluation queries are synthetic, not from real users

### 6.5.3 Evaluation Limitations
- **BC/CS are proxy metrics**: They measure chunk quality properties, not retrieval performance directly — hence Step 2
- **Single embedding model**: Results may vary with different embedding models (e.g., multilingual-e5 vs BGE-M3)
- **FAISS exact search**: Production systems use approximate search (HNSW, IVF) which may affect results

### 6.5.4 RQ5 Limitations (Anticipated)
- **LoRA capacity**: Rank-8 LoRA may be insufficient for full knowledge transfer
- **Domain shift**: Training on government docs + academic papers may not generalize to other domains
- **Evaluation gap**: OmniDocBench covers 9 document types but may not represent all production use cases

## 6.6 Threats to Validity

**Internal**: Thinking contamination in initial GT may have inflated failure rates; re-generated pages need re-validation to confirm fix. Auto-generated queries may introduce systematic bias toward certain content patterns.

**External**: Results on Korean/English may not generalize to other languages. VLM capabilities are evolving rapidly — results are model-version specific.

**Construct**: Structure F1 measures element-type counts, not content accuracy of structural elements. A heading detected as `##` instead of `###` counts as a true positive despite being wrong level. BC/CS measure statistical properties of chunks, not semantic correctness.
