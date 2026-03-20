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

## 6.2 Why the Student Surpasses the Teacher

A surprising result is that WigtnOCR-2B outperforms the 30B teacher on tables (TEDS 0.649 vs 0.523), text (page_avg NED 0.304 vs 0.415), and reading order (NED 0.211 vs 0.227). We attribute this to three factors:

1. **Quality filtering effect**: Training data includes only pseudo-GT pages scoring >= 3/5 on quality validation. The student learns from the teacher's best outputs, effectively filtering out the teacher's failure cases (broken tables, truncated content, hallucinations).

2. **Regularization from LoRA**: Rank-8 LoRA constrains the model's capacity, acting as an implicit regularizer. This may prevent overfitting to the teacher's idiosyncratic errors while retaining the learned document parsing patterns.

3. **Base model complementarity**: The 2B Instruct base model already has reasonable capabilities (TEDS-S 0.667 is competitive). LoRA fine-tuning adds domain-specific document parsing knowledge on top of a solid foundation, combining the base model's general instruction-following with the teacher's document-specific expertise.

## 6.3 The Formula Gap

Formula recognition shows the smallest improvement (CDM F1: 0.865 → 0.884, vs teacher's 0.939). Possible explanations:

- **LaTeX complexity**: Mathematical notation requires precise character-level generation (e.g., `\frac{\partial}{\partial x}`) that may exceed LoRA rank-8's representational capacity
- **Training data composition**: KoGovDoc (government documents) contains few mathematical formulas; ArXiv papers provide formulas but represent only ~19% of training pages
- **Vision encoder frozen**: `freeze_vit: True` prevents the vision encoder from learning to better recognize mathematical symbols

## 6.4 Robustness Improvement

The skip rate reduction (18.8% → 5.8%) is arguably the most practically significant result. A model that fails on nearly 1 in 5 documents is unreliable for production use. The trained model's 5.8% failure rate (comparable to the 30B teacher's 5.5%) enables deployment in production pipelines without requiring a fallback mechanism for most document types.

## 6.5 Error Analysis

### 6.5.1 VLM Error Taxonomy

| Category | Frequency | Severity | Root Cause |
|----------|-----------|----------|------------|
| THINKING_CONTAMINATION | High (36-47%) | Critical | Improper enable_thinking config |
| TRUNCATION | High (25-31%) | High | max_tokens insufficient for complex pages |
| TABLE_CORRUPTION | Medium (10-24%) | Medium | Complex merged cells, nested tables |
| HALLUCINATION | Low | Critical | Unclear/low-quality input images |

### 6.5.2 Document-Specific Challenges

**Korean Government Documents (KoGovDoc)**:
- Dense tables with merged cells and legal numbering (조/항/목)
- kogov_008 (53% of data) contains repetitive tabular formats
- Mixed Korean-English content requires careful prompt design

**English Academic Papers (ArXiv)**:
- Two-column layouts cause reading order ambiguity
- Mathematical notation (LaTeX) interleaved with text
- Figure/table captions vs. body text distinction
- Reference sections create alignment issues

## 6.6 Prompt Engineering for Small VLMs

Key findings from prompt evolution (v1→v2):
1. **Explicit rules are essential**: "MUST use #", "NEVER output without #" — 2B models need direct instructions
2. **Number→level mapping**: "1→##, 2.1→###" outperforms implicit "use appropriate headings"
3. **System/User separation**: Role definition (system) + task instructions (user) improves structure quality
4. **Temperature 0.1**: Near-deterministic with slight flexibility

## 6.7 Limitations

### 6.7.1 Dataset Limitations
- **Language coverage**: Korean + English only; no CJK generalization
- **Document diversity**: Government docs and academic papers; no financial reports, medical records, etc.
- **GT quality**: Pseudo-labels from 30B model, not expert annotations

### 6.7.2 Methodological Limitations
- **Single teacher model**: Results specific to Qwen3-VL-30B; other teachers may produce different quality
- **Single judge model**: Qwen3.5-122B validation scores not cross-validated with human judgment
- **No image-based validation**: Text-only judge cannot detect visual fidelity issues

### 6.7.3 Training Limitations
- **LoRA capacity**: Rank-8 LoRA shows limited formula transfer (CDM F1 gap of 5.5pp); higher rank or full fine-tuning may close this gap
- **Domain shift**: Training on Korean government docs + English academic papers; generalization to other domains is untested
- **Single training run**: Results from a single configuration; hyperparameter sensitivity not explored

## 6.8 Two-Step Causal Evaluation Design Rationale

### 6.8.1 Why Separate Steps?

A direct comparison of "parser A retrieves better than parser B" conflates multiple variables: parsing quality, chunking strategy effects, and embedding model behavior. By separating into two steps on KoGovDoc, we isolate each causal link:

- **Step 1**: Controls chunking strategy, varies parser → Does structure quality affect chunk quality?
- **Step 2**: Uses best chunks from Step 1, measures retrieval → Does chunk quality affect retrieval?

### 6.8.2 Why KoGovDoc for Causal Evaluation?

OmniDocBench provides standardized parsing quality metrics but lacks a natural RAG use case. KoGovDoc — Korean government documents with known GT — provides:
- **Real-world relevance**: Government documents are a primary RAG deployment target
- **Structural diversity**: Tables, legal numbering, mixed Korean-English
- **Controlled comparison**: val.jsonl provides GT for quality measurement while predictions enable chunking/retrieval evaluation

## 6.9 Threats to Validity

**Internal**: Thinking contamination in initial GT may have inflated failure rates; re-generated pages need re-validation to confirm fix. Auto-generated queries for retrieval evaluation may introduce systematic bias.

**External**: Results on Korean/English may not generalize to other languages. VLM capabilities are evolving rapidly — results are model-version specific.

**Construct**: OmniDocBench metrics evaluate element-level quality; page-level holistic quality may differ. BC/CS measure statistical properties of chunks, not semantic correctness.
