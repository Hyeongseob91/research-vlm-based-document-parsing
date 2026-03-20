# 3. Methodology

## 3.1 System Overview

WigtnOCR employs a pseudo-labeling approach to distill document parsing capabilities from a large teacher VLM into a compact student model. The system consists of four pipeline stages:

```
┌──────────────────────────────────────────────────────────────────┐
│                     WigtnOCR Pipeline                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Pseudo GT          Stage 2: Quality            Stage 3 │
│  Generation                  Validation                  & 4     │
│  ┌─────────────┐            ┌─────────────┐            ┌───────┐│
│  │ PDF Pages   │            │ GT Markdown │            │LoRA   ││
│  │     ↓       │            │     ↓       │            │Train  ││
│  │ Qwen3-VL-30B│     →      │ Qwen3.5-122B│     →      │Qwen3  ││
│  │  (Teacher)  │            │   (Judge)   │            │VL-2B  ││
│  │     ↓       │            │     ↓       │            │       ││
│  │ Markdown GT │            │ Score 1-5   │            │WigtnOCR│
│  └─────────────┘            └─────────────┘            └───────┘│
│                                                                  │
│  Models: 3 separate roles (Teacher, Judge, Student)              │
│  No image needed for validation — text-based quality assessment  │
└──────────────────────────────────────────────────────────────────┘
```

## 3.2 Stage 1: Pseudo Ground Truth Generation

### 3.2.1 Architecture

Each PDF page is rendered to a high-resolution image (200 DPI) and sent to the teacher VLM (Qwen3-VL-30B-A3B-Thinking) for direct image-to-markdown conversion.

**Pipeline**:
```
PDF → PyMuPDF page rendering (200 DPI PNG)
    → base64 encoding
    → VLM API (OpenAI-compatible, vLLM serving)
    → Raw response with thinking tags
    → _clean_response() post-processing
    → Clean structured markdown
```

### 3.2.2 Teacher Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen3-VL-30B-A3B-Thinking | MoE architecture, strong document understanding |
| Serving | vLLM v0.13.0, TP=2 | Dual GPU tensor parallelism |
| Temperature | 0.1 | Near-deterministic with slight flexibility |
| Max tokens | 8,192 | Sufficient for most pages |
| enable_thinking | True | Model always thinks; proper tag generation |
| reasoning-parser | qwen3 | Server-side thinking/content separation |

### 3.2.3 Thinking Tag Handling

Qwen3-VL models always perform internal reasoning (thinking). Without proper handling, thinking text contaminates GT output. We address this at two levels:

**Server-level**: vLLM `--reasoning-parser qwen3` automatically separates thinking into `reasoning_content` field, leaving clean `content`.

**Application-level fallback** (`_clean_response()`):
1. Split on `</think>` tag — model reliably outputs closing tag with `enable_thinking: True`
2. Remove residual `<think>` tags via regex
3. Pattern detection for untagged thinking (e.g., "Okay, let's tackle...", "Wait, the image shows...")
4. Strip markdown code fences

### 3.2.4 Language-Specific Prompts

Separate prompt templates for Korean and English documents:

- **Korean** (`PSEUDO_GT_SYSTEM_PROMPT`): Emphasizes exact Korean character preservation, 조/항/목 legal structure, mixed Korean-English handling
- **English** (`PSEUDO_GT_SYSTEM_PROMPT_EN`): Academic paper-specific rules — LaTeX notation, citation preservation, figure captions

### 3.2.5 Batch Processing

Concurrent processing via `ThreadPoolExecutor` with configurable batch size (default: 4). Page-level caching enables incremental processing — existing page files are skipped on restart.

## 3.3 Stage 2: Quality Validation

### 3.3.1 Text-Based Validation Design

A key design decision: the judge model evaluates GT quality from **markdown text only**, without seeing the original document image. This is intentional:

- The judge assesses **internal consistency** — structural coherence, formatting quality, contamination signals
- Image comparison would require a VLM judge, adding cost and complexity
- Text-based evaluation is sufficient to detect the primary failure modes (thinking contamination, truncation, broken tables)

### 3.3.2 Judge Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen3.5-122B-A10B-NVFP4 | Large text LLM, strong reasoning |
| Serving | vLLM with `--reasoning-parser deepseek_r1` | Auto-separates thinking |
| Max tokens | 16,384 | Sufficient budget for thinking + JSON response |
| Temperature | 0.1 | Consistent scoring |

**Note**: `enable_thinking: False` must NOT be used with this model — the 122B uses `deepseek_r1` reasoning parser which auto-separates thinking. Suppressing thinking tags causes empty content responses.

### 3.3.3 Five-Dimension Quality Scoring

Each page is scored on five criteria (1-5 scale):

| Dimension | Description | Failure Indicators |
|-----------|-------------|-------------------|
| structure_quality | Heading hierarchy, list nesting, structural coherence | Broken `#` levels, orphaned list items |
| table_quality | Pipe separators, header rows, consistent column counts | Missing `\|---\|`, mismatched columns |
| completeness_signals | Signs of truncation or placeholder text | Abrupt endings, `<!-- MISSING -->` markers |
| hallucination_signals | AI contamination in output | Thinking tags, chain-of-thought leakage, meta-commentary |
| formatting_consistency | Consistent markdown style throughout | Mixed list markers, inconsistent spacing |

**Overall score**: 1-5 aggregate, with `is_acceptable = (score >= 3)`.

### 3.3.4 Sampling Strategy

| Dataset | Pages | Sample Ratio | Sampled | Rationale |
|---------|-------|-------------|---------|-----------|
| KoGovDoc | 3,637 | 30% | ~1,050 | Large dataset, proportional sampling |
| ArXivPapers | 864 | 100% | 864 | Smaller dataset, full validation |

Pages not sampled for validation are included in training by default (no rejection signal).

## 3.4 Stage 3: Training Data Preparation

### 3.4.1 Quality Filtering

Only pages with validation score ≥ 3 are included. Pages not sampled for validation are included by default.

### 3.4.2 Bias Correction

KoGovDoc dataset exhibits significant size imbalance — a single document (kogov_008, 1,929 pages) represents 53% of total pages. Without correction, the model would overfit to this document's style and content.

**Solution**: Per-document downsampling with `max_doc_ratio = 0.25`. Documents exceeding 25% of total are randomly subsampled to the threshold.

### 3.4.3 Data Format (ms-swift JSONL)

Each training sample pairs a page image with its GT markdown:

```json
{
  "messages": [
    {"role": "system", "content": "<WigtnOCR system prompt>"},
    {"role": "user", "content": "<image>Convert this document page to Markdown."},
    {"role": "assistant", "content": "<GT markdown content>"}
  ],
  "images": ["/absolute/path/to/page_NNNN.png"]
}
```

### 3.4.4 Train/Validation Split

90/10 random split with fixed seed (42) for reproducibility.

## 3.5 Stage 4: LoRA Fine-tuning

### 3.5.1 Student Model

Qwen3-VL-2B-Instruct — the smallest dense model in the Qwen3-VL family. Selected for production deployment viability.

### 3.5.2 LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Framework | ms-swift 4.0 | Official Qwen recommendation |
| tuner_type | LoRA | Parameter-efficient, preserves base capabilities |
| rank | 8 | Sufficient for task adaptation |
| alpha | 32 | Standard 4x rank ratio |
| target_modules | all-linear | All linear layers in LLM |
| freeze_vit | True | Preserve pretrained vision encoder |
| freeze_aligner | True | Preserve vision-language alignment |

### 3.5.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Learning rate | 1e-4 |
| Batch size | 1 per device |
| Gradient accumulation | 4 steps |
| Max sequence length | 4,096 tokens |
| Precision | bfloat16 |
| Multi-GPU | DeepSpeed ZeRO-2 (2x GPU) |
| Optimizer | AdamW (ms-swift default) |

### 3.5.4 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition × 2 (98GB VRAM each) |
| RAM | 128GB DDR5 |
| Serving | vLLM v0.13.0 (inference), ms-swift 4.0.1 (training) |

## 3.6 Evaluation Framework

### 3.6.1 Dual-Benchmark Design

Two benchmarks serve complementary roles:

| Benchmark | Purpose | RQs | Metrics |
|-----------|---------|-----|---------|
| **OmniDocBench** (1,355 pages) | Parsing quality — international standard | RQ1-3 | NED, TEDS, CDM, Reading Order |
| **KoGovDoc Val** (294 samples) | Downstream impact — chunking & retrieval | RQ4-5 | BC, CS, Hit@K, MRR, nDCG |

### 3.6.2 OmniDocBench Parsing Metrics (RQ1-3)

#### NED (Normalized Edit Distance)

$$NED = 1 - \frac{ED(pred, gt)}{\max(len(pred), len(gt))}$$

Lower NED indicates better text recognition quality. Evaluated at sample_avg, page_avg, and edit_whole granularities.

#### TEDS (Tree Edit Distance-based Similarity)

Measures structural similarity between predicted and reference tables via HTML tree edit distance. TEDS-S evaluates structure only (ignoring cell text content).

#### CDM (Character Detection Matching)

OmniDocBench official metric for formula evaluation. Renders predicted and reference LaTeX to images, then computes character-level detection matching (F1 and ExactMatch rate).

#### Reading Order NED

NED computed over the element ordering sequence, evaluating whether elements appear in correct reading order.

### 3.6.3 KoGovDoc Two-Step Causal Evaluation (RQ4-5)

#### Step 1: Structure → Chunking Quality (RQ4)

Three chunking strategies isolate the effect of structural preservation:

| Strategy | Mechanism | Structure Dependency |
|----------|-----------|---------------------|
| **Header-based** | Split at markdown `#` headings | **High** — requires structural markup |
| **Semantic** | Split at embedding distance peaks (BGE-M3) | **Low** — works on any text |
| **Fixed-size** | Split at 512-token boundaries | **None** — baseline |

**Key comparison**: If header-based chunking on VLM output outperforms semantic chunking on unstructured text, this proves that explicit structure provides value beyond what embeddings can infer.

**BC (Boundary Clarity)**: Measures semantic dissimilarity across chunk boundaries:
$$BC = \frac{1}{|B|} \sum_{b \in B} (1 - \cos(e_{b-1}, e_{b+1}))$$

**CS (Chunk Stickiness)**: Measures intra-chunk semantic coherence:
$$CS = \frac{1}{|C|} \sum_{c \in C} \text{mean\_similarity}(c)$$

Both metrics are **label-free** — they require no ground truth, only the parser output and an embedding model.

#### Step 2: Chunking → Retrieval Performance (RQ5)

Using the best chunking strategy from Step 1:

```
Parser output → Chunking → BGE-M3 embedding → FAISS index
Query set → BGE-M3 embedding → Top-K retrieval → Relevance check
```

Auto-generated queries from KoGovDoc GT text, with three query types (passage→query, heading→query, entity→query). A retrieved chunk is relevant if it contains ≥ 50% of the source GT segment (token overlap).

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Hit@K** (K=1,5,10) | Relevant chunk in top-K | "Did we find it?" |
| **MRR** | Mean reciprocal rank of first relevant result | "How quickly?" |
| **nDCG@10** | Normalized discounted cumulative gain | Rank-sensitive relevance |

#### Causal Chain Summary

```
Step 1: Controls chunking strategy, varies parser
    → Isolates: Does structure quality affect chunk quality? (BC/CS)

Step 2: Uses best chunks from Step 1, measures retrieval
    → Isolates: Does chunk quality affect retrieval? (Hit@K/MRR)

Together: Better parsing → Better chunks → Better retrieval
```

### 3.6.4 Model Comparison

| Parser | Type | Model | Benchmarks |
|--------|------|-------|------------|
| 2B base | Direct VLM | Qwen3-VL-2B-Instruct | OmniDocBench |
| **WigtnOCR-2B** | Direct VLM | Qwen3-VL-2B + LoRA | OmniDocBench + KoGovDoc |
| Teacher (30B) | Direct VLM | Qwen3-VL-30B-A3B-Instruct | OmniDocBench |
| Marker | OCR pipeline | Marker v1.10.2 | OmniDocBench |
| PyMuPDF | Text extraction | PyMuPDF | KoGovDoc (unstructured baseline) |
