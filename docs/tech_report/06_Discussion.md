# 6. Discussion

## 6.1 Research Question Analysis

### 6.1.1 RQ1: Lexical Fidelity

**Question**: Does VLM-based parsing achieve better character-level and word-level accuracy compared to traditional OCR methods?

**Finding**: [TODO: Fill based on results]

**Analysis**:
<!-- TODO: Update after experiments -->

The CER and WER results reveal several patterns:

1. **Digital PDFs with Tables**: [Expected observation]
   - pdfplumber extracts raw text but loses structure
   - VLM preserves structure but may introduce minor errors
   - Trade-off between structure and character accuracy

2. **Scanned Documents**: [Expected observation]
   - Traditional OCR (RapidOCR) struggles with image quality
   - VLM leverages visual understanding beyond character recognition
   - Significant advantage for complex layouts

3. **Korean Language Impact**: [Expected observation]
   - MeCab tokenization crucial for fair WER comparison
   - Korean agglutinative nature affects error counting
   - Morpheme-level errors vs character-level errors differ

### 6.1.2 RQ2: Structural Preservation

**Question**: Does VLM-based parsing preserve document structure better, leading to improved semantic chunking?

**Finding**: [TODO: Fill based on results]

**Analysis**:

1. **Boundary Score Improvement**:
   - VLM markdown provides natural break points
   - Headers, blank lines, and table boundaries preserved
   - Chunking algorithm benefits from explicit structure

2. **Chunk Score Improvement**:
   - Structural markers prevent mid-concept splits
   - Tables remain atomic units
   - Lists maintain coherence

3. **Indirect Structural Effect**:
   - Even with structure-agnostic chunking, VLM output benefits
   - Markdown newlines and formatting act as implicit boundaries
   - This "free" structural information improves chunking

### 6.1.3 RQ3: Retrieval Impact

**Question**: Does structural preservation in parsing improve downstream retrieval performance in RAG systems?

**Finding**: [TODO: Fill based on results]

**Analysis**:

1. **Hit Rate Improvement by Query Type**:
   - **Factual queries**: Modest improvement
   - **Table queries**: Significant improvement (expected)
   - **Multi-hop queries**: Variable results
   - **Inferential queries**: [Analysis needed]

2. **Causal Chain**:
   ```
   Better Parsing → Better Boundaries → Better Chunks → Better Retrieval
   (VLM)          (Higher BS)        (Higher CS)     (Higher HR@k)
   ```

3. **Correlation Analysis**:
   - BS correlates with HR@k (r = [TBD])
   - CS correlates with HR@k (r = [TBD])
   - Supports hypothesis that structure improves retrieval

## 6.2 Error Pattern Analysis

### 6.2.1 VLM Error Categories

| Category | Frequency | Severity | Root Cause |
|----------|-----------|----------|------------|
| TABLE_STRUCTURE | TBD% | Critical | Complex merged cells |
| MULTI_COLUMN | TBD% | Critical | Ambiguous reading order |
| HALLUCINATION | TBD% | Major | Over-interpretation |
| HEADER_HIERARCHY | TBD% | Medium | Level confusion |
| DELETION | TBD% | Varies | Low-contrast text |
| SUBSTITUTION | TBD% | Minor | Similar characters |

### 6.2.2 Traditional OCR Error Categories

| Category | Frequency | Severity | Root Cause |
|----------|-----------|----------|------------|
| STRUCTURE_LOSS | TBD% | Critical | No layout understanding |
| TABLE_COLLAPSE | TBD% | Critical | Tables become text streams |
| COLUMN_MIX | TBD% | Critical | Multi-column ordering |
| CHARACTER_ERROR | TBD% | Minor | OCR accuracy |

### 6.2.3 Comparative Error Analysis

**When VLM Wins**:
1. Complex table structures (merged cells, nested headers)
2. Multi-column documents (academic papers, newspapers)
3. Scanned documents with layout complexity
4. Documents with mixed formatting

**When Traditional OCR Wins**:
1. Simple digital PDFs with clean text
2. Speed-critical applications
3. Documents without complex structure
4. Resource-constrained environments

## 6.3 Implications

### 6.3.1 For RAG System Design

1. **Parser Selection Strategy**:
   - Use document classification to route to appropriate parser
   - VLM for complex layouts, OCR for simple documents
   - Hybrid approach optimizes cost-quality trade-off

2. **Chunking Strategy Integration**:
   - VLM output enables structure-aware chunking
   - Consider hierarchical chunking for markdown output
   - Table-as-atomic-unit policy recommended

3. **Quality Assurance**:
   - Monitor parsing quality with CER/WER thresholds
   - Fallback mechanisms for parser failures
   - Human review for high-stakes documents

### 6.3.2 For VLM Prompt Engineering

1. **Transcription vs Extraction**:
   - Transcription-focused prompts reduce hallucination
   - Explicit "do not add" instructions critical
   - [unclear] markers better than guessing

2. **Output Format**:
   - Markdown provides good balance of structure and simplicity
   - JSON/XML for programmatic processing
   - Plain text loses structural benefits

3. **Temperature Settings**:
   - Temperature 0.0 for deterministic, reproducible output
   - Higher temperatures for creative tasks only

### 6.3.3 For Future Research

1. **Dataset Expansion**:
   - More diverse document types needed
   - Larger scale for statistical power
   - Multilingual evaluation

2. **End-to-End Evaluation**:
   - Extend to answer generation quality
   - RAGAs framework integration
   - User satisfaction studies

3. **Efficiency Optimization**:
   - Smaller VLM models (distillation)
   - Batch processing optimization
   - Caching strategies

## 6.4 Limitations

### 6.4.1 Dataset Limitations

1. **Sample Size**: Only 3 documents, 30 Q&A pairs
   - Statistical power limited
   - Results should be considered preliminary
   - Bootstrap CI provides some robustness

2. **Document Diversity**:
   - Limited to Korean and English
   - Specific document types may not generalize
   - No handwritten or highly degraded documents

3. **Ground Truth Quality**:
   - Manual annotation subject to human error
   - Single annotator (no inter-rater reliability)
   - Markdown style choices affect metrics

### 6.4.2 Methodological Limitations

1. **Single VLM Model**:
   - Results may not generalize to other VLMs
   - Qwen3-VL specific behaviors
   - Version and configuration dependencies

2. **Chunking Configuration**:
   - Fixed parameters may not be optimal
   - Structure-agnostic chunking by design
   - Ablation study partially addresses

3. **Embedding Model Choice**:
   - ko-sroberta may have language biases
   - Single embedding model tested
   - Retrieval method (cosine) is basic

### 6.4.3 Practical Limitations

1. **Cost Analysis Incomplete**:
   - GPU costs not quantified
   - Latency measured but not optimized
   - Production deployment considerations missing

2. **Real-World Conditions**:
   - Clean test documents
   - No noisy or low-quality scans
   - Ideal network conditions for VLM API

## 6.5 Threats to Validity

### 6.5.1 Internal Validity

- **Confounding Variables**: Chunking may not be perfectly controlled
- **Measurement Error**: CER/WER sensitive to normalization
- **Selection Bias**: Test documents not randomly sampled

### 6.5.2 External Validity

- **Generalization**: Limited document types
- **Technology Changes**: VLM capabilities evolving rapidly
- **Domain Specificity**: Results may not apply to all domains

### 6.5.3 Construct Validity

- **Metric Relevance**: CER/WER may not fully capture usefulness
- **Retrieval Proxy**: HR@k approximates but != actual utility
- **Ground Truth Definition**: Markdown style choices affect baseline
