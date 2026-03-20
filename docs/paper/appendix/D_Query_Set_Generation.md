# Appendix D: Query Set Generation for Retrieval Evaluation

## D.1 Purpose

Step 2 of our evaluation (RQ4: Chunking → Retrieval) requires a query set with known relevant passages. Since OmniDocBench provides document GT but no retrieval QA pairs, we auto-generate queries from GT text segments. This appendix details the generation methodology, quality controls, and example outputs to ensure reproducibility.

## D.2 Generation Pipeline

```
OmniDocBench GT (1,355 pages)
    │
    ├── 1. Segment Extraction
    │     Split GT into meaningful text segments (~200-500 tokens)
    │     Respect section boundaries where available
    │
    ├── 2. Query Generation (3 types)
    │     For each segment, generate queries via LLM:
    │     - Passage → Question (natural language)
    │     - Heading → Question (structure-based)
    │     - Entity → Lookup (factoid)
    │
    ├── 3. Quality Filtering
    │     Remove trivial, unanswerable, or ambiguous queries
    │     Deduplicate near-identical queries
    │
    └── 4. Relevance Annotation
          Each query paired with source segment as relevant passage
          Token overlap threshold for retrieval judgment: ≥ 50%
```

## D.3 Query Types

### D.3.1 Passage → Question

**Input**: A GT text segment (stripped of markdown formatting)
**Output**: A natural language question answerable by the segment

**Generation Prompt**:
```
Given the following text passage from a document, generate a natural question
that someone might ask which would be answered by this passage.
The question should be specific enough that this passage is the best answer.
Do NOT include information not present in the passage.

Passage: {segment_text}
Question:
```

**Example**:
- Segment: "The self-attention mechanism computes queries (Q), keys (K), and values (V) from the input embeddings using learned linear projections."
- Query: "How does the self-attention mechanism compute Q, K, and V?"

### D.3.2 Heading → Question

**Input**: A section heading from GT
**Output**: A question about that section's content

**Example**:
- Heading: "## 3.2 Training Procedure"
- Query: "What is the training procedure used in this work?"

### D.3.3 Entity → Lookup

**Input**: Key entities extracted from GT segment (model names, metrics, numbers)
**Output**: A factoid lookup query

**Example**:
- Segment contains: "BERT-base achieves 88.5% accuracy on MNLI"
- Query: "What accuracy does BERT-base achieve on MNLI?"

## D.4 Quality Controls

| Control | Method | Threshold |
|---------|--------|-----------|
| Answerability | Query must be answerable from source segment alone | Manual spot-check on 5% sample |
| Specificity | Query should match ≤ 3 segments in the corpus | Embedding similarity check |
| Diversity | No two queries should have cosine similarity > 0.9 | Deduplication pass |
| Language | Query language matches document language | Automatic detection |

## D.5 Target Statistics

| Property | Target |
|----------|--------|
| Total queries | ~500 |
| Passage → Question | ~250 (50%) |
| Heading → Question | ~150 (30%) |
| Entity → Lookup | ~100 (20%) |
| Avg query length | 10-20 tokens |
| Avg relevant passage | 200-500 tokens |

## D.6 Relevance Judgment

A retrieved chunk is **relevant** to a query if:

$$\text{relevance}(chunk, passage) = \frac{|tokens(chunk) \cap tokens(passage)|}{|tokens(passage)|} \geq 0.5$$

This token-overlap threshold ensures the chunk contains the majority of the answer-bearing text, regardless of chunk boundaries.

## D.7 Limitations

- **Synthetic bias**: Auto-generated queries may not represent real user information needs
- **LLM generation quality**: Query quality depends on the generation model
- **Single relevant passage**: Each query has exactly one GT-relevant passage; in practice, multiple passages may be relevant
- **Language imbalance**: OmniDocBench is primarily English; Korean query quality may differ

## D.8 Examples

> **Status**: Pending — will be populated with actual generated queries after implementation.
