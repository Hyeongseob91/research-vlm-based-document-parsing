# Appendix: Chunking Quality Metrics — Implementation Details

## A.1 Overview

We adopt the MoC (Metrics of Chunks) framework (Zhao et al., ACL 2025) to evaluate
chunking quality of document parsing outputs. MoC provides two complementary metrics:

- **Boundary Clarity (BC)**: Measures local boundary separation between adjacent chunks.
- **Chunk Stickiness (CS)**: Measures global structural entropy of the semantic association graph.

Since the original paper does not release evaluation code, we provide a faithful
re-implementation based on the published formulas (Equations 1–3) with transparent
documentation of all engineering decisions made where the paper is underspecified.

## A.2 Formulas (from Zhao et al., ACL 2025)

**Boundary Clarity** (Eq. 1):

$$BC(q, d) = \frac{ppl(q \mid d)}{ppl(q)}$$

where $ppl(q)$ is the unconditional perplexity of sentence $q$, and
$ppl(q \mid d)$ is the contrastive perplexity of $q$ conditioned on chunk $d$.
$BC \to 1.0$ indicates independent chunks (good boundaries).

**Edge Weight** (Eq. 2):

$$Edge(q, d) = \frac{ppl(q) - ppl(q \mid d)}{ppl(q)} = 1 - BC(q, d)$$

**Chunk Stickiness** (Eq. 3):

$$CS(G) = -\sum_{i} \frac{h_i}{2m} \cdot \log_2 \frac{h_i}{2m}$$

where $G$ is the semantic association graph, $h_i$ is the degree of node $i$,
and $m$ is the total number of edges. Lower CS indicates more independent chunks.

## A.3 Implementation Decisions

The following aspects are not fully specified in Zhao et al. (2025).
We document our choices and their rationale.

### A.3.1 Perplexity Computation

| Aspect | Our implementation | Rationale |
|--------|-------------------|-----------|
| Method | vLLM completions API with `echo=True`, `logprobs=1` | Standard autoregressive log-likelihood approach |
| Formula | $ppl = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(t_i \mid t_{<i})\right)$ | Standard token-level perplexity definition |
| Conditioning | Concatenate context + `\n\n` + text as prompt | `\n\n` separator provides natural paragraph boundary |
| Token split | Tokenize context separately to determine split point; only use logprobs after the context portion | Ensures text perplexity is not contaminated by context token probabilities |
| Separator attribution | `\n\n` tokens are attributed to the context side | Conservative: keeps text logprobs clean |

The original paper states perplexity was computed using Qwen2.5 series models
(1.5B/7B/14B) but does not specify the exact computation method.

### A.3.2 Graph Construction for CS

| Parameter | Symbol | Default | Range tested | Rationale |
|-----------|--------|---------|-------------|-----------|
| Edge threshold | $K$ | 0.10 | 0.05–0.20 | Filters weak semantic associations; see sensitivity analysis (Section A.4) |
| Sequential skip | $\delta$ | 1 | 1–3 | Excludes trivially adjacent chunks from the graph |
| Symmetrization | — | max(w_ij, w_ji) | — | Edge(i,j) ≠ Edge(j,i) due to perplexity asymmetry; `max` is conservative (retains edge if either direction shows strong association) |

The original paper mentions the threshold $K$ and skip constraint $\delta$
in Section 3.2 but does not specify concrete values.

### A.3.3 BC Measurement Pairs

BC is computed for **adjacent chunk pairs** $(c_i, c_{i+1})$ by default,
consistent with the paper's focus on boundary quality between consecutive chunks.

## A.4 Sensitivity Analysis Protocol

To demonstrate that our findings are robust to hyperparameter choices,
we report CS across a grid of $(K, \delta)$ values for all compared models.

**Grid:**
- $K \in \{0.05, 0.10, 0.15, 0.20\}$
- $\delta \in \{1, 2, 3\}$

The sensitivity analysis reuses the pre-computed edge weight matrix
(no additional LLM calls required), making it computationally free.

**Expected table format:**

| $K$ | $\delta$ | Baseline CS | Our Model CS | $\Delta$ |
|-----|---------|------------|-------------|----------|
| 0.05 | 1 | — | — | — |
| 0.10 | 1 | — | — | — |
| ... | ... | ... | ... | ... |

**Key claim to verify:** The relative ordering between models should be
consistent across all $(K, \delta)$ configurations. If Model A achieves
lower CS than Model B at $K=0.10, \delta=1$, the same ordering should
hold at $K=0.05, \delta=2$, etc.

## A.5 Retrieval Metrics

We additionally report retrieval quality metrics following the BEIR benchmark
(Thakur et al., NeurIPS 2021):

| Metric | Formula | Description |
|--------|---------|-------------|
| nDCG@k | $\frac{DCG@k}{IDCG@k}$ where $DCG@k = \sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$ | Primary retrieval quality metric |
| Hit@K | $\mathbb{1}[\exists \text{ relevant doc in top-}K]$ | Binary retrieval success |
| MRR@K | $\frac{1}{\lvert Q \rvert}\sum_{q \in Q}\frac{1}{rank_q}$ | Mean Reciprocal Rank |
| Recall@K | $\frac{\lvert \text{relevant} \cap \text{top-}K \rvert}{\lvert \text{relevant} \rvert}$ | Coverage of relevant documents |

These metrics are standard with well-defined formulas and do not require
implementation-specific decisions. We evaluate at $k \in \{1, 3, 5, 10\}$.

## A.6 Reproducibility

All evaluation code is available in the `evaluation/metrics/` directory:
- `chunking.py`: BC, CS, sensitivity analysis
- `retrieval.py`: nDCG, Hit, MRR, Recall

Perplexity computation requires a vLLM-served language model accessible
via the OpenAI-compatible completions API.
