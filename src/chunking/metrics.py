"""
MoC-based Chunking Quality Metrics

This module implements label-free chunking evaluation metrics based on the
MoC (Mixtures of Chunking) paper (arXiv:2503.09600v2).

Key Metrics:
- BC (Boundary Clarity): Measures independence between adjacent chunks
- CS (Chunk Stickiness): Measures overall graph connectivity via Structural Entropy

Implementation:
- Uses Semantic Distance (embedding cosine similarity) instead of perplexity
- OpenAI API cannot provide input token logprobs, so perplexity-based approach is not feasible
- Semantic Distance provides equivalent evaluation with better API compatibility

Advantages over traditional metrics:
- No Ground Truth required
- Repeatable measurements in production
- Strong correlation with RAG performance (BC↔ROUGE-L: 0.88)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections.abc import Sequence


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BCScore:
    """Boundary Clarity evaluation result.

    BC = ppl(q|d) / ppl(q)
    - Higher is better (chunks are more independent)
    - Close to 1.0: chunks are independent
    - Close to 0.0: chunks are highly dependent
    """
    score: float  # Average BC across all adjacent pairs
    pair_scores: list[float]  # Per-pair BC scores
    min_score: float
    max_score: float
    std_dev: float
    pair_count: int

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "pair_count": self.pair_count,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "std_dev": self.std_dev,
        }


@dataclass
class CSScore:
    """Chunk Stickiness evaluation result.

    CS = -Σ (h_i / 2m) * log2(h_i / 2m)  (Structural Entropy)
    - Lower is better (chunks are more independent)
    - h_i: degree of node i
    - m: total number of edges
    """
    score: float  # Structural Entropy
    graph_type: str  # "complete" or "incomplete"
    node_count: int
    edge_count: int
    threshold_k: float

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "graph_type": self.graph_type,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "threshold_k": self.threshold_k,
        }


@dataclass
class ChunkingMetrics:
    """Combined chunking quality metrics (BC + CS)."""
    bc_score: Optional[BCScore] = None
    cs_score: Optional[CSScore] = None

    def to_dict(self) -> dict:
        return {
            "bc": self.bc_score.to_dict() if self.bc_score else None,
            "cs": self.cs_score.to_dict() if self.cs_score else None,
        }


# =============================================================================
# Embedding Client for Semantic Distance Calculation
# =============================================================================

class EmbeddingClient:
    """Embedding-based client for semantic distance calculation.

    Uses sentence-transformers for computing embeddings and cosine similarity.
    This approach is used instead of perplexity because OpenAI API cannot
    provide input token logprobs needed for true perplexity calculation.
    """

    def __init__(
        self,
        model: str = "jhgan/ko-sroberta-multitask",
        device: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """Initialize embedding client.

        Args:
            model: Sentence transformer model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_embeddings: Whether to cache embeddings for repeated texts
        """
        self.model_name = model
        self.device = device
        self.cache_embeddings = cache_embeddings
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic distance calculation. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            return np.zeros(384)  # Default embedding dimension

        # Check cache
        if self.cache_embeddings and text in self._cache:
            return self._cache[text]

        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)

        # Cache result
        if self.cache_embeddings:
            self._cache[text] = embedding

        return embedding

    def get_embeddings_batch(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        # Check which texts need embedding
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            if self.cache_embeddings and text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts in batch
        if uncached_texts:
            model = self._get_model()
            new_embeddings = model.encode(uncached_texts, convert_to_numpy=True)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                results[idx] = emb
                if self.cache_embeddings:
                    self._cache[text] = emb

        return np.array(results)

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity in range [-1, 1] (typically [0, 1] for text)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def semantic_distance(self, text1: str, text2: str) -> float:
        """Calculate semantic distance between two texts.

        Semantic distance = 1 - cosine_similarity

        Args:
            text1: First text
            text2: Second text

        Returns:
            Semantic distance in range [0, 2] (typically [0, 1] for text)
        """
        return 1.0 - self.cosine_similarity(text1, text2)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}


class MockEmbeddingClient:
    """Mock embedding client for testing without actual model.

    Uses simple word overlap heuristics for similarity calculation.
    """

    def __init__(self):
        pass

    def get_embedding(self, text: str) -> np.ndarray:
        """Mock embedding using bag-of-words approach."""
        if not text.strip():
            return np.zeros(100)

        words = text.lower().split()
        # Simple hash-based embedding
        embedding = np.zeros(100)
        for word in words:
            idx = hash(word) % 100
            embedding[idx] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_embeddings_batch(self, texts: list[str]) -> np.ndarray:
        """Mock batch embedding."""
        return np.array([self.get_embedding(t) for t in texts])

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using mock embeddings."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def semantic_distance(self, text1: str, text2: str) -> float:
        """Calculate semantic distance."""
        return 1.0 - self.cosine_similarity(text1, text2)

    def clear_cache(self):
        """No-op for mock client."""
        pass


# =============================================================================
# Legacy LLM Clients (kept for reference, not used)
# =============================================================================

class LLMClient:
    """DEPRECATED: Client for LLM API with perplexity calculation support.

    Note: This class requires vLLM or similar local API with echo=True support.
    OpenAI API does not support this. Use EmbeddingClient instead.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1/completions",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: float = 60.0,
        api_key: str = "dummy"
    ):
        import warnings
        warnings.warn(
            "LLMClient is deprecated. Use EmbeddingClient for semantic distance calculation.",
            DeprecationWarning
        )
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def calculate_perplexity(self, text: str, context: Optional[str] = None) -> float:
        raise NotImplementedError(
            "Perplexity calculation requires local LLM with echo=True support. "
            "Use EmbeddingClient.semantic_distance() instead."
        )

    def calculate_perplexity_batch(self, texts: list[str], contexts: Optional[list[Optional[str]]] = None) -> list[float]:
        raise NotImplementedError("Use EmbeddingClient instead.")


class MockLLMClient:
    """DEPRECATED: Mock LLM client. Use MockEmbeddingClient instead."""

    def __init__(self):
        import warnings
        warnings.warn(
            "MockLLMClient is deprecated. Use MockEmbeddingClient instead.",
            DeprecationWarning
        )

    def calculate_perplexity(self, text: str, context: Optional[str] = None) -> float:
        # Simple heuristic for backward compatibility
        if not text.strip():
            return 1.0
        words = text.lower().split()
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)
        base_ppl = 10 + vocab_diversity * 90
        if context:
            context_words = set(context.lower().split())
            overlap = len(unique_words & context_words)
            overlap_ratio = overlap / max(len(unique_words), 1)
            base_ppl *= (1 - overlap_ratio * 0.5)
        return max(base_ppl, 1.0)

    def calculate_perplexity_batch(self, texts: list[str], contexts: Optional[list[Optional[str]]] = None) -> list[float]:
        if contexts is None:
            contexts = [None] * len(texts)
        return [self.calculate_perplexity(text, context) for text, context in zip(texts, contexts)]


class OpenAIClient:
    """DEPRECATED: OpenAI API cannot provide input token logprobs.

    Use EmbeddingClient for semantic distance calculation instead.
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, timeout: float = 60.0):
        raise NotImplementedError(
            "OpenAI Chat API does not support input token logprobs required for perplexity. "
            "Use EmbeddingClient for semantic distance-based BC/CS calculation."
        )


# =============================================================================
# BC (Boundary Clarity) Calculation - Semantic Distance Based
# =============================================================================

def calculate_bc(
    chunks: Sequence,
    embedding_client: EmbeddingClient | MockEmbeddingClient,
    verbose: bool = False
) -> BCScore:
    """Calculate Boundary Clarity for a list of chunks using semantic distance.

    BC measures how independent adjacent chunks are.
    BC = 1 - cosine_similarity(chunk_i, chunk_i+1)

    Interpretation:
    - Higher BC = more independent chunks (good)
    - BC close to 1.0: chunks are semantically different (independent)
    - BC close to 0.0: chunks are semantically similar (dependent)

    Args:
        chunks: List of Chunk objects or strings
        embedding_client: Embedding client for semantic distance calculation
        verbose: Print progress

    Returns:
        BCScore with average and per-pair scores
    """
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]

    if len(contents) < 2:
        return BCScore(
            score=1.0,
            pair_scores=[],
            min_score=1.0,
            max_score=1.0,
            std_dev=0.0,
            pair_count=0,
        )

    pair_scores = []

    # Pre-compute embeddings for efficiency
    if verbose:
        print("  Pre-computing embeddings...")
    embeddings = embedding_client.get_embeddings_batch(contents)

    for i in range(len(contents) - 1):
        if verbose:
            print(f"  BC calculation: chunk {i} → {i+1}", end="", flush=True)

        # Calculate cosine similarity between adjacent chunks
        emb_i = embeddings[i]
        emb_j = embeddings[i + 1]

        norm_i = np.linalg.norm(emb_i)
        norm_j = np.linalg.norm(emb_j)

        if norm_i == 0 or norm_j == 0:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))

        # BC = semantic distance = 1 - cosine_similarity
        # Higher BC = more independent (good)
        bc = 1.0 - cos_sim

        # Clamp to [0, 1] range
        bc = max(0.0, min(bc, 1.0))
        pair_scores.append(bc)

        if verbose:
            print(f" → BC={bc:.4f} (cos_sim={cos_sim:.4f})")

    if not pair_scores:
        return BCScore(
            score=1.0,
            pair_scores=[],
            min_score=1.0,
            max_score=1.0,
            std_dev=0.0,
            pair_count=0,
        )

    return BCScore(
        score=float(np.mean(pair_scores)),
        pair_scores=pair_scores,
        min_score=float(min(pair_scores)),
        max_score=float(max(pair_scores)),
        std_dev=float(np.std(pair_scores)),
        pair_count=len(pair_scores),
    )


# =============================================================================
# CS (Chunk Stickiness) Calculation - Semantic Distance Based
# =============================================================================

def calculate_edge_weight_semantic(
    emb_i: np.ndarray,
    emb_j: np.ndarray
) -> float:
    """Calculate edge weight between two chunks using cosine similarity.

    Edge weight = cosine_similarity(chunk_i, chunk_j)
    - Close to 1: high semantic similarity (chunks are related)
    - Close to 0: low similarity (independent)

    Args:
        emb_i: Embedding of first chunk
        emb_j: Embedding of second chunk

    Returns:
        Edge weight [0, 1]
    """
    norm_i = np.linalg.norm(emb_i)
    norm_j = np.linalg.norm(emb_j)

    if norm_i == 0 or norm_j == 0:
        return 0.0

    cos_sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))
    # Clamp to [0, 1] - negative similarities treated as 0
    return max(0.0, min(1.0, cos_sim))


def build_chunk_graph(
    chunks: Sequence,
    embedding_client: EmbeddingClient | MockEmbeddingClient,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    verbose: bool = False,
    embeddings: Optional[np.ndarray] = None
) -> dict[int, list[tuple[int, float]]]:
    """Build a weighted graph from chunks based on semantic similarity.

    Args:
        chunks: List of chunks
        embedding_client: Embedding client for similarity calculation
        threshold_k: Only keep edges with weight >= threshold_k
        graph_type: "complete" (all pairs) or "incomplete" (sequential only)
        verbose: Print progress
        embeddings: Pre-computed embeddings (optional)

    Returns:
        Adjacency list: {node_id: [(neighbor_id, weight), ...]}
    """
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]
    n = len(contents)

    if n == 0:
        return {}

    # Get embeddings if not provided
    if embeddings is None:
        if verbose:
            print("  Computing embeddings...")
        embeddings = embedding_client.get_embeddings_batch(contents)

    # Initialize adjacency list
    graph: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

    if graph_type == "complete":
        # Complete graph: all pairs
        total_pairs = n * (n - 1) // 2
        current = 0

        for i in range(n):
            for j in range(i + 1, n):
                current += 1
                if verbose:
                    print(f"  Edge {current}/{total_pairs}: {i} ↔ {j}", end="", flush=True)

                # Calculate cosine similarity as edge weight
                weight = calculate_edge_weight_semantic(embeddings[i], embeddings[j])

                if verbose:
                    print(f" → w={weight:.4f}")

                if weight >= threshold_k:
                    graph[i].append((j, weight))
                    graph[j].append((i, weight))

    elif graph_type == "incomplete":
        # Incomplete graph: only sequential pairs
        for i in range(n - 1):
            j = i + 1
            if verbose:
                print(f"  Edge: {i} → {j}", end="", flush=True)

            weight = calculate_edge_weight_semantic(embeddings[i], embeddings[j])

            if verbose:
                print(f" → w={weight:.4f}")

            if weight >= threshold_k:
                graph[i].append((j, weight))
                graph[j].append((i, weight))

    return graph


def calculate_structural_entropy(graph: dict[int, list[tuple[int, float]]]) -> float:
    """Calculate Structural Entropy of the graph.

    H = -Σ (h_i / 2m) * log2(h_i / 2m)

    Where:
    - h_i: weighted degree of node i
    - m: total edge weight / 2

    Args:
        graph: Adjacency list with weights

    Returns:
        Structural entropy (lower = better chunking)
    """
    if not graph:
        return 0.0

    # Calculate weighted degrees
    degrees = {}
    for node, neighbors in graph.items():
        degrees[node] = sum(w for _, w in neighbors)

    # Total edge weight (divide by 2 for undirected)
    total_weight = sum(degrees.values())
    m = total_weight / 2

    if m <= 0:
        return 0.0

    # Calculate structural entropy
    entropy = 0.0
    for node, h_i in degrees.items():
        if h_i > 0:
            p = h_i / (2 * m)
            entropy -= p * math.log2(p)

    return entropy


def calculate_cs(
    chunks: Sequence,
    embedding_client: EmbeddingClient | MockEmbeddingClient,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    verbose: bool = False,
    embeddings: Optional[np.ndarray] = None
) -> CSScore:
    """Calculate Chunk Stickiness using Structural Entropy with semantic similarity.

    Args:
        chunks: List of chunks
        embedding_client: Embedding client for similarity calculation
        threshold_k: Edge filtering threshold (keep edges with similarity >= threshold)
        graph_type: "complete" or "incomplete"
        verbose: Print progress
        embeddings: Pre-computed embeddings (optional)

    Returns:
        CSScore with structural entropy
    """
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]

    if len(contents) < 2:
        return CSScore(
            score=0.0,
            graph_type=graph_type,
            node_count=len(contents),
            edge_count=0,
            threshold_k=threshold_k,
        )

    # Get embeddings if not provided
    if embeddings is None:
        if verbose:
            print("  Computing embeddings...")
        embeddings = embedding_client.get_embeddings_batch(contents)

    # Build graph
    if verbose:
        print(f"  Building {graph_type} graph (threshold={threshold_k})...")

    graph = build_chunk_graph(
        chunks, embedding_client, threshold_k, graph_type, verbose, embeddings
    )

    # Count edges (divide by 2 for undirected)
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2

    # Calculate structural entropy
    entropy = calculate_structural_entropy(graph)

    return CSScore(
        score=entropy,
        graph_type=graph_type,
        node_count=len(contents),
        edge_count=edge_count,
        threshold_k=threshold_k,
    )


# =============================================================================
# Combined Evaluation
# =============================================================================

def evaluate_chunking(
    chunks: Sequence,
    embedding_client: EmbeddingClient | MockEmbeddingClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    calculate_cs_flag: bool = True,
    verbose: bool = False
) -> ChunkingMetrics:
    """Evaluate chunking quality using BC and CS metrics with semantic distance.

    Args:
        chunks: List of chunks
        embedding_client: Embedding client (uses MockEmbeddingClient if None)
        threshold_k: CS edge filtering threshold (similarity >= threshold)
        graph_type: CS graph type ("complete" or "incomplete")
        calculate_cs_flag: Whether to calculate CS (can be slow for complete graph)
        verbose: Print progress

    Returns:
        ChunkingMetrics with BC and CS scores
    """
    if embedding_client is None:
        print("Warning: No embedding client provided, using MockEmbeddingClient")
        embedding_client = MockEmbeddingClient()

    # Pre-compute embeddings for efficiency
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]

    if verbose:
        print("Pre-computing embeddings for all chunks...")
    embeddings = embedding_client.get_embeddings_batch(contents)

    # Calculate BC
    if verbose:
        print("Calculating BC (Boundary Clarity)...")
    bc_score = calculate_bc(chunks, embedding_client, verbose)

    # Calculate CS
    cs_score = None
    if calculate_cs_flag:
        if verbose:
            print("Calculating CS (Chunk Stickiness)...")
        cs_score = calculate_cs(
            chunks, embedding_client, threshold_k, graph_type, verbose, embeddings
        )

    return ChunkingMetrics(bc_score=bc_score, cs_score=cs_score)


def compare_chunking_quality(
    results: dict[str, Sequence],
    embedding_client: EmbeddingClient | MockEmbeddingClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    verbose: bool = False
) -> dict:
    """Compare chunking quality across multiple parsers.

    Args:
        results: {parser_name: chunks} dictionary
        embedding_client: Embedding client for similarity calculation
        threshold_k: CS threshold
        graph_type: CS graph type
        verbose: Print progress

    Returns:
        Comparison results with metrics per parser
    """
    comparison = {}

    for parser_name, chunks in results.items():
        if verbose:
            print(f"\n=== Evaluating: {parser_name} ===")

        metrics = evaluate_chunking(
            chunks, embedding_client, threshold_k, graph_type,
            calculate_cs_flag=True, verbose=verbose
        )

        comparison[parser_name] = {
            "metrics": metrics.to_dict(),
            "chunk_count": len(chunks),
        }

    return comparison


# =============================================================================
# Convenience Functions
# =============================================================================

def create_embedding_client(
    model: str = "jhgan/ko-sroberta-multitask",
    use_mock: bool = False
) -> EmbeddingClient | MockEmbeddingClient:
    """Factory function to create embedding client.

    Args:
        model: Sentence transformer model name
        use_mock: Use mock client (no actual embeddings)

    Returns:
        EmbeddingClient or MockEmbeddingClient instance
    """
    if use_mock:
        return MockEmbeddingClient()
    return EmbeddingClient(model=model)
