"""
Chunk Retriever Module

Provides semantic search over document chunks
for RAG evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .embedder import EmbeddingConfig, TextEmbedder, create_embedder


@dataclass
class RetrievalConfig:
    """Configuration for chunk retriever."""
    method: str = "cosine_similarity"  # cosine_similarity, dot_product
    top_k: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    threshold: float = 0.0  # Minimum similarity threshold
    embedding_config: Optional[EmbeddingConfig] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "embedding": self.embedding_config.to_dict() if self.embedding_config else None,
        }


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    query: str
    query_id: Optional[str] = None
    retrieved_chunks: list = field(default_factory=list)  # List of (chunk_id, score)
    top_k: int = 5
    expected_chunk_id: Optional[str] = None
    is_hit: bool = False
    hit_rank: Optional[int] = None  # Rank of expected chunk if found

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_id": self.query_id,
            "retrieved_chunks": [
                {"chunk_id": cid, "score": score}
                for cid, score in self.retrieved_chunks
            ],
            "top_k": self.top_k,
            "expected_chunk_id": self.expected_chunk_id,
            "is_hit": self.is_hit,
            "hit_rank": self.hit_rank,
        }


class ChunkRetriever:
    """
    Semantic chunk retriever for document search.

    Embeds chunks and queries, then performs similarity search.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        embedder: Optional[TextEmbedder] = None
    ):
        self.config = config or RetrievalConfig()
        self._embedder = embedder
        self._chunks = []
        self._embeddings = None
        self._chunk_id_to_idx = {}

    @property
    def embedder(self) -> TextEmbedder:
        """Get or create embedder."""
        if self._embedder is None:
            embedding_config = self.config.embedding_config or EmbeddingConfig()
            self._embedder = create_embedder(embedding_config)
        return self._embedder

    def index_chunks(self, chunks: list):
        """
        Index chunks for retrieval.

        Args:
            chunks: List of Chunk objects or dicts with 'id' and 'content'
        """
        self._chunks = []
        self._chunk_id_to_idx = {}

        # Extract chunk info
        texts = []
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'id'):
                chunk_id = chunk.id
                content = chunk.content
            elif isinstance(chunk, dict):
                chunk_id = chunk.get('id', f'chunk_{i}')
                content = chunk.get('content', str(chunk))
            else:
                chunk_id = f'chunk_{i}'
                content = str(chunk)

            self._chunks.append({
                'id': chunk_id,
                'content': content,
                'original': chunk
            })
            self._chunk_id_to_idx[chunk_id] = i
            texts.append(content)

        # Generate embeddings
        self._embeddings = self.embedder.embed(texts)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expected_chunk_id: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            expected_chunk_id: Expected chunk ID for evaluation
            query_id: Optional ID for the query

        Returns:
            RetrievalResult with retrieved chunks and hit information
        """
        if self._embeddings is None or len(self._chunks) == 0:
            raise ValueError("No chunks indexed. Call index_chunks first.")

        top_k = top_k or max(self.config.top_k)

        # Embed query
        query_embedding = self.embedder.embed(query)[0]

        # Calculate similarities
        results = self.embedder.similarity(
            query_embedding,
            self._embeddings,
            top_k=top_k
        )

        # Filter by threshold
        if self.config.threshold > 0:
            results = [(idx, score) for idx, score in results if score >= self.config.threshold]

        # Map indices to chunk IDs
        retrieved_chunks = [
            (self._chunks[idx]['id'], score)
            for idx, score in results
        ]

        # Check if expected chunk is in results
        is_hit = False
        hit_rank = None
        if expected_chunk_id:
            for rank, (chunk_id, _) in enumerate(retrieved_chunks):
                if chunk_id == expected_chunk_id:
                    is_hit = True
                    hit_rank = rank + 1  # 1-indexed rank
                    break

        return RetrievalResult(
            query=query,
            query_id=query_id,
            retrieved_chunks=retrieved_chunks,
            top_k=top_k,
            expected_chunk_id=expected_chunk_id,
            is_hit=is_hit,
            hit_rank=hit_rank,
        )

    def retrieve_batch(
        self,
        queries: list[dict],
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of dicts with 'query', optional 'expected_chunk_id', 'query_id'
            top_k: Number of chunks to retrieve per query

        Returns:
            List of RetrievalResult objects
        """
        results = []
        for q in queries:
            result = self.retrieve(
                query=q['query'],
                top_k=top_k,
                expected_chunk_id=q.get('expected_chunk_id'),
                query_id=q.get('query_id')
            )
            results.append(result)
        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        """Get chunk by ID."""
        idx = self._chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self._chunks[idx]
        return None

    def find_relevant_chunk(
        self,
        answer_span: str,
        threshold: float = 0.5
    ) -> Optional[str]:
        """
        Find the chunk that contains the answer span.

        Args:
            answer_span: Text span that should be in the relevant chunk
            threshold: Minimum overlap ratio to consider a match

        Returns:
            Chunk ID of the most relevant chunk, or None
        """
        best_chunk_id = None
        best_overlap = 0.0

        answer_lower = answer_span.lower()

        for chunk in self._chunks:
            content_lower = chunk['content'].lower()

            # Check if answer span is in chunk
            if answer_lower in content_lower:
                # Calculate overlap ratio
                overlap = len(answer_span) / len(chunk['content'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk_id = chunk['id']

        if best_overlap >= threshold or best_chunk_id:
            return best_chunk_id

        # Fallback: find chunk with highest word overlap
        answer_words = set(answer_lower.split())
        for chunk in self._chunks:
            content_words = set(chunk['content'].lower().split())
            overlap = len(answer_words & content_words) / len(answer_words) if answer_words else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk_id = chunk['id']

        return best_chunk_id if best_overlap >= threshold else None


class SimpleRetriever(ChunkRetriever):
    """
    Simple retriever using TF-IDF without neural embeddings.

    Useful for testing or when GPU is not available.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self._vectorizer = None
        self._chunk_vectors = None

    def index_chunks(self, chunks: list):
        """Index chunks using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn required for SimpleRetriever. "
                "Install with: pip install scikit-learn"
            )

        self._chunks = []
        self._chunk_id_to_idx = {}

        texts = []
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'id'):
                chunk_id = chunk.id
                content = chunk.content
            elif isinstance(chunk, dict):
                chunk_id = chunk.get('id', f'chunk_{i}')
                content = chunk.get('content', str(chunk))
            else:
                chunk_id = f'chunk_{i}'
                content = str(chunk)

            self._chunks.append({
                'id': chunk_id,
                'content': content,
                'original': chunk
            })
            self._chunk_id_to_idx[chunk_id] = i
            texts.append(content)

        self._vectorizer = TfidfVectorizer()
        self._chunk_vectors = self._vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expected_chunk_id: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> RetrievalResult:
        """Retrieve using TF-IDF similarity."""
        if self._chunk_vectors is None:
            raise ValueError("No chunks indexed. Call index_chunks first.")

        from sklearn.metrics.pairwise import cosine_similarity

        top_k = top_k or max(self.config.top_k)

        # Transform query
        query_vector = self._vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self._chunk_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Map to chunk IDs
        retrieved_chunks = [
            (self._chunks[idx]['id'], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= self.config.threshold
        ]

        # Check hit
        is_hit = False
        hit_rank = None
        if expected_chunk_id:
            for rank, (chunk_id, _) in enumerate(retrieved_chunks):
                if chunk_id == expected_chunk_id:
                    is_hit = True
                    hit_rank = rank + 1
                    break

        return RetrievalResult(
            query=query,
            query_id=query_id,
            retrieved_chunks=retrieved_chunks,
            top_k=top_k,
            expected_chunk_id=expected_chunk_id,
            is_hit=is_hit,
            hit_rank=hit_rank,
        )
