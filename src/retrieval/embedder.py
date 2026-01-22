"""
Text Embedding Module

Provides embedding generation for chunks and queries
using sentence-transformers models.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for text embedder."""
    model: str = "jhgan/ko-sroberta-multitask"
    dimension: int = 768
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda, mps
    show_progress: bool = False

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "device": self.device,
        }


class TextEmbedder:
    """
    Text embedding generator using sentence-transformers.

    Supports Korean and English text with multilingual models.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.config.model,
                    device=self.config.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(
        self,
        texts: Union[str, list[str]],
        show_progress: Optional[bool] = None
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            show_progress: Override config show_progress setting

        Returns:
            numpy array of shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        show_progress = show_progress if show_progress is not None else self.config.show_progress

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def embed_chunks(self, chunks: list) -> list[dict]:
        """
        Embed a list of chunks and return with metadata.

        Args:
            chunks: List of Chunk objects or dicts with 'content' key

        Returns:
            List of dicts with 'id', 'content', 'embedding'
        """
        # Extract content from chunks
        texts = []
        chunk_data = []

        for chunk in chunks:
            if hasattr(chunk, 'content'):
                content = chunk.content
                chunk_id = chunk.id if hasattr(chunk, 'id') else str(len(texts))
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            elif isinstance(chunk, dict):
                content = chunk.get('content', str(chunk))
                chunk_id = chunk.get('id', str(len(texts)))
                metadata = chunk.get('metadata', {})
            else:
                content = str(chunk)
                chunk_id = str(len(texts))
                metadata = {}

            texts.append(content)
            chunk_data.append({
                'id': chunk_id,
                'content': content,
                'metadata': metadata
            })

        # Generate embeddings
        embeddings = self.embed(texts)

        # Combine with chunk data
        for i, chunk in enumerate(chunk_data):
            chunk['embedding'] = embeddings[i]

        return chunk_data

    def similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Calculate cosine similarity and return top-k results.

        Args:
            query_embedding: Single query embedding (dimension,)
            corpus_embeddings: Corpus embeddings (n_docs, dimension)
            top_k: Number of results to return

        Returns:
            List of (index, score) tuples sorted by score descending
        """
        # Ensure query is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()

        # Calculate cosine similarity
        if self.config.normalize:
            # Already normalized, just dot product
            scores = np.dot(corpus_embeddings, query_embedding)
        else:
            # Calculate cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
            scores = np.dot(corpus_embeddings, query_embedding) / (corpus_norms * query_norm)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


class MockEmbedder(TextEmbedder):
    """
    Mock embedder for testing without real embeddings.

    Uses simple TF-IDF-like approach for basic similarity.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._vectorizer = None

    @property
    def vectorizer(self):
        """Lazy load TF-IDF vectorizer."""
        if self._vectorizer is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vectorizer = TfidfVectorizer(max_features=self.config.dimension)
            except ImportError:
                raise ImportError(
                    "scikit-learn required for mock embedder. "
                    "Install with: pip install scikit-learn"
                )
        return self._vectorizer

    def embed(
        self,
        texts: Union[str, list[str]],
        show_progress: Optional[bool] = None
    ) -> np.ndarray:
        """Generate TF-IDF based embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        # Fit and transform
        vectors = self.vectorizer.fit_transform(texts)
        embeddings = vectors.toarray()

        # Pad or truncate to desired dimension
        if embeddings.shape[1] < self.config.dimension:
            padding = np.zeros((embeddings.shape[0], self.config.dimension - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        elif embeddings.shape[1] > self.config.dimension:
            embeddings = embeddings[:, :self.config.dimension]

        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        return embeddings


def create_embedder(config: EmbeddingConfig, use_mock: bool = False) -> TextEmbedder:
    """Factory function to create appropriate embedder."""
    if use_mock:
        return MockEmbedder(config)

    try:
        embedder = TextEmbedder(config)
        # Test if model loads
        _ = embedder.model
        return embedder
    except ImportError:
        print("Warning: sentence-transformers not available, using mock embedder")
        return MockEmbedder(config)
