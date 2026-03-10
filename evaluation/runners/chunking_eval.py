"""Chunking evaluation runner.

Evaluates chunking quality using BC/CS metrics on parsed outputs.
"""

from pathlib import Path
from typing import Optional

from wigtnocr.chunking.chunker import ChunkerConfig, create_chunker, Chunk
from wigtnocr.chunking.metrics import (
    evaluate_chunking,
    create_embedding_client,
    ChunkingMetrics,
    EmbeddingClientType,
)


def evaluate_document_chunking(
    parsed_text: str,
    parser_name: str = "default",
    breakpoint_type: str = "percentile",
    breakpoint_threshold: float = 95.0,
    embedding_api_url: str = "http://localhost:8001/embeddings",
    embedding_model: str = "BAAI/bge-m3",
) -> dict:
    """Evaluate chunking quality for a single parsed document.

    Args:
        parsed_text: Parsed document text
        parser_name: Parser identifier
        breakpoint_type: Semantic chunking breakpoint type
        breakpoint_threshold: Breakpoint threshold
        embedding_api_url: Embedding API URL
        embedding_model: Embedding model name

    Returns:
        Dict with chunking metrics
    """
    config = ChunkerConfig(
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    chunker = create_chunker(config, embedding_api_url, embedding_model)
    chunks = chunker.chunk(parsed_text, document_id=parser_name)

    if len(chunks) < 2:
        return {"chunk_count": len(chunks), "bc": None, "cs": None}

    client = create_embedding_client(api_url=embedding_api_url, model=embedding_model)
    metrics = evaluate_chunking(chunks, embedding_client=client)

    return {
        "parser": parser_name,
        "chunk_count": len(chunks),
        "bc": metrics.bc_score.to_dict() if metrics.bc_score else None,
        "cs": metrics.cs_score.to_dict() if metrics.cs_score else None,
    }
