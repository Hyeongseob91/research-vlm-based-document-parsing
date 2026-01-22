"""
Chunking Quality Metrics

This module implements metrics for evaluating chunking quality:
- Boundary Score (BS): How well chunk boundaries align with semantic boundaries
- Chunk Score (CS): How coherent each chunk is internally

These metrics help explain why better parsing leads to better retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BoundaryScore:
    """Results of boundary alignment evaluation."""
    score: float  # Main score: aligned / total_gt
    aligned_boundaries: int
    total_gt_boundaries: int
    total_pred_boundaries: int
    precision: float  # aligned / total_pred
    recall: float  # aligned / total_gt (same as score)
    f1: float
    tolerance: int

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "aligned_boundaries": self.aligned_boundaries,
            "total_gt_boundaries": self.total_gt_boundaries,
            "total_pred_boundaries": self.total_pred_boundaries,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tolerance": self.tolerance,
        }


@dataclass
class ChunkScore:
    """Results of chunk coherence evaluation."""
    score: float  # Average coherence across chunks
    chunk_scores: list[float]  # Per-chunk coherence
    min_score: float
    max_score: float
    std_dev: float

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "chunk_count": len(self.chunk_scores),
            "min_score": self.min_score,
            "max_score": self.max_score,
            "std_dev": self.std_dev,
        }


@dataclass
class ChunkingMetrics:
    """Combined chunking quality metrics."""
    boundary_score: Optional[BoundaryScore] = None
    chunk_score: Optional[ChunkScore] = None

    def to_dict(self) -> dict:
        return {
            "boundary_score": self.boundary_score.to_dict() if self.boundary_score else None,
            "chunk_score": self.chunk_score.to_dict() if self.chunk_score else None,
        }


def calculate_boundary_score(
    predicted_text: str,
    ground_truth_text: str,
    tolerance: int = 50
) -> BoundaryScore:
    """
    Calculate how well predicted chunk boundaries align with ground truth.

    Boundary Score (BS) = |B_pred ∩ B_gt| / |B_gt|

    Args:
        predicted_text: Text after chunking (with chunk markers or just boundaries)
        ground_truth_text: Ground truth text with natural semantic boundaries
        tolerance: Character tolerance for boundary matching

    Returns:
        BoundaryScore with alignment metrics
    """
    # Extract boundary positions
    pred_boundaries = _extract_boundaries(predicted_text)
    gt_boundaries = _extract_boundaries(ground_truth_text)

    if not gt_boundaries:
        # No GT boundaries, return perfect score
        return BoundaryScore(
            score=1.0,
            aligned_boundaries=0,
            total_gt_boundaries=0,
            total_pred_boundaries=len(pred_boundaries),
            precision=1.0 if not pred_boundaries else 0.0,
            recall=1.0,
            f1=1.0 if not pred_boundaries else 0.0,
            tolerance=tolerance,
        )

    # Count aligned boundaries (within tolerance)
    aligned = 0
    used_pred = set()

    for gt_pos in gt_boundaries:
        for i, pred_pos in enumerate(pred_boundaries):
            if i not in used_pred and abs(gt_pos - pred_pos) <= tolerance:
                aligned += 1
                used_pred.add(i)
                break

    # Calculate metrics
    recall = aligned / len(gt_boundaries) if gt_boundaries else 1.0
    precision = aligned / len(pred_boundaries) if pred_boundaries else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return BoundaryScore(
        score=recall,  # Main score is recall (how many GT boundaries we captured)
        aligned_boundaries=aligned,
        total_gt_boundaries=len(gt_boundaries),
        total_pred_boundaries=len(pred_boundaries),
        precision=precision,
        recall=recall,
        f1=f1,
        tolerance=tolerance,
    )


def _extract_boundaries(text: str) -> list[int]:
    """
    Extract semantic boundary positions from text.

    Boundaries are identified by:
    - Double newlines (paragraph breaks)
    - Markdown headers
    - Table boundaries
    - List starts
    """
    boundaries = []

    # Paragraph breaks (double newline)
    for match in re.finditer(r'\n\n+', text):
        boundaries.append(match.start())

    # Markdown headers
    for match in re.finditer(r'^#{1,6}\s', text, re.MULTILINE):
        boundaries.append(match.start())

    # Table rows (simplified)
    for match in re.finditer(r'^\|.+\|$', text, re.MULTILINE):
        if match.start() > 0:
            boundaries.append(match.start())

    # List items
    for match in re.finditer(r'^[-*+]\s|\d+\.\s', text, re.MULTILINE):
        boundaries.append(match.start())

    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))
    return boundaries


def calculate_chunk_score(
    chunks: list,
    embedding_model: str = "jhgan/ko-sroberta-multitask"
) -> ChunkScore:
    """
    Calculate internal coherence score for each chunk.

    Chunk Score (CS) measures how semantically coherent each chunk is.
    Higher score = sentences within chunk are more similar to each other.

    Args:
        chunks: List of Chunk objects or strings
        embedding_model: Model to use for sentence embeddings

    Returns:
        ChunkScore with coherence metrics
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
    except ImportError:
        # Fallback: return mock scores
        return _mock_chunk_score(chunks)

    chunk_scores = []

    for chunk in chunks:
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        score = _calculate_single_chunk_coherence(content, model)
        chunk_scores.append(score)

    if not chunk_scores:
        return ChunkScore(
            score=0.0,
            chunk_scores=[],
            min_score=0.0,
            max_score=0.0,
            std_dev=0.0,
        )

    return ChunkScore(
        score=np.mean(chunk_scores),
        chunk_scores=chunk_scores,
        min_score=min(chunk_scores),
        max_score=max(chunk_scores),
        std_dev=np.std(chunk_scores),
    )


def _calculate_single_chunk_coherence(text: str, model) -> float:
    """
    Calculate coherence score for a single chunk.

    Coherence is measured as average pairwise similarity between sentences.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

    if len(sentences) <= 1:
        return 1.0  # Single sentence = perfectly coherent

    # Get embeddings
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(sim)

    if not similarities:
        return 1.0

    # Coherence = average similarity (higher = more coherent)
    return float(np.mean(similarities))


def _mock_chunk_score(chunks: list) -> ChunkScore:
    """Generate mock chunk scores when embeddings are unavailable."""
    # Simple heuristic: longer chunks with fewer paragraph breaks = more coherent
    scores = []
    for chunk in chunks:
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        paragraphs = content.count('\n\n') + 1
        length = len(content)

        # Heuristic score: penalize many paragraph breaks
        if length > 0:
            score = max(0.0, 1.0 - (paragraphs - 1) * 0.1)
        else:
            score = 0.0
        scores.append(score)

    if not scores:
        return ChunkScore(
            score=0.0,
            chunk_scores=[],
            min_score=0.0,
            max_score=0.0,
            std_dev=0.0,
        )

    return ChunkScore(
        score=np.mean(scores),
        chunk_scores=scores,
        min_score=min(scores),
        max_score=max(scores),
        std_dev=np.std(scores),
    )


def compare_chunking_quality(
    baseline_chunks: list,
    vlm_chunks: list,
    ground_truth_text: str,
    tolerance: int = 50
) -> dict:
    """
    Compare chunking quality between baseline and VLM parsers.

    Args:
        baseline_chunks: Chunks from baseline parser (pdfplumber/OCR)
        vlm_chunks: Chunks from VLM parser
        ground_truth_text: Ground truth document text
        tolerance: Boundary matching tolerance

    Returns:
        Dictionary with comparison metrics
    """
    # Reconstruct text from chunks for boundary comparison
    baseline_text = "\n\n".join(
        c.content if hasattr(c, 'content') else str(c) for c in baseline_chunks
    )
    vlm_text = "\n\n".join(
        c.content if hasattr(c, 'content') else str(c) for c in vlm_chunks
    )

    # Calculate boundary scores
    baseline_bs = calculate_boundary_score(baseline_text, ground_truth_text, tolerance)
    vlm_bs = calculate_boundary_score(vlm_text, ground_truth_text, tolerance)

    # Calculate chunk scores
    baseline_cs = calculate_chunk_score(baseline_chunks)
    vlm_cs = calculate_chunk_score(vlm_chunks)

    return {
        "baseline": {
            "boundary_score": baseline_bs.to_dict(),
            "chunk_score": baseline_cs.to_dict(),
            "chunk_count": len(baseline_chunks),
        },
        "vlm": {
            "boundary_score": vlm_bs.to_dict(),
            "chunk_score": vlm_cs.to_dict(),
            "chunk_count": len(vlm_chunks),
        },
        "improvement": {
            "boundary_score_delta": vlm_bs.score - baseline_bs.score,
            "chunk_score_delta": vlm_cs.score - baseline_cs.score,
            "boundary_score_pct": (
                (vlm_bs.score - baseline_bs.score) / baseline_bs.score * 100
                if baseline_bs.score > 0 else 0
            ),
            "chunk_score_pct": (
                (vlm_cs.score - baseline_cs.score) / baseline_cs.score * 100
                if baseline_cs.score > 0 else 0
            ),
        }
    }
