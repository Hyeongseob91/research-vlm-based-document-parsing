"""Table Edit Distance Similarity (TEDS) metric.

Evaluates table structure preservation quality using tree edit distance.
"""

from typing import Optional


def calculate_teds(pred_html: str, gt_html: str) -> float:
    """Calculate TEDS score between predicted and ground truth HTML tables.

    Args:
        pred_html: Predicted HTML table
        gt_html: Ground truth HTML table

    Returns:
        TEDS score in [0, 1] (1.0 = perfect match)
    """
    try:
        from apted import APTED, Config as AptedConfig
    except ImportError:
        raise ImportError("apted required for TEDS. Install with: pip install apted")

    # TODO: Implement full TEDS with HTML tree parsing
    # For now, delegate to simple string comparison
    if pred_html.strip() == gt_html.strip():
        return 1.0

    # Placeholder - will be replaced with full APTED-based TEDS
    return 0.0
