"""Structure F1 metric for document parsing evaluation.

Measures structural element preservation:
- Headings (# ## ### etc.)
- Lists (- * 1. etc.)
- Tables (| ... |)
- Code blocks (``` ```)
"""

import re
from dataclasses import dataclass


@dataclass
class StructureF1Result:
    """Structure F1 evaluation result."""
    f1: float
    precision: float
    recall: float
    pred_counts: dict[str, int]
    gt_counts: dict[str, int]


def count_structure_elements(text: str) -> dict[str, int]:
    """Count structural elements in markdown text."""
    counts = {
        "headings": 0,
        "lists": 0,
        "tables": 0,
        "code_blocks": 0,
    }

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r'^#{1,6}\s+', stripped):
            counts["headings"] += 1
        elif re.match(r'^[-*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            counts["lists"] += 1
        elif stripped.startswith("|") and stripped.endswith("|"):
            counts["tables"] += 1
        elif stripped.startswith("```"):
            counts["code_blocks"] += 1

    return counts


def calculate_structure_f1(prediction: str, reference: str) -> StructureF1Result:
    """Calculate Structure F1 between predicted and reference markdown.

    Args:
        prediction: Predicted/parsed markdown text
        reference: Ground truth markdown text

    Returns:
        StructureF1Result with F1, precision, recall, and element counts
    """
    pred_counts = count_structure_elements(prediction)
    gt_counts = count_structure_elements(reference)

    total_pred = sum(pred_counts.values())
    total_gt = sum(gt_counts.values())

    if total_pred == 0 and total_gt == 0:
        return StructureF1Result(
            f1=1.0, precision=1.0, recall=1.0,
            pred_counts=pred_counts, gt_counts=gt_counts,
        )

    # Count matching elements (min of pred and gt per category)
    matching = sum(min(pred_counts[k], gt_counts[k]) for k in pred_counts)

    precision = matching / total_pred if total_pred > 0 else 0.0
    recall = matching / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return StructureF1Result(
        f1=f1, precision=precision, recall=recall,
        pred_counts=pred_counts, gt_counts=gt_counts,
    )
