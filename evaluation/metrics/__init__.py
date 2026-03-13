"""Evaluation metrics for document parsing quality."""

from evaluation.metrics.cer import calculate_cer, calculate_wer, calculate_edit_distance
from evaluation.metrics.structure import (
    calculate_structure_eval,
    calculate_structure_f1,  # legacy wrapper
    StructureEvalResult,
    StructureF1Result,  # legacy alias
    extract_elements,
)
from evaluation.metrics.teds import calculate_teds

__all__ = [
    "calculate_cer", "calculate_wer", "calculate_edit_distance",
    "calculate_structure_eval", "calculate_structure_f1",
    "StructureEvalResult", "StructureF1Result",
    "extract_elements",
    "calculate_teds",
]
