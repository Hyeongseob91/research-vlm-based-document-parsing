"""
Error Analysis Module for VLM Document Parsing

This module provides tools for analyzing and categorizing parsing errors:
- Error detection and classification
- Diff visualization
- Case study generation
- Error frequency analysis
"""

from .analyzer import (
    ErrorCategory,
    ErrorSeverity,
    ParsingError,
    ErrorAnalyzer,
)
from .diff_visualizer import (
    DiffResult,
    TextDiff,
    create_html_diff,
)
from .case_study import (
    CaseStudy,
    CaseStudyGenerator,
)

__all__ = [
    # Error analysis
    "ErrorCategory",
    "ErrorSeverity",
    "ParsingError",
    "ErrorAnalyzer",
    # Diff visualization
    "DiffResult",
    "TextDiff",
    "create_html_diff",
    # Case study
    "CaseStudy",
    "CaseStudyGenerator",
]
