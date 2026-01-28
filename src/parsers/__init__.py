# VLM Document Parsing Test - Parsers Package

from .ocr_parser import (
    OCRParser,
    OCRResult,
    ImageOCRParser,
    RapidOCRParser,
    check_rapidocr_available,
)
from .text_structurer import TextStructurer, TextStructurerResult
from .two_stage_parser import TwoStageParser, TwoStageResult

__all__ = [
    # OCR Parser (PyMuPDF - Text-Baseline)
    "OCRParser",
    "OCRResult",
    "ImageOCRParser",
    # RapidOCR Parser (Image-Baseline)
    "RapidOCRParser",
    "check_rapidocr_available",
    # Text Structurer (LLM-based)
    "TextStructurer",
    "TextStructurerResult",
    # Two-Stage Parser (Baseline + VLM Structuring = Advanced)
    "TwoStageParser",
    "TwoStageResult",
]
