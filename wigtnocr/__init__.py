"""
WigtnOCR - Korean Government Document Intelligence
VLM-based document parser specialized for Korean public sector documents.
"""

from wigtnocr.pipeline.two_stage import TwoStageParser, TwoStageResult
from wigtnocr.pipeline.hybrid import HybridParser
from wigtnocr.parsers.pymupdf import OCRParser, OCRResult
from wigtnocr.parsers.rapidocr import RapidOCRParser
from wigtnocr.parsers.vlm import TextStructurer, TextStructurerResult

__version__ = "0.1.0"

__all__ = [
    "WigtnOCR",
    "TwoStageParser", "TwoStageResult",
    "HybridParser",
    "OCRParser", "OCRResult",
    "RapidOCRParser",
    "TextStructurer", "TextStructurerResult",
]


class WigtnOCR:
    """Main entry point for WigtnOCR document parsing.

    Usage:
        >>> from wigtnocr import WigtnOCR
        >>> ocr = WigtnOCR(mode="hybrid")
        >>> result = ocr.parse("document.pdf")
        >>> print(result.markdown)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        vlm_api_url: str = "http://localhost:8010/v1/chat/completions",
        vlm_model: str = "qwen3-vl-2b-instruct",
        vlm_timeout: float = 120.0,
    ):
        self.mode = mode
        self.vlm_api_url = vlm_api_url
        self.vlm_model = vlm_model
        self.vlm_timeout = vlm_timeout

        if mode == "hybrid":
            self._parser = HybridParser(
                vlm_api_url=vlm_api_url,
                vlm_model=vlm_model,
                vlm_timeout=vlm_timeout,
            )
        else:
            self._parser = TwoStageParser(
                structurer_api_url=vlm_api_url,
                structurer_model=vlm_model,
                structurer_timeout=vlm_timeout,
            )

    def parse(self, pdf_path: str) -> TwoStageResult:
        """Parse a PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            TwoStageResult with parsed content
        """
        from pathlib import Path
        pdf_bytes = Path(pdf_path).read_bytes()

        if self.mode == "hybrid":
            return self._parser.parse(pdf_bytes)
        return self._parser.parse_auto(pdf_bytes)
