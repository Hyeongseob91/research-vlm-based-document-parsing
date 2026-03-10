"""Hybrid Parser - Routes documents based on complexity.

Simple digital PDFs → PyMuPDF (CPU, fast)
Complex/scanned PDFs → VLM (GPU, accurate)
"""

import time
from dataclasses import dataclass
from typing import Optional

from wigtnocr.parsers.pymupdf import OCRParser, ImageOCRParser
from wigtnocr.parsers.vlm import TextStructurer
from wigtnocr.pipeline.two_stage import TwoStageParser, TwoStageResult


@dataclass
class ComplexityScore:
    """Document complexity assessment."""
    score: float  # 0.0 (simple) to 1.0 (complex)
    has_images: bool
    has_tables: bool
    text_density: float  # chars per page
    is_scanned: bool
    reason: str


class HybridParser:
    """Routes documents to appropriate parser based on complexity.

    Routing logic:
    - Digital PDF + simple layout → PyMuPDF only (CPU, ~0.3s/page)
    - Complex tables/diagrams/scanned → VLM (GPU, ~40s/page)
    """

    COMPLEXITY_THRESHOLD = 0.5

    def __init__(
        self,
        vlm_api_url: str = "http://localhost:8010/v1/chat/completions",
        vlm_model: str = "qwen3-vl-2b-instruct",
        vlm_timeout: float = 120.0,
        complexity_threshold: float = 0.5,
    ):
        self.ocr_parser = OCRParser()
        self.image_parser = ImageOCRParser()
        self.two_stage = TwoStageParser(
            structurer_api_url=vlm_api_url,
            structurer_model=vlm_model,
            structurer_timeout=vlm_timeout,
        )
        self.complexity_threshold = complexity_threshold

    def assess_complexity(self, pdf_bytes: bytes) -> ComplexityScore:
        """Assess document complexity to decide routing."""
        import fitz

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            total_text = 0
            total_images = 0
            total_tables = 0

            sample_pages = min(5, page_count)
            for i in range(sample_pages):
                page = doc[i]
                text = page.get_text("text")
                total_text += len(text.strip())
                total_images += len(page.get_images())
                tables = page.find_tables()
                total_tables += len(tables)

            doc.close()

            text_density = total_text / max(sample_pages, 1)
            is_scanned = text_density < 50
            has_images = total_images > 0
            has_tables = total_tables > 0

            # Complexity scoring
            score = 0.0
            reasons = []

            if is_scanned:
                score += 0.6
                reasons.append("scanned document")
            if has_tables:
                score += 0.2
                reasons.append(f"{total_tables} tables")
            if has_images:
                score += 0.1
                reasons.append(f"{total_images} images")
            if text_density < 200:
                score += 0.1
                reasons.append("low text density")

            score = min(1.0, score)

            return ComplexityScore(
                score=score,
                has_images=has_images,
                has_tables=has_tables,
                text_density=text_density,
                is_scanned=is_scanned,
                reason=", ".join(reasons) if reasons else "simple digital PDF",
            )

        except Exception as e:
            return ComplexityScore(
                score=1.0,
                has_images=False,
                has_tables=False,
                text_density=0,
                is_scanned=True,
                reason=f"analysis failed: {e}",
            )

    def parse(self, pdf_bytes: bytes) -> TwoStageResult:
        """Parse document with automatic routing."""
        complexity = self.assess_complexity(pdf_bytes)

        if complexity.score >= self.complexity_threshold:
            # Complex document → VLM pipeline
            return self.two_stage.parse_auto(pdf_bytes)
        else:
            # Simple document → PyMuPDF only
            start_time = time.time()
            result = self.ocr_parser.parse_pdf(pdf_bytes)
            return TwoStageResult(
                success=result.success,
                content=result.content,
                stage1_content=result.content,
                stage1_parser="pymupdf",
                stage2_applied=False,
                elapsed_time=time.time() - start_time,
                stage1_time=result.elapsed_time,
                stage2_time=0.0,
                page_count=result.page_count,
                pdf_type="digital",
                error=result.error,
            )
