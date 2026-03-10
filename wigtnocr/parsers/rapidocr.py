"""
RapidOCR Parser - 스캔 PDF OCR (Image-Baseline)

RapidOCR 기반 PDF OCR 파서입니다.
스캔된 PDF나 이미지 기반 PDF에서 OCR을 통해 텍스트를 추출합니다.
"""

import time
import logging
import os
from typing import List, Optional

from wigtnocr.parsers.pymupdf import OCRResult


# ==============================================================================
# RapidOCR 선택적 임포트 및 싱글톤 인스턴스
# ==============================================================================

RAPIDOCR_AVAILABLE = False
_rapidocr_instance = None  # 싱글톤 인스턴스

try:
    # RapidOCR 로깅 억제 (import 전에 설정)
    os.environ.setdefault("RAPIDOCR_LOG_LEVEL", "WARNING")
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    logging.getLogger("rapidocr").setLevel(logging.WARNING)

    from rapidocr_pdf import RapidOCRPDF
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RapidOCRPDF = None


def _get_rapidocr_instance():
    """RapidOCRPDF 싱글톤 인스턴스 반환 (lazy initialization)"""
    global _rapidocr_instance
    if _rapidocr_instance is None and RAPIDOCR_AVAILABLE:
        _rapidocr_instance = RapidOCRPDF()
    return _rapidocr_instance


def check_rapidocr_available() -> bool:
    """RapidOCR 사용 가능 여부 확인"""
    return RAPIDOCR_AVAILABLE


# ==============================================================================
# Main Class: RapidOCRParser (스캔 PDF OCR - Image-Baseline)
# ==============================================================================

class RapidOCRParser:
    """RapidOCR 기반 PDF OCR 파서 (Image-Baseline)

    스캔된 PDF나 이미지 기반 PDF에서 OCR을 통해 텍스트를 추출합니다.
    rapidocr-pdf 패키지를 사용하여 PDF를 직접 처리합니다.
    싱글톤 패턴으로 모델을 한 번만 로드합니다.

    Example:
        >>> parser = RapidOCRParser()
        >>> result = parser.parse_pdf(pdf_bytes)
        >>> print(result.content)
    """

    # ==========================================================================
    # Class Constants
    # ==========================================================================

    MIN_TEXT_LENGTH = 50  # 텍스트 존재 판별 기준

    # ==========================================================================
    # Constructor
    # ==========================================================================

    def __init__(self):
        """RapidOCRParser 초기화 (싱글톤 인스턴스 사용)"""
        if not RAPIDOCR_AVAILABLE:
            raise ImportError(
                "rapidocr-pdf가 설치되지 않았습니다. "
                "pip install rapidocr-pdf rapidocr-onnxruntime 명령으로 설치하세요."
            )
        self._extractor = _get_rapidocr_instance()

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def parse_pdf(self, pdf_bytes: bytes) -> OCRResult:
        """PDF에서 OCR 텍스트 추출 (Entry Point)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            OCRResult: 파싱 결과

        Flow:
            1. RapidOCR로 PDF 처리
            2. 페이지별 텍스트 수집
            3. 콘텐츠 결합
            4. OCRResult 반환
        """
        start_time = time.time()
        pages_text = []

        try:
            # RapidOCR로 PDF 처리
            # 반환값: List[Tuple[page_num, content, score]]
            results = self._extractor(pdf_bytes)

            if not results:
                return OCRResult(
                    success=False,
                    content="",
                    pages=[],
                    tables=[],
                    elapsed_time=time.time() - start_time,
                    page_count=0,
                    has_text=False,
                    error="OCR 결과가 없습니다."
                )

            # 페이지별 텍스트 수집
            page_count = 0
            for item in results:
                if len(item) >= 2:
                    page_num, content = item[0], item[1]
                    pages_text.append(content if content else "")
                    page_count = max(page_count, page_num + 1)

            # 콘텐츠 결합
            content = self._combine_content(pages_text)

            # 텍스트 존재 여부 판별
            total_text = "".join(pages_text)
            has_text = len(total_text.strip()) > self.MIN_TEXT_LENGTH

            return OCRResult(
                success=True,
                content=content,
                pages=pages_text,
                tables=[],  # RapidOCR는 표 추출 미지원
                elapsed_time=time.time() - start_time,
                page_count=page_count,
                has_text=has_text
            )

        except Exception as e:
            return OCRResult(
                success=False,
                content="",
                pages=[],
                tables=[],
                elapsed_time=time.time() - start_time,
                page_count=0,
                has_text=False,
                error=f"OCR 파싱 실패: {str(e)}"
            )

    # ==========================================================================
    # Private Methods (Internal Helpers)
    # ==========================================================================

    def _combine_content(self, pages_text: List[str]) -> str:
        """페이지 텍스트를 하나의 콘텐츠로 결합

        Args:
            pages_text: 페이지별 텍스트 리스트

        Returns:
            결합된 전체 콘텐츠
        """
        content = "\n\n---\n\n".join(
            f"[Page {i+1}]\n{text}"
            for i, text in enumerate(pages_text)
            if text.strip()
        )
        return content
