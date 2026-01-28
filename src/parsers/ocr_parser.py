"""
OCR Parser - PDF 문서 파싱 모듈

디지털 PDF와 스캔 PDF 모두 지원하는 문서 파싱 모듈입니다.

Parsers:
    1. OCRParser (Text-Baseline): PyMuPDF 기반 디지털 PDF 텍스트 추출
    2. RapidOCRParser (Image-Baseline): RapidOCR 기반 스캔 PDF OCR
    3. ImageOCRParser: PDF → 이미지 변환 유틸리티

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  pdf_bytes                                                  │
    │       │                                                     │
    │       ├─────────────────────┬───────────────────┐           │
    │       ▼                     ▼                   ▼           │
    │  ┌──────────┐        ┌───────────┐       ┌───────────┐      │
    │  │ OCRParser │        │ RapidOCR  │       │ ImageOCR  │      │
    │  │ (PyMuPDF) │        │ Parser    │       │ Parser    │      │
    │  │ Text-Base │        │ Image-Base│       │ (PyMuPDF) │      │
    │  └──────────┘        └───────────┘       └───────────┘      │
    │       │                    │                   │            │
    │       ▼                    ▼                   ▼            │
    │  ┌──────────┐        ┌───────────┐       ┌───────────┐      │
    │  │ OCRResult │        │ OCRResult │       │List[bytes]│      │
    │  │ (텍스트)  │        │ (OCR텍스트)│      │(PNG images)│     │
    │  └──────────┘        └───────────┘       └───────────┘      │
    └─────────────────────────────────────────────────────────────┘
"""

import time
from dataclasses import dataclass
from typing import Optional, List
import fitz  # PyMuPDF
from PIL import Image


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class OCRResult:
    """OCR 파싱 결과 데이터 클래스

    Attributes:
        success: 파싱 성공 여부
        content: 추출된 전체 텍스트
        pages: 페이지별 텍스트 리스트
        tables: 추출된 표 리스트
        elapsed_time: 처리 소요 시간 (초)
        page_count: 총 페이지 수
        has_text: 디지털 텍스트 존재 여부 (50자 이상)
        error: 에러 발생 시 메시지
    """
    success: bool
    content: str
    pages: List[str]
    tables: List[str]
    elapsed_time: float
    page_count: int
    has_text: bool
    error: Optional[str] = None


# ==============================================================================
# Main Class 1: OCRParser (디지털 PDF용)
# ==============================================================================

class OCRParser:
    """PyMuPDF 기반 문서 파서

    디지털 PDF에서 텍스트와 표를 추출합니다.
    스캔된 PDF (이미지 기반)는 텍스트 추출이 불가능합니다.

    Example:
        >>> parser = OCRParser()
        >>> result = parser.parse_pdf(pdf_bytes)
        >>> print(result.content)
    """

    # ==========================================================================
    # Class Constants
    # ==========================================================================

    MIN_TEXT_LENGTH = 50  # 디지털 PDF 판별 기준 (문자 수)
    SAMPLE_PAGES = 3      # PDF 타입 감지 시 샘플링할 페이지 수

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def parse_pdf(self, pdf_bytes: bytes) -> OCRResult:
        """PDF에서 텍스트 추출 (Entry Point)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            OCRResult: 파싱 결과

        Flow:
            1. PyMuPDF로 PDF 열기
            2. 각 페이지에서 텍스트/표 추출
            3. 콘텐츠 결합
            4. OCRResult 반환
        """
        start_time = time.time()
        pages_text = []
        tables_text = []

        try:
            # PyMuPDF로 PDF 열기
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)

            # Step 1-2: 페이지 순회하며 텍스트/표 추출
            for page in doc:
                # 텍스트 추출 (레이아웃 보존)
                text = page.get_text("text")
                pages_text.append(text)

                # 표 추출 (PyMuPDF의 find_tables 사용)
                tables = page.find_tables()
                for table in tables:
                    if table:
                        table_str = self._table_to_text(table.extract())
                        tables_text.append(table_str)

            doc.close()

            # Step 3: 콘텐츠 결합
            content = self._combine_content(pages_text, tables_text)

            # Step 4: 텍스트 존재 여부 판별
            total_text = "".join(pages_text)
            has_text = len(total_text.strip()) > self.MIN_TEXT_LENGTH

            return OCRResult(
                success=True,
                content=content,
                pages=pages_text,
                tables=tables_text,
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
                error=f"PDF 파싱 실패: {str(e)}"
            )

    def detect_pdf_type(self, pdf_bytes: bytes) -> str:
        """PDF 타입 감지 (Entry Point)

        처음 N페이지를 샘플링하여 디지털/스캔 PDF 판별.

        Args:
            pdf_bytes: PDF 파일 바이트

        Returns:
            "digital": 텍스트 추출 가능
            "scanned": 이미지 기반 (VLM 필요)
            "unknown": 판별 불가
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            sample_count = min(self.SAMPLE_PAGES, total_pages)
            text_pages = 0

            for i in range(sample_count):
                page = doc[i]
                text = page.get_text("text")
                if len(text.strip()) > self.MIN_TEXT_LENGTH:
                    text_pages += 1

            doc.close()
            return "digital" if text_pages > 0 else "scanned"

        except Exception:
            return "unknown"

    # ==========================================================================
    # Private Methods (Internal Helpers)
    # ==========================================================================

    def _table_to_text(self, table: List[List]) -> str:
        """2D 배열을 파이프 구분 텍스트로 변환

        Args:
            table: PyMuPDF가 추출한 표 (2D 리스트)

        Returns:
            파이프(|)로 구분된 텍스트
        """
        if not table:
            return ""

        rows = []
        for row in table:
            cells = [str(cell) if cell else "" for cell in row]
            rows.append(" | ".join(cells))

        return "\n".join(rows)

    def _combine_content(
        self,
        pages_text: List[str],
        tables_text: List[str]
    ) -> str:
        """페이지 텍스트와 표를 하나의 콘텐츠로 결합

        Args:
            pages_text: 페이지별 텍스트 리스트
            tables_text: 추출된 표 리스트

        Returns:
            결합된 전체 콘텐츠
        """
        content = "\n\n---\n\n".join(
            f"[Page {i+1}]\n{text}"
            for i, text in enumerate(pages_text)
            if text.strip()
        )

        if tables_text:
            content += "\n\n---\n\n[Tables]\n" + "\n\n".join(tables_text)

        return content


# ==============================================================================
# Main Class 2: ImageOCRParser (스캔 PDF → 이미지 변환용)
# ==============================================================================

class ImageOCRParser:
    """PDF를 이미지로 변환하는 유틸리티 클래스

    스캔된 PDF를 VLM이 처리할 수 있도록 PNG 이미지로 변환합니다.
    실제 OCR은 VLM이 수행하며, 이 클래스는 전처리만 담당합니다.

    Example:
        >>> parser = ImageOCRParser()
        >>> images = parser.pdf_to_images(pdf_bytes)
        >>> for img in images:
        ...     vlm_result = vlm_parser.parse(img)
    """

    # ==========================================================================
    # Class Constants
    # ==========================================================================

    DEFAULT_DPI = 150  # 기본 이미지 해상도

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def pdf_to_images(self, pdf_bytes: bytes, dpi: int = None) -> List[bytes]:
        """PDF를 이미지로 변환 (Entry Point)

        PyMuPDF를 사용하여 PDF 페이지를 PNG 이미지로 변환합니다.

        Args:
            pdf_bytes: PDF 파일 바이트
            dpi: 이미지 해상도 (기본값: 150)

        Returns:
            각 페이지의 PNG 이미지 바이트 리스트
        """
        if dpi is None:
            dpi = self.DEFAULT_DPI

        images = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # DPI를 zoom factor로 변환 (72 DPI 기준)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page in doc:
                # 페이지를 이미지로 렌더링
                pix = page.get_pixmap(matrix=matrix)
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)

            doc.close()

        except Exception:
            pass  # 변환 실패 시 빈 리스트 반환

        return images


# ==============================================================================
# Main Class 3: RapidOCRParser (스캔 PDF OCR - Image-Baseline)
# ==============================================================================

# RapidOCR 선택적 임포트 및 싱글톤 인스턴스
import logging
import os

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
