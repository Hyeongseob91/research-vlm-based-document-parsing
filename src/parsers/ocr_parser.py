"""
OCR Parser - pdfplumber 기반 문서 파싱

전통적인 텍스트 추출 방식으로 디지털 PDF에서 텍스트를 추출합니다.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  pdf_bytes                                                  │
    │       │                                                     │
    │       ├─────────────────────┬───────────────────┐           │
    │       ▼                     ▼                   ▼           │
    │  ┌──────────┐        ┌───────────┐       ┌───────────┐      │
    │  │ OCRParser │        │ detect_   │       │ ImageOCR  │      │
    │  │ parse_pdf │        │ pdf_type  │       │ Parser    │      │
    │  └──────────┘        └───────────┘       └───────────┘      │
    │       │                    │                   │            │
    │       ▼                    ▼                   ▼            │
    │  ┌──────────┐        ┌───────────┐       ┌───────────┐      │
    │  │ OCRResult │        │ "digital" │       │List[bytes]│      │
    │  │ (텍스트)  │        │ "scanned" │       │(PNG images)│     │
    │  └──────────┘        └───────────┘       └───────────┘      │
    │                                                │            │
    │                                                ▼            │
    │                                          VLM Parser로       │
    │                                          전달 (스캔 문서)    │
    └─────────────────────────────────────────────────────────────┘
"""

import time
from dataclasses import dataclass
from typing import Optional, List
from io import BytesIO

import pdfplumber
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
    """pdfplumber 기반 문서 파서

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
            1. pdfplumber로 PDF 열기
            2. 각 페이지에서 텍스트/표 추출
            3. 콘텐츠 결합
            4. OCRResult 반환
        """
        start_time = time.time()
        pages_text = []
        tables_text = []

        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)

                # Step 1-2: 페이지 순회하며 텍스트/표 추출
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)

                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_str = self._table_to_text(table)
                            tables_text.append(table_str)

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
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                sample_count = min(self.SAMPLE_PAGES, total_pages)
                text_pages = 0

                for page in pdf.pages[:sample_count]:
                    text = page.extract_text() or ""
                    if len(text.strip()) > self.MIN_TEXT_LENGTH:
                        text_pages += 1

                return "digital" if text_pages > 0 else "scanned"

        except Exception:
            return "unknown"

    # ==========================================================================
    # Private Methods (Internal Helpers)
    # ==========================================================================

    def _table_to_text(self, table: List[List]) -> str:
        """2D 배열을 파이프 구분 텍스트로 변환

        Args:
            table: pdfplumber가 추출한 표 (2D 리스트)

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

        Args:
            pdf_bytes: PDF 파일 바이트
            dpi: 이미지 해상도 (기본값: 150)

        Returns:
            각 페이지의 PNG 이미지 바이트 리스트

        Strategy:
            1. pdf2image 시도 (poppler 기반, 고품질)
            2. 실패 시 pdfplumber fallback (순수 Python)
        """
        if dpi is None:
            dpi = self.DEFAULT_DPI

        try:
            return self._convert_with_pdf2image(pdf_bytes, dpi)
        except ImportError:
            return self._convert_with_pdfplumber(pdf_bytes, dpi)

    # ==========================================================================
    # Private Methods (Internal Helpers)
    # ==========================================================================

    def _convert_with_pdf2image(
        self,
        pdf_bytes: bytes,
        dpi: int
    ) -> List[bytes]:
        """pdf2image를 사용한 변환 (poppler 필요)

        Args:
            pdf_bytes: PDF 바이트
            dpi: 해상도

        Returns:
            PNG 이미지 바이트 리스트

        Raises:
            ImportError: pdf2image 미설치 시
        """
        import pdf2image
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi)

        result = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            result.append(buffer.getvalue())

        return result

    def _convert_with_pdfplumber(
        self,
        pdf_bytes: bytes,
        dpi: int
    ) -> List[bytes]:
        """pdfplumber를 사용한 변환 (Fallback)

        pdf2image가 없을 때 사용하는 대체 방법.

        Args:
            pdf_bytes: PDF 바이트
            dpi: 해상도 (pdfplumber는 resolution으로 매핑)

        Returns:
            PNG 이미지 바이트 리스트
        """
        images = []

        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    img = page.to_image(resolution=dpi)
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    images.append(buffer.getvalue())

        except Exception:
            pass  # 변환 실패 시 빈 리스트 반환

        return images
