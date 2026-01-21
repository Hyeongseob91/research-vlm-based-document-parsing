"""
Docling Parser - RapidOCR 기반 문서 파싱

Docling 라이브러리를 사용하여 Image PDF에서 텍스트를 추출합니다.
내부적으로 RapidOCR을 호출하여 OCR을 수행합니다.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  pdf_bytes / pdf_path                                       │
    │       │                                                     │
    │       ▼                                                     │
    │  ┌──────────────┐                                           │
    │  │   Docling    │                                           │
    │  │   Converter  │                                           │
    │  └──────────────┘                                           │
    │       │                                                     │
    │       ├─── PDF Pipeline ───┐                                │
    │       │                    │                                │
    │       ▼                    ▼                                │
    │  ┌──────────┐       ┌───────────┐                           │
    │  │ RapidOCR │       │  Layout   │                           │
    │  │ (텍스트) │       │  Analysis │                           │
    │  └──────────┘       └───────────┘                           │
    │       │                    │                                │
    │       └────────┬───────────┘                                │
    │                ▼                                            │
    │         DoclingResult                                       │
    │         (텍스트 + 구조)                                      │
    └─────────────────────────────────────────────────────────────┘
"""

import time
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from io import BytesIO
import tempfile
import os


@dataclass
class DoclingResult:
    """Docling 파싱 결과 데이터 클래스

    Attributes:
        success: 파싱 성공 여부
        content: 추출된 전체 텍스트
        markdown: 마크다운 형식 출력 (구조화)
        elapsed_time: 처리 소요 시간 (초)
        page_count: 총 페이지 수
        error: 에러 발생 시 메시지
    """
    success: bool
    content: str
    markdown: str
    elapsed_time: float
    page_count: int
    error: Optional[str] = None


class DoclingParser:
    """Docling 기반 문서 파서

    Docling을 사용하여 Image PDF에서 텍스트를 추출합니다.
    내부적으로 RapidOCR을 호출하여 OCR을 수행합니다.

    Example:
        >>> parser = DoclingParser()
        >>> result = parser.parse_pdf(pdf_bytes)
        >>> print(result.content)
    """

    def __init__(self, ocr_enabled: bool = True):
        """DoclingParser 초기화

        Args:
            ocr_enabled: OCR 활성화 여부 (기본값: True)
        """
        self.ocr_enabled = ocr_enabled
        self._converter = None

    def _get_converter(self):
        """Docling Converter 인스턴스 반환 (lazy initialization)"""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat
                from docling.document_converter import PdfFormatOption

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = self.ocr_enabled
                pipeline_options.do_table_structure = True

                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
            except ImportError as e:
                raise ImportError(
                    f"Docling 라이브러리가 설치되지 않았습니다. "
                    f"pip install docling 명령으로 설치하세요. Error: {e}"
                )

        return self._converter

    def parse_pdf(self, pdf_bytes: bytes) -> DoclingResult:
        """PDF에서 텍스트 추출 (Entry Point - bytes)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            DoclingResult: 파싱 결과
        """
        start_time = time.time()

        try:
            # 임시 파일로 저장 (Docling은 파일 경로 필요)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                result = self.parse_pdf_path(tmp_path)
                result.elapsed_time = time.time() - start_time
                return result
            finally:
                # 임시 파일 삭제
                os.unlink(tmp_path)

        except Exception as e:
            return DoclingResult(
                success=False,
                content="",
                markdown="",
                elapsed_time=time.time() - start_time,
                page_count=0,
                error=f"PDF 파싱 실패: {str(e)}"
            )

    def parse_pdf_path(self, pdf_path: str) -> DoclingResult:
        """PDF에서 텍스트 추출 (Entry Point - path)

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            DoclingResult: 파싱 결과
        """
        start_time = time.time()

        try:
            converter = self._get_converter()

            # Docling 변환 실행
            result = converter.convert(pdf_path)

            # 텍스트 추출
            content = result.document.export_to_text()

            # 마크다운 추출
            markdown = result.document.export_to_markdown()

            # 페이지 수 (Docling은 직접 제공하지 않으므로 추정)
            page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 1

            return DoclingResult(
                success=True,
                content=content,
                markdown=markdown,
                elapsed_time=time.time() - start_time,
                page_count=page_count
            )

        except ImportError as e:
            return DoclingResult(
                success=False,
                content="",
                markdown="",
                elapsed_time=time.time() - start_time,
                page_count=0,
                error=f"Docling 라이브러리 오류: {str(e)}"
            )
        except Exception as e:
            return DoclingResult(
                success=False,
                content="",
                markdown="",
                elapsed_time=time.time() - start_time,
                page_count=0,
                error=f"PDF 파싱 실패: {str(e)}"
            )


def check_docling_available() -> bool:
    """Docling 라이브러리 사용 가능 여부 확인"""
    try:
        from docling.document_converter import DocumentConverter
        return True
    except ImportError:
        return False
