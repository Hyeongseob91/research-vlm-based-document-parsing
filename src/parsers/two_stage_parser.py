"""
Two-Stage Parser - 2단계 파싱 파이프라인

Stage 1: OCR 텍스트 추출 (pdfplumber 또는 Docling/RapidOCR)
Stage 2: LLM 텍스트 구조화 (TextStructurer)

연구 가설:
    "VLM을 통한 텍스트 구조화(2-Stage)가 단순 추출(1-Stage) 대비
    Semantic Chunking 품질(BC↑, CS↓)을 개선한다."

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  PDF Input                                                  │
    │       │                                                     │
    │       ├─────────────────┬───────────────────┐               │
    │       ▼                 ▼                   ▼               │
    │  detect_pdf_type()                                          │
    │       │                                                     │
    │       ├─── "digital" ──►  parse_text_pdf()                  │
    │       │                      │                              │
    │       │                      ▼                              │
    │       │                   pdfplumber (Stage 1)              │
    │       │                      │                              │
    │       │                      ▼                              │
    │       │                   TextStructurer (Stage 2)          │
    │       │                      │                              │
    │       │                      ▼                              │
    │       │                   TwoStageResult                    │
    │       │                                                     │
    │       └─── "scanned" ──►  parse_scanned_pdf()               │
    │                              │                              │
    │                              ▼                              │
    │                           Docling/RapidOCR (Stage 1)        │
    │                              │                              │
    │                              ▼                              │
    │                           TextStructurer (Stage 2)          │
    │                              │                              │
    │                              ▼                              │
    │                           TwoStageResult                    │
    └─────────────────────────────────────────────────────────────┘

Usage:
    >>> parser = TwoStageParser()
    >>> result = parser.parse_auto(pdf_bytes)
    >>> print(result.content)  # 구조화된 마크다운
    >>> print(f"Stage 1: {result.stage1_time:.2f}s")
    >>> print(f"Stage 2: {result.stage2_time:.2f}s")
"""

import time
from dataclasses import dataclass
from typing import Optional

from .ocr_parser import OCRParser
from .text_structurer import TextStructurer, TextStructurerResult

# Docling은 선택적 의존성
try:
    from .docling_parser import DoclingParser, check_docling_available
    DOCLING_AVAILABLE = check_docling_available()
except ImportError:
    DOCLING_AVAILABLE = False
    DoclingParser = None
    check_docling_available = lambda: False


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class TwoStageResult:
    """2-Stage 파싱 결과 데이터 클래스

    Attributes:
        success: 전체 파이프라인 성공 여부
        content: 최종 구조화된 텍스트 (Stage 2 출력)
        stage1_content: Stage 1 (OCR) 원본 텍스트
        stage1_parser: Stage 1 파서 이름 ("pdfplumber" | "docling")
        stage2_applied: VLM 구조화 적용 여부
        elapsed_time: 전체 처리 시간 (초)
        stage1_time: Stage 1 처리 시간 (초) - Tech Report용
        stage2_time: Stage 2 처리 시간 (초) - Tech Report용
        page_count: 총 페이지 수
        pdf_type: PDF 타입 ("digital" | "scanned" | "unknown")
        error: 에러 발생 시 메시지
    """
    success: bool
    content: str
    stage1_content: str
    stage1_parser: str
    stage2_applied: bool
    elapsed_time: float
    stage1_time: float
    stage2_time: float
    page_count: int
    pdf_type: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """결과를 딕셔너리로 변환 (JSON 저장용)"""
        return {
            "success": self.success,
            "content_length": len(self.content),
            "stage1_content_length": len(self.stage1_content),
            "stage1_parser": self.stage1_parser,
            "stage2_applied": self.stage2_applied,
            "elapsed_time": self.elapsed_time,
            "stage1_time": self.stage1_time,
            "stage2_time": self.stage2_time,
            "page_count": self.page_count,
            "pdf_type": self.pdf_type,
            "error": self.error,
        }


# ==============================================================================
# Main Class
# ==============================================================================

class TwoStageParser:
    """2-Stage 문서 파서

    Stage 1: OCR 텍스트 추출
        - Text PDF: pdfplumber
        - Scanned PDF: Docling + RapidOCR

    Stage 2: LLM 텍스트 구조화
        - TextStructurer (현재: Qwen3-VL API)

    Example:
        >>> parser = TwoStageParser()
        >>> result = parser.parse_auto(pdf_bytes)
        >>> print(result.content)
        >>> print(f"Stage 1: {result.stage1_time:.2f}s, Stage 2: {result.stage2_time:.2f}s")
    """

    # ==========================================================================
    # Constructor
    # ==========================================================================

    def __init__(
        self,
        structurer_api_url: str = "http://localhost:8005/v1/chat/completions",
        structurer_model: str = "qwen3-vl-2b-instruct",
        structurer_timeout: float = 120.0,
        skip_stage2_on_failure: bool = True
    ):
        """TwoStageParser 초기화

        Args:
            structurer_api_url: TextStructurer API 엔드포인트 URL
            structurer_model: TextStructurer 모델 ID
            structurer_timeout: API 요청 타임아웃 (초)
            skip_stage2_on_failure: Stage 1 실패 시 Stage 2 스킵 여부
        """
        self.ocr_parser = OCRParser()
        self.structurer = TextStructurer(
            api_url=structurer_api_url,
            model=structurer_model,
            timeout=structurer_timeout
        )
        self.skip_stage2_on_failure = skip_stage2_on_failure

        # Docling 파서 (설치된 경우에만)
        self.docling_parser = DoclingParser(ocr_enabled=True) if DOCLING_AVAILABLE else None

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def parse_auto(self, pdf_bytes: bytes) -> TwoStageResult:
        """PDF 타입 자동 감지 후 파싱 (Entry Point)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            TwoStageResult: 2-Stage 파싱 결과
        """
        # PDF 타입 감지
        pdf_type = self.ocr_parser.detect_pdf_type(pdf_bytes)

        if pdf_type == "digital":
            result = self.parse_text_pdf(pdf_bytes)
            result.pdf_type = pdf_type
            return result
        elif pdf_type == "scanned":
            result = self.parse_scanned_pdf(pdf_bytes)
            result.pdf_type = pdf_type
            return result
        else:
            # unknown: pdfplumber로 시도
            result = self.parse_text_pdf(pdf_bytes)
            result.pdf_type = "unknown"
            return result

    def parse_text_pdf(self, pdf_bytes: bytes) -> TwoStageResult:
        """Text PDF 파싱 (pdfplumber → VLM 구조화)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            TwoStageResult: 2-Stage 파싱 결과
        """
        total_start = time.time()

        # =====================================================================
        # Stage 1: pdfplumber로 텍스트 추출
        # =====================================================================
        stage1_start = time.time()
        ocr_result = self.ocr_parser.parse_pdf(pdf_bytes)
        stage1_time = time.time() - stage1_start

        if not ocr_result.success:
            return TwoStageResult(
                success=False,
                content="",
                stage1_content="",
                stage1_parser="pdfplumber",
                stage2_applied=False,
                elapsed_time=time.time() - total_start,
                stage1_time=stage1_time,
                stage2_time=0.0,
                page_count=0,
                error=f"Stage 1 실패: {ocr_result.error}"
            )

        stage1_content = ocr_result.content

        # Stage 1 결과가 비어있으면 Stage 2 스킵
        if not stage1_content.strip():
            return TwoStageResult(
                success=False,
                content="",
                stage1_content=stage1_content,
                stage1_parser="pdfplumber",
                stage2_applied=False,
                elapsed_time=time.time() - total_start,
                stage1_time=stage1_time,
                stage2_time=0.0,
                page_count=ocr_result.page_count,
                error="Stage 1 결과가 비어있습니다."
            )

        # =====================================================================
        # Stage 2: VLM 텍스트 구조화
        # =====================================================================
        stage2_start = time.time()
        structurer_result = self.structurer.structure(stage1_content)
        stage2_time = time.time() - stage2_start

        if not structurer_result.success:
            # Stage 2 실패 시 Stage 1 결과 반환 (skip_stage2_on_failure에 따라)
            if self.skip_stage2_on_failure:
                return TwoStageResult(
                    success=True,  # Stage 1은 성공
                    content=stage1_content,  # Stage 1 결과 사용
                    stage1_content=stage1_content,
                    stage1_parser="pdfplumber",
                    stage2_applied=False,
                    elapsed_time=time.time() - total_start,
                    stage1_time=stage1_time,
                    stage2_time=stage2_time,
                    page_count=ocr_result.page_count,
                    error=f"Stage 2 실패 (Stage 1 결과 반환): {structurer_result.error}"
                )
            else:
                return TwoStageResult(
                    success=False,
                    content="",
                    stage1_content=stage1_content,
                    stage1_parser="pdfplumber",
                    stage2_applied=False,
                    elapsed_time=time.time() - total_start,
                    stage1_time=stage1_time,
                    stage2_time=stage2_time,
                    page_count=ocr_result.page_count,
                    error=f"Stage 2 실패: {structurer_result.error}"
                )

        # 성공
        return TwoStageResult(
            success=True,
            content=structurer_result.content,
            stage1_content=stage1_content,
            stage1_parser="pdfplumber",
            stage2_applied=True,
            elapsed_time=time.time() - total_start,
            stage1_time=stage1_time,
            stage2_time=stage2_time,
            page_count=ocr_result.page_count
        )

    def parse_scanned_pdf(self, pdf_bytes: bytes) -> TwoStageResult:
        """Scanned PDF 파싱 (Docling/RapidOCR → VLM 구조화)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            TwoStageResult: 2-Stage 파싱 결과
        """
        total_start = time.time()

        # =====================================================================
        # Stage 1: Docling + RapidOCR로 텍스트 추출
        # =====================================================================
        if not DOCLING_AVAILABLE or self.docling_parser is None:
            return TwoStageResult(
                success=False,
                content="",
                stage1_content="",
                stage1_parser="docling",
                stage2_applied=False,
                elapsed_time=time.time() - total_start,
                stage1_time=0.0,
                stage2_time=0.0,
                page_count=0,
                error="Docling이 설치되지 않았습니다. pip install docling"
            )

        stage1_start = time.time()
        docling_result = self.docling_parser.parse_pdf(pdf_bytes)
        stage1_time = time.time() - stage1_start

        if not docling_result.success:
            return TwoStageResult(
                success=False,
                content="",
                stage1_content="",
                stage1_parser="docling",
                stage2_applied=False,
                elapsed_time=time.time() - total_start,
                stage1_time=stage1_time,
                stage2_time=0.0,
                page_count=0,
                error=f"Stage 1 실패: {docling_result.error}"
            )

        # Docling은 markdown 출력을 우선 사용
        stage1_content = docling_result.markdown or docling_result.content

        # Stage 1 결과가 비어있으면 Stage 2 스킵
        if not stage1_content.strip():
            return TwoStageResult(
                success=False,
                content="",
                stage1_content=stage1_content,
                stage1_parser="docling",
                stage2_applied=False,
                elapsed_time=time.time() - total_start,
                stage1_time=stage1_time,
                stage2_time=0.0,
                page_count=docling_result.page_count,
                error="Stage 1 결과가 비어있습니다."
            )

        # =====================================================================
        # Stage 2: VLM 텍스트 구조화
        # =====================================================================
        stage2_start = time.time()
        structurer_result = self.structurer.structure(stage1_content)
        stage2_time = time.time() - stage2_start

        if not structurer_result.success:
            if self.skip_stage2_on_failure:
                return TwoStageResult(
                    success=True,
                    content=stage1_content,
                    stage1_content=stage1_content,
                    stage1_parser="docling",
                    stage2_applied=False,
                    elapsed_time=time.time() - total_start,
                    stage1_time=stage1_time,
                    stage2_time=stage2_time,
                    page_count=docling_result.page_count,
                    error=f"Stage 2 실패 (Stage 1 결과 반환): {structurer_result.error}"
                )
            else:
                return TwoStageResult(
                    success=False,
                    content="",
                    stage1_content=stage1_content,
                    stage1_parser="docling",
                    stage2_applied=False,
                    elapsed_time=time.time() - total_start,
                    stage1_time=stage1_time,
                    stage2_time=stage2_time,
                    page_count=docling_result.page_count,
                    error=f"Stage 2 실패: {structurer_result.error}"
                )

        # 성공
        return TwoStageResult(
            success=True,
            content=structurer_result.content,
            stage1_content=stage1_content,
            stage1_parser="docling",
            stage2_applied=True,
            elapsed_time=time.time() - total_start,
            stage1_time=stage1_time,
            stage2_time=stage2_time,
            page_count=docling_result.page_count
        )

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def detect_pdf_type(self, pdf_bytes: bytes) -> str:
        """PDF 타입 감지 (OCRParser 위임)

        Args:
            pdf_bytes: PDF 파일 바이트

        Returns:
            "digital" | "scanned" | "unknown"
        """
        return self.ocr_parser.detect_pdf_type(pdf_bytes)

    @property
    def docling_available(self) -> bool:
        """Docling 사용 가능 여부"""
        return DOCLING_AVAILABLE
