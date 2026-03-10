"""
Text Structurer - LLM 기반 텍스트 구조화

OCR로 추출된 텍스트를 LLM을 사용하여 구조화된 마크다운으로 변환합니다.
현재는 Qwen3-VL API를 사용하지만, 추후 텍스트 전용 LLM으로 교체 가능하도록 설계되었습니다.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  raw_text (OCR 추출 텍스트)                                   │
    │       │                                                     │
    │       ▼                                                     │
    │  TextStructurer.structure()                                 │
    │       │                                                     │
    │       ▼                                                     │
    │  ┌───────────────────┐                                      │
    │  │  STRUCTURING_PROMPT │                                    │
    │  │  + raw_text         │                                    │
    │  └───────────────────┘                                      │
    │       │                                                     │
    │       ▼                                                     │
    │  ┌───────────┐                                              │
    │  │  LLM API  │  (현재: Qwen3-VL, 추후: Qwen3-1.7B 등)         │
    │  │  Request  │                                              │
    │  └───────────┘                                              │
    │       │                                                     │
    │       ▼                                                     │
    │  TextStructurerResult                                       │
    │  (구조화된 마크다운)                                          │
    └─────────────────────────────────────────────────────────────┘

Usage:
    >>> structurer = TextStructurer()
    >>> result = structurer.structure("OCR로 추출된 텍스트...")
    >>> print(result.content)  # 구조화된 마크다운
"""

import time
import httpx
from dataclasses import dataclass
from typing import Optional


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class TextStructurerResult:
    """텍스트 구조화 결과 데이터 클래스

    Attributes:
        success: 구조화 성공 여부
        content: 구조화된 마크다운 텍스트
        original_text: 원본 텍스트 (구조화 전)
        elapsed_time: 처리 소요 시간 (초)
        model: 사용된 LLM 모델 ID
        error: 에러 발생 시 메시지
    """
    success: bool
    content: str
    original_text: str
    elapsed_time: float
    model: str
    error: Optional[str] = None


# ==============================================================================
# Main Class
# ==============================================================================

class TextStructurer:
    """LLM 기반 텍스트 구조화 클래스

    OCR로 추출된 비정형 텍스트를 구조화된 마크다운으로 변환합니다.
    현재는 Qwen3-VL API를 사용하지만, 추후 텍스트 전용 LLM으로 교체 가능합니다.

    Example:
        >>> structurer = TextStructurer()
        >>> result = structurer.structure("1. 제목\n내용...")
        >>> print(result.content)
    """

    # ==========================================================================
    # Class Constants (Prompt for Text Structuring)
    # ==========================================================================

    SYSTEM_PROMPT = """You are a Markdown formatting expert. Your task is to convert plain text into well-structured Markdown format.

CRITICAL RULES - You MUST follow these:
1. ALWAYS use # symbols for headings. This is mandatory.
2. Document title → # Title
3. Section numbers like "1 Introduction" or "1. Introduction" → ## 1. Introduction
4. Subsections like "3.1 Method" → ### 3.1 Method
5. Sub-subsections like "3.1.1 Details" → #### 3.1.1 Details
6. Tables with aligned columns → Markdown table with | separators
7. Bullet points → - item
8. Numbered lists → 1. item

NEVER output plain text headings without # symbols."""

    STRUCTURING_PROMPT = """Format this text as Markdown. Add # symbols to section headings.

HEADING LEVELS (based on numbering depth):
- Paper/document title (no number) → # Actual Title
- Top sections (1, 2, 3...) → ## 1. Section Name
- Subsections (2.1, 3.2...) → ### 2.1 Subsection
- Sub-subsections (3.1.1, 4.2.1...) → #### 3.1.1 Name

OTHER RULES:
- Tables: use | col1 | col2 | with |---|---| separator
- Lists: use - or 1.
- PRESERVE all original text content exactly
- Output Markdown only

TEXT:
{text}

OUTPUT:"""

    # ==========================================================================
    # Constructor
    # ==========================================================================

    def __init__(
        self,
        api_url: str = "http://localhost:8005/v1/chat/completions",
        model: str = "qwen3-vl-2b-instruct",
        timeout: float = 120.0,
        max_chunk_chars: int = 8000,  # 긴 문서 청킹용
        chunk_overlap: int = 200
    ):
        """TextStructurer 초기화

        Args:
            api_url: LLM API 엔드포인트 URL
            model: 사용할 모델 ID
            timeout: API 요청 타임아웃 (초)
            max_chunk_chars: 한 번에 처리할 최대 문자 수 (긴 문서 청킹용)
            chunk_overlap: 청크 간 오버랩 (문맥 유지)
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def structure(
        self,
        text: str,
        custom_prompt: Optional[str] = None
    ) -> TextStructurerResult:
        """텍스트 구조화 (Entry Point)

        Args:
            text: OCR로 추출된 원본 텍스트
            custom_prompt: 커스텀 프롬프트 (None이면 기본 프롬프트 사용)

        Returns:
            TextStructurerResult: 구조화 결과

        Flow:
            1. 프롬프트 구성 (텍스트 삽입)
            2. API 페이로드 구성
            3. LLM API 호출
            4. 응답 파싱 및 후처리
            5. TextStructurerResult 반환
        """
        start_time = time.time()

        # 입력 검증
        if not text or not text.strip():
            return TextStructurerResult(
                success=False,
                content="",
                original_text=text,
                elapsed_time=time.time() - start_time,
                model=self.model,
                error="입력 텍스트가 비어있습니다."
            )

        # Step 1: 프롬프트 구성
        if custom_prompt:
            prompt = custom_prompt.format(text=text)
        else:
            prompt = self.STRUCTURING_PROMPT.format(text=text)

        # Step 2: API 페이로드 구성
        payload = self._build_payload(prompt)

        # Step 3-4: API 호출 및 결과 처리
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()

                # 응답에서 텍스트 추출
                raw_content = result["choices"][0]["message"]["content"]

                # 후처리: Thinking 태그 제거
                content = self._extract_content(raw_content)

                return TextStructurerResult(
                    success=True,
                    content=content,
                    original_text=text,
                    elapsed_time=time.time() - start_time,
                    model=self.model
                )

        except httpx.TimeoutException:
            return TextStructurerResult(
                success=False,
                content="",
                original_text=text,
                elapsed_time=time.time() - start_time,
                model=self.model,
                error="API 요청 타임아웃"
            )
        except httpx.HTTPStatusError as e:
            return TextStructurerResult(
                success=False,
                content="",
                original_text=text,
                elapsed_time=time.time() - start_time,
                model=self.model,
                error=f"API 오류: {e.response.status_code}"
            )
        except Exception as e:
            return TextStructurerResult(
                success=False,
                content="",
                original_text=text,
                elapsed_time=time.time() - start_time,
                model=self.model,
                error=f"구조화 실패: {str(e)}"
            )

    # ==========================================================================
    # Private Methods (Internal Helpers)
    # ==========================================================================

    def _build_payload(self, prompt: str) -> dict:
        """LLM API 요청 페이로드 구성

        Args:
            prompt: 구조화 프롬프트 (텍스트 포함)

        Returns:
            API 요청 페이로드 딕셔너리
        """
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 8192,  # 긴 문서 지원
            "temperature": 0.1   # 일관된 출력
        }

    def _extract_content(self, raw_content: str) -> str:
        """응답에서 실제 콘텐츠 추출

        Thinking 모델의 경우 </think> 태그 이후 내용만 추출합니다.

        Args:
            raw_content: LLM API 원본 응답

        Returns:
            정리된 콘텐츠
        """
        # Thinking 태그 처리
        if "</think>" in raw_content:
            parts = raw_content.split("</think>", 1)
            content = parts[1].strip() if len(parts) > 1 else ""
        else:
            content = raw_content

        # 마크다운 코드 펜스 제거 (만약 있다면)
        content = content.strip()
        if content.startswith("```markdown"):
            content = content[len("```markdown"):].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        return content
