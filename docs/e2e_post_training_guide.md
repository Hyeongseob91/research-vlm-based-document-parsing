# Qwen3-VL-2B End-to-End Post-Training 가이드

> **문서 버전**: v1.0
> **작성일**: 2026-02-26
> **대상 모델**: Qwen3-VL-2B-Instruct
> **전제 조건**: 기존 평가 프레임워크 (`src/eval_parsers.py`) 실행 가능한 환경

---

## 1. 개요

### 1.1 문서 목적

이 문서는 기존 Two-Stage 파이프라인(OCR → VLM 구조화)을 **End-to-End VLM 파이프라인**(이미지 → VLM 직접 파싱)으로 확장하기 위한 Post-Training 세팅 가이드이다.

기존 Tech Report(`docs/tech_report/`)에서 4파서 비교 실험(RQ1-RQ3)을 수행한 결과를 기반으로, **RQ4: End-to-End VLM이 Two-Stage 파이프라인을 대체할 수 있는가?** 를 검증하기 위한 실험 설계와 구현 방법을 다룬다.

### 1.2 기존 Tech Report와의 관계

```
Tech Report (기존)          이 가이드 (확장)
├── RQ1: 텍스트 추출 품질    ├── 기존 4파서 결과 기준선으로 활용
├── RQ2: 구조 보존 효과      ├── E2E 파서의 Structure F1 비교 대상
├── RQ3: 다운스트림 효과     ├── 청킹 품질 비교 확장
└── (없음)                   └── RQ4: E2E VLM vs Two-Stage 비교
```

### 1.3 원본 가이드 대비 주요 수정사항

이 가이드는 Claude 웹 기반으로 생성된 초기 버전을 기존 코드베이스와 통합하며 다음 사항을 수정했다.

**P0 (Critical — 실행 불가 방지)**:

| # | 항목 | 원본 가이드 | 수정 |
|---|------|------------|------|
| 1 | 모델 클래스 | Qwen3-VL 확인 없이 진행 | `AutoProcessor` 로드 시 `Qwen2.5VLForConditionalGeneration` vs `Qwen3VLForConditionalGeneration` 확인 절차 추가 |
| 2 | save_steps | `50` | `5` (총 ~15 스텝에서 체크포인트 0개 문제 해결) |
| 3 | GT 데이터 | 신규 GT 제작 전제 | 기존 `data/test_*/gt_data_*.md` 3개 활용 계획 명시 |
| 4 | 코드 통합 | 독립 스크립트 | 기존 `src/parsers/`, `src/eval_parsers.py` 패턴 준수 |

**P1 (Important — 성능/안정성)**:

| # | 항목 | 원본 가이드 | 수정 |
|---|------|------------|------|
| 1 | lora_rank | `64` | `16` (12개 샘플 기준 과적합 방지) |
| 2 | freeze_merger | `False` | `True` (소량 데이터 안정성) |
| 3 | TEDS 평가 | 미포함 | 기존 `compute_teds()` 활용 |
| 4 | 시간 추정 | 없음 | 13-17시간 현실적 추정 포함 |

---

## 2. 배경: 왜 End-to-End인가

### 2.1 현재 아키텍처의 한계

기존 Two-Stage 파이프라인은 Stage 1(OCR/텍스트 추출)에서 발생한 에러가 Stage 2(VLM 구조화)로 전파된다.

```
Two-Stage (현재):
PDF → [OCR/PyMuPDF] → 텍스트 → [VLM 구조화] → Markdown
         ↑                          ↑
    에러 발생 지점              에러 전파 + 증폭
```

**실측 근거** (`docs/tech_report/05_Results.md`):

| 문제 | 실측값 | 파일 위치 |
|------|--------|----------|
| OCR 에러 전파 | Image-Advanced CER 536.50% (test_1, 한국어 스캔) | `05_Results.md` §5.1.1 |
| VLM이 CER을 증가 | Text-Baseline 51.25% → Text-Advanced 64.11% (+13pp) | `05_Results.md` §5.1.1 |
| 구조 보존은 VLM 의존 | Text-Advanced Structure F1 79.25% (test_3) | `05_Results.md` §5.2.1 |

### 2.2 End-to-End 접근의 가설

End-to-End VLM은 OCR 단계를 제거하고 이미지에서 직접 마크다운을 생성한다:

```
End-to-End (제안):
PDF → [pdf_to_images()] → 이미지 → [VLM 직접 파싱] → Markdown
                                      ↑
                              Vision Encoder가 직접 인식
                              OCR 에러 전파 없음
```

**핵심 가설**: VLM의 Vision Encoder가 텍스트 인식과 구조 추출을 동시에 수행하므로, OCR 에러 전파 문제가 근본적으로 제거된다. 특히 한국어 스캔 문서(test_1)에서 CER 536%와 같은 극단적 에러가 개선될 수 있다.

### 2.3 VLM Vision 능력 미활용 문제

현재 Two-Stage에서 VLM은 **텍스트만** 입력받아 구조화한다 (`TextStructurer` — `src/parsers/text_structurer.py`). VLM의 Vision Encoder는 사실상 사용되지 않는다.

```python
# 현재: TextStructurer._build_payload() — 텍스트만 전달
{"role": "user", "content": prompt}  # 텍스트 전용

# 제안: E2EVLMParser — 이미지 직접 전달
{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    {"type": "text", "text": prompt}
]}
```

---

## 3. 아키텍처: 4파서 → 6파서 확장

### 3.1 파서 구성 테이블

| # | 파서 이름 | Stage 1 | Stage 2 | 신규 |
|---|----------|---------|---------|------|
| 1 | Text-Baseline | PyMuPDF | — | 기존 |
| 2 | Image-Baseline | RapidOCR | — | 기존 |
| 3 | Text-Advanced | PyMuPDF | VLM 구조화 (텍스트) | 기존 |
| 4 | Image-Advanced | RapidOCR | VLM 구조화 (텍스트) | 기존 |
| **5** | **E2E-VLM-Zero** | ImageOCRParser (이미지) | VLM 직접 파싱 | **신규** |
| **6** | **E2E-VLM-Finetuned** | ImageOCRParser (이미지) | LoRA 파인튜닝된 VLM | **신규** |

### 3.2 파이프라인 다이어그램

```
                         ┌───────────────────────────────────────────┐
                         │         기존 파서 (1-4)                    │
                         │                                           │
  PDF ──► OCRParser ─────┤  #1 Text-Baseline  : PyMuPDF 텍스트       │
      │   or RapidOCR    │  #2 Image-Baseline : RapidOCR 텍스트      │
      │                  │  #3 Text-Advanced  : PyMuPDF → VLM(텍스트) │
      │                  │  #4 Image-Advanced : RapidOCR → VLM(텍스트)│
      │                  └───────────────────────────────────────────┘
      │
      │                  ┌───────────────────────────────────────────┐
      │                  │         신규 파서 (5-6)                    │
      └──► ImageOCRParser│                                           │
           .pdf_to_images│  #5 E2E-VLM-Zero     : 이미지 → VLM API   │
           ()            │  #6 E2E-VLM-Finetuned : 이미지 → LoRA VLM │
                         └───────────────────────────────────────────┘
```

**핵심**: 신규 파서는 `ImageOCRParser.pdf_to_images()` (`src/parsers/ocr_parser.py:268`)를 재사용하여 PDF를 PNG 이미지로 변환한 후, VLM API에 이미지를 직접 전달한다.

---

## 4. 환경 설정

### 4.1 기존 환경 (평가 인프라)

기존 `pyproject.toml` 의존성만으로 평가 파이프라인은 추가 설치 없이 실행 가능하다:

```bash
# 기존 프로젝트 의존성 확인
pip install -e ".[all]"
```

주요 의존성: `httpx`, `PyMuPDF`, `jiwer`, `apted`, `mistletoe` — 모두 `pyproject.toml`에 포함.

### 4.2 파인튜닝 환경 (추가 설치)

파인튜닝은 별도 환경에서 수행한다. [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) 프레임워크를 사용한다.

```bash
# 파인튜닝 환경 (별도 디렉토리)
git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git
cd Qwen-VL-Series-Finetune
pip install -r requirements.txt
```

### 4.3 P0: 모델 클래스 확인 절차

Qwen3-VL과 Qwen2.5-VL은 모델 클래스가 다르므로 반드시 확인해야 한다:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "Qwen/Qwen3-VL-2B-Instruct"

# Step 1: config.json의 model_type 확인
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"model_type: {config.model_type}")
# 예상: "qwen3_vl" (Qwen3) 또는 "qwen2_5_vl" (Qwen2.5)

# Step 2: 올바른 클래스 사용
if config.model_type == "qwen3_vl":
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id)
elif config.model_type == "qwen2_5_vl":
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
else:
    # 안전한 fallback
    model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
```

> **주의**: `trust_remote_code=True` 없이 로드하면 custom 모델 클래스 인식에 실패할 수 있다. 파인튜닝 프레임워크(2U1)에서도 동일하게 model_type을 확인해야 한다.

### 4.4 vLLM 서빙 (추론용)

```bash
# Zero-Shot 모델
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --port 8005 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=15  # 최대 15페이지 이미지

# LoRA 파인튜닝 모델
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --port 8006 \
    --enable-lora \
    --lora-modules e2e-finetuned=./output/e2e_vlm_lora \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=15
```

> **트러블슈팅**: `--limit-mm-per-prompt image=N`을 지정하지 않으면 멀티이미지 입력이 기본값(1)으로 제한된다. test_3(Attention Is All You Need)은 15페이지이므로 최소 `image=15`가 필요하다.

---

## 5. 학습 데이터 준비

### 5.1 P0: 기존 GT 활용 계획

기존 3개의 Ground Truth를 활용한다. 신규 GT를 처음부터 만드는 것이 아니라, **기존 GT를 페이지 단위로 분할**하여 학습 데이터를 구성한다.

| GT 파일 | 문서 | 언어 | 페이지 수 | 유형 |
|---------|------|------|----------|------|
| `data/test_1/gt_data_1.md` | 공공AX 프로젝트 공모안내서 | 한국어 | 5 | 스캔 |
| `data/test_2/gt_data_2.md` | Chain-of-Thought Prompting | 영어 | 4 | 스캔 |
| `data/test_3/gt_data_3.md` | Attention Is All You Need | 영어 | 15 | 디지털 |

**총 학습 가능 페이지**: 5 + 4 + 15 = **24페이지** → 약 12페어 (이미지-GT 쌍, 일부 페이지 제외 가능)

### 5.2 페이지 이미지 생성

`ImageOCRParser.pdf_to_images()` (`src/parsers/ocr_parser.py:268`)를 재사용한다:

```python
from src.parsers.ocr_parser import ImageOCRParser
from pathlib import Path

parser = ImageOCRParser()

for test_dir in ["test_1", "test_2", "test_3"]:
    pdf_path = Path(f"data/{test_dir}")
    # PDF 파일 찾기
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        continue

    pdf_bytes = pdf_files[0].read_bytes()
    images = parser.pdf_to_images(pdf_bytes, dpi=150)  # DEFAULT_DPI=150

    # 페이지별 이미지 저장
    img_dir = pdf_path / "page_images"
    img_dir.mkdir(exist_ok=True)
    for i, img_bytes in enumerate(images):
        (img_dir / f"page_{i+1:03d}.png").write_bytes(img_bytes)
        print(f"  {test_dir}/page_{i+1:03d}.png ({len(img_bytes)/1024:.1f} KB)")
```

### 5.3 페이지 레벨 GT 분할 전략

기존 GT는 문서 전체에 대한 단일 마크다운이므로, 페이지 단위로 분할해야 한다.

**분할 방법**:
1. GT 마크다운을 페이지 구분자(`---`, 페이지 번호, 섹션 경계)로 분할
2. 각 페이지 이미지와 매칭
3. 수동 검수 (필수 — 자동 분할은 정확하지 않음)

**현실적 시간 추정**:
- test_1 (5페이지, 한국어): ~1시간
- test_2 (4페이지, 영어): ~0.5시간
- test_3 (15페이지, 영어): ~2-3시간
- 검수 및 포맷 정리: ~1시간
- **총: 4-6시간**

### 5.4 LLaVA 포맷 학습 데이터 생성

2U1/Qwen-VL-Series-Finetune은 LLaVA 포맷 JSON을 사용한다:

```python
import json
import base64
from pathlib import Path

def create_training_data(test_dirs: list, output_path: str):
    """기존 GT와 페이지 이미지로 LLaVA 포맷 학습 데이터 생성"""
    dataset = []

    for test_dir in test_dirs:
        img_dir = Path(f"data/{test_dir}/page_images")
        gt_dir = Path(f"data/{test_dir}/page_gt")  # 분할된 페이지별 GT

        if not img_dir.exists() or not gt_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("page_*.png")):
            page_num = img_path.stem  # "page_001"
            gt_path = gt_dir / f"{page_num}.md"

            if not gt_path.exists():
                continue

            gt_text = gt_path.read_text(encoding="utf-8")

            sample = {
                "id": f"{test_dir}_{page_num}",
                "image": str(img_path),
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nConvert this document page to well-structured Markdown. Preserve all text content, headings, tables, lists, and mathematical formulas exactly."
                    },
                    {
                        "from": "gpt",
                        "value": gt_text
                    }
                ]
            }
            dataset.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Created {len(dataset)} training samples → {output_path}")
    return dataset

# 실행
create_training_data(
    ["test_1", "test_2", "test_3"],
    "data/e2e_train_data.json"
)
```

### 5.5 arXiv 데이터셋 확장 경로

기존 `src/dataset/` 파이프라인을 활용하면 대규모 학습 데이터를 구축할 수 있다:

```bash
# arXiv 논문 다운로드 + LaTeX → Markdown GT 변환
python -m src.dataset.build_arxiv_dataset --limit 50
```

이 파이프라인은 `arxiv_downloader.py` → `latex_to_markdown.py` → `validate_gt.py` 순서로 실행되며, 각 논문에 대해 PDF와 GT 마크다운을 자동 생성한다.

---

## 6. E2EVLMParser 구현 코드

### 6.1 파서 구현: `src/parsers/e2e_vlm_parser.py`

기존 `TextStructurer` (`src/parsers/text_structurer.py`)와 `TwoStageResult` (`src/parsers/two_stage_parser.py`) 패턴을 따른다.

```python
"""
E2E VLM Parser - 이미지 직접 파싱

PDF → 이미지 → VLM API로 직접 마크다운 생성.
OCR 단계 없이 Vision Encoder가 텍스트 인식과 구조 추출을 동시 수행.

Architecture:
    PDF → ImageOCRParser.pdf_to_images() → VLM API (이미지 입력) → Markdown
"""

import base64
import time
import httpx
from dataclasses import dataclass
from typing import Optional, List

from .ocr_parser import ImageOCRParser


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class E2EVLMResult:
    """E2E VLM 파싱 결과 데이터 클래스

    Attributes:
        success: 파싱 성공 여부
        content: 최종 마크다운 텍스트 (전체 문서)
        pages: 페이지별 마크다운 리스트
        elapsed_time: 전체 처리 시간 (초)
        image_conversion_time: PDF → 이미지 변환 시간 (초)
        vlm_time: VLM API 호출 시간 (초)
        page_count: 총 페이지 수
        model: 사용된 VLM 모델 ID
        error: 에러 발생 시 메시지
    """
    success: bool
    content: str
    pages: List[str]
    elapsed_time: float
    image_conversion_time: float
    vlm_time: float
    page_count: int
    model: str
    error: Optional[str] = None


# ==============================================================================
# Main Class
# ==============================================================================

class E2EVLMParser:
    """End-to-End VLM 문서 파서

    PDF를 이미지로 변환한 후 VLM API에 직접 전달하여
    마크다운을 생성한다. OCR 단계를 건너뛴다.

    Example:
        >>> parser = E2EVLMParser()
        >>> result = parser.parse_pdf(pdf_bytes)
        >>> print(result.content)
    """

    # ==========================================================================
    # Class Constants
    # ==========================================================================

    SYSTEM_PROMPT = """You are a document parsing expert. Convert the given document page image to well-structured Markdown format.

CRITICAL RULES:
1. ALWAYS use # symbols for headings based on document hierarchy
2. Document title → # Title
3. Section numbers like "1 Introduction" → ## 1. Introduction
4. Subsections like "3.1 Method" → ### 3.1 Method
5. Tables with aligned columns → Markdown table with | separators
6. Bullet points → - item
7. Numbered lists → 1. item
8. Mathematical formulas → LaTeX notation ($inline$ or $$block$$)
9. Multi-column layouts → merge into single-column reading order
10. Skip page headers, footers, and page numbers

PRESERVE all original text content exactly. Output Markdown only."""

    USER_PROMPT = "Convert this document page to well-structured Markdown. Preserve all text, headings, tables, lists, and mathematical formulas exactly."

    # ==========================================================================
    # Constructor
    # ==========================================================================

    def __init__(
        self,
        api_url: str = "http://localhost:8005/v1/chat/completions",
        model: str = "qwen3-vl-2b-instruct",
        timeout: float = 120.0,
        dpi: int = 150,
    ):
        """E2EVLMParser 초기화

        Args:
            api_url: VLM API 엔드포인트 URL
            model: 사용할 모델 ID
            timeout: API 요청 타임아웃 (초)
            dpi: PDF → 이미지 변환 해상도
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.dpi = dpi
        self.image_parser = ImageOCRParser()

    # ==========================================================================
    # Public Methods (Entry Points)
    # ==========================================================================

    def parse_pdf(self, pdf_bytes: bytes) -> E2EVLMResult:
        """PDF를 E2E VLM으로 파싱 (Entry Point)

        Args:
            pdf_bytes: PDF 파일 바이트 데이터

        Returns:
            E2EVLMResult: 파싱 결과

        Flow:
            1. PDF → 이미지 변환 (ImageOCRParser 재사용)
            2. 각 페이지 이미지를 VLM API에 전달
            3. 페이지별 마크다운 결합
            4. E2EVLMResult 반환
        """
        total_start = time.time()

        # Step 1: PDF → 이미지 변환
        img_start = time.time()
        images = self.image_parser.pdf_to_images(pdf_bytes, dpi=self.dpi)
        image_conversion_time = time.time() - img_start

        if not images:
            return E2EVLMResult(
                success=False,
                content="",
                pages=[],
                elapsed_time=time.time() - total_start,
                image_conversion_time=image_conversion_time,
                vlm_time=0.0,
                page_count=0,
                model=self.model,
                error="PDF → 이미지 변환 실패"
            )

        # Step 2-3: 각 페이지를 VLM으로 파싱
        vlm_start = time.time()
        pages_md = []
        errors = []

        for i, img_bytes in enumerate(images):
            try:
                page_md = self._parse_single_page(img_bytes)
                pages_md.append(page_md)
            except Exception as e:
                errors.append(f"Page {i+1}: {str(e)}")
                pages_md.append("")

        vlm_time = time.time() - vlm_start

        # Step 4: 페이지별 결과 결합
        content = "\n\n---\n\n".join(
            f"<!-- Page {i+1} -->\n{md}"
            for i, md in enumerate(pages_md)
            if md.strip()
        )

        error_msg = "; ".join(errors) if errors else None

        return E2EVLMResult(
            success=len(pages_md) > 0 and any(md.strip() for md in pages_md),
            content=content,
            pages=pages_md,
            elapsed_time=time.time() - total_start,
            image_conversion_time=image_conversion_time,
            vlm_time=vlm_time,
            page_count=len(images),
            model=self.model,
            error=error_msg
        )

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _parse_single_page(self, img_bytes: bytes) -> str:
        """단일 페이지 이미지를 VLM으로 파싱

        Args:
            img_bytes: PNG 이미지 바이트

        Returns:
            마크다운 텍스트
        """
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.USER_PROMPT
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()

        raw_content = result["choices"][0]["message"]["content"]
        return self._extract_content(raw_content)

    def _extract_content(self, raw_content: str) -> str:
        """응답에서 실제 콘텐츠 추출

        TextStructurer._extract_content() 패턴 재사용.
        Thinking 모델의 </think> 태그 및 마크다운 펜스 제거.
        """
        # Thinking 태그 처리
        if "</think>" in raw_content:
            parts = raw_content.split("</think>", 1)
            content = parts[1].strip() if len(parts) > 1 else ""
        else:
            content = raw_content

        # 마크다운 코드 펜스 제거
        content = content.strip()
        if content.startswith("```markdown"):
            content = content[len("```markdown"):].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        return content
```

### 6.2 `src/parsers/__init__.py` export 추가

기존 export 패턴에 따라 추가한다:

```python
# 추가할 내용 (기존 __init__.py 끝에)
from .e2e_vlm_parser import E2EVLMParser, E2EVLMResult

__all__ = [
    # ... 기존 항목 유지 ...
    # E2E VLM Parser (End-to-End)
    "E2EVLMParser",
    "E2EVLMResult",
]
```

---

## 7. eval_parsers.py 통합

### 7.1 test_e2e_vlm() 함수 추가

기존 `test_image_advanced()` (`src/eval_parsers.py:892`) 패턴을 따른다:

```python
def test_e2e_vlm(pdf_bytes: bytes, verbose: bool = False,
                  model: str = "qwen3-vl-2b-instruct",
                  api_url: str = "http://localhost:8005/v1/chat/completions") -> dict:
    """E2E VLM Parser (이미지 직접 파싱) 테스트"""
    from src.parsers.e2e_vlm_parser import E2EVLMParser

    print("\n" + "=" * 60)
    print(f"E2E-VLM ({model})")
    print("=" * 60)

    parser = E2EVLMParser(api_url=api_url, model=model)
    result = parser.parse_pdf(pdf_bytes)

    print("\n결과:")
    print(f"   - 성공: {'O' if result.success else 'X'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - 이미지 변환: {result.image_conversion_time:.2f}s")
    print(f"   - VLM 파싱: {result.vlm_time:.2f}s")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - 최종 출력 길이: {len(result.content)} chars")

    if result.error:
        print(f"   - 에러: {result.error}")

    if verbose and result.content:
        print("\n파싱 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "image_conversion_time": result.image_conversion_time,
        "vlm_time": result.vlm_time,
        "error": result.error
    }
```

### 7.2 run_single_test()에 파서 등록

`src/eval_parsers.py:1400` 부근, 기존 파서 4개 이후에 2줄 추가:

```python
    # ... 기존 파서 1-4 ...

    # 5. E2E-VLM-Zero (이미지 → VLM 직접 파싱)
    if not skip_advanced:
        try:
            results["E2E-VLM-Zero"] = test_e2e_vlm(input_bytes, verbose)
        except Exception as e:
            print(f"E2E-VLM-Zero 오류: {e}")
            results["E2E-VLM-Zero"] = {"success": False, "error": str(e)}

    # 6. E2E-VLM-Finetuned (LoRA 파인튜닝 모델)
    if not skip_advanced:
        try:
            results["E2E-VLM-Finetuned"] = test_e2e_vlm(
                input_bytes, verbose,
                model="e2e-finetuned",
                api_url="http://localhost:8006/v1/chat/completions"
            )
        except Exception as e:
            print(f"E2E-VLM-Finetuned 오류: {e}")
            results["E2E-VLM-Finetuned"] = {"success": False, "error": str(e)}
```

### 7.3 변경 불필요 확인

다음 함수들은 **제네릭 인터페이스**이므로 수정이 필요 없다:

- `evaluate_results()` (`src/eval_parsers.py:946`): `results` 딕셔너리의 키를 순회하며 CER/WER/Structure F1/TEDS를 계산. 파서 이름에 의존하지 않음.
- `save_results_to_files()` (`src/eval_parsers.py:1059`): 마찬가지로 제네릭.
- `print_summary()`: 마찬가지로 제네릭.

### 7.4 실행 명령어

```bash
# 전체 6파서 테스트 실행
python -m src.eval_parsers --all

# E2E 파서만 테스트 (Baseline 스킵으로 불가 — run_single_test 수정 필요)
# 현재는 --all로 전체 실행 후 결과에서 E2E 항목 확인
```

---

## 8. LoRA 파인튜닝 설정

### 8.1 스텝 수 계산

학습 데이터 규모를 기반으로 정확한 스텝 수를 계산한다:

```
총 샘플 수: 12 (24페이지 중 유효 페이지 추정)
배치 사이즈: 1 (GPU 메모리 제약)
gradient_accumulation_steps: 4
에폭 수: 5

유효 배치 사이즈 = batch_size × gradient_accumulation_steps = 1 × 4 = 4
스텝/에폭 = ceil(12 / 4) = 3
총 스텝 수 = 3 × 5 = 15
```

### 8.2 P0: save_steps 수정

| 파라미터 | 원본 가이드 | 수정 | 이유 |
|---------|------------|------|------|
| `save_steps` | `50` | `5` | 총 15 스텝에서 50 스텝마다 저장하면 체크포인트 **0개** 생성 |

```
원본: save_steps=50 → 체크포인트 생성: [50, 100, ...] → 15 스텝 이내 없음 ❌
수정: save_steps=5  → 체크포인트 생성: [5, 10, 15]  → 3개 체크포인트 ✓
```

### 8.3 P1: LoRA 파라미터 수정

| 파라미터 | 원본 가이드 | 수정 | 이유 |
|---------|------------|------|------|
| `lora_rank` | `64` | `16` | 12개 샘플로 rank 64는 과적합 위험 |
| `lora_alpha` | `128` | `32` | alpha = 2 × rank 관례 유지 |
| `freeze_merger` | `False` | `True` | 소량 데이터에서 merger 학습은 불안정 |

**rank 선택 근거**:
- 학습 데이터 12개 < rank 64의 학습 가능 파라미터 수 → 과적합
- rank 16 × 대상 레이어 수 ≈ 수백만 파라미터 (12개 샘플에 적합)
- 데이터 100개 이상 확보 시 rank 32-64로 점진적 확대

### 8.4 단계별 동결 전략

| 데이터 규모 | lora_rank | freeze_vision | freeze_merger | 근거 |
|------------|-----------|---------------|---------------|------|
| ~12 샘플 | 16 | True | True | Vision/Merger 고정, LLM만 적응 |
| 50-100 샘플 | 32 | True | False | Merger 해동으로 시각-언어 정렬 개선 |
| 200+ 샘플 | 64 | False | False | 전체 파인튜닝 (Vision 포함) |

### 8.5 파인튜닝 설정 파일

2U1/Qwen-VL-Series-Finetune 형식:

```yaml
# config/e2e_vlm_lora.yaml
model_id: "Qwen/Qwen3-VL-2B-Instruct"
output_dir: "./output/e2e_vlm_lora"

# LoRA 설정
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 동결 전략 (12 샘플)
freeze_vision_tower: true
freeze_merger: true

# 학습 하이퍼파라미터
num_train_epochs: 5
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true

# 저장 설정 (P0 수정)
save_steps: 5
save_total_limit: 3
logging_steps: 1

# 데이터
dataset_path: "data/e2e_train_data.json"
max_length: 4096
```

### 8.6 파인튜닝 실행

```bash
cd Qwen-VL-Series-Finetune

# 단일 GPU
python train.py --config config/e2e_vlm_lora.yaml

# 멀티 GPU (DeepSpeed)
deepspeed --num_gpus=2 train.py \
    --config config/e2e_vlm_lora.yaml \
    --deepspeed ds_config_zero2.json
```

---

## 9. 시스템 프롬프트 설계

### 9.1 기존 TextStructurer 프롬프트와의 비교

기존 `TextStructurer.SYSTEM_PROMPT` (`src/parsers/text_structurer.py:87`)는 **텍스트 입력**을 전제로 설계되었다. E2E 파서는 **이미지 입력**이므로 추가 규칙이 필요하다.

### 9.2 E2E VLM 시스템 프롬프트

```
기존 TextStructurer 규칙 (보존):
  ✓ 헤딩 레벨 (# ~ ####)
  ✓ 테이블 (| 구분)
  ✓ 리스트 (-, 1.)

E2E 추가 규칙:
  + Multi-column 레이아웃 → 단일 컬럼 읽기 순서
  + 페이지 헤더/푸터/페이지 번호 → 제거
  + 수식 → LaTeX 표기 ($inline$, $$block$$)
  + 그림/차트 → 캡션만 보존, 이미지 내용 설명 제외
```

위 규칙은 Section 6의 `E2EVLMParser.SYSTEM_PROMPT`에 이미 반영되어 있다.

### 9.3 프롬프트 최적화 방향

| 시도 | 프롬프트 변형 | 검증 방법 |
|------|-------------|----------|
| v1 | 기본 (Section 6 버전) | Structure F1, CER 측정 |
| v2 | Few-shot (1-2 예시 포함) | v1 대비 개선 여부 |
| v3 | 한국어 특화 규칙 추가 | test_1 CER 개선 여부 |

기존 프롬프트 변형 실험은 `docs/tech_report/appendix/A_Prompt_Variations.md`를 참조.

---

## 10. 평가 및 성공 기준

### 10.1 평가 메트릭

기존 `evaluate_results()` (`src/eval_parsers.py:946`)가 이미 계산하는 메트릭:

| 메트릭 | 함수 | 측정 대상 |
|--------|------|----------|
| CER (Full) | `calculate_cer()` | 전체 문자 정확도 |
| CER (Body) | `calculate_cer()` + `split_body_references()` | References 제외 본문 정확도 |
| WER | `calculate_wer()` | 단어 정확도 |
| Structure F1 | `calculate_structure_f1()` | 구조 보존 (헤딩, 리스트, 테이블) |
| TEDS | `calculate_teds()` | 테이블 구조 유사도 |

**P1 수정**: 원본 가이드에 없던 TEDS를 포함. 기존 `compute_teds()` (apted 기반)가 이미 구현되어 있으므로 추가 코드 불필요.

### 10.2 6-way 비교 테이블 템플릿

```
| Parser              | CER (Full) | CER (Body) | WER    | Struct F1 | TEDS  | Latency |
|---------------------|-----------|------------|--------|-----------|-------|---------|
| Text-Baseline       |           |            |        |           |       |         |
| Image-Baseline      |           |            |        |           |       |         |
| Text-Advanced       |           |            |        |           |       |         |
| Image-Advanced      |           |            |        |           |       |         |
| E2E-VLM-Zero        |           |            |        |           |       |         |
| E2E-VLM-Finetuned   |           |            |        |           |       |         |
```

### 10.3 성공 기준

| 기준 | 조건 | 비교 대상 | 근거 |
|------|------|----------|------|
| **최소 성공** | E2E-VLM-Finetuned Structure F1 > 79.25% | Text-Advanced (test_3) | 기존 최고 성능 |
| 추가 성공 | E2E-VLM-Zero Structure F1 > 16.67% | Image-Advanced (test_2) | Zero-shot도 구조 보존 |
| 보너스 | E2E-VLM-Zero CER < 536.50% (test_1) | Image-Advanced (test_1) | 한국어 hallucination 개선 |
| 이상적 | E2E-VLM-Finetuned CER < Text-Advanced CER | 각 test_* | OCR 에러 전파 제거 입증 |

---

## 11. Curriculum Learning

### 11.1 3단계 커리큘럼

기존 test_1/2/3의 난이도를 활용하여 점진적 학습을 설계한다:

| 단계 | 데이터 | 난이도 | 학습 목표 |
|------|--------|--------|----------|
| **Phase 1**: 기초 | test_3 (Attention Is All You Need) | 쉬움 — 디지털 PDF, 영어, 정형 구조 | 기본 마크다운 구조화 |
| **Phase 2**: 중급 | test_2 (Chain-of-Thought) | 중간 — 스캔 PDF, 영어, 수식 포함 | 스캔 문서 인식 + 수식 |
| **Phase 3**: 고급 | test_1 (공공AX 프로젝트) | 어려움 — 스캔 PDF, 한국어, 비정형 | 한국어 + 비정형 레이아웃 |

### 11.2 기존 test_* 매핑

```
Phase 1 → data/test_3/ (15 pages, 영어 디지털)
  └── 학습 후: Structure F1 > 79.25% 목표

Phase 2 → data/test_2/ (4 pages, 영어 스캔)
  └── 학습 후: CER < 33.09% (기존 Image-Advanced)

Phase 3 → data/test_1/ (5 pages, 한국어 스캔)
  └── 학습 후: CER < 536.50% (기존 Image-Advanced hallucination 개선)
```

### 11.3 arXiv 데이터셋 확장 경로

```
Phase 1 확장: test_3 + arXiv 영어 디지털 논문 50편
  └── python -m src.dataset.build_arxiv_dataset --limit 50

Phase 2 확장: + 스캔된 arXiv 논문 (PDF → 이미지 → 재스캔 시뮬레이션)
  └── 추가 스크립트 필요 (PDF → 이미지 → JPEG 압축 → 노이즈 추가)

Phase 3 확장: + 한국어 문서 (공공데이터포털 등)
  └── 별도 데이터 수집 필요
```

---

## 12. 로드맵 + Tech Report 연결

### 12.1 RQ4 추가 제안

기존 Tech Report에 RQ4를 추가:

> **RQ4**: End-to-End VLM이 OCR 에러 전파 문제를 해결하고 Two-Stage 파이프라인 대비 구조 보존 품질을 향상시킬 수 있는가?

이를 위해 `docs/tech_report/05_Results.md`에 Section 5.5를 확장하여 6파서 비교 결과를 추가한다.

### 12.2 현실적 시간 추정

| 단계 | 작업 | 예상 시간 |
|------|------|----------|
| 1 | 페이지 이미지 생성 + GT 분할 | 4-6시간 |
| 2 | E2EVLMParser 구현 + 테스트 | 2-3시간 |
| 3 | eval_parsers.py 통합 | 1시간 |
| 4 | Zero-Shot 평가 실행 | 1-2시간 |
| 5 | LoRA 파인튜닝 | 2-3시간 (GPU 의존) |
| 6 | Finetuned 모델 평가 | 1-2시간 |
| 7 | 결과 분석 + Tech Report 업데이트 | 2-3시간 |
| **총** | | **13-20시간** |

### 12.3 EMNLP 2026 System Demo 타임라인

| 마일스톤 | 기한 | 산출물 |
|---------|------|--------|
| E2E Zero-Shot 베이스라인 | 3월 1주 | 6-way 비교 결과 (test_1/2/3) |
| LoRA 파인튜닝 v1 | 3월 2주 | Finetuned 모델 + 평가 |
| arXiv 확장 데이터 | 3월 3주 | 50+ 논문 학습 데이터 |
| LoRA 파인튜닝 v2 | 3월 4주 | 확장 데이터로 재학습 |
| Tech Report RQ4 | 4월 1주 | 05_Results.md 업데이트 |
| 논문 초고 | 5월 | EMNLP 2026 System Demo 제출 |

---

## 13. 트러블슈팅

### 13.1 vLLM 멀티이미지 설정

**증상**: `E2EVLMParser.parse_pdf()`에서 페이지 1만 파싱되고 나머지 실패.

**원인**: vLLM의 기본 이미지 입력 제한이 1.

**해결**:
```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --limit-mm-per-prompt image=15  # 이 옵션 필수
```

> 현재 `E2EVLMParser`는 페이지별 개별 API 호출 방식이므로 이 문제가 발생하지 않는다. 향후 전체 페이지를 한 번에 전달하는 batch 모드를 구현할 경우 필요.

### 13.2 모델 클래스 불일치

**증상**: `KeyError: 'qwen3_vl'` 또는 `ValueError: Unrecognized model`.

**원인**: transformers 버전이 Qwen3-VL을 지원하지 않음.

**해결**:
```bash
pip install transformers>=4.45.0  # Qwen3-VL 지원 버전 확인
# 또는
pip install git+https://github.com/huggingface/transformers.git
```

### 13.3 save_steps 체크포인트 검증

**검증 방법**: 파인튜닝 완료 후 체크포인트 존재 확인.

```bash
ls output/e2e_vlm_lora/checkpoint-*/
# 예상 출력:
#   output/e2e_vlm_lora/checkpoint-5/
#   output/e2e_vlm_lora/checkpoint-10/
#   output/e2e_vlm_lora/checkpoint-15/
```

체크포인트가 없다면 `save_steps` 값이 총 스텝 수보다 큰 것이므로 재확인:

```
총 스텝 = ceil(샘플 수 / 유효배치) × 에폭 수
        = ceil(12 / 4) × 5 = 15
save_steps는 15 이하여야 함
```

### 13.4 GPU 메모리 부족 (OOM)

**증상**: `CUDA out of memory` during training.

**해결 순서**:
1. `gradient_accumulation_steps` 증가 (4 → 8), `per_device_train_batch_size` = 1 유지
2. `max_length` 감소 (4096 → 2048)
3. 이미지 해상도 감소 (`dpi=150` → `dpi=100`)
4. DeepSpeed ZeRO-2 적용

### 13.5 Thinking 태그 잔류

**증상**: VLM 출력에 `<think>...</think>` 태그가 포함됨.

**해결**: `E2EVLMParser._extract_content()`에서 이미 처리 (`TextStructurer._extract_content()` 패턴 재사용). 만약 여전히 잔류한다면:

```python
# API 호출 시 thinking 비활성화 (vLLM 지원 시)
payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
```

---

## 14. 원본 대비 수정 사항 요약 테이블

| # | 카테고리 | 항목 | 원본 | 수정 | 우선도 | 이유 |
|---|---------|------|------|------|--------|------|
| 1 | 환경 | 모델 클래스 | 확인 없음 | `config.model_type` 검증 추가 | P0 | Qwen2.5/3 클래스 다름 |
| 2 | 학습 | save_steps | 50 | 5 | P0 | 총 15 스텝에서 체크포인트 0개 |
| 3 | 데이터 | GT 출처 | 신규 제작 | 기존 gt_data_*.md 활용 | P0 | 불필요한 재작업 방지 |
| 4 | 코드 | 통합 방식 | 독립 스크립트 | src/parsers/ 패턴 준수 | P0 | 기존 평가 인프라 재사용 |
| 5 | 학습 | lora_rank | 64 | 16 | P1 | 12 샘플 과적합 방지 |
| 6 | 학습 | freeze_merger | False | True | P1 | 소량 데이터 안정성 |
| 7 | 평가 | TEDS | 미포함 | 포함 | P1 | 기존 compute_teds() 활용 |
| 8 | 계획 | 시간 추정 | 없음 | 13-20시간 | P1 | 현실적 기대치 설정 |

---

## 참조

### 기존 코드베이스 파일 맵

| 파일 | 이 가이드에서의 용도 |
|------|-------------------|
| `src/parsers/ocr_parser.py` | `ImageOCRParser.pdf_to_images()` 재사용, `OCRResult` 패턴 참조 |
| `src/parsers/text_structurer.py` | `TextStructurer` 프롬프트/API 패턴 참조, `_extract_content()` 재사용 |
| `src/parsers/two_stage_parser.py` | `TwoStageResult` dataclass 패턴 참조 |
| `src/parsers/__init__.py` | export 패턴 참조 |
| `src/eval_parsers.py` L892-939 | `test_image_advanced()` 함수 패턴 참조 |
| `src/eval_parsers.py` L1369-1445 | `run_single_test()` 파서 등록 위치 |
| `src/eval_parsers.py` L946-1056 | `evaluate_results()` 제네릭 인터페이스 확인 |
| `src/dataset/build_arxiv_dataset.py` | arXiv 데이터 확장 파이프라인 |
| `data/test_1/gt_data_1.md` | 한국어 스캔 문서 GT |
| `data/test_2/gt_data_2.md` | 영어 스캔 문서 GT |
| `data/test_3/gt_data_3.md` | 영어 디지털 문서 GT |
| `docs/tech_report/05_Results.md` | 기존 벤치마크 수치 (CER, Structure F1) |
| `docs/tech_report/appendix/A_Prompt_Variations.md` | 프롬프트 변형 실험 참조 |
