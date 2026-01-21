# VLM Document Parsing Quality Test

문서 파싱 품질 평가 프레임워크입니다.
**구조화된 파싱 결과가 Semantic Chunking 품질에 미치는 영향**을 검증합니다.

## 목표

```
┌─────────────────────────────────────────────────────────────┐
│  Document (PDF/Image)                                       │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Parsing (3 Methods)                     │   │
│  │  ┌───────────┬───────────┬───────────┐              │   │
│  │  │    VLM    │ pdfplumber│  RapidOCR │              │   │
│  │  │(Qwen3-VL) │  (Text)   │  (Image)  │              │   │
│  │  └───────────┴───────────┴───────────┘              │   │
│  │              ↓ CER, WER, Latency                     │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Semantic Chunking                          │   │
│  │              ↓ BS, CS Score                          │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  구조화된 데이터가 Chunking 품질에 영향을 미치는가?         │
└─────────────────────────────────────────────────────────────┘
```

## 테스트 설계

### 테스트 데이터

| ID | 파일 | 유형 | Ground Truth |
|----|------|------|--------------|
| test_1 | test_data_1.pdf | Digital PDF | Gemini 3.0 Pro |
| test_2 | test_data_2.jpg | Image | Gemini 3.0 Pro |
| test_3 | test_data_3.pdf | Digital PDF | Gemini 3.0 Pro |
| test_4 | 2025_한국부자보고서.pdf | Digital PDF | Gemini 3.0 Pro |

### Parser 비교

| Parser | 입력 | 출력 | 특징 |
|--------|------|------|------|
| **VLM** (Qwen3-VL) | Image | Markdown | 구조화된 출력, GPU 필요 |
| **pdfplumber** | PDF (Text) | Plain Text | 빠름, 디지털 PDF 전용 |
| **RapidOCR** (Docling) | PDF (Image) | Text | 스캔 문서 지원 |

### 평가 지표

**Phase 1: Parsing 품질**
- **CER** (Character Error Rate): 문자 단위 오류율
- **WER** (Word Error Rate): 단어 단위 오류율
- **Latency**: 처리 시간

**Phase 2: Chunking 품질** (예정)
- **BS** (Boundary Score): 청크 경계 품질
- **CS** (Chunk Score): 청크 내용 품질

## 프로젝트 구조

```
test-vlm-document-parsing/
├── src/
│   ├── app.py                 # Streamlit 비교 UI
│   ├── test_parsers.py        # CLI 테스트 스크립트
│   └── parsers/
│       ├── __init__.py
│       ├── vlm_parser.py      # VLM (Qwen3-VL)
│       ├── ocr_parser.py      # pdfplumber + ImageOCR
│       └── docling_parser.py  # Docling (RapidOCR)
├── data/
│   ├── test_1/                # PDF 테스트 1
│   ├── test_2/                # Image 테스트
│   ├── test_3/                # PDF 테스트 2
│   └── test_4/                # PDF 테스트 3
├── pyproject.toml
└── README.md
```

## 실행 방법

### 1. 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e .
```

### 2. CLI 테스트

```bash
# 단일 파일 테스트
python -m src.test_parsers --pdf data/test_1/test_data_1.pdf --gt data/test_1/gt_data_1.md

# VLM 서버 확인 (Qwen3-VL)
curl http://localhost:8005/v1/models

# 옵션
--skip-vlm       # VLM 테스트 스킵
--skip-docling   # Docling 테스트 스킵
--verbose        # 상세 출력
--tokenizer okt  # 한국어 토크나이저 (mecab/okt)
--output-dir     # 결과 저장 경로
```

### 3. Streamlit UI

```bash
cd src
streamlit run app.py --server.port 8501
```

## 의존성

```toml
[project.dependencies]
streamlit>=1.45.0
httpx>=0.27.0
pdfplumber>=0.11.0
Pillow>=10.0.0
jiwer>=3.0.0          # CER/WER 계산
python-Levenshtein>=0.25.0
pyyaml>=6.0.0
pandas>=2.0.0
pdf2image>=1.16.0
konlpy>=0.6.0         # 한국어 토크나이저

[project.optional-dependencies]
docling = ["docling>=2.0.0"]  # RapidOCR
```

## 예상 결과

| Parser | CER | WER | Latency | 구조화 |
|--------|-----|-----|---------|--------|
| VLM | 낮음 | 낮음 | 느림 | Markdown |
| pdfplumber | 중간 | 중간 | 빠름 | Plain Text |
| RapidOCR | 높음 | 높음 | 중간 | Plain Text |

**가설**: VLM의 구조화된 Markdown 출력이 Semantic Chunking에서 더 나은 BS/CS 점수를 보일 것이다.
