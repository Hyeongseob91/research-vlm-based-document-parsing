# VLM 문서 파싱 품질 평가 프레임워크

> **"구조적 무결성이 의미 검색을 개선하는가?"**

Vision-Language Model(VLM)의 구조화된 마크다운 출력이 기존 OCR 방식 대비 Semantic Chunking 및 RAG(Retrieval-Augmented Generation) 성능에 미치는 영향을 정량적으로 분석하는 종합 평가 프레임워크입니다.

---

## 빠른 시작

```bash
# 의존성 설치
uv sync

# 파서 비교 벤치마크 실행
python -m src.test_parsers --pdf data/test_1/test_data_1.pdf --gt data/test_1/gt_data_1.md

# 전체 평가 파이프라인 실행
python -m src.run_benchmark --config experiments/config.yaml
```

---

## 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [설치](#설치)
- [사용법](#사용법)
- [평가 지표](#평가-지표)
- [프로젝트 구조](#프로젝트-구조)
- [기술 보고서](#기술-보고서)
- [실험 결과](#실험-결과)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

---

## 개요

### 문제 정의

기존 RAG 파이프라인은 주로 일반 텍스트 추출에 의존하며, 이는 중요한 문서 구조를 보존하지 못합니다:

- **표 구조 손실**: 행-열 관계가 파괴됨
- **다단 오류**: 2단 레이아웃에서 읽기 순서 혼란
- **헤더 계층 손실**: 섹션 관계가 보존되지 않음
- **의미적 단절**: 청킹이 잘못된 위치에서 분리됨

### 가설

> VLM 기반 파싱은 문서 레이아웃을 보존하는 구조화된 마크다운을 생성하여, 더 나은 의미적 청킹 경계(높은 Boundary Score)와 향상된 청크 일관성(높은 Chunk Score)을 달성하고, 궁극적으로 더 나은 검색 정확도를 제공합니다.

### 핵심 연구 질문

| RQ | 질문 | 지표 |
|----|------|------|
| **RQ1** | VLM이 더 나은 어휘적 정확도를 달성하는가? | CER, WER |
| **RQ2** | VLM이 구조를 더 잘 보존하는가? | Boundary Score, Chunk Score |
| **RQ3** | 더 나은 파싱이 검색을 개선하는가? | Hit Rate@k, MRR |

---

## 주요 기능

### 다중 파서 지원
- **VLM Parser** (Qwen3-VL): 레이아웃 이해를 통한 구조화된 마크다운 출력
- **pdfplumber**: 빠른 디지털 PDF 텍스트 추출
- **Docling + RapidOCR**: 스캔 문서 OCR

### 종합 평가
- **어휘 지표**: 한국어 형태소 분석(MeCab)을 포함한 CER, WER
- **구조 지표**: Boundary Score, Chunk Score
- **검색 지표**: 통계적 유의성 검정을 포함한 Hit Rate@k, MRR
- **오류 분석**: 자동 오류 감지, 분류, 사례 연구 생성

### 연구 인프라
- **기술 보고서 템플릿**: 완전한 학술 보고서 구조
- **실험 구성**: YAML 기반 재현 가능한 실험
- **Q&A 생성**: LLM 기반 평가 데이터셋 생성
- **벤치마크 러너**: 자동화된 종단 간 평가 파이프라인

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        문서 입력                                     │
│                   (PDF / 이미지 / 스캔)                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        파싱 레이어                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │    VLM      │  │ pdfplumber  │  │   Docling   │                 │
│  │ (Qwen3-VL)  │  │   (디지털)   │  │ (RapidOCR)  │                 │
│  │  마크다운    │  │  일반 텍스트 │  │  일반 텍스트 │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        청킹 레이어                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   고정      │  │   재귀적    │  │   의미적    │                 │
│  │   크기      │  │   문자      │  │   (주제)    │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        평가 레이어                                   │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │   어휘    │  │   구조    │  │   검색    │  │   오류    │       │
│  │ CER, WER  │  │  BS, CS   │  │ HR@k, MRR │  │   분석    │       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 설치

### 사전 요구사항

- Python 3.11+
- CUDA 지원 GPU (VLM 추론용)
- MeCab (한국어 토큰화용)

### uv 사용 (권장)

```bash
git clone https://github.com/your-repo/test-vlm-document-parsing.git
cd test-vlm-document-parsing

# uv로 설치
uv sync

# 한국어 NLP 지원 설치
uv sync --extra korean
```

### pip 사용

```bash
pip install -e .

# 한국어 지원
pip install -e ".[korean]"

# 모든 기능
pip install -e ".[all]"
```

### MeCab 설치 (Ubuntu)

```bash
sudo apt-get install mecab mecab-ko mecab-ko-dic libmecab-dev
pip install mecab-python3
```

---

## 사용법

### 1. 기본 파서 비교

```bash
# 문서에 대해 모든 파서 비교
python -m src.test_parsers \
    --pdf data/test_1/test_data_1.pdf \
    --gt data/test_1/gt_data_1.md \
    --tokenizer mecab

# VLM 스킵 (GPU가 없을 때)
python -m src.test_parsers --pdf document.pdf --gt ground_truth.md --skip-vlm
```

### 2. 전체 벤치마크 파이프라인

```bash
# 설정 파일로 전체 평가 실행
python -m src.run_benchmark --config experiments/config.yaml

# 단일 문서에 대해 실행
python -m src.run_benchmark \
    --pdf data/test_1/test_data_1.pdf \
    --gt data/test_1/gt_data_1.md \
    --qa data/qa_pairs.json \
    --output results/
```

### 3. Q&A 데이터셋 생성

```bash
# Ground Truth 문서에서 Q&A 쌍 생성
python -m experiments.generate_qa \
    --config experiments/config.yaml \
    --output data/qa_pairs.json

# 특정 LLM 제공자 사용
python -m experiments.generate_qa --provider openai --questions-per-doc 15
```

### 4. Streamlit 웹 UI

```bash
streamlit run src/app.py --server.port 8501
```

---

## 평가 지표

### 1단계: 어휘적 정확도

| 지표 | 공식 | 설명 |
|------|------|------|
| **CER** | `(S + D + I) / N` | 문자 오류율 |
| **WER** | `(S + D + I) / N` | 단어 오류율 (형태소 토큰화 적용) |

- `S`: 대체 (잘못된 문자/단어)
- `D`: 삭제 (누락된 내용)
- `I`: 삽입 (환각으로 추가된 내용)
- `N`: Ground Truth의 총 문자/단어 수

### 2단계: 구조적 무결성

| 지표 | 공식 | 설명 |
|------|------|------|
| **Boundary Score (BS)** | `\|B_pred ∩ B_gt\| / \|B_gt\|` | 의미적 경계 정렬도 |
| **Chunk Score (CS)** | `avg(coherence(chunk))` | 청크 내 의미적 일관성 |

### 3단계: 검색 성능

| 지표 | 공식 | 설명 |
|------|------|------|
| **Hit Rate@k** | `hits_in_top_k / total_queries` | 상위 k개 결과에 관련 청크 포함 여부 |
| **MRR** | `(1/N) * Σ(1/rank_i)` | 평균 역순위 |

### 품질 임계값

| 지표 | 우수 | 양호 | 허용 |
|------|------|------|------|
| CER | < 2% | < 5% | < 10% |
| WER | < 5% | < 10% | < 20% |
| BS | > 0.9 | > 0.8 | > 0.7 |
| Hit Rate@5 | > 90% | > 75% | > 60% |

---

## 프로젝트 구조

```
test-vlm-document-parsing/
├── src/
│   ├── app.py                    # Streamlit 웹 UI
│   ├── test_parsers.py           # CLI 파서 비교 도구
│   ├── run_benchmark.py          # 전체 평가 파이프라인
│   │
│   ├── parsers/                  # 파서 구현
│   │   ├── vlm_parser.py         # Qwen3-VL 통합
│   │   ├── ocr_parser.py         # pdfplumber 파서
│   │   └── docling_parser.py     # Docling + RapidOCR
│   │
│   ├── chunking/                 # 텍스트 청킹 모듈
│   │   ├── chunker.py            # 청킹 전략
│   │   └── metrics.py            # BS, CS 계산
│   │
│   ├── retrieval/                # 검색 평가
│   │   ├── embedder.py           # 텍스트 임베딩
│   │   ├── retriever.py          # 의미 검색
│   │   └── evaluator.py          # HR@k, MRR 지표
│   │
│   ├── error_analysis/           # 오류 분석 모듈
│   │   ├── analyzer.py           # 오류 감지
│   │   ├── diff_visualizer.py    # HTML diff 생성
│   │   └── case_study.py         # 사례 연구 생성기
│   │
│   └── evaluation/               # 통합 평가 인터페이스
│       └── __init__.py           # 통합 모듈
│
├── experiments/
│   ├── config.yaml               # 실험 설정
│   └── generate_qa.py            # Q&A 데이터셋 생성기
│
├── data/
│   ├── test_1/                   # 한국어 정부 문서
│   ├── test_2/                   # 영수증 이미지
│   ├── test_3/                   # 영어 학술 논문
│   └── qa_pairs.json             # 생성된 Q&A 데이터셋
│
├── docs/
│   └── tech_report/              # 기술 보고서 섹션
│       ├── 00_Abstract.md
│       ├── 01_Introduction.md
│       ├── 02_Related_Work.md
│       ├── 03_Methodology.md
│       ├── 04_Experimental_Setup.md
│       ├── 05_Results.md
│       ├── 06_Discussion.md
│       ├── 07_Conclusion.md
│       ├── 08_References.md
│       ├── appendix/
│       └── figures/
│
├── results/                      # 평가 출력
│   ├── retrieval/
│   ├── structure/
│   ├── errors/
│   └── ablation/
│
├── _drafts/                      # 레거시 평가 모듈
├── pyproject.toml
├── README.md                     # 영어 문서
└── README.ko.md                  # 한국어 문서
```

---

## 기술 보고서

프레임워크에는 완전한 학술 기술 보고서 템플릿이 포함되어 있습니다:

| 섹션 | 내용 |
|------|------|
| **Abstract** | 연구 요약, 주요 발견 |
| **Introduction** | 문제 정의, 연구 질문, 기여점 |
| **Related Work** | VLM, OCR, RAG 문헌 리뷰 |
| **Methodology** | 평가 프레임워크, 지표 공식 |
| **Experimental Setup** | 데이터셋, 파서 설정, 파라미터 |
| **Results** | 표 및 분석 (템플릿) |
| **Discussion** | RQ 답변, 오류 패턴, 한계점 |
| **Conclusion** | 발견, 하이브리드 전략 권장 |
| **References** | 26개 학술 인용 |
| **Appendix** | 프롬프트 변형, 전체 결과, 사례 연구 |

---

## 실험 결과

### 예비 결과 (test_3 - 스캔 PDF)

| 파서 | CER | WER | 지연 시간 |
|------|-----|-----|----------|
| **VLM (Qwen3-VL)** | 56.29% | 70.85% | 15.61초 |
| pdfplumber | 99.62% | 100% | 18.12초 |
| RapidOCR | N/A | N/A | 6.85초 |

**핵심 발견**: 스캔 문서의 경우 VLM이 유일한 실행 가능한 옵션입니다.

### 하이브리드 전략 권장

```
문서 입력
     │
     ▼
  스캔? ──예──► VLM (필수)
     │
   아니오
     │
     ▼
  복잡한 레이아웃? ──예──► VLM (권장)
  (표, 다단)
     │
   아니오
     │
     ▼
  pdfplumber (빠름, 충분함)
```

---

## 프롬프트 엔지니어링

### 문제: VLM 환각

초기 프롬프트는 VLM이 요약과 설명을 추가하게 하여 삽입 오류를 증가시켰습니다.

### 해결책: 전사 중심 프롬프트 (v2)

```
당신은 작가가 아닌 문서 전사 엔진입니다.
주어진 문서 이미지를 마크다운으로 변환하되,
이미지에 보이는 내용만 엄격하게 전사하세요.

## 하드 제약:
- 텍스트를 추가, 다시 표현, 요약, 추론 또는 번역하지 마세요.
- 수행 중인 작업을 설명, 코멘트하거나 묘사하지 마세요.
- 읽을 수 없는 경우 추측 대신 `[읽을 수 없음]`을 작성하세요.
- 값이 누락된 경우 `[비어 있음]`을 사용하세요. 값을 만들어내지 마세요.

## 출력:
(여기에 전사된 마크다운 내용만 작성하세요.)
```

---

## 설정

모든 실험은 `experiments/config.yaml`을 통해 제어됩니다:

```yaml
# 청킹 (공정한 비교를 위해 고정)
chunking:
  strategy: "recursive_character"
  chunk_size: 500
  chunk_overlap: 50

# 임베딩
embedding:
  model: "jhgan/ko-sroberta-multitask"
  device: "cuda"

# 검색
retrieval:
  top_k: [1, 3, 5, 10]

# 품질 임계값
thresholds:
  cer:
    excellent: 0.02
    good: 0.05
```

---

## API 참조

### UnifiedEvaluator

```python
from src.evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator(config)

# 전체 평가
results = evaluator.evaluate_full(
    parsed_text=parsed_content,
    ground_truth=gt_content,
    qa_pairs=qa_data,
    document_id="test_1",
    parser_name="vlm"
)

# 개별 지표
lexical = evaluator.evaluate_lexical(parsed, gt)
structural = evaluator.evaluate_structural(parsed, gt)
retrieval = evaluator.evaluate_retrieval(parsed, qa_pairs)
errors = evaluator.analyze_errors(parsed, gt)
```

---

## 알려진 문제 및 오류 패턴

### VLM 환각

**증상**: VLM이 원문에 없는 설명/요약을 추가하여 삽입 오류 급증

**완화 방법**:
- 전사 중심 프롬프트 사용
- `max_tokens` 제한으로 과도한 생성 방지
- `temperature: 0.0`으로 창의성 억제

### RapidOCR 한국어 인식 약함

**증상**: 한국어 문서에서 인식률이 현저히 낮음

**완화 방법**:
- 스캔 품질 개선 (DPI 증가)
- 한국어 문서는 VLM 또는 pdfplumber 우선 사용

### 다단 레이아웃 순서 꼬임

**증상**: 2단 레이아웃에서 텍스트 순서가 뒤섞임

**완화 방법**:
- VLM 우선 사용 (레이아웃 이해 능력)
- 후처리로 순서 재정렬

---

## 기여하기

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 열기

---

## 참고 문헌

- [jiwer](https://github.com/jitsi/jiwer) - CER/WER 계산
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF 텍스트 추출
- [Docling](https://github.com/DS4SD/docling) - 문서 이해
- [KoNLPy](https://konlpy.org/) - 한국어 NLP 툴킷
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - Vision-Language Model
- [sentence-transformers](https://www.sbert.net/) - 텍스트 임베딩

---

## 라이선스

MIT License

---

## 인용

```bibtex
@software{vlm_document_parsing,
  title = {VLM Document Parsing Quality Test Framework},
  year = {2025},
  url = {https://github.com/your-repo/test-vlm-document-parsing}
}
```
