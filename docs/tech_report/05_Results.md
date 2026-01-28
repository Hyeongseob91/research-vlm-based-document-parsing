# 5. Results

> **Updated**: 2026-01-28
> **Status**: Parsing Test 결과 반영 완료

## 5.1 Lexical Accuracy Results

### 5.1.1 Character Error Rate (CER)

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 (Korean/Scanned) | N/A | 91.87% | N/A | 536.50% |
| test_2 (English/Scanned) | 99.59% | 40.80% | 120.54% | **33.09%** |
| test_3 (English/Digital) | 51.25% | **40.79%** | 64.11% | 57.71% |

**Key Observations**:
1. **스캔 PDF에서 Text-Baseline 한계**: PyMuPDF는 스캔 PDF에서 텍스트 추출 불가 (test_1, test_2)
2. **Image-Advanced가 영어 스캔 문서에서 최적**: test_2에서 CER 33.09%로 가장 낮음
3. **VLM 구조화의 Hallucination 위험**: test_1에서 Image-Advanced CER 536.50% (원본보다 긴 텍스트 생성)
4. **디지털 PDF에서 Image-Baseline 우수**: test_3에서 CER 40.79%로 Text-Baseline(51.25%)보다 낮음

### 5.1.2 Word Error Rate (WER)

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 (Korean/Scanned) | N/A | 99.42% | N/A | 322.63% |
| test_2 (English/Scanned) | 99.69% | 55.59% | 262.94% | **37.31%** |
| test_3 (English/Digital) | 57.19% | **41.24%** | 69.34% | 63.27% |

**Tokenization**:
- English: Whitespace tokenization
- Korean: Whitespace tokenization (MeCab 미적용)

### 5.1.3 Parser Architecture Comparison

| Parser | Stage 1 | Stage 2 | 적합 문서 유형 |
|--------|---------|---------|----------------|
| Text-Baseline | PyMuPDF | - | 디지털 PDF (텍스트 레이어 존재) |
| Image-Baseline | RapidOCR | - | 스캔 PDF, 이미지 기반 문서 |
| Text-Advanced | PyMuPDF | VLM 구조화 | 디지털 PDF + 구조 추출 필요 시 |
| Image-Advanced | RapidOCR | VLM 구조화 | 스캔 PDF + 구조 추출 필요 시 |

## 5.2 Structural Integrity Results

### 5.2.1 Structure F1 Score

> **정의**: 마크다운 구조 요소(Heading, List, Table) 검출 정확도

| Document | Text-Baseline | Image-Baseline | Text-Advanced | Image-Advanced |
|----------|---------------|----------------|---------------|----------------|
| test_1 (Korean/Scanned) | N/A | 0.00% | N/A | 0.00% |
| test_2 (English/Scanned) | 0.00% | 0.00% | 9.30% | **16.67%** |
| test_3 (English/Digital) | 0.00% | 0.00% | **79.25%** | 77.78% |

**Key Observations**:
1. **Baseline 파서는 구조 검출 불가**: 모든 Baseline 파서의 Structure F1 = 0%
2. **VLM 구조화 효과 입증**: Advanced 파서에서 Structure F1 크게 향상
3. **디지털 PDF에서 Text-Advanced 최적**: test_3에서 F1 79.25% 달성 (Precision 72.4%, Recall 87.5%)

### 5.2.2 Structure F1 Detail (test_3)

| Parser | Precision | Recall | TP | FP | FN | Hyp Elements | Ref Elements |
|--------|-----------|--------|----|----|----|--------------|--------------|
| Text-Baseline | 0.00% | 0.00% | 0 | 11 | 24 | 11 | 24 |
| Image-Baseline | 0.00% | 0.00% | 0 | 0 | 24 | 0 | 24 |
| Text-Advanced | 72.41% | 87.50% | 21 | 8 | 3 | 29 | 24 |
| Image-Advanced | 70.00% | 87.50% | 21 | 9 | 3 | 30 | 24 |

**해석**:
- **Recall 87.5%**: GT의 24개 구조 요소 중 21개 검출
- **Precision ~71%**: 검출한 요소 중 약 71%가 정확
- **FN 3개**: 놓친 구조 요소 (세부 섹션 헤딩 누락 추정)

### 5.2.3 Structure Element Types

평가 대상 구조 요소:

| Element Type | Pattern | Example |
|--------------|---------|---------|
| Heading | `^#{1,6}\s+` | `# Title`, `## Section` |
| Unordered List | `^[\s]*[-*+]\s+` | `- item` |
| Ordered List | `^[\s]*\d+\.\s+` | `1. first` |
| Table Row | `^\|.+\|$` | `\| col1 \| col2 \|` |

## 5.3 Latency Analysis

### 5.3.1 Processing Time (seconds)

| Parser | test_1 | test_2 | test_3 | 특성 |
|--------|--------|--------|--------|------|
| Text-Baseline | 1.35 | 3.58 | **2.31** | 가장 빠름 (디지털 PDF) |
| Image-Baseline | 18.07 | 23.65 | **0.27** | 스캔 PDF에서 느림 |
| Text-Advanced | 1.83 | 39.01 | 42.92 | VLM 호출 오버헤드 |
| Image-Advanced | 51.26 | 37.06 | 35.75 | OCR + VLM 이중 처리 |

### 5.3.2 Stage별 시간 분석 (test_3)

| Parser | Stage 1 | Stage 2 (VLM) | Total |
|--------|---------|---------------|-------|
| Text-Advanced | 2.28s | 40.64s | 42.92s |
| Image-Advanced | 0.27s | 35.48s | 35.75s |

**Observation**: VLM 구조화 단계가 전체 시간의 90% 이상 차지

### 5.3.3 Cost-Quality Trade-off

| Parser | Avg CER | Avg Latency | 적합 시나리오 |
|--------|---------|-------------|---------------|
| Text-Baseline | 75.4% | 2.4s | 빠른 텍스트 검색 (디지털 PDF) |
| Image-Baseline | 57.8% | 14.0s | 스캔 문서 OCR |
| Text-Advanced | 94.9% | 27.9s | 구조화된 출력 필요 시 |
| Image-Advanced | 209.1%* | 41.4s | 스캔 문서 + 구조화 (주의 필요) |

*test_1의 Hallucination 포함 평균

## 5.4 Document Type Analysis

### 5.4.1 스캔 PDF (test_1, test_2)

| Metric | Best Parser | Score | 비고 |
|--------|-------------|-------|------|
| CER (Korean) | Image-Baseline | 91.87% | 한글 OCR 한계 |
| CER (English) | Image-Advanced | 33.09% | VLM 구조화 효과 |
| Structure F1 | Image-Advanced | 16.67% | 부분적 구조 검출 |

**권장**: 영어 스캔 문서 → Image-Advanced, 한글 스캔 문서 → Image-Baseline (Hallucination 방지)

### 5.4.2 디지털 PDF (test_3)

| Metric | Best Parser | Score | 비고 |
|--------|-------------|-------|------|
| CER | Image-Baseline | 40.79% | 텍스트 레이어 무시, OCR 사용 |
| Structure F1 | Text-Advanced | 79.25% | 구조화 최적 |
| Latency | Text-Baseline | 2.31s | 빠른 처리 |

**권장**: 구조 필요 시 Text-Advanced, 속도 우선 시 Text-Baseline

## 5.5 Key Findings Summary

### 5.5.1 Research Question Answers

| Research Question | Finding | Evidence |
|-------------------|---------|----------|
| **RQ1**: VLM 구조화가 텍스트 정확도를 향상시키는가? | **부분적** | 영어 스캔 문서에서 CER 개선 (40.8% → 33.1%), 단 Hallucination 위험 |
| **RQ2**: VLM 구조화가 구조 보존에 효과적인가? | **Yes** | Structure F1: 0% → 79% (test_3) |
| **RQ3**: 문서 유형별 최적 파서가 다른가? | **Yes** | 스캔 PDF: Image 계열, 디지털 PDF: Text 계열 |

### 5.5.2 Trade-off 분석

```
                    텍스트 정확도 (CER ↓)
                           ▲
                           │
         Baseline          │         (이상적)
         ┌─────┐           │
         │ 좋음 │           │
         └─────┘           │
                           │
    ─────────────────────────────────────▶ 구조화 품질 (F1 ↑)
                           │
                           │         Advanced
                           │         ┌─────┐
                           │         │ 좋음 │
                           │         └─────┘
```

- **Baseline**: 원본 텍스트 보존 우수, 구조 정보 없음
- **Advanced**: 구조화 품질 우수, 일부 텍스트 변형 발생

### 5.5.3 Parser Selection Guide

| 사용 목적 | 권장 파서 | 이유 |
|----------|----------|------|
| 텍스트 검색/인덱싱 | Baseline | 높은 텍스트 정확도 |
| RAG/Chunking | Advanced | 구조 기반 청킹 가능 |
| 문서 변환 (HTML/MD) | Advanced | 마크다운 구조 활용 |
| 실시간 처리 | Baseline | 낮은 Latency |

---

## Appendix: Raw Data

### A. test_1 (Korean Scanned PDF)
```
Document: 공공AX 프로젝트 공모안내서
Pages: 5
Type: Scanned PDF

Text-Baseline:  Extraction failed (scanned PDF)
Image-Baseline: CER=91.87%, WER=99.42%, F1=0.00%, Latency=18.07s
Text-Advanced:  Extraction failed (scanned PDF)
Image-Advanced: CER=536.50%, WER=322.63%, F1=0.00%, Latency=51.26s
```

### B. test_2 (English Scanned PDF)
```
Document: Chain-of-Thought Prompting (NeurIPS 2022)
Pages: 4
Type: Scanned PDF

Text-Baseline:  CER=99.59%, WER=99.69%, F1=0.00%, Latency=3.58s
Image-Baseline: CER=40.80%, WER=55.59%, F1=0.00%, Latency=23.65s
Text-Advanced:  CER=120.54%, WER=262.94%, F1=9.30%, Latency=39.01s
Image-Advanced: CER=33.09%, WER=37.31%, F1=16.67%, Latency=37.06s
```

### C. test_3 (English Digital PDF)
```
Document: Attention Is All You Need (NeurIPS 2017)
Pages: 15
Type: Digital PDF

Text-Baseline:  CER=51.25%, WER=57.19%, F1=0.00%, Latency=2.31s
Image-Baseline: CER=40.79%, WER=41.24%, F1=0.00%, Latency=0.27s
Text-Advanced:  CER=64.11%, WER=69.34%, F1=79.25%, Latency=42.92s
Image-Advanced: CER=57.71%, WER=63.27%, F1=77.78%, Latency=35.75s
```
