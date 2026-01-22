# Appendix C: Error Case Studies

## C.1 Overview

This appendix presents detailed case studies of parsing errors, categorized by error type and severity. Each case study includes:
- Error description and context
- Ground truth vs. parsed output comparison
- Root cause analysis
- Impact on downstream tasks

## C.2 Error Taxonomy

### C.2.1 Error Categories

| Category | Code | Description | Severity |
|----------|------|-------------|----------|
| Table Structure | TABLE_STRUCT | Table row/column structure corrupted | Critical |
| Multi-Column | MULTI_COL | Multi-column reading order incorrect | Critical |
| Hallucination | HALLUC | Content generated that doesn't exist | Major |
| Header Hierarchy | HEADER_HIER | Header level incorrectly assigned | Medium |
| Text Deletion | DELETION | Text present in source but missing | Varies |
| Text Substitution | SUBST | Characters incorrectly recognized | Minor |
| Formatting Loss | FORMAT | Formatting elements lost | Minor |

### C.2.2 Severity Definitions

| Severity | Definition | RAG Impact |
|----------|------------|------------|
| Critical | Renders content unusable | Retrieval failure likely |
| Major | Significant content error | Retrieval accuracy degraded |
| Medium | Structural error | Chunking may be affected |
| Minor | Cosmetic/minor error | Minimal impact |

## C.3 Case Studies

### Case Study 1: Table Structure Preservation (VLM Success)

**Document**: test_1 (Korean Government Document)
**Category**: TABLE_STRUCT
**Parser Comparison**: VLM vs pdfplumber

#### Ground Truth

```markdown
| 구분 | 지원 내용 | 지원 금액 |
|------|----------|----------|
| 기본 지원 | AI 도입 컨설팅 | 최대 5천만원 |
| 추가 지원 | 기술 고도화 | 최대 3천만원 |
```

#### VLM Output

```markdown
| 구분 | 지원 내용 | 지원 금액 |
|------|----------|----------|
| 기본 지원 | AI 도입 컨설팅 | 최대 5천만원 |
| 추가 지원 | 기술 고도화 | 최대 3천만원 |
```

#### pdfplumber Output

```
구분 지원 내용 지원 금액
기본 지원 AI 도입 컨설팅 최대 5천만원
추가 지원 기술 고도화 최대 3천만원
```

#### Analysis

**VLM Performance**: Successfully preserved table structure with proper markdown syntax

**pdfplumber Failure**: Lost table structure; text extracted as continuous stream

**Root Cause**: pdfplumber relies on PDF line rules for table detection; table without visible borders was not recognized as a table

**RAG Impact**:
- VLM: Query "지원 금액은 얼마인가요?" correctly retrieves table chunk
- pdfplumber: Same query may fail to find structured information

---

### Case Study 2: Multi-Column Reading Order (VLM Success)

**Document**: test_3 (Academic Paper - Two-column layout)
**Category**: MULTI_COL
**Parser Comparison**: VLM vs pdfplumber

#### Ground Truth (Simplified)

```markdown
## 1 Introduction

Chain-of-thought prompting is a technique...

## 2 Related Work

Prior work has explored various prompting methods...
```

#### VLM Output

```markdown
## 1 Introduction

Chain-of-thought prompting is a technique...

## 2 Related Work

Prior work has explored various prompting methods...
```

#### pdfplumber Output

```
1 Introduction
Chain-of-thought prompting is a technique... 2 Related Work
Prior work has explored various prompting methods... [text from column 2 mixed]
```

#### Analysis

**VLM Performance**: Correctly interpreted two-column layout and reading order

**pdfplumber Issue**: Extracted text in spatial order (left-to-right across page) rather than logical reading order

**Root Cause**: pdfplumber extracts text boxes by position; multi-column layouts require layout analysis

**RAG Impact**:
- VLM: Sections remain coherent for chunking
- pdfplumber: Section boundaries corrupted, semantic chunking fails

---

### Case Study 3: Hallucination Detection (VLM Failure)

**Document**: test_2 (Receipt)
**Category**: HALLUC
**Parser**: VLM with Prompt v1 (Extraction Expert)

#### Ground Truth

```markdown
## 김태준의 탕탕집
서울시 강남구 역삼동 123-45

| 메뉴 | 수량 | 금액 |
|------|------|------|
| 닭볶음탕 | 1 | 35,000 |
| 공기밥 | 2 | 2,000 |

합계: 37,000원
```

#### VLM Output (Prompt v1)

```markdown
## 김태준의 탕탕집
서울시 강남구 역삼동 123-45
(매일 신선한 재료로 정성껏 준비합니다)  ← HALLUCINATION

| 메뉴 | 수량 | 금액 |
|------|------|------|
| 닭볶음탕 (2-3인분 추천) | 1 | 35,000 |  ← HALLUCINATION
| 공기밥 | 2 | 2,000 |

합계: 37,000원
```

#### Analysis

**VLM Failure**: Added promotional text and menu description not present in original

**Root Cause**: Prompt v1 ("extraction expert") encouraged interpretation and "helpful" additions

**Mitigation**: Prompt v2 ("transcription engine") with explicit "do not add" instructions

**RAG Impact**:
- Hallucinated content may be retrieved for unrelated queries
- False information could be presented as factual

---

### Case Study 4: OCR Total Failure (Scanned Document)

**Document**: test_2 (Receipt Image)
**Category**: Multiple
**Parser Comparison**: VLM vs RapidOCR

#### Ground Truth

```markdown
## 김태준의 탕탕집
...
```

#### VLM Output

```markdown
## 김태준의 탕탕집
서울시 강남구 역삼동 123-45
...
[Accurate transcription]
```

#### RapidOCR Output

```
[Empty or severely corrupted text]
```

#### Analysis

**VLM Performance**: Successfully recognized and transcribed Korean text from image

**RapidOCR Failure**: Failed to extract meaningful text from receipt image

**Root Cause**:
- Image quality issues (lighting, angle)
- Font style not well supported
- Korean character recognition limitations

**RAG Impact**: Complete retrieval failure for RapidOCR; VLM essential for scanned documents

---

### Case Study 5: Header Hierarchy Confusion

**Document**: test_1 (Government Document)
**Category**: HEADER_HIER
**Parser**: VLM

#### Ground Truth

```markdown
# 공공 AX 지원 사업 안내

## 1. 사업 개요

### 1.1 목적

공공기관의 AI 전환을 지원합니다.

### 1.2 지원 대상

중소기업 및 스타트업
```

#### VLM Output

```markdown
# 공공 AX 지원 사업 안내

## 1. 사업 개요

## 1.1 목적  ← Should be ### (h3)

공공기관의 AI 전환을 지원합니다.

## 1.2 지원 대상  ← Should be ### (h3)

중소기업 및 스타트업
```

#### Analysis

**Issue**: Subsection headers (1.1, 1.2) marked as h2 instead of h3

**Root Cause**: VLM may not distinguish visual hierarchy without explicit formatting cues

**Impact**:
- Hierarchical chunking may create incorrect section boundaries
- Minor impact on flat chunking strategies

---

### Case Study 6: Text Deletion (Partial Failure)

**Document**: test_3 (Academic Paper)
**Category**: DELETION
**Parser**: VLM

#### Ground Truth (Section)

```markdown
We use a few-shot prompting approach where we provide the model with
exemplars of (input, chain of thought, output) triples. The exemplars
are fixed for all test cases in a given task.
```

#### VLM Output

```markdown
We use a few-shot prompting approach where we provide the model with
exemplars of (input, chain of thought, output) triples. [unclear]
```

#### Analysis

**Issue**: Final sentence marked as [unclear] rather than transcribed

**Root Cause**: Low-contrast text or partial image quality issue

**Positive Note**: [unclear] marker indicates uncertainty rather than hallucination

**Impact**: Minor information loss; better than incorrect transcription

---

### Case Study 7: Character Substitution (Korean)

**Document**: test_1 (Government Document)
**Category**: SUBST
**Parser**: VLM

#### Ground Truth

```markdown
지원 규모: 최대 5천만원
```

#### VLM Output

```markdown
지원 규모: 최대 5천만원
```

#### Analysis

**Result**: No substitution errors in this example

**Common Korean Substitutions**:
- 원 ↔ 완 (similar characters)
- 천 ↔ 쳔 (archaic/modern)
- Numbers: 5 ↔ S (font-dependent)

**Impact**: Usually minor; context often allows correction

---

## C.4 Error Frequency Summary

<!-- TODO: Fill with actual experimental results -->

### C.4.1 By Category

| Category | VLM Count | pdfplumber Count | Docling Count |
|----------|-----------|------------------|---------------|
| TABLE_STRUCT | TBD | TBD | TBD |
| MULTI_COL | TBD | TBD | TBD |
| HALLUC | TBD | TBD | TBD |
| HEADER_HIER | TBD | TBD | TBD |
| DELETION | TBD | TBD | TBD |
| SUBST | TBD | TBD | TBD |

### C.4.2 By Severity

| Severity | VLM | pdfplumber | Docling |
|----------|-----|------------|---------|
| Critical | TBD | TBD | TBD |
| Major | TBD | TBD | TBD |
| Medium | TBD | TBD | TBD |
| Minor | TBD | TBD | TBD |

## C.5 Key Insights

### C.5.1 VLM Strengths
1. Table structure preservation
2. Multi-column reading order
3. Scanned document handling
4. Context-aware interpretation

### C.5.2 VLM Weaknesses
1. Hallucination risk (mitigated by prompt v2)
2. Header hierarchy confusion
3. Occasional text deletion with [unclear]
4. Higher latency

### C.5.3 Traditional OCR Strengths
1. Fast processing
2. No hallucination risk
3. Deterministic output
4. Low resource requirements

### C.5.4 Traditional OCR Weaknesses
1. Complete structure loss
2. Multi-column failures
3. Scanned document limitations
4. No semantic understanding

## C.6 Recommendations

Based on error analysis:

1. **Use VLM for**: Complex tables, multi-column, scanned documents
2. **Use OCR for**: Simple digital PDFs, speed-critical applications
3. **Always use Prompt v2**: Minimizes hallucination
4. **Implement [unclear] handling**: Parse and flag uncertain regions
5. **Monitor Critical errors**: TABLE_STRUCT and MULTI_COL warrant VLM
