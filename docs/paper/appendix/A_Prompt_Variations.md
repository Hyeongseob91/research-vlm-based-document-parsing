# Appendix A: Prompt Engineering Evolution

## A.1 Purpose

This appendix documents the iterative prompt engineering process for VLM-based document parsing. It serves as a practical guide for researchers working with small (2B) VLMs, demonstrating that explicit structural rules dramatically outperform implicit instructions.

## A.2 Prompt Evolution Timeline

### A.2.1 Version 1: Implicit Structure (Baseline)

**Context**: Initial attempt with generic document extraction framing.

**System Prompt**:
```
You are an expert document extraction assistant. Your task is to extract all information
from the given document image and present it in a clear, organized markdown format.

Please extract:
1. All text content
2. Tables (preserve structure)
3. Lists and enumerations
4. Headers and section titles

Format the output as clean markdown.
```

**Result**: Structure F1 = **0%** (test_3)
- Model produced text but no markdown heading markers (`#`)
- Tables extracted as plain text streams
- 2B model interpreted "organized markdown" as "neat text" without structural markup

### A.2.2 Version 2: Explicit Rules (Selected for Production)

**Context**: After observing 0% Structure F1, diagnosed root cause as model ignoring implicit structure expectations. Redesigned with mandatory rules.

**System Prompt** (English variant):
```
You are a document transcription engine. Convert the document image to structured markdown.

CRITICAL RULES:
- MUST use # symbols for ALL headings (# for title, ## for sections, ### for subsections)
- MUST use | for ALL table structures with proper header separators |---|
- NEVER output text without proper markdown structure
- Preserve ALL visible text exactly as shown

HEADING LEVEL MAPPING:
- Document title → #
- Numbered sections (1, 2, 3) → ##
- Sub-sections (1.1, 2.1) → ###
- Sub-sub-sections (1.1.1) → ####

TABLE RULES:
- Every table MUST have | header | separator |---|
- Preserve column alignment
- Keep merged cells as-is
```

**Result**: Structure F1 = **79.25%** (test_3)
- 21/24 structural elements correctly detected
- Headings, tables, lists all generated with proper markdown syntax

### A.2.3 Korean Variant

**System Prompt** (Korean documents):
```
당신은 문서 전사 엔진입니다. 문서 이미지를 구조화된 마크다운으로 변환합니다.

핵심 규칙:
- 모든 제목에 반드시 # 기호 사용 (제목 → #, 장 → ##, 절 → ###)
- 모든 표에 반드시 | 구분자와 |---| 헤더 구분선 사용
- 한국어 원문 그대로 전사 (한자, 영문 혼용 포함)
- 법령 번호 체계 유지: 조 → ##, 항 → ###, 목 → ####
```

### A.2.4 Teacher (30B) GT Generation Prompt

Separate prompt used for pseudo GT generation with the 30B teacher model. Includes additional rules for:
- Thinking tag avoidance
- Complete page coverage (no truncation)
- Language-specific formatting conventions

Full prompts available in `training/prompts/templates.py`.

## A.3 Key Lessons

| Lesson | v1 (Failed) | v2 (Succeeded) |
|--------|------------|----------------|
| Heading instruction | "use headers" | "MUST use # symbols" |
| Level mapping | Implicit | "1→##, 1.1→###, 1.1.1→####" |
| Table formatting | "preserve structure" | "MUST have \|---\| separator" |
| Prohibition | None | "NEVER output without #" |
| Model framing | "expert assistant" | "transcription engine" |

**Core insight**: 2B VLMs require **explicit, rule-based instructions** with mandatory keywords (MUST, NEVER, ALWAYS). Implicit expectations ("clean markdown", "organized format") are insufficient for small models to generate structural markup.

## A.4 Ablation Results

> **Status**: Pending — will be populated with OmniDocBench evaluation results.

| Prompt Version | Structure F1 | CER | Hallucination Rate |
|---------------|-------------|-----|-------------------|
| v1 (Implicit) | ___ | ___ | ___ |
| v2 (Explicit) | ___ | ___ | ___ |
| v2-Korean | ___ | ___ | ___ |
