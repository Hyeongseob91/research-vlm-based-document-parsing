# Appendix A: Prompt Variations

## A.1 Overview

This appendix documents the prompt variations tested during the VLM document parsing experiments. Each version represents a different approach to instructing the VLM for document parsing.

## A.2 Prompt Versions

### A.2.1 Version 1: Extraction Expert (Baseline)

**Approach**: Frame the model as an expert extractor

**Prompt**:
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

**Characteristics**:
- Encourages interpretation ("extract", "present", "organized")
- May lead to restructuring or summarization
- Risk of hallucination through "helpful" additions

**Expected Behavior**:
- Good structure recognition
- May add explanatory text
- Higher hallucination rate

### A.2.2 Version 2: Transcription Engine (Selected)

**Approach**: Frame the model as a faithful transcription system

**Prompt**:
```
You are a document transcription engine. Your sole purpose is to convert the given
image into markdown text format. You MUST only transcribe what is actually visible
in the image - do not add any additional information, explanations, or content that
is not present in the original document.

Rules:
1. Transcribe ALL visible text exactly as shown
2. Use markdown formatting to preserve structure (headers, lists, tables)
3. For tables, use markdown table syntax with proper alignment
4. Preserve the original language (Korean, English, etc.)
5. Do NOT add explanations, summaries, or interpretations
6. Do NOT add information that is not visible in the image
7. If text is unclear, indicate with [unclear] rather than guessing
8. Maintain original paragraph breaks and spacing intent
```

**Characteristics**:
- Emphasizes faithful reproduction
- Explicit "do not add" instructions
- [unclear] markers for uncertainty
- Minimal interpretation allowed

**Expected Behavior**:
- Lower hallucination rate
- Faithful text reproduction
- Clear uncertainty markers

### A.2.3 Version 3: Minimal Instruction

**Approach**: Minimal prompt to establish baseline VLM behavior

**Prompt**:
```
Convert this document image to markdown format.
```

**Characteristics**:
- No specific instructions
- Relies on model's default behavior
- Useful for understanding base capabilities

**Expected Behavior**:
- Model's natural interpretation
- Variable quality
- Baseline for prompt engineering evaluation

### A.2.4 Version 4: XML Structured Output

**Approach**: Use explicit XML structure for output organization

**Prompt**:
```
Transcribe the document image into structured XML format.

Output structure:
<document>
  <metadata>
    <language>[detected language]</language>
    <page_type>[table-heavy|text-heavy|mixed]</page_type>
  </metadata>
  <content>
    <section level="[1-6]" title="[header text if any]">
      <text>[paragraph content]</text>
      <table>
        <header>[col1|col2|...]</header>
        <row>[cell1|cell2|...]</row>
      </table>
      <list type="[ordered|unordered]">
        <item>[list item]</item>
      </list>
    </section>
  </content>
</document>

Rules:
- Only include what is visible in the image
- Use [unclear] for text that cannot be determined
- Preserve original language
```

**Characteristics**:
- Highly structured output
- Explicit element types
- Easier programmatic parsing
- More verbose output

**Expected Behavior**:
- Consistent structure
- Explicit element classification
- Longer processing time
- May force structure where none exists

### A.2.5 Version 5: Korean-Specific (Experimental)

**Approach**: Optimized for Korean document characteristics

**Prompt**:
```
당신은 한국어 문서 전사 시스템입니다. 주어진 이미지의 모든 텍스트를 마크다운 형식으로
정확하게 전사해 주세요.

규칙:
1. 보이는 모든 텍스트를 그대로 전사
2. 표는 마크다운 표 문법 사용
3. 제목은 # 헤더 문법 사용
4. 원본에 없는 내용 추가 금지
5. 불확실한 텍스트는 [불명확] 표시
6. 한자, 영어 등 원문 그대로 유지
```

**Characteristics**:
- Native Korean instructions
- May improve Korean text handling
- Cultural/formatting awareness

**Expected Behavior**:
- Better Korean honorific handling
- Improved Korean table formatting
- Potential improvement in Korean-specific layouts

## A.3 Comparison Matrix

| Version | Hallucination Risk | Structure Quality | Speed | Recommended Use |
|---------|-------------------|-------------------|-------|-----------------|
| v1 (Extraction) | High | Good | Fast | Not recommended |
| v2 (Transcription) | Low | Good | Medium | **Primary choice** |
| v3 (Minimal) | Medium | Variable | Fastest | Baseline only |
| v4 (XML) | Low | Excellent | Slow | Programmatic parsing |
| v5 (Korean) | Low | Good | Medium | Korean documents |

## A.4 Ablation Study Results

<!-- TODO: Fill with actual experimental results -->

### A.4.1 CER by Prompt Version

| Document | v1 | v2 | v3 | v4 | v5 |
|----------|-----|-----|-----|-----|-----|
| test_1 | TBD% | TBD% | TBD% | TBD% | TBD% |
| test_2 | TBD% | TBD% | TBD% | TBD% | TBD% |
| test_3 | TBD% | TBD% | TBD% | TBD% | TBD% |

### A.4.2 Hallucination Rate by Prompt Version

| Document | v1 | v2 | v3 | v4 | v5 |
|----------|-----|-----|-----|-----|-----|
| test_1 | TBD% | TBD% | TBD% | TBD% | TBD% |
| test_2 | TBD% | TBD% | TBD% | TBD% | TBD% |
| test_3 | TBD% | TBD% | TBD% | TBD% | TBD% |

**Hallucination Definition**: Text present in output but not in ground truth (excluding formatting)

### A.4.3 Processing Time by Prompt Version

| Document | v1 | v2 | v3 | v4 | v5 |
|----------|------|------|------|------|------|
| test_1 | TBD s | TBD s | TBD s | TBD s | TBD s |
| test_2 | TBD s | TBD s | TBD s | TBD s | TBD s |
| test_3 | TBD s | TBD s | TBD s | TBD s | TBD s |

## A.5 Recommendations

### A.5.1 General Use

Use **Version 2 (Transcription Engine)** for most applications:
- Best balance of accuracy and hallucination prevention
- Clear uncertainty handling with [unclear] markers
- Appropriate for quality-sensitive applications

### A.5.2 Special Cases

| Scenario | Recommended Version | Rationale |
|----------|-------------------|-----------|
| Quick baseline | v3 (Minimal) | Fastest, acceptable quality |
| Korean documents | v5 (Korean) | Native language optimization |
| Programmatic processing | v4 (XML) | Structured parsing |
| Speed critical | v3 (Minimal) | Minimal overhead |
| Quality critical | v2 (Transcription) | Best accuracy/hallucination balance |

## A.6 Prompt Engineering Lessons

1. **Explicit Negation**: "Do NOT add" is more effective than implicit expectations
2. **Role Framing**: "Transcription engine" induces more literal behavior than "expert"
3. **Uncertainty Handling**: Providing [unclear] option reduces hallucination
4. **Structure Guidance**: Explicit markdown/XML instructions improve consistency
5. **Language Matching**: Native language prompts may improve results for non-English documents
