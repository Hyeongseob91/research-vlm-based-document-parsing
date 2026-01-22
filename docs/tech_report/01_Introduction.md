# 1. Introduction

## 1.1 Problem Definition

Retrieval-Augmented Generation (RAG) systems rely on accurate document parsing to create meaningful text chunks for semantic search. However, traditional Optical Character Recognition (OCR) methods often fail to preserve critical structural elements:

- **Table structures** collapse into unreadable text streams
- **Multi-column layouts** produce incorrect reading order
- **Header hierarchies** lose semantic relationships
- **Lists and enumerations** merge into continuous paragraphs

These structural failures propagate through the RAG pipeline, degrading both chunking quality and retrieval accuracy.

## 1.2 Research Questions

This study addresses three primary research questions:

### RQ1: Lexical Fidelity
> Does VLM-based parsing achieve better character-level and word-level accuracy compared to traditional OCR methods?

**Metrics**: CER (Character Error Rate), WER (Word Error Rate)

### RQ2: Structural Preservation
> Does VLM-based parsing preserve document structure better, leading to improved semantic chunking?

**Metrics**: Boundary Score (BS), Chunk Score (CS)

### RQ3: Retrieval Impact
> Does structural preservation in parsing improve downstream retrieval performance in RAG systems?

**Metrics**: Hit Rate@k, MRR (Mean Reciprocal Rank)

## 1.3 Core Hypothesis

> "Structured parsing via Vision-Language Models produces better semantic chunks than traditional OCR, resulting in measurable improvements in retrieval accuracy."

This hypothesis posits that even when using identical chunking algorithms, the input quality (structured markdown vs. plain text) significantly affects output quality.

## 1.4 Contributions

This work makes the following contributions:

1. **Evaluation Framework**: A comprehensive multi-metric framework for comparing document parsers, including:
   - Lexical metrics (CER, WER) with Korean morphological analysis
   - Structural metrics (Boundary Score, Chunk Score)
   - Retrieval metrics (Hit Rate, MRR)

2. **Empirical Analysis**: Quantitative comparison of three parsing approaches:
   - VLM (Qwen3-VL) with structured markdown output
   - pdfplumber (digital PDF text extraction)
   - Docling + RapidOCR (scanned document OCR)

3. **Hybrid Strategy**: Data-driven recommendations for when to use VLM vs. traditional OCR based on document characteristics

4. **Error Taxonomy**: Categorized analysis of parsing failures with case studies

5. **Reproducible Benchmark**: Open-source evaluation toolkit with ground truth datasets

## 1.5 Scope and Limitations

### In Scope
- Korean and English documents
- Digital PDFs and scanned documents
- Tables, multi-column layouts, headers, lists
- RAG retrieval evaluation (not full end-to-end answer generation)

### Out of Scope
- Handwritten text recognition
- Complex diagrams and charts
- Real-time streaming applications
- Production deployment optimization

## 1.6 Document Structure

The remainder of this report is organized as follows:

- **Section 2**: Related work in document parsing and VLM applications
- **Section 3**: Methodology and evaluation framework
- **Section 4**: Experimental setup and dataset description
- **Section 5**: Results and analysis
- **Section 6**: Discussion and implications
- **Section 7**: Conclusion and future work
- **Section 8**: References
