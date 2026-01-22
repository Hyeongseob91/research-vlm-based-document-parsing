# Abstract

## Research Summary

This technical report presents a comprehensive evaluation of Vision-Language Models (VLMs) for document parsing in Retrieval-Augmented Generation (RAG) pipelines. We investigate whether structured markdown output from VLMs (specifically Qwen3-VL) improves semantic chunking quality and downstream retrieval performance compared to traditional OCR methods (pdfplumber, Docling+RapidOCR).

## Problem Statement

Traditional OCR pipelines often fail to preserve document structure (tables, headers, multi-column layouts), leading to degraded semantic chunking and retrieval accuracy. This study quantifies the impact of structural preservation on RAG system performance.

## Methodology

We employ a multi-phase evaluation framework:
1. **Lexical Accuracy**: Character Error Rate (CER) and Word Error Rate (WER) with Korean morphological analysis
2. **Structural Integrity**: Boundary Score (BS) and Chunk Score (CS) metrics
3. **Retrieval Performance**: Hit Rate@k and Mean Reciprocal Rank (MRR) on generated Q&A pairs

## Key Findings

<!-- TODO: Fill in after experiments -->
- **CER/WER Results**: [Pending experimental results]
- **Retrieval Improvement**: [Pending experimental results]
- **Structural Analysis**: [Pending experimental results]

## Conclusion

<!-- TODO: Fill in after experiments -->
[Preliminary findings suggest that VLM-based parsing with structural markdown output improves retrieval accuracy by X% for documents with complex layouts. A hybrid parsing strategy is recommended.]

---

**Keywords**: Vision-Language Models, Document Parsing, RAG, Semantic Chunking, OCR, Qwen-VL

**Word Count**: ~200 words (target)
