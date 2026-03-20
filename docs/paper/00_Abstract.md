# Abstract

Document parsing is a critical bottleneck in Retrieval-Augmented Generation (RAG) pipelines. Traditional OCR methods extract text but fail to preserve structural elements such as tables, headings, and hierarchical layouts, resulting in degraded chunking quality and retrieval accuracy. Recent Vision-Language Models (VLMs) offer direct image-to-markdown conversion but large models (30B+) are too costly for production deployment.

This paper presents **WigtnOCR**, a pseudo-labeling framework that distills document parsing capabilities from a large teacher VLM (Qwen3-VL-30B) into a compact student model (Qwen3-VL-2B) via LoRA fine-tuning. Our approach consists of four stages: (1) large-scale pseudo ground truth (GT) generation from PDF page images using the teacher VLM, (2) automated GT quality validation using a separate judge LLM (Qwen3.5-122B), (3) quality-filtered training data preparation with bias correction, and (4) parameter-efficient LoRA fine-tuning of the student model.

We construct two large-scale datasets: **KoGovDoc** (10 Korean government documents, 3,637 pages) and **ArXivPapers** (39 English academic papers, 864 pages), validated through text-based quality assessment across five dimensions. The validation pipeline achieves 74-75% acceptance rate at score threshold 3/5, with thinking-tag contamination identified as the primary failure mode (36-47% of failures).

We propose a **two-step causal evaluation methodology**:
- **OmniDocBench** (1,355 pages): Standardized parsing quality evaluation — WigtnOCR-2B matches or exceeds the 30B teacher in 4/5 metric categories (text NED 0.288 vs 0.289, table TEDS 0.649 vs 0.523 **+12.6pp**, reading order NED 0.211 vs 0.227). Compared to Marker, WigtnOCR-2B achieves superior table and formula recognition.
- **KoGovDoc** (294 val samples): Two-step causal chain evaluation — (Step 1) VLM-structured markdown produces higher-quality chunks than unstructured text, measured by Boundary Clarity (BC) and Chunk Stickiness (CS); (Step 2) higher-quality chunks yield better retrieval performance, measured by Hit@K, MRR, and nDCG.

A surprising finding is that the student outperforms the teacher in 4 of 5 categories, which we attribute to the quality filtering effect: by training only on pseudo-GT with validation score ≥ 3, the student learns from the teacher's best outputs while avoiding its failure cases.

---

**Keywords**: Vision-Language Models, Document Parsing, Knowledge Distillation, Pseudo-Labeling, LoRA Fine-tuning, RAG, Semantic Chunking, OmniDocBench, Qwen-VL
