# Abstract

Document parsing is a critical bottleneck in Retrieval-Augmented Generation (RAG) pipelines. Traditional OCR methods extract text but fail to preserve structural elements such as tables, headings, and hierarchical layouts, resulting in degraded chunking quality and retrieval accuracy. Recent Vision-Language Models (VLMs) offer direct image-to-markdown conversion but large models (30B+) are too costly for production deployment.

This paper presents **WigtnOCR**, a pseudo-labeling framework that distills document parsing capabilities from a large teacher VLM (Qwen3-VL-30B) into a compact student model (Qwen3-VL-2B) via LoRA fine-tuning. Our approach consists of four stages: (1) large-scale pseudo ground truth (GT) generation from PDF page images using the teacher VLM, (2) automated GT quality validation using a separate judge LLM (Qwen3.5-122B), (3) quality-filtered training data preparation with bias correction, and (4) parameter-efficient LoRA fine-tuning of the student model.

We construct two large-scale datasets: **KoGovDoc** (10 Korean government documents, 3,637 pages) and **ArXivPapers** (39 English academic papers, 864 pages), validated through text-based quality assessment across five dimensions. The validation pipeline achieves 74-75% acceptance rate at score threshold 3/5, with thinking-tag contamination identified as the primary failure mode (36-47% of failures).

We propose a **two-step evaluation methodology** to establish the causal chain from parsing quality to retrieval performance:
- **Step 1 (Structure → Chunking)**: Demonstrates that VLM-structured markdown produces higher-quality chunks than traditional OCR output, measured by Boundary Clarity (BC) and Chunk Stickiness (CS) across header-based, semantic, and fixed-size chunking strategies.
- **Step 2 (Chunking → Retrieval)**: Demonstrates that higher-quality chunks yield better retrieval performance, measured by Hit@K, MRR, and nDCG on auto-generated query sets.

Evaluation on **OmniDocBench** (1,355 pages, 9 document types) compares three VLM configurations (2B base, 2B LoRA, 30B teacher) against traditional baselines, establishing that pseudo-label distillation enables a 2B model to approach the parsing quality and downstream retrieval performance of models 15x larger.

---

**Keywords**: Vision-Language Models, Document Parsing, Pseudo-Labeling, LoRA Fine-tuning, RAG, Retrieval Evaluation, Semantic Chunking, Qwen-VL
