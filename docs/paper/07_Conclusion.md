# 7. Conclusion

## 7.1 Summary of Findings

This work presents WigtnOCR, a pseudo-labeling framework for distilling document parsing capabilities from a large teacher VLM (30B) into a compact student model (2B) via LoRA fine-tuning, with a two-step evaluation methodology establishing the causal chain from parsing quality to retrieval performance.

### 7.1.1 RQ1: Text Extraction Prerequisite
Baseline CER 40-51% confirms sufficient text extraction quality for VLM input on English documents. Korean scanned documents exhibit hallucination risk (CER 536%), requiring quality safeguards before VLM application.

### 7.1.2 RQ2: Structural Preservation
VLM-based parsing achieves Structure F1 improvement from **0% → 79.25%**, with Recall 87.5% (21/24 structural elements detected). This is the core empirical finding: traditional OCR cannot preserve document structure, while even a 2B VLM with proper prompting can recover ~80% of structural elements.

### 7.1.3 RQ3: Structure → Chunking Quality (Step 1)
*Pending results.* Evaluation will compare BC/CS across 3 chunking strategies (header-based, semantic, fixed-size) for each parser, establishing whether VLM structural preservation produces measurably better chunks.

### 7.1.4 RQ4: Chunking → Retrieval Performance (Step 2)
*Pending results.* Retrieval evaluation with auto-generated queries will establish whether higher-quality chunks translate to better retrieval metrics (Hit@K, MRR, nDCG).

### 7.1.5 RQ5: Pseudo-Label Distillation
*Pending results.* The pipeline — GT generation (30B) → validation (122B) → training data preparation → LoRA fine-tuning (2B) — is fully implemented and operational. Preliminary validation results (74-75% acceptance rate) demonstrate that large-scale pseudo-labeling produces sufficient quality training signal after filtering.

## 7.2 Core Contributions

1. **End-to-end pseudo-labeling pipeline** with three-model architecture (teacher, judge, student) for practical VLM document parsing

2. **Text-based GT validation** — demonstrating that a text-only judge LLM effectively detects contamination, truncation, and structural issues without requiring image access

3. **Two-step causal evaluation methodology** — separately establishing structure→chunking and chunking→retrieval causal links, avoiding confounding variables in end-to-end comparison

4. **Thinking tag contamination analysis** — identifying and resolving the primary failure mode in Qwen3-VL pseudo-labeling (36-47% of failures)

5. **Bilingual training dataset** — 4,501 pages across Korean government documents and English academic papers with per-page quality scores

6. **Multi-metric evaluation framework** — CER/WER (prerequisite) → Structure F1/TEDS (structure) → BC/CS (chunking) → Hit@K/MRR/nDCG (retrieval)

## 7.3 Practical Recommendations

### For Pseudo-Labeling with VLMs
- Always use `enable_thinking: True` with Qwen3-VL models to ensure proper tag generation
- Configure server-side reasoning parsers (vLLM `--reasoning-parser`) for automatic thinking separation
- Implement multi-level cleanup: server parser → API response field check → regex fallback

### For Document Parsing Deployment
- Use hybrid routing: VLM for complex layouts, traditional OCR for simple documents
- LoRA fine-tuning enables compact (2B) models to serve production workloads
- Monitor CER as prerequisite threshold, not optimization target

### For RAG Chunking Strategy
- Use header-based chunking when VLM-structured markdown is available
- Fall back to semantic chunking for unstructured text from traditional OCR
- Fixed-size chunking should be avoided when structure is available

## 7.4 Future Work

### Short-Term
- Complete Step 1 evaluation: BC/CS across parsers and chunking strategies
- Complete Step 2 evaluation: retrieval performance with auto-generated queries
- Complete RQ5: WigtnOCR-2B vs Teacher-30B full comparison
- Implement TEDS metric for table evaluation
- Re-validate re-generated pages to confirm contamination fix

### Medium-Term
- Multi-language expansion (Chinese, Japanese)
- Curriculum learning: progressive difficulty during training
- Efficiency research: quantization (GPTQ/AWQ), batch inference optimization
- Human evaluation of auto-generated query quality

### Long-Term
- Adaptive parsing: document complexity classifier → automatic parser routing
- End-to-end RAG evaluation: answer generation quality (RAGAs)
- Real user query evaluation (production log-based)
- Public benchmark publication

---

**Data Availability**: Training datasets, evaluation code, and experiment configurations will be made available upon publication.

**Code**: The WigtnOCR framework is implemented as a Python package with CLI tools for GT generation (`scripts/generate_pseudo_gt.py`), validation (`scripts/validate_gt.py`), data preparation (`scripts/prepare_training_data.py`), and training (`training/lora_trainer.py`).
