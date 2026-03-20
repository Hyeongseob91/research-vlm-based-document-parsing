# 7. Conclusion

## 7.1 Summary of Findings

This work presents WigtnOCR, a pseudo-labeling framework for distilling document parsing capabilities from a large teacher VLM (30B) into a compact student model (2B) via LoRA fine-tuning.

### 7.1.1 RQ1: Distillation Effectiveness

Evaluation on OmniDocBench (1,276 pages evaluated) confirms that pseudo-label distillation successfully transfers document parsing capabilities:

- **Text**: NED improves from 0.364 (base) to 0.288 (trained), a 20.9% improvement
- **Tables**: TEDS improves from 0.561 to 0.649, a 15.7% improvement
- **Formulas**: CDM F1 improves from 0.865 to 0.884, CDM ExpRate from 0.504 to 0.600
- **Reading Order**: NED improves from 0.300 to 0.211, a 29.7% improvement
- **Robustness**: Skip rate drops from 18.8% to 5.8%, a 69.1% reduction

### 7.1.2 RQ2: Quality Filtering Enables Student to Surpass Teacher

The student outperforms the 30B teacher in 4 of 5 metric categories:
- **Text**: WigtnOCR-2B (NED 0.288) matches the teacher (0.289)
- **Tables**: WigtnOCR-2B (TEDS 0.649) surpasses the teacher (0.523) by **12.6pp**
- **Reading Order**: WigtnOCR-2B (NED 0.211) exceeds the teacher (0.227)
- **Formulas**: CDM F1 gap remains (0.884 vs 0.939)

Quality-filtered pseudo-labels (score >= 3) provide a stronger training signal than the teacher's average output — the student learns from the teacher's best work.

### 7.1.3 RQ3: Cost-Quality Trade-off

WigtnOCR-2B surpasses Marker (OCR pipeline) on tables (TEDS 0.649 vs 0.586) and formulas (CDM F1 0.884 vs 0.863), while Marker leads on text extraction (NED 0.218 vs 0.288) and reading order (0.165 vs 0.211). VLMs and OCR pipelines offer complementary strengths.

## 7.2 Core Contributions

1. **End-to-end pseudo-labeling pipeline** with three-model architecture (teacher, judge, student) for practical VLM document parsing, demonstrated to produce a student that matches or exceeds the teacher in 4/5 evaluation categories

2. **Empirical evidence that quality-filtered distillation can surpass the teacher**: WigtnOCR-2B achieves TEDS 0.649 vs teacher's 0.523 on tables, suggesting curated pseudo-labels provide stronger training signal than the teacher's average output

3. **Text-based GT validation** — demonstrating that a text-only judge LLM effectively detects contamination, truncation, and structural issues without requiring image access

4. **Thinking tag contamination analysis** — identifying and resolving the primary failure mode in Qwen3-VL pseudo-labeling (36-47% of failures), with practical guidelines

5. **Bilingual training dataset** — 4,501 pages across Korean government documents and English academic papers with per-page quality scores

6. **Two-step causal evaluation** — separately establishing structure→chunking (BC/CS) and chunking→retrieval (Hit@K/MRR) causal links on KoGovDoc

## 7.3 Practical Recommendations

### For Pseudo-Labeling with VLMs
- Always use `enable_thinking: True` with Qwen3-VL models to ensure proper tag generation
- Configure server-side reasoning parsers (vLLM `--reasoning-parser`) for automatic thinking separation
- Implement multi-level cleanup: server parser → API response field check → regex fallback

### For Document Parsing Deployment
- LoRA fine-tuning enables compact (2B) models to serve production workloads with quality comparable to 30B models
- Monitor skip rate as a key production reliability metric
- Consider hybrid VLM + OCR pipeline for optimal coverage across element types

### For RAG Chunking Strategy
- Use header-based chunking when VLM-structured markdown is available
- Fall back to semantic chunking for unstructured text from traditional OCR
- Fixed-size chunking should be avoided when structure is available

## 7.4 Future Work

### Short-Term
- Language-stratified evaluation on OmniDocBench (EN/ZH separate) for direct comparison with published results
- Higher LoRA rank (16, 32) experiments to close the formula recognition gap
- Re-validate re-generated pages to confirm contamination fix

### Medium-Term
- Multi-language expansion (Chinese, Japanese)
- Curriculum learning: progressive difficulty during training
- Efficiency research: quantization (GPTQ/AWQ), batch inference optimization
- Domain-specific evaluation (financial, medical, legal documents)

### Long-Term
- Adaptive parsing: document complexity classifier → automatic parser routing
- End-to-end RAG evaluation: answer generation quality (RAGAs)
- Real user query evaluation (production log-based)
- Public benchmark publication

---

**Data Availability**: Training datasets, evaluation code, and experiment configurations will be made available upon publication.

**Code**: The WigtnOCR framework is implemented as a Python package with CLI tools for GT generation (`scripts/generate_pseudo_gt.py`), validation (`scripts/validate_gt.py`), data preparation (`scripts/prepare_training_data.py`), and training (`training/lora_trainer.py`).
