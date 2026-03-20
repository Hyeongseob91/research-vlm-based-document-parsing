# 8. References

## Document Understanding and Layout Analysis

1. **Smith, R.** (2007). An Overview of the Tesseract OCR Engine. *Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)*, 629-633.

2. **Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M.** (2020). LayoutLM: Pre-training of Text and Layout for Document Image Understanding. *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1192-1200.

3. **Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., ... & Zhou, M.** (2021). LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics*, 2579-2591.

4. **Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F.** (2022). LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. *Proceedings of the 30th ACM International Conference on Multimedia*, 4083-4091.

## Vision-Language Models

5. **Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., ... & Zhou, J.** (2023). Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. *arXiv preprint arXiv:2308.12966*.

6. **Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., ... & Zhou, J.** (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. *arXiv preprint arXiv:2409.12191*.

7. **OpenAI.** (2023). GPT-4V(ision) System Card. *OpenAI Technical Report*.

8. **Anthropic.** (2024). Claude 3 Model Card and System Prompt. *Anthropic Technical Documentation*.

9. **Team Gemini.** (2024). Gemini: A Family of Highly Capable Multimodal Models. *arXiv preprint arXiv:2312.11805*.

## Document Parsing and OCR

10. **Blecher, L., Cucurull, G., Scialom, T., & Stojnic, R.** (2023). Nougat: Neural Optical Understanding for Academic Documents. *arXiv preprint arXiv:2308.13418*.

11. **Singer-Vine, J.** (2022). pdfplumber: Plumb a PDF for detailed information about each text character, rectangle, line, et cetera. *GitHub Repository*. https://github.com/jsvine/pdfplumber

12. **IBM Research.** (2024). Docling: Document Processing Pipeline. *GitHub Repository*. https://github.com/DS4SD/docling

## Knowledge Distillation

13. **Hinton, G., Vinyals, O., & Dean, J.** (2015). Distilling the Knowledge in Neural Networks. *arXiv preprint arXiv:1503.02531*.

14. **Lee, D. H.** (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. *ICML 2013 Workshop on Challenges in Representation Learning*.

## Evaluation Metrics

15. **Levenshtein, V. I.** (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707-710.

16. **Zhong, X., ShafieiBavani, E., & Jimeno Yepes, A.** (2020). Image-based table recognition: data, model, and evaluation. *arXiv preprint arXiv:2011.13534*. (TEDS metric)

17. **Smock, B., Pesala, R., & Abraham, R.** (2022). GriTS: Grid table similarity metric for table structure recognition. *arXiv preprint arXiv:2203.12555*.

## OmniDocBench

18. **OmniDocBench Authors.** (2025). OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations. *CVPR 2025*.

## Chunking and Retrieval Evaluation

19. **Zhao, Z., et al.** (2025). Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System. *ACL 2025*. (MoC framework — BC/CS metrics)

20. **Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I.** (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021*.

## Chain-of-Thought and Prompting

21. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D.** (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

## Korean NLP

22. **Park, E., & Cho, J.** (2014). KoNLPy: Korean natural language processing in Python. *Proceedings of the 26th Annual Conference on Human & Cognitive Language Technology*, 133-136.

---

## Software and Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| vLLM | 0.13.0 | VLM serving |
| ms-swift | 4.0.1 | LoRA training |
| PyMuPDF | Latest | PDF rendering |
| jiwer | 3.0.0 | Edit distance |
| httpx | Latest | API client |
| BGE-M3 | Latest | Multilingual embeddings (chunking/retrieval) |
| FAISS | Latest | Vector similarity search |
| Sentence-Transformers | Latest | Embedding inference |

---

*Note: arXiv papers are cited with their arXiv identifiers. Some may have been published in peer-reviewed venues subsequent to their preprint release.*
