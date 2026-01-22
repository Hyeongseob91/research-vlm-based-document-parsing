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

## Text Similarity and Evaluation Metrics

13. **Levenshtein, V. I.** (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707-710.

14. **Morris, A. C., Maier, V., & Green, P. D.** (2004). From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition. *Proceedings of INTERSPEECH 2004*.

15. **Papineni, K., Roukos, S., Ward, T., & Zhu, W. J.** (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 311-318.

## Korean NLP

16. **Park, E., & Cho, J.** (2014). KoNLPy: Korean natural language processing in Python. *Proceedings of the 26th Annual Conference on Human & Cognitive Language Technology*, 133-136.

17. **Kudo, T., Yamamoto, K., & Matsumoto, Y.** (2004). Applying Conditional Random Fields to Japanese Morphological Analysis. *Proceedings of EMNLP 2004*, 230-237.

## Retrieval and RAG

18. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D.** (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

19. **Reimers, N., & Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982-3992.

20. **Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T.** (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 6769-6781.

## Evaluation Frameworks

21. **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S.** (2023). RAGAs: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.

22. **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P.** (2023). Lost in the Middle: How Language Models Use Long Contexts. *arXiv preprint arXiv:2307.03172*.

## Semantic Chunking

23. **LangChain.** (2024). Text Splitters Documentation. *LangChain Documentation*. https://python.langchain.com/docs/modules/data_connection/document_transformers/

24. **Gao, L., Ma, X., Lin, J., & Callan, J.** (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, 1762-1777.

## Chain-of-Thought and Prompting

25. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D.** (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

26. **Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y.** (2022). Large Language Models are Zero-Shot Reasoners. *Advances in Neural Information Processing Systems*, 35, 22199-22213.

---

## Software and Tools Used

| Tool | Version | Purpose | Reference |
|------|---------|---------|-----------|
| Python | 3.11+ | Runtime | python.org |
| pdfplumber | 0.11.0 | PDF extraction | [11] |
| jiwer | 3.0.0 | CER/WER calculation | pypi.org/project/jiwer |
| KoNLPy | 0.6.0 | Korean NLP | [16] |
| Sentence-Transformers | Latest | Embeddings | [19] |
| LangChain | Latest | Chunking | [23] |
| Streamlit | 1.45.0 | Web UI | streamlit.io |

---

*Note: arXiv papers are cited with their arXiv identifiers. Some may have been published in peer-reviewed venues subsequent to their preprint release.*
