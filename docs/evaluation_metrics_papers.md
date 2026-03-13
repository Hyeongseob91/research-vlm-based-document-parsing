# WigtnOCR Evaluation Framework - Reference Papers

WigtnOCR 평가 프레임워크 설계 시 참조해야 할 논문 목록.
Claude / Gemini 학술 검토 보고서 기반으로 정리.

---

## 1. Document Parsing Benchmarks

### OmniDocBench (Primary Benchmark)

- **Title**: OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations
- **Authors**: Ouyang et al.
- **Venue**: CVPR 2025
- **arXiv**: [2412.07626](https://arxiv.org/abs/2412.07626)
- **Code**: https://github.com/opendatalab/OmniDocBench
- **Dataset**: https://huggingface.co/datasets/opendatalab/OmniDocBench
- **Key**: 1,355 PDF pages, 9 document types, 4 layout types, 3 languages. 15 block-level + 4 span-level element annotations (20k+ / 80k+). 공식 메트릭: NED (text), TEDS (table), Edit Distance (formula), NED (reading order). `end2end` 평가 방식 권장.
- **Relevance**: WigtnOCR의 primary benchmark. 공식 eval 코드를 직접 사용하여 재현성 확보 필요.

### Upstage dp-bench (Supplementary Benchmark)

- **Title**: dp-bench Document Parsing Benchmark
- **Authors**: Upstage AI
- **Dataset**: https://huggingface.co/datasets/upstage/dp-bench
- **Key**: 200 samples (Library of Congress 90 + OER 90 + Internal 20), 12 element types. **NID (Normalized Indel Distance)** 제안 — Edit Distance에서 substitution 제외, 길이 차이에 더 민감. Table/Figure/Chart 제외하고 텍스트 요소에 집중.
- **Relevance**: NID 메트릭은 OmniDocBench NED의 보완적 메트릭으로 활용 가능.

---

## 2. Table Evaluation Metrics

### TEDS (Primary - Table Metric)

- **Title**: Image-based Table Recognition: Data, Model, and Evaluation
- **Authors**: Zhong et al. (IBM Research)
- **arXiv**: [1911.10683](https://arxiv.org/abs/1911.10683)
- **Code**: https://github.com/ibm-aur-nlp/PubTabNet
- **Key**: Tree-Edit-Distance-based Similarity. HTML 트리 구조 비교. PubTabNet 데이터셋(568k table images) 함께 제안. Score 0-1 (1 = perfect). **TEDS-S** (Structure Only) 변형은 셀 내용 제외, 구조만 평가.
- **Relevance**: OmniDocBench 공식 테이블 메트릭. TEDS + TEDS-S 둘 다 리포트 권장.

### GriTS (Supplementary - Table Metric)

- **Title**: GriTS: Grid Table Similarity Metric for Table Structure Recognition
- **Authors**: Smock et al. (Microsoft)
- **Venue**: ICDAR 2023
- **arXiv**: [2203.12555](https://arxiv.org/abs/2203.12555)
- **Code**: https://github.com/microsoft/table-transformer
- **Key**: 테이블을 행렬(matrix)로 직접 비교. Topology(행/열 span), Content(셀 내용), Location(픽셀 좌표) 3수준 평가. 2D-LCS 문제를 2D-MSS로 일반화. TEDS보다 셀 병합/구조 정렬에서 더 정밀.
- **Relevance**: TEDS 대안으로 병행 사용 시 contribution 명확화.

---

## 3. Chunking Quality Metrics

### MoC - BC/CS Metrics (Primary - Chunking)

- **Title**: MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System
- **Authors**: Zhao et al.
- **Venue**: ACL 2025 (63rd Annual Meeting)
- **arXiv**: [2503.09600](https://arxiv.org/abs/2503.09600)
- **ACL**: https://aclanthology.org/2025.acl-long.258/
- **Key**: **BC (Boundary Clarity)** — 청크 경계의 분리 능력 (Perplexity 기반). **CS (Chunk Stickiness)** — 청크 내부 응집력/논리적 독립성. BC 높을수록, CS 낮을수록 좋음. Granularity-aware Mixture-of-Chunkers 프레임워크 제안.
- **Relevance**: 청킹 품질을 직접 정량화하는 유일한 peer-reviewed 메트릭. ACL 2025 게재로 학술적 근거 확보. 다만 2025.03 발표로 매우 신규 — "novel metric 채택" framing 권장.

---

## 4. RAG Evaluation Frameworks

### RAGAS (RAG Evaluation Standard)

- **Title**: RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **Authors**: Es et al.
- **Venue**: EACL 2024 (System Demonstrations)
- **arXiv**: [2309.15217](https://arxiv.org/abs/2309.15217)
- **ACL**: https://aclanthology.org/2024.eacl-demo.16/
- **Code**: https://github.com/explodinggradients/ragas
- **Key**: Reference-free RAG 평가. Faithfulness, Answer Relevance, Context Relevance. LangChain / LlamaIndex 통합 지원.
- **Relevance**: End-to-End RAGAS 메트릭으로 "파싱 품질이 최종 RAG 성능에 미치는 영향" 검증 가능.

### ARES (Automated RAG Evaluation)

- **Title**: ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
- **Authors**: Saad-Falcon, Khattab, Potts, Zaharia (Stanford)
- **arXiv**: [2311.09476](https://arxiv.org/abs/2311.09476)
- **Code**: https://github.com/stanford-futuredata/ARES
- **Key**: Lightweight LM judges + PPI (Prediction-Powered Inference). 수백 개 human annotation으로도 평가 가능. KILT, SuperGLUE, AIS 8개 태스크에서 검증.
- **Relevance**: LLM-as-Judge 방식의 학술적 근거. GT에서 LLM으로 쿼리 생성하는 방식의 타당성 지지.

---

## 5. Retrieval Evaluation

### BEIR (IR Benchmark Standard)

- **Title**: BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models
- **Authors**: Thakur, Reimers, Ruckle, Srivastava, Gurevych
- **Venue**: NeurIPS 2021 (Datasets and Benchmarks Track)
- **arXiv**: [2104.08663](https://arxiv.org/abs/2104.08663)
- **Code**: https://github.com/beir-cellar/beir
- **Key**: 18개 공개 데이터셋, zero-shot IR 평가 de facto standard. **nDCG@10** primary metric. Lexical, sparse, dense, late-interaction, re-ranking 10개 시스템 평가.
- **Relevance**: nDCG@10을 primary retrieval metric으로 설정하는 근거. WigtnOCR retrieval 평가 시 BEIR 프로토콜 참조.

### HyDE (Query Generation Method)

- **Title**: Precise Zero-Shot Dense Retrieval without Relevance Labels
- **Authors**: Gao, Ma, Lin, Callan (CMU)
- **Venue**: ACL 2023
- **arXiv**: [2212.10496](https://arxiv.org/abs/2212.10496)
- **Code**: https://github.com/texttron/hyde
- **Key**: Hypothetical Document Embeddings. Query → LLM이 가상 문서 생성 → 임베딩 → 유사 실제 문서 검색. Zero-shot에서 fine-tuned retriever 수준 성능.
- **Relevance**: SoundMind Analysis Platform의 HyDE Query Enhancement 구현 근거. 쿼리 자동 생성 방식의 학술적 타당성 지지.

---

## 6. Document AI / Layout Analysis

### LayoutLMv3

- **Title**: LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
- **Authors**: Huang, Lv, Cui, Lu, Wei (Microsoft)
- **Venue**: ACM Multimedia 2022
- **arXiv**: [2204.08387](https://arxiv.org/abs/2204.08387)
- **Model**: https://huggingface.co/microsoft/layoutlmv3-base
- **Key**: Text + Image + Layout 통합 사전학습. MLM + MIM + WPA (Word-Patch Alignment). CNN 없이 이미지 임베딩 — 파라미터 절약. Document AI text-centric / image-centric 태스크 모두 지원.
- **Relevance**: Reading Order Similarity 평가 시 참조. PyMuPDF 대비 VLM의 읽기 순서 우위 증명에 활용.

---

## 7. Supplementary References

### RAGBench

- **Title**: RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems
- **arXiv**: [2407.11005](https://arxiv.org/abs/2407.11005)
- **Key**: 100k examples, industry corpora (user manuals 등). Explainable RAG 평가.

### PubTabNet (TEDS Dataset)

- **Title**: Image-based Table Recognition: Data, Model, and Evaluation
- **arXiv**: [1911.10683](https://arxiv.org/abs/1911.10683)
- **Dataset**: https://github.com/ibm-aur-nlp/PubTabNet
- **Key**: 568k table images with HTML GT. TEDS 메트릭 원논문.

---

## Quick Reference: Metric → Paper Mapping

| Metric | Paper | Venue |
|--------|-------|-------|
| NED (Normalized Edit Distance) | OmniDocBench | CVPR 2025 |
| TEDS / TEDS-S | PubTabNet (Zhong et al.) | arXiv 2019 |
| GriTS | Smock et al. | ICDAR 2023 |
| NID (Normalized Indel Distance) | Upstage dp-bench | HuggingFace 2024 |
| BC (Boundary Clarity) | MoC (Zhao et al.) | ACL 2025 |
| CS (Chunk Stickiness) | MoC (Zhao et al.) | ACL 2025 |
| nDCG@10, MRR, Hit@K | BEIR (Thakur et al.) | NeurIPS 2021 |
| Faithfulness, Answer Relevance | RAGAS (Es et al.) | EACL 2024 |
| Context Relevance, Recall | ARES (Saad-Falcon et al.) | arXiv 2023 |
| Reading Order Similarity | LayoutLMv3 (Huang et al.) | ACM MM 2022 |
| HyDE (Query Enhancement) | Gao et al. | ACL 2023 |

---

## Claude vs Gemini 검토 보고서 교차 비교

| 항목 | Claude 권장 | Gemini 권장 | 합의 |
|------|-----------|-----------|------|
| Text 메트릭 | NED(primary) + BLEU + METEOR + CER/WER(보조) | NED + F1-Score (Element-wise) | NED primary 합의 |
| Table 메트릭 | TEDS + TEDS-S | TEDS + GriTS 병행 | TEDS 필수, TEDS-S + GriTS 둘 다 추가 권장 |
| Structure 평가 | Reading Order NED + NID | Tree Edit Distance 또는 Layout-aware F1 | Structure F1(마크다운 카운트) 방식은 학술적으로 약함 — 둘 다 대안 제시 |
| Chunking 메트릭 | BC/CS (MoC) + Recall@K | BC/CS + LLM-based Semantic Integrity | BC/CS 합의, 보조 메트릭은 다름 |
| Retrieval 메트릭 | nDCG@10(primary) + Recall@K 추가 | nDCG@10 + RR + MAP + Cross-Encoder Reranking | nDCG@10 primary 합의 |
| 추가 권장 | 문서 유형별 subset 분석, Context Precision/Recall | Reading Order Similarity, Token Efficiency, End-to-End RAGAS | 상호 보완적 |
