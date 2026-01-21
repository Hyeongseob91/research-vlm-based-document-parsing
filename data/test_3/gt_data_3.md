# [Ground Truth] Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval

## 1. 문서 메타데이터 (Document Metadata)
* [cite_start]**제목**: Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval [cite: 67]
* [cite_start]**저자**: Joaquín Polonuer, Lucas Vittor, Iñaki Arango, Ayush Noori, David A. Clifton, Luciano Del Corro, Marinka Zitnik [cite: 68]
* [cite_start]**소속**: Harvard Medical School, University of Oxford, Universidad de Buenos Aires 등 [cite: 69, 70]
* [cite_start]**발행 정보**: arXiv:2601.13969v1 [cs.AI], 2026년 1월 20일 발행 [cite: 66]
* [cite_start]**교신 저자**: delcorrol@udesa.edu.ar, marinka@hms.harvard.edu [cite: 73]

## 2. 초록 (Abstract)
* [cite_start]**연구 배경**: 지식 그래프(KG) 검색에서 광범위한 탐색(Breadth)과 다단계 관계 추적(Depth) 사이의 균형을 맞추는 것이 핵심 과제임[cite: 75].
* [cite_start]**제안 방법**: ARK(ADAPTIVE RETRIEVER OF KNOWLEDGE)라는 에이전틱 KG 리트리버를 도입함[cite: 77].
* [cite_start]**주요 기능**: 전역 어휘 검색(Global Lexical Search)과 인접 노드 탐색(One-hop Neighborhood Exploration)의 두 가지 도구를 사용하여 탐색을 수행함[cite: 77].
* [cite_start]**주요 성과**: STaRK 벤치마크에서 평균 Hit@1 59.1%, 평균 MRR 67.4를 달성하여 기존 방식 대비 Hit@1 기준 최대 31.4% 향상됨[cite: 80].

## 3. 핵심 방법론 (Methodology)

### 3.1 검색 도구 (Tools)
* [cite_start]**Global Search (Search(q, k))**: 그래프 전체 노드 중 텍스트 속성 $d_V(u)$에 대해 BM25 유사도 $rel(q, d_V(u))$가 가장 높은 상위 $k$개 노드를 반환함[cite: 199, 201].
* [cite_start]**Neighborhood Exploration (Neighbors(v, q, F))**: 특정 노드 $v$의 인접 노드 중 타입 필터 $F$를 만족하고 쿼리 $q$와 유사한 상위 $k$개 노드를 반환함[cite: 203, 210].
* **수식 정의**:
  $$N_F(v) := \{u \in N(v) | [cite_start]\phi_V(u) \in F_V, \phi_E(\{u, v\}) \in F_E\}$$ [cite: 206]

### 3.2 병렬 탐색 및 증류 (Parallel Exploration & Distillation)
* [cite_start]**병렬화**: $n$개의 독립적인 에이전트를 동시에 실행하고 투표 수(Vote count) 기반의 순위 융합 규칙을 적용하여 견고성을 높임[cite: 213, 218].
* [cite_start]**모델 증류**: 교사 모델(GPT-4.1)의 도구 사용 궤적을 8B 모델(Qwen3-8B)에 모방 학습(Imitation Learning)시켜 성능 손실을 최소화하며 비용을 절감함[cite: 221, 229, 255].

## 4. 실험 결과 (Experimental Results)

### 4.1 데이터셋 통계 (Table 4)
| Dataset | Entity types | Relation types | Entities | Relations | Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AMAZON** | 4 | 5 | 1,035,542 | 9,443,802 | 592,067,882 |
| **MAG** | 4 | 4 | 1,872,968 | 39,802,116 | 212,602,571 |
| **PRIME** | 10 | 18 | 129,375 | 8,100,498 | 31,844,769 |
[cite_start][cite: 743]

### 4.2 STaRK 테스트 셋 성능 비교 (Table 1 - 요약)
| Category | Method | AMAZON (Hit@1) | MAG (Hit@1) | PRIME (Hit@1) | Avg (Hit@1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Retrieval-based** | BM25 | 44.94 | 25.85 | 12.75 | 27.85 |
| **Training-free** | KAR | 54.20 | 50.47 | 30.35 | 45.01 |
| **Agent-based** | Think-on-Graph | 20.67 | 23.33 | 16.67 | 20.22 |
| **Proposed** | **ARK (GPT-4.1)** | **55.82** | **73.40** | **48.20** | **59.14** |
| **Distilled** | ARK distilled (8B) | 54.99 | 61.66 | 31.87 | 49.51 |
[cite_start][cite: 257]

### 4.3 도구 세트 설계의 영향 (Table 2)
* [cite_start]**Full (전체 기능)**: MAG 기준 Hit@1 79.2 [cite: 310]
* [cite_start]**w/o Neighbors (이웃 탐색 제거)**: MAG 기준 Hit@1 30.5 (가장 큰 하락폭) [cite: 310]
* [cite_start]**Neighbors w/o q (쿼리 순위화 제거)**: MAG 기준 Hit@1 72.1 [cite: 310]

## 5. 지식 그래프 탐색 에이전트 시스템 프롬프트 (Appendix A.3)
에이전트는 다음과 같은 툴을 사용하여 탐색을 수행함:
1. [cite_start]**search_in_graph**: 초기 광범위 검색용[cite: 766].
2. [cite_start]**search_in_neighborhood**: 1-hop 인접 노드 탐색 및 관계 확인용[cite: 772].
3. [cite_start]**add_to_answer**: 관련 노드를 근거와 함께 정답 리스트에 추가[cite: 779].
4. [cite_start]**finish**: 탐색 완료 선언[cite: 784].

---
**출처**: Polonuer et al., "Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval", arXiv:2601.13969v1, 2026.