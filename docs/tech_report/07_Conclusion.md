# 7. Conclusion

## 7.1 Summary of Findings

This study investigated the impact of Vision-Language Model (VLM) based document parsing on semantic chunking and retrieval performance in RAG systems. Our key findings are:

### 7.1.1 Lexical Accuracy (RQ1)

<!-- TODO: Update with final results -->
- VLM parsing achieved [TBD]% lower CER compared to traditional OCR
- For scanned documents, VLM demonstrated [TBD]% improvement in WER
- Trade-off exists: VLM may introduce minor errors but preserves structure

### 7.1.2 Structural Preservation (RQ2)

- VLM output shows [TBD] higher Boundary Score
- Chunk coherence (CS) improved by [TBD] with VLM parsing
- Markdown formatting provides implicit structural cues even with structure-agnostic chunking

### 7.1.3 Retrieval Performance (RQ3)

- Hit Rate@5 improved by [TBD]% with VLM-based parsing
- Largest improvements observed for table-related queries
- MRR increased from [TBD] to [TBD]

## 7.2 Core Contributions

1. **Multi-Metric Evaluation Framework**: A comprehensive framework measuring lexical, structural, and retrieval performance for document parsers

2. **Empirical Validation**: Quantitative evidence that structural preservation in parsing improves downstream RAG performance

3. **Prompt Engineering Insights**: Transcription-focused prompts (v2) reduce hallucination compared to extraction-focused prompts

4. **Error Taxonomy**: Categorized analysis of parsing failures with severity ratings

5. **Hybrid Strategy Recommendation**: Data-driven guidelines for parser selection based on document characteristics

## 7.3 Hybrid Parsing Strategy

Based on our findings, we recommend the following decision tree for document parsing:

```
                    Document Input
                         │
                         ▼
                    ┌─────────┐
                    │ Scanned?│──────Yes────► VLM (Required)
                    └────┬────┘
                         │No
                         ▼
                    ┌──────────────┐
                    │ Complex      │
                    │ Layout?      │──────Yes────► VLM (Recommended)
                    │ (Tables,     │
                    │ Multi-column)│
                    └──────┬───────┘
                           │No
                           ▼
                    pdfplumber (Fast, Sufficient)
```

### 7.3.1 Document Routing Criteria

| Characteristic | Route to VLM | Route to OCR |
|----------------|--------------|--------------|
| Scanned/Image | Always | Never |
| Tables present | Yes | Simple only |
| Multi-column | Yes | No |
| Headers important | Yes | Optional |
| Speed critical | No | Yes |
| Cost sensitive | No | Yes |

### 7.3.2 Expected Outcomes

| Strategy | Avg Quality | Avg Cost | Best For |
|----------|-------------|----------|----------|
| VLM Only | Highest | Highest | Quality-critical |
| OCR Only | Lowest | Lowest | Speed-critical |
| Hybrid | High | Medium | Balanced |

## 7.4 Practical Recommendations

### 7.4.1 For RAG System Developers

1. **Implement Document Classification**: Route documents to appropriate parser based on complexity
2. **Use Transcription Prompts**: Avoid hallucination with explicit transcription instructions
3. **Enable Structure-Aware Chunking**: Leverage markdown structure for better chunk boundaries
4. **Monitor Quality Metrics**: Track CER/WER to catch parsing degradation

### 7.4.2 For Researchers

1. **Expand Evaluation Datasets**: Include more document types and languages
2. **Study End-to-End Impact**: Measure answer quality, not just retrieval
3. **Compare VLM Models**: Evaluate across different VLM architectures
4. **Investigate Efficiency**: Smaller models, quantization, batching

## 7.5 Future Work

### 7.5.1 Short-Term (3-6 months)

1. **Dataset Expansion**:
   - Add more Korean government documents
   - Include financial reports, legal documents
   - Create larger Q&A evaluation set (100+ pairs)

2. **End-to-End RAG Evaluation**:
   - Integrate with LLM answer generation
   - Measure answer quality (RAGAs metrics)
   - User preference studies

3. **Ablation Completion**:
   - Full prompt variation study
   - Resolution optimization
   - Embedding model comparison

### 7.5.2 Medium-Term (6-12 months)

1. **Hybrid System Implementation**:
   - Automatic document classifier
   - Parser selection logic
   - Production-ready pipeline

2. **Multilingual Expansion**:
   - Chinese, Japanese document support
   - Cross-lingual retrieval evaluation
   - Language-specific tokenization

3. **Efficiency Research**:
   - Smaller VLM models (distillation)
   - Batch processing optimization
   - Edge deployment feasibility

### 7.5.3 Long-Term (1+ years)

1. **Adaptive Parsing**:
   - Learn optimal parser per document type
   - Reinforcement learning for parser selection
   - Self-improving quality metrics

2. **Benchmark Publication**:
   - Public benchmark dataset
   - Leaderboard for document parsers
   - Community evaluation standards

## 7.6 Closing Remarks

This study provides quantitative evidence that **structural preservation matters** in document parsing for RAG systems. While VLM-based parsing is not a silver bullet, it offers significant advantages for complex documents. The key insight is that **investing in parsing quality pays dividends downstream** in retrieval accuracy.

We advocate for a **hybrid approach** that balances quality and efficiency:
- Use VLM when structure is important
- Use traditional OCR when speed is paramount
- Monitor and route based on document characteristics

The evaluation framework and recommendations presented here aim to guide practitioners in building more effective RAG systems through informed parser selection.

---

**Acknowledgments**

<!-- TODO: Add acknowledgments if applicable -->

**Data Availability**

The ground truth datasets, evaluation code, and experiment configurations are available at:
- Repository: [To be published]
- License: [To be determined]
