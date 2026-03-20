# Appendix F: Per-Document-Type Results on OmniDocBench

## F.1 Purpose

OmniDocBench covers 9 document types with varying structural complexity. The main text reports aggregate results; this appendix provides per-type breakdowns to identify which document types benefit most from VLM distillation and where the 2B LoRA model matches or falls short of the 30B teacher.

## F.2 OmniDocBench Document Types

| Type | Count | Characteristics | Expected VLM Advantage |
|------|-------|----------------|----------------------|
| Academic Paper | ___ | Two-column, equations, references | High (reading order, structure) |
| Financial Report | ___ | Dense tables, charts | High (table structure) |
| Textbook | ___ | Hierarchical sections, figures | Medium (clear structure) |
| Government Document | ___ | Legal numbering, tables | High (complex structure) |
| Magazine/News | ___ | Multi-column, images | Medium (layout) |
| Slide/Presentation | ___ | Sparse text, diagrams | Low (minimal structure) |
| Form/Invoice | ___ | Key-value pairs, boxes | Medium (table-like) |
| Handwritten | ___ | Variable layout | Special (OCR challenge) |
| Other | ___ | Mixed | Varies |

## F.3 Text NED by Document Type

> **Status**: Pending — requires language/type-stratified evaluation.

| Document Type | 2B base | WigtnOCR-2B | 30B teacher | Marker |
|---------------|---------|-------------|-------------|--------|
| Academic Paper | ___ | ___ | ___ | ___ |
| Financial Report | ___ | ___ | ___ | ___ |
| Textbook | ___ | ___ | ___ | ___ |
| Government Doc | ___ | ___ | ___ | ___ |
| Magazine/News | ___ | ___ | ___ | ___ |
| **Average** | 0.364 | 0.288 | 0.289 | 0.218 |

## F.4 Table TEDS by Document Type

| Document Type | 2B base | WigtnOCR-2B | 30B teacher | Marker |
|---------------|---------|-------------|-------------|--------|
| Academic Paper | ___ | ___ | ___ | ___ |
| Financial Report | ___ | ___ | ___ | ___ |
| Government Doc | ___ | ___ | ___ | ___ |
| **Average** | 0.561 | 0.649 | 0.523 | 0.586 |

## F.5 Analysis Questions

This per-type breakdown enables answering:

1. **Which document types benefit most from distillation?** — Types with largest improvement from 2B base to WigtnOCR-2B
2. **Where does the 2B LoRA model fall short?** — Types where WigtnOCR-2B << 30B teacher
3. **Training data coverage** — Whether types well-represented in training data (academic papers, government docs) show stronger distillation results
