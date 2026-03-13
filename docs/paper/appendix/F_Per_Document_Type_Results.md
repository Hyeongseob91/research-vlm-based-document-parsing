# Appendix F: Per-Document-Type Results on OmniDocBench

## F.1 Purpose

OmniDocBench covers 9 document types with varying structural complexity. The main text reports aggregate results; this appendix provides per-type breakdowns to identify which document types benefit most from VLM structural parsing and where the 2B LoRA model matches or falls short of the 30B teacher.

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

## F.3 Parsing Quality by Document Type

> **Status**: Pending — will be populated after OmniDocBench evaluation.

### F.3.1 Structure F1 by Type

| Document Type | PyMuPDF | 2B base | WigtnOCR-2B | 30B teacher |
|---------------|---------|---------|-------------|-------------|
| Academic Paper | ___ | ___ | ___ | ___ |
| Financial Report | ___ | ___ | ___ | ___ |
| Textbook | ___ | ___ | ___ | ___ |
| Government Doc | ___ | ___ | ___ | ___ |
| Magazine/News | ___ | ___ | ___ | ___ |
| Slide | ___ | ___ | ___ | ___ |
| Form/Invoice | ___ | ___ | ___ | ___ |
| **Average** | ___ | ___ | ___ | ___ |

### F.3.2 CER by Type

| Document Type | PyMuPDF | 2B base | WigtnOCR-2B | 30B teacher |
|---------------|---------|---------|-------------|-------------|
| Academic Paper | ___ | ___ | ___ | ___ |
| Financial Report | ___ | ___ | ___ | ___ |
| Textbook | ___ | ___ | ___ | ___ |
| Government Doc | ___ | ___ | ___ | ___ |
| Magazine/News | ___ | ___ | ___ | ___ |
| Slide | ___ | ___ | ___ | ___ |
| Form/Invoice | ___ | ___ | ___ | ___ |
| **Average** | ___ | ___ | ___ | ___ |

## F.4 Chunking Quality by Document Type (Step 1)

### F.4.1 BC (Header-based chunking) by Type

| Document Type | PyMuPDF | 2B base | WigtnOCR-2B | 30B teacher |
|---------------|---------|---------|-------------|-------------|
| Academic Paper | N/A | ___ | ___ | ___ |
| Financial Report | N/A | ___ | ___ | ___ |
| Textbook | N/A | ___ | ___ | ___ |
| **Average** | N/A | ___ | ___ | ___ |

## F.5 Retrieval Performance by Document Type (Step 2)

### F.5.1 Hit@5 by Type

| Document Type | PyMuPDF | 2B base | WigtnOCR-2B | 30B teacher |
|---------------|---------|---------|-------------|-------------|
| Academic Paper | ___ | ___ | ___ | ___ |
| Financial Report | ___ | ___ | ___ | ___ |
| Textbook | ___ | ___ | ___ | ___ |
| **Average** | ___ | ___ | ___ | ___ |

## F.6 Analysis Questions

This per-type breakdown enables answering:

1. **Which document types benefit most from VLM parsing?** — Types with high SF1 gap between baseline and VLM
2. **Where does the 2B LoRA model fall short?** — Types where WigtnOCR-2B << 30B teacher
3. **Is the causal chain type-dependent?** — Whether SF1→BC→Hit@K correlation holds across all types
4. **Training data coverage** — Whether types well-represented in training data (academic papers, government docs) show stronger distillation results
