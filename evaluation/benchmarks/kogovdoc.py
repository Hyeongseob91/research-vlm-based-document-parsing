"""KoGovDoc-Bench — Korean Government Document Benchmark.

50-100 manually curated Korean public sector documents with
hand-crafted ground truth for structure-aware parsing evaluation.
"""

import json
from pathlib import Path
from typing import Optional


def load_kogovdoc_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
) -> list[dict]:
    """Load KoGovDoc benchmark dataset.

    Args:
        dataset_path: Root path of datasets/documents/
        limit: Maximum number of documents to load

    Returns:
        List of document dicts with keys:
        - doc_id: str
        - pdf_path: Path
        - gt_markdown: str
        - metadata: dict (doc_type, page_count, etc.)
    """
    documents = []

    if not dataset_path.exists():
        return documents

    for doc_dir in sorted(dataset_path.iterdir()):
        if not doc_dir.is_dir() or not doc_dir.name.startswith("kogov_"):
            continue

        # Find PDF and GT files
        pdf_files = list(doc_dir.glob("*.pdf"))
        gt_files = list(doc_dir.glob("gt*.md"))
        metadata_file = doc_dir / "metadata.json"

        if not pdf_files:
            continue

        gt_markdown = ""
        if gt_files:
            gt_markdown = gt_files[0].read_text(encoding="utf-8")

        metadata = {}
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        documents.append({
            "doc_id": doc_dir.name,
            "pdf_path": pdf_files[0],
            "gt_markdown": gt_markdown,
            "metadata": metadata,
        })

        if limit and len(documents) >= limit:
            break

    return documents
