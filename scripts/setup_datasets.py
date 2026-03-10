#!/usr/bin/env python3
"""Dataset setup verification script."""

import json
from pathlib import Path


def main():
    datasets_dir = Path("datasets")

    print("=" * 60)
    print("WigtnOCR Dataset Status")
    print("=" * 60)

    # Papers (English)
    papers_dir = datasets_dir / "papers"
    if papers_dir.exists():
        paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]
        print(f"\nPapers (English): {len(paper_dirs)} documents")
        for d in sorted(paper_dirs)[:5]:
            pdfs = list(d.glob("*.pdf"))
            gts = list(d.glob("gt*.md"))
            print(f"  {d.name}: PDF={'Y' if pdfs else 'N'}, GT={'Y' if gts else 'N'}")
        if len(paper_dirs) > 5:
            print(f"  ... and {len(paper_dirs) - 5} more")

    # Documents (Korean)
    docs_dir = datasets_dir / "documents"
    if docs_dir.exists():
        doc_dirs = [d for d in docs_dir.iterdir() if d.is_dir()]
        print(f"\nDocuments (Korean): {len(doc_dirs)} documents")
        for d in sorted(doc_dirs):
            pdfs = list(d.glob("*.pdf"))
            gts = list(d.glob("gt*.md"))
            meta_file = d / "metadata.json"
            doc_type = ""
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                doc_type = f" [{meta.get('doc_type', '')}]"
            print(f"  {d.name}: PDF={'Y' if pdfs else 'N'}, GT={'Y' if gts else 'N'}{doc_type}")

    # OmniDocBench
    omni_dir = datasets_dir / "omnidocbench"
    if omni_dir.exists():
        json_files = list(omni_dir.glob("*.json"))
        print(f"\nOmniDocBench: {len(json_files)} JSON files")

    print()


if __name__ == "__main__":
    main()
