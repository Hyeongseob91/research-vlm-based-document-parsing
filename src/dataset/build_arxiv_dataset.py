"""
arXiv dataset builder — orchestrator.

Downloads arXiv papers, converts LaTeX → Markdown GT,
validates quality, and organizes into test_arxiv_NNN/ folders.

Usage:
    python -m src.dataset.build_arxiv_dataset --limit 5
    python -m src.dataset.build_arxiv_dataset --all
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from src.dataset.arxiv_downloader import download_paper, check_source_availability
from src.dataset.latex_to_markdown import latex_to_markdown
from src.dataset.validate_gt import validate_gt


_REQUEST_DELAY = 3.0


def build_dataset(
    paper_list_path: Path = Path("src/dataset/arxiv_paper_list.json"),
    output_dir: Path = Path("data"),
    raw_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    cer_threshold: float = 0.15,
    skip_download: bool = False,
) -> dict:
    """
    Build the arXiv test dataset.

    Args:
        paper_list_path: Path to arxiv_paper_list.json.
        output_dir: Where to create test_arxiv_NNN folders.
        raw_dir: Where to store raw downloads (default: output_dir/_arxiv_raw).
        limit: Max papers to process.
        cer_threshold: CER threshold for GT validation.
        skip_download: Skip download if raw files already exist.

    Returns:
        Build manifest dict.
    """
    if raw_dir is None:
        raw_dir = output_dir / "_arxiv_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(paper_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data["papers"]
    if limit:
        papers = papers[:limit]

    manifest = {
        "total_candidates": len(papers),
        "downloaded": 0,
        "converted": 0,
        "validated": 0,
        "failed": [],
        "papers": [],
    }

    test_index = _get_next_index(output_dir)

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        title = paper.get("title", arxiv_id)
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(papers)}] {title} ({arxiv_id})")
        print(f"{'=' * 60}")

        paper_result = {
            "arxiv_id": arxiv_id,
            "title": title,
            "category": paper.get("category", ""),
            "status": "pending",
            "test_folder": None,
            "errors": [],
        }

        # Step 1: Download
        paper_raw_dir = raw_dir / f"arxiv_{arxiv_id.replace('/', '_')}"

        if skip_download and paper_raw_dir.exists():
            print("  Using cached download...")
            pdf_path = paper_raw_dir / "paper.pdf"
            # Find tex file
            from src.dataset.arxiv_downloader import find_main_tex
            source_dir = paper_raw_dir / "latex_source"
            tex_path = find_main_tex(source_dir) if source_dir.exists() else None

            if not pdf_path.exists() or tex_path is None:
                paper_result["status"] = "download_failed"
                paper_result["errors"].append("Cached files incomplete")
                manifest["failed"].append(paper_result)
                manifest["papers"].append(paper_result)
                continue
        else:
            import time
            print("  Step 1: Checking source availability...")
            if not check_source_availability(arxiv_id):
                paper_result["status"] = "no_source"
                paper_result["errors"].append("LaTeX source not available on arXiv")
                manifest["failed"].append(paper_result)
                manifest["papers"].append(paper_result)
                print("  → SKIP: No LaTeX source")
                time.sleep(_REQUEST_DELAY)
                continue

            time.sleep(_REQUEST_DELAY)
            download_result = download_paper(arxiv_id, raw_dir, delay=_REQUEST_DELAY)

            if not download_result["success"]:
                paper_result["status"] = "download_failed"
                paper_result["errors"].append(download_result.get("error", "Unknown"))
                manifest["failed"].append(paper_result)
                manifest["papers"].append(paper_result)
                print(f"  → SKIP: {download_result.get('error')}")
                continue

            pdf_path = Path(download_result["pdf_path"])
            tex_path = Path(download_result["tex_path"])

        manifest["downloaded"] += 1
        print(f"  Step 1: Download OK")

        # Step 2: Convert LaTeX → Markdown
        print("  Step 2: Converting LaTeX → Markdown...")
        try:
            markdown = latex_to_markdown(tex_path)
        except Exception as e:
            markdown = None
            paper_result["errors"].append(f"Conversion error: {e}")

        if markdown is None or len(markdown.strip()) < 100:
            paper_result["status"] = "conversion_failed"
            paper_result["errors"].append("LaTeX → Markdown conversion produced no output")
            manifest["failed"].append(paper_result)
            manifest["papers"].append(paper_result)
            print("  → SKIP: Conversion failed")
            continue

        manifest["converted"] += 1
        print(f"  Step 2: Conversion OK ({len(markdown)} chars)")

        # Step 3: Create test folder
        folder_name = f"test_arxiv_{test_index:03d}"
        test_folder = output_dir / folder_name
        test_folder.mkdir(parents=True, exist_ok=True)

        # Copy PDF
        shutil.copy2(pdf_path, test_folder / "paper.pdf")

        # Write GT markdown
        gt_path = test_folder / "gt_paper.md"
        gt_path.write_text(markdown, encoding="utf-8")

        # Write metadata
        meta = {
            "arxiv_id": arxiv_id,
            "title": title,
            "category": paper.get("category", ""),
            "expected_structures": paper.get("expected_structures", []),
        }
        (test_folder / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Step 4: Validate
        print("  Step 3: Validating GT quality...")
        validation = validate_gt(gt_path, test_folder / "paper.pdf", cer_threshold)

        if validation["overall_passed"]:
            manifest["validated"] += 1
            paper_result["status"] = "success"
            print("  → SUCCESS")
        else:
            paper_result["status"] = "validation_warning"
            paper_result["errors"].extend(
                validation["structural"]["issues"]
            )
            print(f"  → WARNING: {validation['structural']['issues']}")

        paper_result["test_folder"] = folder_name

        if validation.get("comparative") and validation["comparative"].get("cer") is not None:
            paper_result["cer"] = validation["comparative"]["cer"]
            cer_status = "PASS" if validation["comparative"]["passed"] else "FAIL"
            print(f"  CER vs PyMuPDF: {validation['comparative']['cer']:.3f} ({cer_status})")

        if validation.get("table_count"):
            paper_result["gt_table_count"] = validation["table_count"]["gt_table_count"]
            print(f"  Tables in GT: {validation['table_count']['gt_table_count']}")

        manifest["papers"].append(paper_result)
        test_index += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Candidates: {manifest['total_candidates']}")
    print(f"  Downloaded: {manifest['downloaded']}")
    print(f"  Converted:  {manifest['converted']}")
    print(f"  Validated:  {manifest['validated']}")
    print(f"  Failed:     {len(manifest['failed'])}")

    # Save manifest
    manifest_path = output_dir / "arxiv_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")

    return manifest


def _get_next_index(output_dir: Path) -> int:
    """Find the next available test_arxiv_NNN index."""
    existing = list(output_dir.glob("test_arxiv_*"))
    if not existing:
        return 1
    indices = []
    for d in existing:
        try:
            idx = int(d.name.split("_")[-1])
            indices.append(idx)
        except ValueError:
            continue
    return max(indices, default=0) + 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build arXiv test dataset for structure preservation evaluation"
    )
    parser.add_argument(
        "--paper-list",
        type=Path,
        default=Path("src/dataset/arxiv_paper_list.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process")
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download if raw files already cached",
    )
    args = parser.parse_args()

    build_dataset(
        paper_list_path=args.paper_list,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        limit=args.limit,
        cer_threshold=args.cer_threshold,
        skip_download=args.skip_download,
    )
