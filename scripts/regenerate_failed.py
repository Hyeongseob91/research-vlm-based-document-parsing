#!/usr/bin/env python3
"""Regenerate failed GT pages identified by validation.

Reads validation_report.json to find failed pages (score < 3),
deletes them, and re-generates using the VLM with enable_thinking: True.

Usage:
    # Regenerate all failed pages for documents
    python scripts/regenerate_failed.py --dataset documents

    # Regenerate all failed pages for papers
    python scripts/regenerate_failed.py --dataset papers

    # Regenerate both
    python scripts/regenerate_failed.py --dataset all

    # Dry run (show what would be regenerated)
    python scripts/regenerate_failed.py --dataset documents --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.gt_generator import (
    pdf_page_to_base64,
    call_vlm_with_image,
    _clean_response,
)

DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-30b-thinking"


def collect_failed_pages(dataset_dir: Path) -> dict[str, list[int]]:
    """Read validation_report.json and collect failed page numbers per document."""
    report_path = dataset_dir / "validation_report.json"
    if not report_path.exists():
        print(f"  No validation report found: {report_path}")
        return {}

    with open(report_path) as f:
        report = json.load(f)

    failed = {}
    for doc in report["documents"]:
        if doc["failed_pages"]:
            failed[doc["doc_id"]] = doc["failed_pages"]
    return failed


def regenerate_page(
    pdf_path: Path,
    page_num_1based: int,
    page_file: Path,
    api_url: str,
    model: str,
    dpi: int,
    max_tokens: int,
    timeout: float,
    lang: str,
) -> tuple[bool, float, str]:
    """Regenerate a single page. Returns (success, elapsed, error_or_chars)."""
    page_num_0based = page_num_1based - 1
    start = time.time()
    try:
        image_b64 = pdf_page_to_base64(pdf_path, page_num_0based, dpi=dpi)
        markdown = call_vlm_with_image(
            image_b64=image_b64,
            api_url=api_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            lang=lang,
        )
        page_file.write_text(markdown, encoding="utf-8")
        elapsed = time.time() - start
        return True, elapsed, f"{len(markdown)} chars"
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)


def regenerate_dataset(
    dataset_name: str,
    api_url: str,
    model: str,
    dpi: int,
    max_tokens: int,
    timeout: float,
    batch_size: int,
    dry_run: bool,
):
    """Regenerate all failed pages for a dataset."""
    dataset_dir = Path("datasets") / dataset_name
    lang = "en" if dataset_name == "papers" else "ko"

    failed_map = collect_failed_pages(dataset_dir)
    if not failed_map:
        print(f"  No failed pages for {dataset_name}")
        return

    total_pages = sum(len(pages) for pages in failed_map.values())
    print(f"\n{'=' * 60}")
    print(f"Regenerating failed GT pages")
    print(f"  Dataset: {dataset_name}")
    print(f"  Documents with failures: {len(failed_map)}")
    print(f"  Total pages to regenerate: {total_pages}")
    print(f"  Model: {model}")
    print(f"  Language: {lang}")
    if dry_run:
        print(f"  *** DRY RUN — no files will be modified ***")
    print(f"{'=' * 60}\n")

    if dry_run:
        for doc_id, pages in sorted(failed_map.items()):
            print(f"  {doc_id}: {len(pages)}p — {pages}")
        return

    total_start = time.time()
    total_ok = 0
    total_fail = 0
    still_failed = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    for doc_id, pages in sorted(failed_map.items()):
        doc_dir = dataset_dir / doc_id
        # Find PDF
        pdf_path = doc_dir / "doc.pdf"
        if not pdf_path.exists():
            pdf_path = doc_dir / "paper.pdf"
        if not pdf_path.exists():
            print(f"  [{doc_id}] SKIP — no PDF found")
            continue

        pages_dir = doc_dir / "gt_pages"
        print(f"\n--- {doc_id} ({len(pages)}p) ---")

        # Delete old failed page files first
        for page_num in pages:
            page_file = pages_dir / f"page_{page_num:04d}.md"
            if page_file.exists():
                page_file.unlink()

        # Regenerate with thread pool
        doc_fails = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {}
            for page_num in pages:
                page_file = pages_dir / f"page_{page_num:04d}.md"
                future = executor.submit(
                    regenerate_page,
                    pdf_path=pdf_path,
                    page_num_1based=page_num,
                    page_file=page_file,
                    api_url=api_url,
                    model=model,
                    dpi=dpi,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    lang=lang,
                )
                futures[future] = page_num

            for future in as_completed(futures):
                page_num = futures[future]
                success, elapsed, info = future.result()
                if success:
                    total_ok += 1
                    print(f"  [{doc_id}] page {page_num} OK ({elapsed:.1f}s, {info})")
                else:
                    total_fail += 1
                    doc_fails.append(page_num)
                    print(f"  [{doc_id}] page {page_num} FAIL ({elapsed:.1f}s) — {info}")

        if doc_fails:
            still_failed[doc_id] = sorted(doc_fails)

        # Rebuild gt.md for this document
        _rebuild_gt_md(doc_dir, pages_dir)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"REGENERATION COMPLETE")
    print(f"  Success: {total_ok}/{total_pages}")
    print(f"  Still failed: {total_fail}")
    print(f"  Time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    if still_failed:
        print(f"  Remaining failures:")
        for doc_id, pages in still_failed.items():
            print(f"    {doc_id}: {pages}")
    print(f"{'=' * 60}")


def _rebuild_gt_md(doc_dir: Path, pages_dir: Path):
    """Rebuild gt.md from all page files."""
    page_files = sorted(pages_dir.glob("page_*.md"))
    if not page_files:
        return

    merged = []
    for i, page_file in enumerate(page_files):
        if i > 0:
            merged.append("\n\n---\n\n")
        merged.append(page_file.read_text(encoding="utf-8"))

    gt_path = doc_dir / "gt.md"
    gt_path.write_text("".join(merged), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Regenerate failed GT pages")
    parser.add_argument("--dataset", choices=["documents", "papers", "all"], required=True)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be regenerated")
    args = parser.parse_args()

    datasets = ["documents", "papers"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        regenerate_dataset(
            dataset_name=ds,
            api_url=args.api_url,
            model=args.model,
            dpi=args.dpi,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
