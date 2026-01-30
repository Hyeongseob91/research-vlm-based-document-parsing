"""
arXiv paper downloader.

Downloads PDF and LaTeX source from arXiv for GT generation.
Includes source availability pre-check, rate limiting, and retry logic.
"""

import io
import json
import tarfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# arXiv rate limiting: 3-second delay between requests
_REQUEST_DELAY = 3.0
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5.0


def _download_with_retry(url: str, max_retries: int = _MAX_RETRIES) -> bytes:
    """Download URL content with retry logic."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "vlm-doc-parsing-eval/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = _RETRY_BACKOFF * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    return b""


def check_source_availability(arxiv_id: str) -> bool:
    """Check if LaTeX source is available for an arXiv paper."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "vlm-doc-parsing-eval/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # LaTeX sources are served as application/x-eprint-tar,
            # application/gzip, or application/x-gzip
            return "pdf" not in content_type.lower()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def download_pdf(arxiv_id: str, output_path: Path) -> bool:
    """Download PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        data = _download_with_retry(url)
        if not data or len(data) < 1000:
            return False
        output_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"  PDF download failed: {e}")
        return False


def download_and_extract_source(
    arxiv_id: str, output_dir: Path
) -> Optional[Path]:
    """
    Download and extract LaTeX source from arXiv.

    Returns path to the main .tex file, or None on failure.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        data = _download_with_retry(url)
        if not data:
            return None
    except Exception as e:
        print(f"  Source download failed: {e}")
        return None

    source_dir = output_dir / "latex_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Try extracting as tar/gzip
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            # Security: prevent path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    continue
                tar.extract(member, path=source_dir)
    except tarfile.TarError:
        # Some sources are single .tex files (gzipped)
        import gzip
        try:
            decompressed = gzip.decompress(data)
            single_tex = source_dir / "main.tex"
            single_tex.write_bytes(decompressed)
        except Exception:
            # Raw .tex file
            single_tex = source_dir / "main.tex"
            single_tex.write_bytes(data)

    # Find main .tex file
    return find_main_tex(source_dir)


def find_main_tex(source_dir: Path) -> Optional[Path]:
    """
    Find the main .tex file in extracted LaTeX source.

    Searches for \\documentclass to identify the main file.
    """
    tex_files = list(source_dir.rglob("*.tex"))
    if not tex_files:
        return None

    # Look for \documentclass
    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            if "\\documentclass" in content:
                return tex_file
        except Exception:
            continue

    # Fallback: look for common names
    for name in ["main.tex", "paper.tex", "article.tex", "manuscript.tex"]:
        for tex_file in tex_files:
            if tex_file.name == name:
                return tex_file

    # Last resort: largest .tex file
    return max(tex_files, key=lambda f: f.stat().st_size)


def download_paper(
    arxiv_id: str, output_dir: Path, delay: float = _REQUEST_DELAY
) -> dict:
    """
    Download a single arXiv paper (PDF + LaTeX source).

    Returns a status dict with success flag, paths, and any errors.
    """
    result = {
        "arxiv_id": arxiv_id,
        "success": False,
        "pdf_path": None,
        "tex_path": None,
        "source_available": False,
        "error": None,
    }

    paper_dir = output_dir / f"arxiv_{arxiv_id.replace('/', '_')}"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check source availability
    print(f"  Checking source for {arxiv_id}...")
    result["source_available"] = check_source_availability(arxiv_id)
    time.sleep(delay)

    if not result["source_available"]:
        result["error"] = "LaTeX source not available"
        return result

    # Step 2: Download PDF
    print(f"  Downloading PDF...")
    pdf_path = paper_dir / "paper.pdf"
    if not download_pdf(arxiv_id, pdf_path):
        result["error"] = "PDF download failed"
        return result
    result["pdf_path"] = str(pdf_path)
    time.sleep(delay)

    # Step 3: Download and extract LaTeX source
    print(f"  Downloading LaTeX source...")
    tex_path = download_and_extract_source(arxiv_id, paper_dir)
    if tex_path is None:
        result["error"] = "LaTeX source extraction failed"
        return result
    result["tex_path"] = str(tex_path)
    time.sleep(delay)

    result["success"] = True
    return result


def download_all(
    paper_list_path: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Download all papers from the paper list.

    Args:
        paper_list_path: Path to arxiv_paper_list.json.
        output_dir: Base output directory.
        limit: Max papers to download (None for all).

    Returns:
        List of download result dicts.
    """
    with open(paper_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data["papers"]
    if limit:
        papers = papers[:limit]

    results = []
    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        print(f"[{i + 1}/{len(papers)}] {paper.get('title', arxiv_id)}")

        result = download_paper(arxiv_id, output_dir)
        result["title"] = paper.get("title", "")
        result["category"] = paper.get("category", "")
        results.append(result)

        success_count = sum(1 for r in results if r["success"])
        print(f"  → {'OK' if result['success'] else 'FAILED: ' + str(result['error'])}")
        print(f"  Progress: {success_count}/{len(results)} successful\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download arXiv papers")
    parser.add_argument(
        "--paper-list",
        type=Path,
        default=Path("src/dataset/arxiv_paper_list.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/_arxiv_raw"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = download_all(args.paper_list, args.output_dir, args.limit)

    # Save download report
    report_path = args.output_dir / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    ok = sum(1 for r in results if r["success"])
    print(f"\nDone: {ok}/{total} papers downloaded successfully")
    print(f"Report: {report_path}")
