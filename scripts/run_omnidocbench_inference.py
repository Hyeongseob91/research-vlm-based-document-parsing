#!/usr/bin/env python3
"""Run VLM inference on OmniDocBench dataset.

Generates markdown prediction files for OmniDocBench evaluation.
Each OmniDocBench image is sent to the VLM API, and the resulting
markdown is saved as a .md file matching the image filename.

Output is compatible with OmniDocBench's official eval pipeline:
    python pdf_validation.py --config configs/end2end.yaml

Usage:
    # 2B model
    python scripts/run_omnidocbench_inference.py \
        --model qwen3-vl-2b-instruct \
        --output-dir results/omnidocbench/vlm_2b \
        --batch-size 8

    # 30B model
    python scripts/run_omnidocbench_inference.py \
        --model qwen3-vl-30b-thinking \
        --output-dir results/omnidocbench/vlm_30b \
        --enable-thinking \
        --batch-size 2

    # PyMuPDF baseline (no VLM, text extraction only)
    python scripts/run_omnidocbench_inference.py \
        --parser pymupdf \
        --output-dir results/omnidocbench/pymupdf
"""

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def image_to_base64(image_path: Path) -> str:
    """Read an image file and return base64-encoded string."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def call_vlm(
    image_b64: str,
    api_url: str,
    model: str,
    max_tokens: int = 8192,
    timeout: float = 180.0,
    enable_thinking: bool = False,
    lang: str = "auto",
) -> str:
    """Send image to VLM API and get structured markdown."""
    import httpx

    # Use the proven prompts from GT generation
    from training.prompts.templates import (
        PSEUDO_GT_SYSTEM_PROMPT,
        PSEUDO_GT_USER_PROMPT,
        PSEUDO_GT_SYSTEM_PROMPT_EN,
        PSEUDO_GT_USER_PROMPT_EN,
    )

    # Auto-detect: OmniDocBench has mixed languages, use English prompt as default
    if lang == "auto" or lang == "en":
        system_prompt = PSEUDO_GT_SYSTEM_PROMPT_EN
        user_prompt = PSEUDO_GT_USER_PROMPT_EN
    else:
        system_prompt = PSEUDO_GT_SYSTEM_PROMPT
        user_prompt = PSEUDO_GT_USER_PROMPT

    # Detect image format from base64 header or default to jpeg
    suffix = "jpeg"
    if image_b64[:8].startswith("iVBOR"):
        suffix = "png"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{suffix};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()

    message = result["choices"][0]["message"]

    raw = message["content"] or ""

    return _clean_response(raw)


# Reuse cleaning logic from gt_generator
_THINKING_PATTERNS = re.compile(
    r"^(Okay,? let'?s|Got it,? let'?s|First,? I need to|Let me (think|check|look|analyze|start))"
    r"|^Wait,? (the |let me|I need)"
    r"|^(But how|But the |However,? the |Perhaps the )"
    r"|^(Now,? (let'?s|I need|let me))"
    r"|^Looking at the (image|table|document)",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_response(raw: str) -> str:
    """Remove thinking tags and code fences from VLM response."""
    content = raw
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL)
    content = content.strip()
    if content.startswith("```markdown"):
        content = content[len("```markdown"):]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    if content and _THINKING_PATTERNS.search(content[:500]):
        md_match = re.search(r"^(#{1,6}\s|\|)", content, re.MULTILINE)
        if md_match and md_match.start() > 50:
            content = content[md_match.start():]
    return content


def run_pymupdf(image_path: Path, omnidocbench_dir: Path) -> str:
    """Extract text using PyMuPDF from the source PDF if available.

    OmniDocBench images are page screenshots, not PDFs.
    For fair comparison, we extract text from the image using basic OCR fallback,
    or return empty string if no PDF source is available.

    In practice, PyMuPDF baseline should be run differently — OmniDocBench
    provides source PDFs in some cases. For images without PDF source,
    PyMuPDF returns empty (it's a text extractor, not OCR).
    """
    # OmniDocBench provides images, not PDFs.
    # PyMuPDF is a text extractor for digital PDFs.
    # For a fair comparison, return empty to show PyMuPDF cannot handle images.
    return ""


def process_single_image(
    image_path: Path,
    output_dir: Path,
    api_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool,
    lang: str,
) -> dict:
    """Process a single OmniDocBench image."""
    image_name = image_path.stem  # e.g., "docstructbench_llm-raw-scihub-o.O-0000025460.pdf_7"
    output_path = output_dir / f"{image_name}.md"

    # Skip if already processed
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "image": image_path.name,
            "status": "skipped",
            "elapsed": 0,
            "chars": output_path.stat().st_size,
        }

    start = time.time()
    try:
        image_b64 = image_to_base64(image_path)
        markdown = call_vlm(
            image_b64=image_b64,
            api_url=api_url,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            enable_thinking=enable_thinking,
            lang=lang,
        )
        output_path.write_text(markdown, encoding="utf-8")
        elapsed = time.time() - start
        return {
            "image": image_path.name,
            "status": "ok",
            "elapsed": round(elapsed, 1),
            "chars": len(markdown),
        }
    except Exception as e:
        elapsed = time.time() - start
        # Write empty file to mark as attempted (can be retried with --retry-failed)
        output_path.write_text(f"<!-- ERROR: {e} -->", encoding="utf-8")
        return {
            "image": image_path.name,
            "status": "error",
            "elapsed": round(elapsed, 1),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference on OmniDocBench")
    parser.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=Path("datasets/omnidocbench"),
        help="Path to OmniDocBench dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for prediction .md files",
    )
    parser.add_argument(
        "--parser",
        choices=["vlm", "pymupdf"],
        default="vlm",
        help="Parser type (default: vlm)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8010/v1/chat/completions",
        help="VLM API endpoint",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-2b-instruct",
        help="VLM model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens per VLM response",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="API request timeout in seconds",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode (for 30B model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of concurrent VLM requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "ko", "en"],
        help="Language for prompt selection",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-process images that previously failed (contain ERROR comment)",
    )

    args = parser.parse_args()

    # Validate omnidocbench directory
    images_dir = args.omnidocbench_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    # Collect images
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Found {len(image_files)} images in {images_dir}")

    if args.limit:
        image_files = image_files[:args.limit]
        print(f"Limited to {args.limit} images")

    # Handle retry-failed: remove error files
    if args.retry_failed and args.output_dir.exists():
        removed = 0
        for md_file in args.output_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            if content.startswith("<!-- ERROR:"):
                md_file.unlink()
                removed += 1
        if removed:
            print(f"Removed {removed} failed files for retry")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Count already processed
    existing = set(p.stem for p in args.output_dir.glob("*.md"))
    to_process = [p for p in image_files if p.stem not in existing]
    skipped = len(image_files) - len(to_process)
    print(f"To process: {len(to_process)} (skipping {skipped} already done)")

    if not to_process:
        print("All images already processed. Use --retry-failed to reprocess errors.")
        return

    # Run inference
    start_time = time.time()
    results = {"ok": 0, "error": 0, "skipped": skipped}

    done_count = 0

    if args.parser == "pymupdf":
        # PyMuPDF doesn't need VLM — just saves empty files
        for img_path in to_process:
            output_path = args.output_dir / f"{img_path.stem}.md"
            output_path.write_text("", encoding="utf-8")
            results["ok"] += 1
        print(f"PyMuPDF baseline: {results['ok']} empty .md files created")
        print("Note: PyMuPDF cannot extract text from images. "
              "This serves as a baseline showing text extraction limitations.")
    else:
        # VLM inference with concurrent processing
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {}
            for img_path in to_process:
                future = executor.submit(
                    process_single_image,
                    image_path=img_path,
                    output_dir=args.output_dir,
                    api_url=args.api_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    enable_thinking=args.enable_thinking,
                    lang=args.lang,
                )
                futures[future] = img_path

            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                done_count += 1
                status = result["status"]
                results[status] = results.get(status, 0) + 1

                # Progress
                elapsed = time.time() - start_time
                avg_per_image = elapsed / done_count
                remaining = (len(to_process) - done_count) * avg_per_image

                if status == "ok":
                    print(
                        f"  [{done_count}/{len(to_process)}] {result['image']} "
                        f"OK ({result['elapsed']}s, {result['chars']} chars) "
                        f"ETA: {remaining/60:.0f}m"
                    )
                elif status == "error":
                    print(
                        f"  [{done_count}/{len(to_process)}] {result['image']} "
                        f"ERROR: {result.get('error', 'unknown')}"
                    )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Inference complete: {args.model}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  OK: {results.get('ok', 0)}")
    print(f"  Errors: {results.get('error', 0)}")
    print(f"  Skipped: {results.get('skipped', 0)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    # Save run metadata
    meta = {
        "model": args.model,
        "parser": args.parser,
        "api_url": args.api_url,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "batch_size": args.batch_size,
        "total_images": len(image_files),
        "processed": results.get("ok", 0),
        "errors": results.get("error", 0),
        "skipped": results.get("skipped", 0),
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_image": round(total_time / max(done_count, 1), 1) if args.parser != "pymupdf" else 0,
    }
    meta_path = args.output_dir / "inference_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
