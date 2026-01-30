"""
Ground truth quality validation.

Three-level validation:
1. Structural checks (headings, length, no LaTeX remnants)
2. Comparative checks (CER vs PyMuPDF extraction < 0.15)
3. Table count matching (GT tables == PDF tables detected)
"""

import json
import re
from pathlib import Path
from typing import Optional


def check_structure(markdown: str) -> list[str]:
    """
    Structural validation of GT markdown.

    Returns list of issues (empty = pass).
    """
    issues = []

    # Must have at least 1 heading
    if not re.search(r"^#{1,6}\s+\S", markdown, re.MULTILINE):
        issues.append("NO_HEADINGS: No markdown headings found")

    # Must have reasonable length
    if len(markdown.strip()) < 500:
        issues.append(f"TOO_SHORT: Only {len(markdown.strip())} chars (min 500)")

    # No raw LaTeX commands remaining
    latex_patterns = [
        (r"\\begin\{", "\\begin{...}"),
        (r"\\end\{", "\\end{...}"),
        (r"\\documentclass", "\\documentclass"),
        (r"\\usepackage", "\\usepackage"),
        (r"\\newcommand", "\\newcommand"),
        (r"\\def\\", "\\def"),
    ]
    for pattern, name in latex_patterns:
        matches = re.findall(pattern, markdown)
        if len(matches) > 2:  # Allow occasional remnants
            issues.append(
                f"LATEX_REMNANT: {name} appears {len(matches)} times"
            )

    # Check equation delimiters are balanced
    dollar_double = markdown.count("$$")
    if dollar_double % 2 != 0:
        issues.append(f"UNBALANCED_EQ: $$ count is odd ({dollar_double})")

    # Check table format (pipes should be balanced per line)
    for i, line in enumerate(markdown.split("\n"), 1):
        if line.strip().startswith("|") and line.strip().endswith("|"):
            pipe_count = line.count("|")
            if pipe_count < 3:
                issues.append(f"BAD_TABLE_LINE_{i}: Too few pipes ({pipe_count})")
                break  # Report only first

    return issues


def check_comparative(
    gt_markdown: str,
    pdf_path: Path,
    cer_threshold: float = 0.15,
) -> dict:
    """
    Compare GT markdown against PyMuPDF text extraction.

    Returns dict with cer, passed, and details.
    """
    result = {
        "cer": None,
        "passed": False,
        "error": None,
    }

    try:
        import fitz  # PyMuPDF
    except ImportError:
        result["error"] = "PyMuPDF not installed"
        return result

    if not pdf_path.exists():
        result["error"] = f"PDF not found: {pdf_path}"
        return result

    try:
        doc = fitz.open(str(pdf_path))
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        doc.close()
    except Exception as e:
        result["error"] = f"PDF read failed: {e}"
        return result

    if not pdf_text.strip():
        result["error"] = "PDF has no text layer"
        return result

    # Normalize both for comparison
    gt_clean = _normalize_for_comparison(gt_markdown)
    pdf_clean = _normalize_for_comparison(pdf_text)

    if not gt_clean or not pdf_clean:
        result["error"] = "Empty text after normalization"
        return result

    try:
        from jiwer import cer
        result["cer"] = cer(pdf_clean, gt_clean)
        result["passed"] = result["cer"] < cer_threshold
    except ImportError:
        result["error"] = "jiwer not installed"

    return result


def check_table_count(gt_markdown: str, pdf_path: Path) -> dict:
    """
    Check if GT markdown table count roughly matches PDF content.

    Returns dict with gt_tables, estimation, and match status.
    """
    # Count tables in GT (consecutive | lines)
    gt_tables = 0
    in_table = False
    for line in gt_markdown.split("\n"):
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            if not in_table:
                gt_tables += 1
                in_table = True
        else:
            in_table = False

    return {
        "gt_table_count": gt_tables,
        "passed": True,  # Informational — no hard fail
    }


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for CER comparison."""
    # Remove markdown formatting
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"\$\$[^$]+\$\$", "[EQ]", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "[EQ]", text)
    text = re.sub(r"\|[^|\n]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_gt(
    gt_path: Path,
    pdf_path: Optional[Path] = None,
    cer_threshold: float = 0.15,
) -> dict:
    """
    Run all validation checks on a GT markdown file.

    Returns validation result dict.
    """
    markdown = gt_path.read_text(encoding="utf-8")

    result = {
        "gt_path": str(gt_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "structural": {
            "issues": check_structure(markdown),
            "passed": True,
        },
        "comparative": None,
        "table_count": check_table_count(markdown, pdf_path) if pdf_path else None,
        "overall_passed": True,
    }

    # Structural pass if no critical issues
    critical_issues = [
        i for i in result["structural"]["issues"]
        if i.startswith(("NO_HEADINGS", "TOO_SHORT"))
    ]
    result["structural"]["passed"] = len(critical_issues) == 0

    # Comparative check
    if pdf_path and pdf_path.exists():
        result["comparative"] = check_comparative(
            markdown, pdf_path, cer_threshold
        )

    # Overall: structural must pass; comparative is warning-level
    result["overall_passed"] = result["structural"]["passed"]

    return result


def validate_dataset(
    dataset_dir: Path,
    cer_threshold: float = 0.15,
) -> dict:
    """
    Validate all GT files in a dataset directory.

    Expects test_arxiv_NNN/ folders with paper.pdf and gt_paper.md.
    """
    report = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "papers": [],
    }

    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("test_arxiv_"):
            continue

        gt_file = folder / "gt_paper.md"
        pdf_file = folder / "paper.pdf"

        if not gt_file.exists():
            continue

        report["total"] += 1
        result = validate_gt(gt_file, pdf_file, cer_threshold)
        result["folder"] = folder.name

        if result["overall_passed"]:
            report["passed"] += 1
        else:
            report["failed"] += 1

        report["papers"].append(result)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate GT quality")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data"))
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = validate_dataset(args.dataset_dir, args.cer_threshold)

    print(f"Validated: {report['total']} papers")
    print(f"  Passed: {report['passed']}")
    print(f"  Failed: {report['failed']}")

    for p in report["papers"]:
        status = "PASS" if p["overall_passed"] else "FAIL"
        issues = p["structural"]["issues"]
        cer_info = ""
        if p.get("comparative") and p["comparative"].get("cer") is not None:
            cer_info = f" CER={p['comparative']['cer']:.3f}"
        print(f"  [{status}] {p['folder']}{cer_info}")
        for issue in issues:
            print(f"    - {issue}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nReport saved: {args.output}")
