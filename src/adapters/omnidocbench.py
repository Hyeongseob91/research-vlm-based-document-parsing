"""
OmniDocBench dataset adapter.

Converts OmniDocBench (CVPR 2025) JSON ground truth into the pipeline's
markdown GT format. Supports 1,355 PDF pages, 9 document types, and 3 languages
(English, Chinese, mixed).

OmniDocBench JSON GT structure:
{
  "page_info": { "page_no": 0, "height": ..., "width": ... },
  "layout_dets": [
    {
      "category_type": "text" | "title" | "table" | "figure" | "equation" | ...,
      "order": int,
      "text": "...",
      "html": "<table>...</table>",  # present only for "table" category
      ...
    }
  ]
}
"""

import json
import re
from pathlib import Path
from typing import Optional


# Helper: extract text from a block (text field or line_with_spans fallback)
def _get_block_text(block: dict) -> str:
    text = block.get("text")
    if text:
        return text.strip()
    # Some blocks store text inside line_with_spans
    spans = block.get("line_with_spans", [])
    if spans:
        return " ".join(s.get("text", "") for s in spans).strip()
    return ""


# Helper: extract latex from equation blocks
def _get_equation_text(block: dict) -> str:
    latex = block.get("latex", "")
    if latex:
        # Already wrapped in $$ by GT
        stripped = latex.strip()
        if stripped.startswith("$$"):
            return stripped
        return f"$$\n{stripped}\n$$"
    return _get_block_text(block)


# OmniDocBench category_type → Markdown conversion mapping
_CATEGORY_HANDLERS = {
    "title": lambda block: f"## {_get_block_text(block)}",
    "text_block": lambda block: _get_block_text(block),
    "equation_isolated": lambda block: _get_equation_text(block),
    "equation_caption": lambda block: _get_block_text(block),
    "equation_explanation": lambda block: _get_block_text(block),
    "figure_caption": lambda block: f"*{_get_block_text(block)}*",
    "table_caption": lambda block: f"*{_get_block_text(block)}*",
    "table_footnote": lambda block: _get_block_text(block),
    "figure_footnote": lambda block: _get_block_text(block),
    "page_footnote": lambda block: _get_block_text(block),
    "header": lambda block: _get_block_text(block),
    "footer": lambda block: _get_block_text(block),
    "page_number": lambda block: "",  # skip page numbers
    "reference": lambda block: _get_block_text(block),
    "code_txt": lambda block: f"```\n{_get_block_text(block)}\n```",
    "code_txt_caption": lambda block: _get_block_text(block),
    "figure": lambda block: "",  # images cannot be converted to markdown
    "abandon": lambda block: "",  # intentionally ignored blocks
    "text_mask": lambda block: "",  # masked regions
    "table_mask": lambda block: "",  # masked regions
    "need_mask": lambda block: "",  # masked regions
    # Legacy names (keep for compatibility)
    "text": lambda block: _get_block_text(block),
    "equation": lambda block: _get_equation_text(block),
}


def _table_block_to_markdown(block: dict) -> str:
    """
    Convert an OmniDocBench table block to markdown.

    Uses HTML → markdown table conversion when HTML is present;
    otherwise falls back to the raw text field.
    """
    html = block.get("html", "")
    if html:
        return _html_table_to_markdown(html)
    return block.get("text", "").strip()


def _html_table_to_markdown(html: str) -> str:
    """
    Convert an HTML <table> into a simple markdown table.
    """
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.current_row = []
            self.current_cell = ""
            self.in_cell = False
            self.is_header_row = False
            self.had_header = False

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.current_row = []
                self.is_header_row = False
            elif tag in ("td", "th"):
                self.in_cell = True
                self.current_cell = ""
                if tag == "th":
                    self.is_header_row = True

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self.in_cell = False
                self.current_row.append(self.current_cell.strip())
            elif tag == "tr":
                if self.current_row:
                    self.rows.append((self.current_row, self.is_header_row))

        def handle_data(self, data):
            if self.in_cell:
                self.current_cell += data

    parser = _TableParser()
    parser.feed(html)

    if not parser.rows:
        return ""

    lines = []
    header_sep_added = False
    for i, (cells, is_header) in enumerate(parser.rows):
        line = "| " + " | ".join(cells) + " |"
        lines.append(line)
        # Add separator after the first row (markdown table requires it)
        if i == 0 and not header_sep_added:
            sep = "| " + " | ".join("---" for _ in cells) + " |"
            lines.append(sep)
            header_sep_added = True

    return "\n".join(lines)


def convert_omnidocbench_page_to_markdown(page_data: dict) -> str:
    """
    Convert a single OmniDocBench page JSON to markdown.

    Args:
        page_data: OmniDocBench JSON page data (layout_dets, page_info, etc.).

    Returns:
        Markdown string for the page.
    """
    layout_dets = page_data.get("layout_dets", [])

    sorted_blocks = sorted(layout_dets, key=lambda b: b.get("order") or 0)

    parts = []
    for block in sorted_blocks:
        cat = block.get("category_type", "")

        if cat == "table":
            md = _table_block_to_markdown(block)
        elif cat in _CATEGORY_HANDLERS:
            md = _CATEGORY_HANDLERS[cat](block)
        else:
            md = block.get("text", "").strip()

        if md:
            parts.append(md)

    return "\n\n".join(parts)


def extract_table_html_from_page(page_data: dict) -> list[str]:
    """
    Extract table HTML ground truth from an OmniDocBench page.

    Enables using GT HTML directly for TEDS computation.

    Returns:
        List of HTML <table> strings.
    """
    tables = []
    layout_dets = page_data.get("layout_dets", [])
    sorted_blocks = sorted(layout_dets, key=lambda b: b.get("order") or 0)

    for block in sorted_blocks:
        if block.get("category_type") == "table" and block.get("html"):
            tables.append(block["html"])

    return tables


def load_omnidocbench_dataset(
    dataset_path: Path,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Load the OmniDocBench dataset.

    Args:
        dataset_path: Root path of the OmniDocBench dataset
            (e.g. omnidocbench/OmniDocBench_demo.json or a directory
            of per-page JSON files).
        limit: Maximum number of pages to load; None for no limit.

    Returns:
        List of page dicts, each with:
        - page_id: str
        - image_path: Path or None
        - gt_markdown: str
        - gt_table_htmls: list[str]
        - raw_data: original JSON item
        - metadata: dict (doc_type, language, etc.)
    """
    pages = []

    # Single JSON file
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        else:
            items = [data]

        for item in items:
            page_info = item.get("page_info", {})
            page_id = page_info.get("page_no", len(pages))
            image_name = page_info.get("image_path", "")

            image_path = (
                dataset_path.parent / image_name if image_name else None
            )

            gt_md = convert_omnidocbench_page_to_markdown(item)
            gt_tables = extract_table_html_from_page(item)

            page_attr = page_info.get("page_attribute", {})
            pages.append({
                "page_id": str(page_id),
                "image_path": image_path,
                "gt_markdown": gt_md,
                "gt_table_htmls": gt_tables,
                "raw_data": item,
                "metadata": {
                    "doc_type": page_attr.get("data_source", "unknown"),
                    "language": page_attr.get("language", "unknown"),
                    "layout": page_attr.get("layout", "unknown"),
                },
            })

            if limit and len(pages) >= limit:
                break

    # Directory: scan all JSON files
    elif dataset_path.is_dir():
        json_files = sorted(dataset_path.glob("*.json"))

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    pi = item.get("page_info", {})
                    page_id = pi.get("page_no", len(pages))
                    pa = pi.get("page_attribute", {})
                    gt_md = convert_omnidocbench_page_to_markdown(item)
                    gt_tables = extract_table_html_from_page(item)

                    pages.append({
                        "page_id": f"{jf.stem}_{page_id}",
                        "image_path": None,
                        "gt_markdown": gt_md,
                        "gt_table_htmls": gt_tables,
                        "raw_data": item,
                        "metadata": {
                            "doc_type": pa.get("data_source", "unknown"),
                            "language": pa.get("language", "unknown"),
                            "layout": pa.get("layout", "unknown"),
                        },
                    })

                    if limit and len(pages) >= limit:
                        break
            else:
                pi = data.get("page_info", {})
                page_id = pi.get("page_no", len(pages))
                pa = pi.get("page_attribute", {})
                gt_md = convert_omnidocbench_page_to_markdown(data)
                gt_tables = extract_table_html_from_page(data)

                pages.append({
                    "page_id": f"{jf.stem}_{page_id}",
                    "image_path": None,
                    "gt_markdown": gt_md,
                    "gt_table_htmls": gt_tables,
                    "raw_data": data,
                    "metadata": {
                        "doc_type": pa.get("data_source", "unknown"),
                        "language": pa.get("language", "unknown"),
                        "layout": pa.get("layout", "unknown"),
                    },
                })

            if limit and len(pages) >= limit:
                break

    return pages
