"""Text normalization for OmniDocBench evaluation.

Normalizes text, formulas, and tables before metric computation.
Based on OmniDocBench's data_preprocess.py.
"""

import re
import unicodedata
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def clean_string(text: str) -> str:
    """Normalize text to alphanumeric + CJK characters only.

    Removes whitespace, punctuation, and special characters.
    Used for NED comparison of text blocks.
    """
    text = text.replace("\\t", "").replace("\\n", "")
    text = text.replace("\t", "").replace("\n", "")
    # Keep only word chars and CJK
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """NFKC normalize for consistent comparison."""
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# Formula normalization
# ---------------------------------------------------------------------------

_FORMULA_STRIP = [
    r"\\mathbf", r"\\mathrm", r"\\mathit", r"\\mathbb", r"\\mathcal",
    r"\\mathscr", r"\\mathfrak", r"\\textbf", r"\\text", r"\\boldmath",
    r"\\boldsymbol", r"\\operatorname", r"\\left", r"\\right",
    r"\\displaystyle", r"\\notag", r"\\quad", r"\\qquad", r"\\space",
    r"\\thinspace", r"\\thickspace", r"\\negthinspace",
    r"\\hfill", r"\\hfil",
]

_FORMULA_REMOVE_PATTERNS = [
    (r"\\tag\{.*?\}", ""),
    (r"\\hspace\{.*?\}", ""),
    (r"\\vspace\{.*?\}", ""),
    (r"\\label\{.*?\}", ""),
    (r"\\begin\{.*?\}", ""),
    (r"\\end\{.*?\}", ""),
    (r"\\arraycolsep\s*=\s*\S+", ""),
    (r"\\rule\{.*?\}\{.*?\}", ""),
]


def normalize_formula(text: str) -> str:
    """Normalize LaTeX formula for comparison.

    Strips styling commands, delimiters, and whitespace.
    """
    # Strip delimiters (must use prefix/suffix check, not str.strip)
    text = text.strip()
    for prefix, suffix in [("$$", "$$"), (r"\[", r"\]"), ("$", "$")]:
        if text.startswith(prefix) and text.endswith(suffix):
            text = text[len(prefix):-len(suffix)] if len(suffix) > 0 else text[len(prefix):]
            break
    text = text.strip()

    # Remove styling commands
    for cmd in _FORMULA_STRIP:
        text = text.replace(cmd, "")

    # Remove patterns
    for pattern, repl in _FORMULA_REMOVE_PATTERNS:
        text = re.sub(pattern, repl, text)

    # Normalize
    text = text.lower().strip().rstrip(".")
    return text


# ---------------------------------------------------------------------------
# Table normalization
# ---------------------------------------------------------------------------

def normalize_html_table(html_str: str) -> str:
    """Normalize HTML table for TEDS comparison.

    - Converts <th> to <td>
    - Removes styling attributes
    - Normalizes whitespace
    - Wraps in standard HTML structure
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: basic regex cleanup
        return _basic_html_normalize(html_str)

    soup = BeautifulSoup(html_str, "html.parser")

    # Convert th → td
    for th in soup.find_all("th"):
        th.name = "td"

    # Unwrap thead
    for thead in soup.find_all("thead"):
        thead.unwrap()

    # Remove styling attributes
    for tag in soup.find_all(True):
        for attr in ["style", "height", "width", "align", "class", "valign", "bgcolor"]:
            if tag.has_attr(attr):
                del tag[attr]

    # Remove span wrappers
    for span in soup.find_all("span"):
        span.unwrap()

    # Remove tbody
    for tbody in soup.find_all("tbody"):
        tbody.unwrap()

    # Get table content
    table = soup.find("table")
    if table:
        table_html = str(table)
    else:
        table_html = str(soup)

    # Normalize whitespace
    table_html = re.sub(r"\s+", " ", table_html).strip()

    return f'<html><body><table border="1">{_inner_table(table_html)}</table></body></html>'


def _inner_table(html_str: str) -> str:
    """Extract inner content of <table> tags."""
    match = re.search(r"<table[^>]*>(.*)</table>", html_str, re.DOTALL)
    return match.group(1) if match else html_str


def _basic_html_normalize(html_str: str) -> str:
    """Basic HTML normalization without BeautifulSoup."""
    html_str = re.sub(r"<th(\s|>)", r"<td\1", html_str)
    html_str = re.sub(r"</th>", "</td>", html_str)
    html_str = re.sub(r"</?thead>", "", html_str)
    html_str = re.sub(r"</?tbody>", "", html_str)
    html_str = re.sub(r'\s*(style|class|align|width|height|bgcolor|valign)="[^"]*"', "", html_str)
    html_str = re.sub(r"\s+", " ", html_str).strip()

    inner = _inner_table(html_str)
    return f'<html><body><table border="1">{inner}</table></body></html>'


# ---------------------------------------------------------------------------
# Markdown table → HTML conversion
# ---------------------------------------------------------------------------

def markdown_table_to_html(md_table: str) -> str:
    """Convert a markdown table string to HTML table.

    Args:
        md_table: Markdown table with | delimiters and |---| separator.

    Returns:
        HTML table string.
    """
    lines = [l.strip() for l in md_table.strip().split("\n") if l.strip()]
    if not lines:
        return ""

    rows = []
    for line in lines:
        # Skip separator row
        if re.match(r"^\|[\s\-:]+\|$", line):
            continue
        # Parse cells
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last from leading/trailing |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    html_parts = ["<table>"]
    for i, row in enumerate(rows):
        html_parts.append("<tr>")
        for cell in row:
            html_parts.append(f"<td>{cell}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")

    return "".join(html_parts)
