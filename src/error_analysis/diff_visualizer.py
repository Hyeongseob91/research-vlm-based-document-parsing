"""
Diff Visualization for Document Parsing

Creates visual comparisons between ground truth and parsed output.
"""

import difflib
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffResult:
    """Result of text diff operation."""
    additions: list[str]
    deletions: list[str]
    unchanged: list[str]
    html_diff: str
    similarity_ratio: float


class TextDiff:
    """
    Creates visual diffs between ground truth and parsed text.

    Supports HTML and terminal output formats.
    """

    def __init__(
        self,
        context_lines: int = 3,
        word_diff: bool = True
    ):
        self.context_lines = context_lines
        self.word_diff = word_diff

    def diff(
        self,
        ground_truth: str,
        parsed_text: str,
        title: str = "Diff"
    ) -> DiffResult:
        """
        Generate diff between ground truth and parsed text.

        Args:
            ground_truth: Reference text
            parsed_text: Parsed output to compare
            title: Title for the diff output

        Returns:
            DiffResult with various diff representations
        """
        gt_lines = ground_truth.splitlines(keepends=True)
        parsed_lines = parsed_text.splitlines(keepends=True)

        # Calculate similarity
        matcher = difflib.SequenceMatcher(None, ground_truth, parsed_text)
        similarity = matcher.ratio()

        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            gt_lines,
            parsed_lines,
            fromfile='Ground Truth',
            tofile='Parsed Output',
            lineterm=''
        ))

        # Categorize changes
        additions = []
        deletions = []
        unchanged = []

        for line in diff_lines[2:]:  # Skip header
            if line.startswith('+') and not line.startswith('+++'):
                additions.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                deletions.append(line[1:])
            elif line.startswith(' '):
                unchanged.append(line[1:])

        # Generate HTML diff
        html_diff = self._generate_html_diff(gt_lines, parsed_lines, title)

        return DiffResult(
            additions=additions,
            deletions=deletions,
            unchanged=unchanged,
            html_diff=html_diff,
            similarity_ratio=similarity,
        )

    def _generate_html_diff(
        self,
        gt_lines: list[str],
        parsed_lines: list[str],
        title: str
    ) -> str:
        """Generate HTML visualization of diff."""
        differ = difflib.HtmlDiff(wrapcolumn=80)
        html = differ.make_file(
            gt_lines,
            parsed_lines,
            fromdesc='Ground Truth',
            todesc='Parsed Output',
            context=True,
            numlines=self.context_lines
        )

        # Add custom styling
        custom_css = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            table.diff { border-collapse: collapse; width: 100%; }
            .diff_header { background-color: #e8e8e8; }
            .diff_next { background-color: #c0c0c0; }
            .diff_add { background-color: #aaffaa; }
            .diff_chg { background-color: #ffff77; }
            .diff_sub { background-color: #ffaaaa; }
            td { padding: 2px 4px; vertical-align: top; }
            td.diff_header { font-weight: bold; }
            .summary { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        </style>
        """

        # Insert custom CSS after <head>
        html = html.replace('</head>', custom_css + '</head>')

        return html


def create_html_diff(
    ground_truth: str,
    parsed_text: str,
    output_path: str,
    title: str = "Document Parsing Diff"
) -> DiffResult:
    """
    Create and save HTML diff to file.

    Args:
        ground_truth: Reference text
        parsed_text: Parsed output
        output_path: Path to save HTML file
        title: Title for the diff

    Returns:
        DiffResult with diff information
    """
    differ = TextDiff()
    result = differ.diff(ground_truth, parsed_text, title)

    # Add summary section
    summary_html = f"""
    <div class="summary">
        <h2>Comparison Summary</h2>
        <p><strong>Similarity:</strong> {result.similarity_ratio:.1%}</p>
        <p><strong>Lines Added:</strong> {len(result.additions)}</p>
        <p><strong>Lines Deleted:</strong> {len(result.deletions)}</p>
        <p><strong>Lines Unchanged:</strong> {len(result.unchanged)}</p>
    </div>
    """

    # Insert summary before the diff table
    html = result.html_diff.replace('<body>', f'<body>\n<h1>{title}</h1>\n{summary_html}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return result


def create_side_by_side_comparison(
    ground_truth: str,
    parsed_outputs: dict[str, str],
    output_path: str,
    title: str = "Parser Comparison"
) -> str:
    """
    Create side-by-side comparison of multiple parsers.

    Args:
        ground_truth: Reference text
        parsed_outputs: Dict mapping parser name to output
        output_path: Path to save HTML file
        title: Title for comparison

    Returns:
        Path to generated HTML file
    """
    # Start HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
        }}
        .container {{
            display: flex;
            gap: 20px;
            overflow-x: auto;
        }}
        .column {{
            flex: 1;
            min-width: 300px;
            max-width: 500px;
        }}
        .column h2 {{
            background: #333;
            color: white;
            padding: 10px;
            margin: 0;
            border-radius: 5px 5px 0 0;
        }}
        .content {{
            border: 1px solid #ddd;
            padding: 15px;
            background: #fafafa;
            max-height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.4;
        }}
        .gt-column .content {{
            background: #f0fff0;
        }}
        .stats {{
            margin-top: 10px;
            padding: 10px;
            background: #e8e8e8;
            border-radius: 0 0 5px 5px;
            font-size: 14px;
        }}
        .highlight-add {{ background-color: #aaffaa; }}
        .highlight-del {{ background-color: #ffaaaa; }}
        h1 {{
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="container">
        <div class="column gt-column">
            <h2>Ground Truth</h2>
            <div class="content">{_escape_html(ground_truth)}</div>
            <div class="stats">Characters: {len(ground_truth)}</div>
        </div>
"""

    # Add columns for each parser
    for parser_name, output in parsed_outputs.items():
        similarity = difflib.SequenceMatcher(None, ground_truth, output).ratio()
        html += f"""
        <div class="column">
            <h2>{parser_name}</h2>
            <div class="content">{_escape_html(output)}</div>
            <div class="stats">
                Characters: {len(output)} |
                Similarity: {similarity:.1%}
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )
