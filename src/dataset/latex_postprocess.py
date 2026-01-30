"""
Post-processing for pandoc-generated Markdown.

Cleans up LaTeX artifacts, normalizes formatting, and ensures
consistency with the project's GT markdown format.
"""

import re


def postprocess_markdown(text: str) -> str:
    """Apply all post-processing steps to pandoc output."""
    text = remove_latex_remnants(text)
    text = normalize_equations(text)
    text = normalize_tables(text)
    text = normalize_figure_captions(text)
    text = normalize_headings(text)
    text = normalize_whitespace(text)
    return text


def remove_latex_remnants(text: str) -> str:
    """Remove leftover LaTeX commands that pandoc didn't convert."""
    # Raw LaTeX blocks that pandoc wraps
    text = re.sub(r"```\{=latex\}\n.*?\n```", "", text, flags=re.DOTALL)

    # Inline raw LaTeX
    text = re.sub(r"`\{=latex\}[^`]*`", "", text)

    # Remaining \command{...} patterns
    text = re.sub(r"\\(?:textsc|texttt|textsf)\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:footnote)\{([^}]+)\}", r" (\1)", text)
    text = re.sub(r"\\(?:url|href)\{([^}]+)\}", r"\1", text)

    # Remove \label, \tag
    text = re.sub(r"\\(?:label|tag)\{[^}]*\}", "", text)

    # Remove standalone LaTeX commands with no arguments
    text = re.sub(r"\\(?:centering|noindent|newpage|clearpage|pagebreak)\b", "", text)

    # Remove \begin{...} / \end{...} for non-converted environments
    text = re.sub(r"\\(?:begin|end)\{(?:center|flushleft|flushright)\}", "", text)

    # --- Phase 2: Pandoc div 블록 제거 ---
    # ::: {.center}, ::: {.wrapfigure}, ::: 단독 라인
    text = re.sub(r"^:::\s*\{[^}]*\}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^:::\s*$", "", text, flags=re.MULTILINE)

    # Pandoc 참조 속성: {#fig:xxx}, {reference-type="ref" reference="sec:xxx"}
    text = re.sub(r"\{#[^}]*\}", "", text)
    text = re.sub(r'\{reference-type="[^"]*"\s+reference="[^"]*"\}', "", text)
    # [\[tab:config\]]{#tab:config label="tab:config"} 형태
    text = re.sub(r"\[\\?\[[\w:.-]+\\?\]\]\{[^}]*\}", "", text)

    # 인용 브라켓: [@hochreiter1997], [@author1; @author2]
    text = re.sub(r"\[@[^\]]*\]", "", text)

    # 각주 정의 블록: [^1]: footnote text (줄 시작)
    text = re.sub(r"^\[\^\w+\]:.*$", "", text, flags=re.MULTILINE)
    # 각주 참조: [^4], [^note]
    text = re.sub(r"\[\^[^\]]+\]", "", text)

    return text


def normalize_equations(text: str) -> str:
    """Ensure equations are properly formatted with $$ delimiters."""
    # Pandoc sometimes produces $$...$$ on single line — split for readability
    def split_display_eq(match):
        inner = match.group(1).strip()
        if "\n" in inner:
            return f"$$\n{inner}\n$$"
        return f"$$\n{inner}\n$$"

    text = re.sub(r"\$\$(.+?)\$\$", split_display_eq, text, flags=re.DOTALL)

    # Ensure blank lines around $$ blocks
    text = re.sub(r"([^\n])\n\$\$", r"\1\n\n$$", text)
    text = re.sub(r"\$\$\n([^\n])", r"$$\n\n\1", text)

    return text


def normalize_tables(text: str) -> str:
    """
    Normalize markdown tables.

    Ensures separator row exists after header, cleans up alignment.
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect table row (starts with |)
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            # Collect all consecutive table lines
            table_lines = []
            while i < len(lines) and (
                lines[i].strip().startswith("|")
                or re.match(r"^\s*\|[-:| ]+\|\s*$", lines[i])
            ):
                table_lines.append(lines[i])
                i += 1

            # Ensure separator after first row
            if len(table_lines) >= 1:
                result.append(table_lines[0])
                if len(table_lines) >= 2 and re.match(
                    r"^\s*\|[-:| ]+\|\s*$", table_lines[1]
                ):
                    # Separator already present
                    for tl in table_lines[1:]:
                        result.append(tl)
                else:
                    # Generate separator
                    cols = table_lines[0].count("|") - 1
                    if cols > 0:
                        sep = "| " + " | ".join("---" for _ in range(cols)) + " |"
                        result.append(sep)
                    for tl in table_lines[1:]:
                        result.append(tl)
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


def normalize_figure_captions(text: str) -> str:
    """
    Retain figure captions in italic format.

    Pandoc may produce: Figure N: caption text
    Convert to: *Figure N: caption text*
    """
    text = re.sub(
        r"^(Figure \d+[.:].+)$",
        r"*\1*",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(Table \d+[.:].+)$",
        r"*\1*",
        text,
        flags=re.MULTILINE,
    )
    return text


def normalize_headings(text: str) -> str:
    """Ensure ATX-style headings with proper spacing."""
    # Fix headings without space after #
    text = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", text, flags=re.MULTILINE)

    # Ensure blank line before headings
    text = re.sub(r"([^\n])\n(#{1,6} )", r"\1\n\n\2", text)

    return text


def normalize_whitespace(text: str) -> str:
    """Clean up excessive whitespace."""
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Remove trailing whitespace on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Ensure single newline at end
    text = text.strip() + "\n"

    return text
