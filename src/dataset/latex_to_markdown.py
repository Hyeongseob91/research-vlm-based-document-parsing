"""
LaTeX to Markdown conversion pipeline.

Three stages:
  A. LaTeX preprocessing: merge \\input files, strip preamble, expand macros
  B. pandoc conversion: LaTeX → Markdown via pypandoc
  C. Post-processing: cleanup artifacts (see latex_postprocess.py)
"""

import re
import subprocess
from pathlib import Path
from typing import Optional


def merge_inputs(tex_path: Path) -> str:
    """
    Recursively resolve \\input{} and \\include{} commands
    into a single LaTeX string.
    """
    base_dir = tex_path.parent
    content = tex_path.read_text(encoding="utf-8", errors="ignore")
    return _resolve_inputs(content, base_dir, depth=0)


def _resolve_inputs(content: str, base_dir: Path, depth: int) -> str:
    """Recursively resolve \\input and \\include up to 10 levels."""
    if depth > 10:
        return content

    def replacer(match):
        filename = match.group(1)
        # Add .tex extension if missing
        if not filename.endswith(".tex"):
            filename += ".tex"

        # Search in base_dir and subdirectories
        candidates = [
            base_dir / filename,
            base_dir / Path(filename).name,
        ]
        # Also check without .tex if already has extension
        if filename.endswith(".tex"):
            candidates.append(base_dir / filename[:-4])

        for candidate in candidates:
            if candidate.exists():
                sub_content = candidate.read_text(
                    encoding="utf-8", errors="ignore"
                )
                return _resolve_inputs(sub_content, candidate.parent, depth + 1)

        # File not found — leave a comment
        return f"% [UNRESOLVED: {match.group(0)}]"

    # Match \input{file} and \include{file}
    pattern = r"\\(?:input|include)\{([^}]+)\}"
    return re.sub(pattern, replacer, content)


def merge_bbl(content: str, tex_path: Path) -> str:
    """Merge .bbl bibliography file if present."""
    base_dir = tex_path.parent
    bbl_files = list(base_dir.glob("*.bbl"))
    if not bbl_files:
        return content

    bbl_content = bbl_files[0].read_text(encoding="utf-8", errors="ignore")

    # Replace \bibliography{...} with actual bbl content
    content = re.sub(
        r"\\bibliography\{[^}]*\}",
        lambda _: bbl_content,
        content,
    )
    return content


def strip_preamble(content: str) -> str:
    """
    Extract content between \\begin{document} and \\end{document}.
    Preserves title/author/abstract if present in the body.
    """
    begin_match = re.search(r"\\begin\{document\}", content)
    end_match = re.search(r"\\end\{document\}", content)

    if begin_match:
        start = begin_match.end()
    else:
        start = 0

    if end_match:
        end = end_match.start()
    else:
        end = len(content)

    body = content[start:end].strip()

    # Extract title from preamble if \maketitle is in body
    if "\\maketitle" in body:
        title_match = re.search(r"\\title\{([^}]+)\}", content[:start] if begin_match else content)
        author_match = re.search(r"\\author\{([^}]+)\}", content[:start] if begin_match else content)
        header = ""
        if title_match:
            header += f"# {title_match.group(1).strip()}\n\n"
        if author_match:
            header += f"{author_match.group(1).strip()}\n\n"
        body = re.sub(r"\\maketitle", "", body)
        body = header + body

    return body


def strip_conditionals(content: str) -> str:
    """Remove LaTeX conditional blocks (\\if..\\fi) that crash pandoc."""
    # Remove \iffalse ... \fi blocks entirely
    content = re.sub(r"\\iffalse\b.*?\\fi\b", "", content, flags=re.DOTALL)

    # For other \if variants, keep the content but strip the commands
    # Match \if<word> ... \else ... \fi  or  \if<word> ... \fi
    # This is approximate — deeply nested conditionals may not fully resolve
    for _ in range(5):  # Multiple passes for nested conditionals
        prev = content
        # Remove \if...\else...\fi keeping the \if branch
        content = re.sub(
            r"\\if[a-zA-Z@]*\b[^\n]*\n(.*?)\\else\b(.*?)\\fi\b",
            r"\1",
            content,
            flags=re.DOTALL,
        )
        # Remove \if...\fi (no else)
        content = re.sub(
            r"\\if[a-zA-Z@]*\b[^\n]*\n(.*?)\\fi\b",
            r"\1",
            content,
            flags=re.DOTALL,
        )
        if content == prev:
            break

    # Remove stray \fi, \else
    content = re.sub(r"^\\fi\b.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\\else\b.*$", "", content, flags=re.MULTILINE)

    return content


def strip_custom_commands(content: str) -> str:
    """Remove \\newcommand, \\def, \\renewcommand definitions."""
    # Remove multi-line \newcommand blocks with nested braces
    # Match \newcommand or variants, then consume balanced braces
    lines = content.split("\n")
    result = []
    skip_depth = 0
    in_command_def = False

    for line in lines:
        stripped = line.strip()

        # Detect start of command definition
        if re.match(
            r"\\(?:new|renew|provide)command|\\def\\|\\DeclareMathOperator|\\newcolumntype|\\newlength|\\setlength|\\newcounter|\\setcounter",
            stripped,
        ):
            # Count braces to handle multi-line definitions
            open_b = line.count("{") - line.count("}")
            if open_b <= 0:
                # Single-line definition, skip it
                continue
            else:
                in_command_def = True
                skip_depth = open_b
                continue

        if in_command_def:
            skip_depth += line.count("{") - line.count("}")
            if skip_depth <= 0:
                in_command_def = False
            continue

        result.append(line)

    return "\n".join(result)


def expand_common_macros(content: str) -> str:
    """Expand commonly used LaTeX macros that pandoc doesn't handle."""
    # First strip conditionals and custom commands
    content = strip_conditionals(content)
    content = strip_custom_commands(content)

    # Bold/italic
    content = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", content)
    content = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", content)
    content = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", content)
    content = re.sub(r"\\underline\{([^}]+)\}", r"\1", content)

    # References — simplify
    content = re.sub(r"\\cite[tp]?\{([^}]+)\}", r"[\1]", content)
    content = re.sub(r"\\ref\{([^}]+)\}", r"[\1]", content)
    content = re.sub(r"\\eqref\{([^}]+)\}", r"([\1])", content)
    content = re.sub(r"\\autoref\{([^}]+)\}", r"[\1]", content)
    content = re.sub(r"\\cref\{([^}]+)\}", r"[\1]", content)

    # Remove labels
    content = re.sub(r"\\label\{[^}]+\}", "", content)

    # Remove \centering, \vspace, \hspace, \noindent, \smallskip etc.
    content = re.sub(
        r"\\(?:centering|noindent|raggedright|raggedleft)\b", "", content
    )
    content = re.sub(r"\\[vh]space\*?\{[^}]*\}", "", content)
    content = re.sub(r"\\(?:small|med|big)skip\b", "", content)
    content = re.sub(r"\\(?:newpage|clearpage|pagebreak)\b", "", content)

    # Figure environments — keep caption, remove includegraphics
    content = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}", "", content)

    # Remove comment lines
    content = re.sub(r"^%.*$", "", content, flags=re.MULTILINE)

    # Remove \thanks{...}
    content = re.sub(r"\\thanks\{[^}]*\}", "", content)

    # Remove style-only commands
    content = re.sub(r"\\(?:small|large|Large|LARGE|huge|Huge|footnotesize|scriptsize|tiny|normalsize)\b", "", content)

    # Remove orphaned \end{...} and \begin{...} for non-content environments
    non_content_envs = (
        "center", "flushleft", "flushright", "minipage",
        "wrapfigure", "figure\\*?", "table\\*?", "tikzpicture",
    )
    for env in non_content_envs:
        content = re.sub(rf"\\begin\{{{env}\}}",  "", content)
        content = re.sub(rf"\\end\{{{env}\}}", "", content)

    # Remove unknown single-line commands that pandoc can't handle
    content = re.sub(r"\\(?:AND|affaddr|affiliation)\b[^\n]*", "", content)
    content = re.sub(r"\\color\{[^}]*\}", "", content)

    return content


def convert_with_pandoc(
    tex_content: str,
    source_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Convert LaTeX content to Markdown using pandoc CLI.

    Uses subprocess to call pandoc directly (no pypandoc dependency needed
    if pandoc is installed).
    """
    cmd = [
        "pandoc",
        "-f", "latex",
        "-t", "markdown",
        "--wrap=none",
        "--markdown-headings=atx",
    ]

    if source_dir:
        cmd.extend(["--resource-path", str(source_dir)])

    try:
        result = subprocess.run(
            cmd,
            input=tex_content,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            # pandoc may produce partial output even on error
            if result.stdout and len(result.stdout.strip()) > 200:
                print(f"  pandoc warning (partial output used): {result.stderr[:200]}")
                return result.stdout
            print(f"  pandoc error: {result.stderr[:500]}")
            return None
        return result.stdout
    except FileNotFoundError:
        # Try pypandoc as fallback
        try:
            import pypandoc
            return pypandoc.convert_text(
                tex_content,
                "markdown",
                format="latex",
                extra_args=["--wrap=none", "--markdown-headings=atx"],
            )
        except ImportError:
            print("  Error: pandoc not found. Install pandoc or pypandoc.")
            return None
    except subprocess.TimeoutExpired:
        print("  pandoc timed out")
        return None


def latex_to_markdown(
    tex_path: Path,
    output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Full pipeline: LaTeX file → Markdown string.

    Args:
        tex_path: Path to the main .tex file.
        output_path: Optional path to write the markdown output.

    Returns:
        Markdown string or None on failure.
    """
    from src.dataset.latex_postprocess import postprocess_markdown

    source_dir = tex_path.parent

    # Strategy 1: Give pandoc the merged full document (best quality)
    content = merge_inputs(tex_path)
    content = merge_bbl(content, tex_path)

    # Light cleanup: strip conditionals and custom commands that crash pandoc,
    # but keep document structure intact for pandoc to parse
    content = strip_conditionals(content)
    content = strip_custom_commands(content)

    markdown = convert_with_pandoc(content, source_dir)

    # Strategy 2: If full document fails, try with our manual preprocessing
    if markdown is None or len(markdown.strip()) < 200:
        content = merge_inputs(tex_path)
        content = merge_bbl(content, tex_path)
        content = strip_preamble(content)
        content = expand_common_macros(content)

        # Wrap body in document structure
        wrapped = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            + content
            + "\n\\end{document}\n"
        )
        markdown = convert_with_pandoc(wrapped, source_dir)

    if markdown is None:
        return None

    # Stage C: Post-processing
    markdown = postprocess_markdown(markdown)

    if output_path:
        output_path.write_text(markdown, encoding="utf-8")

    return markdown
