"""Markdown processing utilities."""

import re


def extract_headings(markdown: str) -> list[dict]:
    """Extract heading hierarchy from markdown text."""
    headings = []
    for line in markdown.split("\n"):
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            headings.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
            })
    return headings


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from VLM output."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        return parts[1].strip() if len(parts) > 1 else ""
    return text


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping."""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text
