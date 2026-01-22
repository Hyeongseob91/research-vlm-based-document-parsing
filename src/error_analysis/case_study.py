"""
Case Study Generator for Error Analysis

Creates detailed case study reports for selected parsing errors.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .analyzer import ParsingError, ErrorCategory, ErrorSeverity, ErrorAnalyzer
from .diff_visualizer import create_html_diff


@dataclass
class CaseStudy:
    """Represents a single error case study."""
    id: str
    title: str
    category: ErrorCategory
    severity: ErrorSeverity
    document_id: str
    parser_name: str
    error: ParsingError
    analysis: str
    impact: str
    recommendation: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category.value,
            "severity": self.severity.value,
            "document_id": self.document_id,
            "parser_name": self.parser_name,
            "error": self.error.to_dict(),
            "analysis": self.analysis,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        """Generate markdown representation of case study."""
        return f"""### Case Study {self.id}: {self.title}

**Category**: {self.category.value}
**Severity**: {self.severity.value}
**Document**: {self.document_id}
**Parser**: {self.parser_name}

#### Ground Truth

```
{self.error.gt_text}
```

#### Parsed Output

```
{self.error.parsed_text}
```

#### Analysis

{self.analysis}

#### Impact on RAG

{self.impact}

#### Recommendation

{self.recommendation}

---
"""


class CaseStudyGenerator:
    """
    Generates case studies from parsing errors.

    Selects representative errors and creates detailed analysis.
    """

    # Templates for analysis based on error category
    ANALYSIS_TEMPLATES = {
        ErrorCategory.TABLE_STRUCTURE: """
The table structure was {status} during parsing. This error occurs when:
- Table borders are not clearly defined in the PDF
- Complex table structures (merged cells, nested headers) are present
- OCR engines fail to recognize table layout

**Root Cause**: {root_cause}

**Parsing Behavior**: The {parser} parser {behavior}.
""",
        ErrorCategory.MULTI_COLUMN: """
Multi-column reading order was incorrectly interpreted. This happens when:
- Two-column academic paper layout confuses the parser
- Reading order heuristics fail
- Visual column separation is not detected

**Root Cause**: {root_cause}

**Parsing Behavior**: The {parser} parser {behavior}.
""",
        ErrorCategory.HALLUCINATION: """
The parser generated content not present in the original document. This is a critical error that:
- Introduces false information into the knowledge base
- May mislead downstream RAG applications
- Indicates prompt engineering issues (for VLM)

**Root Cause**: {root_cause}

**Parsing Behavior**: The {parser} parser {behavior}.
""",
        ErrorCategory.DELETION: """
Content was missing from the parsed output. This occurs when:
- Low-contrast text is not recognized
- Text overlaps with images or other elements
- Certain fonts or encodings are not supported

**Root Cause**: {root_cause}

**Parsing Behavior**: The {parser} parser {behavior}.
""",
    }

    IMPACT_TEMPLATES = {
        ErrorSeverity.CRITICAL: """
**Critical Impact**: This error renders the affected content unusable for retrieval.
- Queries targeting this content will fail
- Semantic chunking may be severely compromised
- User queries will return incomplete or incorrect information
""",
        ErrorSeverity.MAJOR: """
**Major Impact**: This error significantly degrades retrieval quality.
- Related queries may return incorrect results
- Chunk coherence is affected
- Some user queries will fail
""",
        ErrorSeverity.MEDIUM: """
**Medium Impact**: This error affects structural integrity.
- Chunking boundaries may be suboptimal
- Some related content may be split incorrectly
- Retrieval accuracy is partially affected
""",
        ErrorSeverity.MINOR: """
**Minor Impact**: This error has limited effect on retrieval.
- Character-level accuracy is affected
- Most queries should still succeed
- User experience minimally impacted
""",
    }

    def __init__(
        self,
        max_case_studies: int = 10,
        output_dir: str = "results/errors"
    ):
        self.max_case_studies = max_case_studies
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_errors(
        self,
        errors: list[ParsingError],
        document_id: str,
        parser_name: str,
        ground_truth: Optional[str] = None,
        parsed_text: Optional[str] = None
    ) -> list[CaseStudy]:
        """
        Generate case studies from detected errors.

        Args:
            errors: List of detected ParsingError objects
            document_id: ID of the source document
            parser_name: Name of the parser that produced these errors
            ground_truth: Full ground truth text (for context)
            parsed_text: Full parsed text (for context)

        Returns:
            List of CaseStudy objects
        """
        # Select representative errors
        selected = self._select_representative_errors(errors)

        case_studies = []
        for i, error in enumerate(selected):
            case_study = self._create_case_study(
                error=error,
                index=i + 1,
                document_id=document_id,
                parser_name=parser_name,
            )
            case_studies.append(case_study)

        return case_studies

    def _select_representative_errors(
        self,
        errors: list[ParsingError]
    ) -> list[ParsingError]:
        """Select diverse, representative errors for case studies."""
        selected = []
        seen_categories = set()
        seen_severities = set()

        # First pass: ensure coverage of categories and severities
        for error in sorted(errors, key=lambda e: e.severity.value):
            if error.category not in seen_categories or error.severity not in seen_severities:
                selected.append(error)
                seen_categories.add(error.category)
                seen_severities.add(error.severity)

                if len(selected) >= self.max_case_studies // 2:
                    break

        # Second pass: fill remaining slots with most severe errors
        remaining_slots = self.max_case_studies - len(selected)
        for error in errors:
            if error not in selected and remaining_slots > 0:
                selected.append(error)
                remaining_slots -= 1

        return selected[:self.max_case_studies]

    def _create_case_study(
        self,
        error: ParsingError,
        index: int,
        document_id: str,
        parser_name: str
    ) -> CaseStudy:
        """Create a single case study from an error."""
        # Generate title
        title = self._generate_title(error)

        # Generate analysis
        analysis = self._generate_analysis(error, parser_name)

        # Generate impact assessment
        impact = self._generate_impact(error)

        # Generate recommendation
        recommendation = self._generate_recommendation(error, parser_name)

        return CaseStudy(
            id=f"CS-{document_id}-{index:02d}",
            title=title,
            category=error.category,
            severity=error.severity,
            document_id=document_id,
            parser_name=parser_name,
            error=error,
            analysis=analysis,
            impact=impact,
            recommendation=recommendation,
        )

    def _generate_title(self, error: ParsingError) -> str:
        """Generate descriptive title for case study."""
        category_titles = {
            ErrorCategory.TABLE_STRUCTURE: "Table Structure Corruption",
            ErrorCategory.MULTI_COLUMN: "Multi-Column Reading Order Error",
            ErrorCategory.HALLUCINATION: "Content Hallucination",
            ErrorCategory.HEADER_HIERARCHY: "Header Level Mismatch",
            ErrorCategory.DELETION: "Content Deletion",
            ErrorCategory.SUBSTITUTION: "Character Recognition Error",
            ErrorCategory.FORMATTING: "Formatting Loss",
            ErrorCategory.ORDERING: "Content Ordering Error",
            ErrorCategory.OTHER: "Parsing Error",
        }
        return f"{category_titles.get(error.category, 'Error')} ({error.severity.value})"

    def _generate_analysis(self, error: ParsingError, parser_name: str) -> str:
        """Generate detailed analysis for error."""
        template = self.ANALYSIS_TEMPLATES.get(
            error.category,
            "Error detected during parsing. Further investigation required."
        )

        # Determine root cause and behavior based on category
        root_cause = self._infer_root_cause(error)
        behavior = self._describe_parser_behavior(error, parser_name)

        analysis = template.format(
            status=self._error_status(error),
            root_cause=root_cause,
            parser=parser_name,
            behavior=behavior,
        )

        return analysis.strip()

    def _generate_impact(self, error: ParsingError) -> str:
        """Generate impact assessment."""
        return self.IMPACT_TEMPLATES.get(
            error.severity,
            "Impact assessment not available."
        ).strip()

    def _generate_recommendation(
        self,
        error: ParsingError,
        parser_name: str
    ) -> str:
        """Generate recommendation for addressing error."""
        recommendations = {
            ErrorCategory.TABLE_STRUCTURE: """
1. For VLM: Use explicit table parsing instructions in prompt
2. Consider pre-processing to enhance table borders
3. For complex tables, use specialized table extraction tools
4. Validate table structure post-parsing
""",
            ErrorCategory.MULTI_COLUMN: """
1. For VLM: This is a strength - VLM typically handles multi-column well
2. For OCR: Consider layout analysis pre-processing
3. Route multi-column documents to VLM parser
4. Implement column detection and reordering
""",
            ErrorCategory.HALLUCINATION: """
1. Use transcription-focused prompts (v2) instead of extraction prompts
2. Add explicit "do not add content" instructions
3. Implement post-parsing validation against source
4. Consider temperature=0 for deterministic output
""",
            ErrorCategory.DELETION: """
1. Check image resolution (use 300 DPI)
2. Verify text contrast in source document
3. Consider OCR confidence thresholds
4. Implement missing content detection
""",
        }

        return recommendations.get(
            error.category,
            "Review parser configuration and consider alternative approaches."
        ).strip()

    def _error_status(self, error: ParsingError) -> str:
        """Describe the error status."""
        if error.category == ErrorCategory.TABLE_STRUCTURE:
            return "corrupted" if error.severity == ErrorSeverity.CRITICAL else "partially preserved"
        return "affected"

    def _infer_root_cause(self, error: ParsingError) -> str:
        """Infer root cause from error details."""
        if error.category == ErrorCategory.TABLE_STRUCTURE:
            if "missing" in error.description.lower():
                return "Table detection failed - no table structure recognized"
            return "Table cell alignment or structure parsing failed"

        if error.category == ErrorCategory.HALLUCINATION:
            return "Model generated content beyond source document"

        if error.category == ErrorCategory.DELETION:
            if len(error.gt_text) > 100:
                return "Large section of text was not recognized"
            return "Text extraction failed for this region"

        return "Parser encountered unexpected content structure"

    def _describe_parser_behavior(
        self,
        error: ParsingError,
        parser_name: str
    ) -> str:
        """Describe how the parser behaved."""
        if parser_name.lower() == "vlm":
            if error.category == ErrorCategory.HALLUCINATION:
                return "interpreted the instruction too liberally and added contextual information"
            return "attempted to preserve structure but encountered limitations"

        if "pdfplumber" in parser_name.lower():
            return "extracted raw text without structural understanding"

        return "processed the document according to its default behavior"

    def save_case_studies(
        self,
        case_studies: list[CaseStudy],
        format: str = "markdown"
    ) -> str:
        """
        Save case studies to file.

        Args:
            case_studies: List of CaseStudy objects
            format: Output format ("markdown" or "json")

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "markdown":
            output_path = self.output_dir / f"case_studies_{timestamp}.md"
            content = "# Error Case Studies\n\n"
            content += f"Generated: {datetime.now().isoformat()}\n\n"
            content += f"Total Case Studies: {len(case_studies)}\n\n"
            content += "---\n\n"

            for cs in case_studies:
                content += cs.to_markdown()
                content += "\n"

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

        else:  # json
            output_path = self.output_dir / f"case_studies_{timestamp}.json"
            data = {
                "generated_at": datetime.now().isoformat(),
                "total_case_studies": len(case_studies),
                "case_studies": [cs.to_dict() for cs in case_studies],
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def generate_summary_report(
        self,
        all_errors: dict[str, list[ParsingError]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate summary report across all parsers.

        Args:
            all_errors: Dict mapping parser name to error list
            output_path: Optional path for output

        Returns:
            Markdown summary report
        """
        report = "# Error Analysis Summary Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"

        # Overall statistics
        report += "## Overall Statistics\n\n"
        report += "| Parser | Total Errors | Critical | Major | Medium | Minor |\n"
        report += "|--------|--------------|----------|-------|--------|-------|\n"

        for parser_name, errors in all_errors.items():
            analyzer = ErrorAnalyzer()
            summary = analyzer.summarize_errors(errors)

            severities = summary["by_severity"]
            report += (
                f"| {parser_name} | {summary['total_errors']} | "
                f"{severities.get('critical', 0)} | "
                f"{severities.get('major', 0)} | "
                f"{severities.get('medium', 0)} | "
                f"{severities.get('minor', 0)} |\n"
            )

        # Error category breakdown
        report += "\n## Error Categories\n\n"
        report += "| Parser | Table | Multi-Col | Halluc. | Header | Delete | Subst. |\n"
        report += "|--------|-------|-----------|---------|--------|--------|--------|\n"

        for parser_name, errors in all_errors.items():
            analyzer = ErrorAnalyzer()
            summary = analyzer.summarize_errors(errors)
            cats = summary["by_category"]

            report += (
                f"| {parser_name} | "
                f"{cats.get('table_structure', 0)} | "
                f"{cats.get('multi_column', 0)} | "
                f"{cats.get('hallucination', 0)} | "
                f"{cats.get('header_hierarchy', 0)} | "
                f"{cats.get('deletion', 0)} | "
                f"{cats.get('substitution', 0)} |\n"
            )

        # Recommendations
        report += "\n## Key Recommendations\n\n"
        report += "Based on the error analysis:\n\n"

        # Determine recommendations based on error patterns
        all_cats = {}
        for errors in all_errors.values():
            for error in errors:
                cat = error.category.value
                all_cats[cat] = all_cats.get(cat, 0) + 1

        if all_cats.get("table_structure", 0) > 5:
            report += "1. **Table-heavy documents**: Route to VLM parser for better structure preservation\n"

        if all_cats.get("hallucination", 0) > 0:
            report += "2. **Hallucination prevention**: Use transcription-focused prompts (v2)\n"

        if all_cats.get("multi_column", 0) > 3:
            report += "3. **Multi-column documents**: VLM shows significant advantage over OCR\n"

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report
