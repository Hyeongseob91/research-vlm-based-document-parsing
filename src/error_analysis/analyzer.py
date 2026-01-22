"""
Error Analyzer for Document Parsing

Detects and categorizes parsing errors by comparing
parsed output against ground truth.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import difflib


class ErrorCategory(Enum):
    """Categories of parsing errors."""
    TABLE_STRUCTURE = "table_structure"  # Table row/column corrupted
    MULTI_COLUMN = "multi_column"  # Reading order error
    HALLUCINATION = "hallucination"  # Content added that doesn't exist
    HEADER_HIERARCHY = "header_hierarchy"  # Header level wrong
    DELETION = "deletion"  # Content missing
    SUBSTITUTION = "substitution"  # Character recognition error
    FORMATTING = "formatting"  # Formatting lost
    ORDERING = "ordering"  # Content order wrong
    OTHER = "other"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"  # Renders content unusable
    MAJOR = "major"  # Significant content error
    MEDIUM = "medium"  # Structural error
    MINOR = "minor"  # Cosmetic error


@dataclass
class ParsingError:
    """Represents a single parsing error."""
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    gt_text: str  # Ground truth snippet
    parsed_text: str  # Parsed output snippet
    position: int  # Approximate position in document
    context: str = ""  # Surrounding context
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "gt_text": self.gt_text,
            "parsed_text": self.parsed_text,
            "position": self.position,
            "context": self.context,
            "metadata": self.metadata,
        }


class ErrorAnalyzer:
    """
    Analyzes parsing output to detect and categorize errors.

    Compares parsed text against ground truth to identify:
    - Missing content (deletions)
    - Added content (hallucinations)
    - Changed content (substitutions)
    - Structure errors (tables, headers)
    """

    # Patterns for detecting structure
    TABLE_PATTERN = re.compile(r'^\|.+\|$', re.MULTILINE)
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    LIST_PATTERN = re.compile(r'^[-*+]\s+|^\d+\.\s+', re.MULTILINE)

    def __init__(
        self,
        context_window: int = 100,
        min_error_length: int = 5
    ):
        self.context_window = context_window
        self.min_error_length = min_error_length

    def analyze(
        self,
        parsed_text: str,
        ground_truth: str,
        parser_name: str = "unknown"
    ) -> list[ParsingError]:
        """
        Analyze parsed text against ground truth.

        Args:
            parsed_text: Output from parser
            ground_truth: Ground truth text
            parser_name: Name of parser for metadata

        Returns:
            List of detected ParsingError objects
        """
        errors = []

        # Detect different error types
        errors.extend(self._detect_deletions(parsed_text, ground_truth))
        errors.extend(self._detect_hallucinations(parsed_text, ground_truth))
        errors.extend(self._detect_substitutions(parsed_text, ground_truth))
        errors.extend(self._detect_table_errors(parsed_text, ground_truth))
        errors.extend(self._detect_header_errors(parsed_text, ground_truth))
        errors.extend(self._detect_ordering_errors(parsed_text, ground_truth))

        # Add parser metadata
        for error in errors:
            error.metadata["parser"] = parser_name

        # Sort by position
        errors.sort(key=lambda e: e.position)

        return errors

    def _detect_deletions(
        self,
        parsed_text: str,
        ground_truth: str
    ) -> list[ParsingError]:
        """Detect content present in GT but missing in parsed."""
        errors = []

        # Use diff to find deletions
        matcher = difflib.SequenceMatcher(None, ground_truth, parsed_text)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                deleted_text = ground_truth[i1:i2].strip()
                if len(deleted_text) >= self.min_error_length:
                    # Determine severity based on what was deleted
                    severity = self._assess_deletion_severity(deleted_text)
                    context = ground_truth[max(0, i1-self.context_window):min(len(ground_truth), i2+self.context_window)]

                    errors.append(ParsingError(
                        category=ErrorCategory.DELETION,
                        severity=severity,
                        description=f"Content missing from parsed output",
                        gt_text=deleted_text[:200],  # Truncate long deletions
                        parsed_text="[MISSING]",
                        position=i1,
                        context=context,
                    ))

        return errors

    def _detect_hallucinations(
        self,
        parsed_text: str,
        ground_truth: str
    ) -> list[ParsingError]:
        """Detect content present in parsed but not in GT."""
        errors = []

        # Use diff to find insertions
        matcher = difflib.SequenceMatcher(None, ground_truth, parsed_text)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                inserted_text = parsed_text[j1:j2].strip()
                if len(inserted_text) >= self.min_error_length:
                    # Filter out likely formatting additions
                    if not self._is_formatting_only(inserted_text):
                        context = parsed_text[max(0, j1-self.context_window):min(len(parsed_text), j2+self.context_window)]

                        errors.append(ParsingError(
                            category=ErrorCategory.HALLUCINATION,
                            severity=ErrorSeverity.MAJOR,
                            description="Content added that doesn't exist in original",
                            gt_text="[NOT IN ORIGINAL]",
                            parsed_text=inserted_text[:200],
                            position=j1,
                            context=context,
                        ))

        return errors

    def _detect_substitutions(
        self,
        parsed_text: str,
        ground_truth: str
    ) -> list[ParsingError]:
        """Detect character-level substitutions."""
        errors = []

        matcher = difflib.SequenceMatcher(None, ground_truth, parsed_text)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                gt_text = ground_truth[i1:i2].strip()
                parsed = parsed_text[j1:j2].strip()

                # Only report if lengths are similar (not a major structural change)
                if 0.5 < len(parsed) / max(len(gt_text), 1) < 2.0:
                    errors.append(ParsingError(
                        category=ErrorCategory.SUBSTITUTION,
                        severity=ErrorSeverity.MINOR,
                        description="Character recognition error",
                        gt_text=gt_text[:100],
                        parsed_text=parsed[:100],
                        position=i1,
                    ))

        return errors

    def _detect_table_errors(
        self,
        parsed_text: str,
        ground_truth: str
    ) -> list[ParsingError]:
        """Detect table structure errors."""
        errors = []

        # Find tables in ground truth
        gt_tables = list(self.TABLE_PATTERN.finditer(ground_truth))
        parsed_tables = list(self.TABLE_PATTERN.finditer(parsed_text))

        # Check for missing tables
        gt_table_count = len(gt_tables)
        parsed_table_count = len(parsed_tables)

        if gt_table_count > 0 and parsed_table_count == 0:
            errors.append(ParsingError(
                category=ErrorCategory.TABLE_STRUCTURE,
                severity=ErrorSeverity.CRITICAL,
                description=f"All {gt_table_count} tables missing from parsed output",
                gt_text="[TABLES IN GT]",
                parsed_text="[NO TABLES FOUND]",
                position=gt_tables[0].start() if gt_tables else 0,
            ))
        elif gt_table_count != parsed_table_count:
            errors.append(ParsingError(
                category=ErrorCategory.TABLE_STRUCTURE,
                severity=ErrorSeverity.MAJOR,
                description=f"Table count mismatch: GT has {gt_table_count}, parsed has {parsed_table_count}",
                gt_text=f"{gt_table_count} tables",
                parsed_text=f"{parsed_table_count} tables",
                position=0,
            ))

        # Check table structure integrity
        for gt_match in gt_tables:
            gt_row = gt_match.group()
            gt_cols = gt_row.count('|') - 1

            # Find corresponding row in parsed
            row_found = False
            for parsed_match in parsed_tables:
                parsed_row = parsed_match.group()
                parsed_cols = parsed_row.count('|') - 1

                if gt_cols != parsed_cols:
                    errors.append(ParsingError(
                        category=ErrorCategory.TABLE_STRUCTURE,
                        severity=ErrorSeverity.CRITICAL,
                        description=f"Column count mismatch: GT has {gt_cols}, parsed has {parsed_cols}",
                        gt_text=gt_row[:100],
                        parsed_text=parsed_row[:100],
                        position=gt_match.start(),
                    ))

        return errors

    def _detect_header_errors(
        self,
        parsed_text: str,
        ground_truth: str
    ) -> list[ParsingError]:
        """Detect header hierarchy errors."""
        errors = []

        gt_headers = self.HEADER_PATTERN.findall(ground_truth)
        parsed_headers = self.HEADER_PATTERN.findall(parsed_text)

        # Compare header levels
        for i, (gt_level, gt_title) in enumerate(gt_headers):
            gt_level_num = len(gt_level)

            # Try to find matching header in parsed
            for parsed_level, parsed_title in parsed_headers:
                # Fuzzy title match
                if self._titles_match(gt_title, parsed_title):
                    parsed_level_num = len(parsed_level)
                    if gt_level_num != parsed_level_num:
                        errors.append(ParsingError(
                            category=ErrorCategory.HEADER_HIERARCHY,
                            severity=ErrorSeverity.MEDIUM,
                            description=f"Header level mismatch: GT is H{gt_level_num}, parsed is H{parsed_level_num}",
                            gt_text=f"{'#'*gt_level_num} {gt_title}",
                            parsed_text=f"{'#'*parsed_level_num} {parsed_title}",
                            position=0,
                        ))
                    break

        return errors

    def _detect_ordering_errors(
        self,
        parsed_text: str,
        ground_truth: str
    ) -> list[ParsingError]:
        """Detect content ordering errors (multi-column issues)."""
        errors = []

        # Split into paragraphs
        gt_paragraphs = [p.strip() for p in ground_truth.split('\n\n') if p.strip()]
        parsed_paragraphs = [p.strip() for p in parsed_text.split('\n\n') if p.strip()]

        # Check if paragraphs appear in different order
        gt_order = {p[:50]: i for i, p in enumerate(gt_paragraphs)}
        parsed_order = {p[:50]: i for i, p in enumerate(parsed_paragraphs)}

        inversions = 0
        for key in gt_order:
            if key in parsed_order:
                # Check if relative ordering is preserved
                for other_key in gt_order:
                    if other_key in parsed_order:
                        gt_before = gt_order[key] < gt_order[other_key]
                        parsed_before = parsed_order[key] < parsed_order[other_key]
                        if gt_before != parsed_before:
                            inversions += 1

        if inversions > 3:
            errors.append(ParsingError(
                category=ErrorCategory.MULTI_COLUMN,
                severity=ErrorSeverity.CRITICAL,
                description=f"Content ordering error: {inversions} order inversions detected",
                gt_text="[ORIGINAL ORDER]",
                parsed_text="[REORDERED]",
                position=0,
                metadata={"inversions": inversions},
            ))

        return errors

    def _assess_deletion_severity(self, deleted_text: str) -> ErrorSeverity:
        """Assess severity of a deletion based on content."""
        # Check if it's a table
        if self.TABLE_PATTERN.search(deleted_text):
            return ErrorSeverity.CRITICAL

        # Check if it's a header
        if self.HEADER_PATTERN.search(deleted_text):
            return ErrorSeverity.MAJOR

        # Check length
        if len(deleted_text) > 200:
            return ErrorSeverity.MAJOR
        elif len(deleted_text) > 50:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.MINOR

    def _is_formatting_only(self, text: str) -> bool:
        """Check if text is only formatting characters."""
        # Remove markdown formatting
        stripped = re.sub(r'[#*_\-|`\n\s]', '', text)
        return len(stripped) < self.min_error_length

    def _titles_match(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar enough to be the same."""
        # Normalize
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()

        if t1 == t2:
            return True

        # Check similarity
        ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
        return ratio > 0.8

    def summarize_errors(self, errors: list[ParsingError]) -> dict:
        """Generate summary statistics for errors."""
        summary = {
            "total_errors": len(errors),
            "by_category": {},
            "by_severity": {},
            "critical_count": 0,
        }

        for error in errors:
            # By category
            cat = error.category.value
            if cat not in summary["by_category"]:
                summary["by_category"][cat] = 0
            summary["by_category"][cat] += 1

            # By severity
            sev = error.severity.value
            if sev not in summary["by_severity"]:
                summary["by_severity"][sev] = 0
            summary["by_severity"][sev] += 1

            if error.severity == ErrorSeverity.CRITICAL:
                summary["critical_count"] += 1

        return summary
