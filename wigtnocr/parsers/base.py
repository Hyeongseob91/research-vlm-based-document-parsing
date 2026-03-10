"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ParseResult:
    """Base parsing result."""
    success: bool
    content: str
    pages: List[str]
    elapsed_time: float
    page_count: int
    error: Optional[str] = None


class BaseParser(ABC):
    """Abstract base class for all parsers."""

    @abstractmethod
    def parse_pdf(self, pdf_bytes: bytes) -> ParseResult:
        """Parse PDF document bytes."""
        ...
