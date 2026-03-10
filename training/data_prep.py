"""Training data preparation.

Converts PDF + GT markdown pairs into VLM training format:
- PDF page → image (PNG)
- GT markdown → target text
- Metadata → training sample JSON
"""

import json
from pathlib import Path
from typing import Optional


def prepare_training_data(
    dataset_dir: Path,
    output_dir: Path,
    dpi: int = 150,
    max_pages_per_doc: Optional[int] = None,
) -> dict:
    """Prepare training data from dataset directory.

    Args:
        dataset_dir: Directory with document folders (PDF + gt.md)
        output_dir: Output directory for training samples
        dpi: Image resolution for PDF rendering
        max_pages_per_doc: Limit pages per document

    Returns:
        Summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement data preparation
    # 1. Iterate through document folders
    # 2. Convert PDF pages to images
    # 3. Split GT markdown by page
    # 4. Create training JSON with image path + target markdown

    raise NotImplementedError("Data preparation not yet implemented")
