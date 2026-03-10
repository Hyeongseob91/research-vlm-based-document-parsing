"""DocVQA Benchmark Adapter.

Document Visual Question Answering benchmark for evaluating
parsing quality through downstream QA performance.
"""

from pathlib import Path
from typing import Optional


def load_docvqa_dataset(
    dataset_path: Path,
    split: str = "val",
    limit: Optional[int] = None,
) -> list[dict]:
    """Load DocVQA dataset.

    Args:
        dataset_path: Root path of DocVQA dataset
        split: Dataset split ("train", "val", "test")
        limit: Maximum number of samples

    Returns:
        List of QA sample dicts
    """
    # TODO: Implement DocVQA loading
    # Will support standard DocVQA JSON format
    raise NotImplementedError("DocVQA adapter not yet implemented")
