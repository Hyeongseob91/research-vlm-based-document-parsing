"""LoRA Fine-tuning Script for WigtnOCR.

Fine-tunes Qwen3-VL-2B-Instruct with LoRA on Korean government documents.

Usage:
    python -m training.lora_trainer --config configs/training.yaml
"""

from pathlib import Path
from typing import Optional

from training.config import TrainingConfig


def train(config: Optional[TrainingConfig] = None):
    """Run LoRA fine-tuning.

    Args:
        config: Training configuration (uses defaults if None)
    """
    if config is None:
        config = TrainingConfig()

    # TODO: Implement LoRA fine-tuning
    # 1. Load base model (Qwen3-VL-2B-Instruct)
    # 2. Apply LoRA adapters
    # 3. Load training data
    # 4. Train with replay data mixed in
    # 5. Save LoRA weights

    raise NotImplementedError(
        "LoRA training not yet implemented. "
        "Requires: transformers, peft, trl, accelerate"
    )


if __name__ == "__main__":
    train()
