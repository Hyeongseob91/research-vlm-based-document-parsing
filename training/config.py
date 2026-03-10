"""Training configuration for LoRA fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA hyperparameters."""
    r: int = 64
    alpha: int = 128
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
    ])
    dropout: float = 0.05


@dataclass
class TrainingConfig:
    """Full training configuration."""
    # Model
    base_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "checkpoints/wigtnocr-ko-gov-2b"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Data
    train_data_dir: str = "datasets/documents"
    replay_ratio: float = 0.1  # General VQA data mixed in to prevent catastrophic forgetting

    # Hardware
    bf16: bool = True
    max_seq_length: int = 4096
