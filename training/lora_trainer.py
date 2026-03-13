"""LoRA Fine-tuning for WigtnOCR using ms-swift.

Wraps ms-swift CLI to fine-tune Qwen3-VL-2B-Instruct with LoRA
on document parsing data (PDF page images → structured Markdown).

Usage:
    # Train with default config
    python -m training.lora_trainer

    # Train with custom config
    python -m training.lora_trainer --config configs/training.yaml

    # Resume from checkpoint
    python -m training.lora_trainer --resume output/wigtnocr-2b-lora/vx-xxx/checkpoint-xxx
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "output_dir": "output/wigtnocr-2b-lora",
    "train_dataset": "datasets/training/train.jsonl",
    "val_dataset": "datasets/training/val.jsonl",
    # LoRA
    "tuner_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": "all-linear",
    # Freeze vision encoder
    "freeze_vit": True,
    "freeze_aligner": True,
    # Training
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "max_length": 4096,
    # Performance
    "torch_dtype": "bfloat16",
    "attn_impl": "flash_attn",
    "gradient_checkpointing": True,
    "packing": True,
    "padding_free": True,
    "deepspeed": "zero2",
    "nproc_per_node": 2,
    # Eval & Save
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 3,
    "logging_steps": 5,
    "dataset_num_proc": 4,
    "dataloader_num_workers": 4,
}


def load_config(config_path: Path) -> dict:
    """Load training config from YAML, merge with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            # Flatten nested YAML structure
            if "model" in yaml_config and isinstance(yaml_config["model"], dict):
                config["model"] = yaml_config["model"].get("base", config["model"])
                config["output_dir"] = yaml_config["model"].get("output", config["output_dir"])
            if "lora" in yaml_config:
                lora = yaml_config["lora"]
                config["lora_rank"] = lora.get("r", config["lora_rank"])
                config["lora_alpha"] = lora.get("alpha", config["lora_alpha"])
                if "target_modules" in lora:
                    config["target_modules"] = lora["target_modules"]
            if "training" in yaml_config:
                t = yaml_config["training"]
                for key in ["learning_rate", "num_epochs", "batch_size",
                            "gradient_accumulation_steps", "warmup_ratio"]:
                    if key in t:
                        mapped = {
                            "num_epochs": "num_train_epochs",
                            "batch_size": "per_device_train_batch_size",
                        }.get(key, key)
                        config[mapped] = t[key]
            if "data" in yaml_config:
                d = yaml_config["data"]
                if "train_dir" in d:
                    config["train_dataset"] = str(Path(d["train_dir"]) / "training" / "train.jsonl")
                    config["val_dataset"] = str(Path(d["train_dir"]) / "training" / "val.jsonl")
    return config


def build_swift_command(config: dict, resume_from: str = None) -> list[str]:
    """Build ms-swift CLI command from config dict."""
    cmd = ["swift", "sft"]

    # Map config keys to swift CLI args
    arg_map = {
        "model": "--model",
        "output_dir": "--output_dir",
        "train_dataset": "--dataset",
        "val_dataset": "--val_dataset",
        "tuner_type": "--tuner_type",
        "lora_rank": "--lora_rank",
        "lora_alpha": "--lora_alpha",
        "target_modules": "--target_modules",
        "torch_dtype": "--torch_dtype",
        "attn_impl": "--attn_impl",
        "num_train_epochs": "--num_train_epochs",
        "learning_rate": "--learning_rate",
        "per_device_train_batch_size": "--per_device_train_batch_size",
        "per_device_eval_batch_size": "--per_device_eval_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "warmup_ratio": "--warmup_ratio",
        "max_length": "--max_length",
        "eval_steps": "--eval_steps",
        "save_steps": "--save_steps",
        "save_total_limit": "--save_total_limit",
        "logging_steps": "--logging_steps",
        "deepspeed": "--deepspeed",
        "dataset_num_proc": "--dataset_num_proc",
        "dataloader_num_workers": "--dataloader_num_workers",
    }

    bool_args = {
        "freeze_vit": "--freeze_vit",
        "freeze_aligner": "--freeze_aligner",
        "gradient_checkpointing": "--gradient_checkpointing",
        "packing": "--packing",
        "padding_free": "--padding_free",
    }

    for key, flag in arg_map.items():
        if key in config:
            val = config[key]
            if isinstance(val, list):
                cmd.extend([flag, " ".join(val)])
            else:
                cmd.extend([flag, str(val)])

    for key, flag in bool_args.items():
        if config.get(key):
            cmd.extend([flag, "true"])

    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])

    return cmd


def train(config_path: Path = None, resume_from: str = None, dry_run: bool = False):
    """Run LoRA fine-tuning via ms-swift.

    Args:
        config_path: Path to training.yaml config
        resume_from: Checkpoint path to resume from
        dry_run: Print command without executing
    """
    config = load_config(config_path)

    # Validate data files exist
    train_path = Path(config["train_dataset"])
    if not train_path.exists():
        print(f"Training data not found: {train_path}")
        print("Run data preparation first:")
        print("  python scripts/prepare_training_data.py --dataset all")
        sys.exit(1)

    cmd = build_swift_command(config, resume_from)

    # Environment variables
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NPROC_PER_NODE": str(config.get("nproc_per_node", 2)),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "IMAGE_MAX_TOKEN_NUM": "1024",
    }

    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    cmd_str = " ".join(cmd)
    full_cmd = f"{env_str} {cmd_str}"

    print(f"\n{'=' * 60}")
    print("WigtnOCR LoRA Training")
    print(f"  Model: {config['model']}")
    print(f"  Train data: {config['train_dataset']}")
    print(f"  Output: {config['output_dir']}")
    print(f"  LoRA rank: {config['lora_rank']}, alpha: {config['lora_alpha']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch: {config['per_device_train_batch_size']} x {config['gradient_accumulation_steps']} accum")
    print(f"  GPUs: {env_vars['CUDA_VISIBLE_DEVICES']} ({config.get('nproc_per_node', 2)} procs)")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print(f"[DRY RUN] Command:\n{full_cmd}")
        return

    # Execute
    import os
    env = os.environ.copy()
    env.update(env_vars)

    print(f"Executing: {cmd_str}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nTraining complete! Checkpoints saved to: {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WigtnOCR LoRA Fine-tuning (ms-swift)")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"),
                        help="Training config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    args = parser.parse_args()

    train(config_path=args.config, resume_from=args.resume, dry_run=args.dry_run)
