"""Unified CLI for WigtnOCR evaluation.

Usage:
    # Evaluate parsers on all datasets
    python -m evaluation.cli parse --dataset documents

    # Evaluate chunking on results
    python -m evaluation.cli chunk --results-dir results/

    # Run full benchmark
    python -m evaluation.cli benchmark --dataset all
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="WigtnOCR Evaluation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")

    # Parse evaluation
    parse_parser = subparsers.add_parser("parse", help="Run parser evaluation")
    parse_parser.add_argument("--dataset", choices=["papers", "documents", "omnidocbench", "all"], default="all")
    parse_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parse_parser.add_argument("--vlm-url", default="http://localhost:8010/v1/chat/completions")
    parse_parser.add_argument("--vlm-model", default="qwen3-vl-2b-instruct")
    parse_parser.add_argument("--skip-advanced", action="store_true")

    # Chunk evaluation
    chunk_parser = subparsers.add_parser("chunk", help="Run chunking evaluation")
    chunk_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    chunk_parser.add_argument("--embedding-url", default="http://localhost:8001/embeddings")

    args = parser.parse_args()

    if args.command == "parse":
        from evaluation.runners.parser_eval import run_batch_evaluation

        datasets_dir = Path("datasets")
        if args.dataset in ("documents", "all"):
            print("=== Evaluating Korean Documents ===")
            run_batch_evaluation(
                datasets_dir / "documents", args.output_dir / "documents",
                vlm_api_url=args.vlm_url, vlm_model=args.vlm_model,
                skip_advanced=args.skip_advanced,
            )
        if args.dataset in ("papers", "all"):
            print("=== Evaluating Papers ===")
            run_batch_evaluation(
                datasets_dir / "papers", args.output_dir / "papers",
                vlm_api_url=args.vlm_url, vlm_model=args.vlm_model,
                skip_advanced=args.skip_advanced,
            )

    elif args.command == "chunk":
        print("Chunking evaluation - use evaluation.runners.chunking_eval")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
