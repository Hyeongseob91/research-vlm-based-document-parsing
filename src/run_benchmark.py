#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for VLM Document Parsing Evaluation

This script orchestrates the full evaluation pipeline:
1. Parse documents with multiple parsers
2. Chunk parsed content
3. Generate/load Q&A pairs
4. Evaluate retrieval performance
5. Calculate structural metrics
6. Generate comparison reports

Usage:
    python -m src.run_benchmark --config experiments/config.yaml
    python -m src.run_benchmark --parser vlm --pdf data/test_1/test_data_1.pdf
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import VLMParser, OCRParser, DoclingParser
from src.chunking import (
    ChunkerConfig,
    ChunkingStrategy,
    create_chunker,
    calculate_boundary_score,
    compare_chunking_quality,
)
from src.retrieval import (
    EmbeddingConfig,
    RetrievalConfig,
    ChunkRetriever,
    RetrievalEvaluator,
    compare_retrieval_performance,
)


class BenchmarkRunner:
    """Orchestrates the full evaluation benchmark."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "parsers": {
                "vlm": {
                    "api_url": "http://localhost:8000/v1/chat/completions",
                    "model": "Qwen3-VL-2B-Instruct",
                    "temperature": 0.0,
                },
                "pdfplumber": {"enabled": True},
                "docling": {"enabled": True},
            },
            "chunking": {
                "strategy": "recursive_character",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
            "embedding": {
                "model": "jhgan/ko-sroberta-multitask",
                "device": "cpu",
            },
            "retrieval": {
                "top_k": [1, 3, 5, 10],
            },
            "output": {
                "results_dir": "results",
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # Deep merge
                self._deep_merge(default_config, user_config)

        return default_config

    def _deep_merge(self, base: dict, override: dict):
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def run_full_benchmark(
        self,
        test_cases: list[dict],
        qa_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Run complete benchmark on test cases.

        Args:
            test_cases: List of test case configs with pdf_path, gt_path, etc.
            qa_path: Path to Q&A pairs JSON file
            output_dir: Output directory for results

        Returns:
            Dictionary with all benchmark results
        """
        output_dir = output_dir or self.config["output"]["results_dir"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load Q&A pairs if provided
        qa_pairs = self._load_qa_pairs(qa_path) if qa_path else None

        all_results = {
            "timestamp": self.timestamp,
            "config": self.config,
            "test_cases": [],
        }

        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"Processing: {test_case.get('name', test_case.get('id', 'Unknown'))}")
            print(f"{'='*60}")

            case_result = self.run_single_benchmark(
                test_case=test_case,
                qa_pairs=self._filter_qa_pairs(qa_pairs, test_case.get("id")),
                output_dir=str(output_path / test_case.get("id", "unknown")),
            )
            all_results["test_cases"].append(case_result)

        # Aggregate results
        all_results["summary"] = self._aggregate_results(all_results["test_cases"])

        # Save overall results
        with open(output_path / f"benchmark_{self.timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

        return all_results

    def run_single_benchmark(
        self,
        test_case: dict,
        qa_pairs: Optional[list] = None,
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Run benchmark on a single test case.

        Args:
            test_case: Test case config with pdf_path, gt_path
            qa_pairs: Q&A pairs for this document
            output_dir: Output directory

        Returns:
            Dictionary with benchmark results
        """
        pdf_path = test_case.get("pdf_path")
        gt_path = test_case.get("gt_path")
        document_type = test_case.get("document_type", "digital")

        results = {
            "test_case": test_case,
            "parsers": {},
            "comparison": {},
        }

        # Load ground truth
        gt_content = ""
        if gt_path and os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_content = f.read()

        # Parse with each parser
        parsed_contents = {}

        # VLM Parser
        if pdf_path:
            print("\n[1/3] Running VLM Parser...")
            vlm_result = self._run_vlm_parser(pdf_path)
            if vlm_result:
                parsed_contents["vlm"] = vlm_result["content"]
                results["parsers"]["vlm"] = {
                    "success": vlm_result["success"],
                    "latency": vlm_result["latency"],
                    "content_length": len(vlm_result["content"]),
                }

            # pdfplumber (for digital PDFs)
            if document_type == "digital":
                print("\n[2/3] Running pdfplumber...")
                ocr_result = self._run_ocr_parser(pdf_path)
                if ocr_result:
                    parsed_contents["pdfplumber"] = ocr_result["content"]
                    results["parsers"]["pdfplumber"] = {
                        "success": ocr_result["success"],
                        "latency": ocr_result["latency"],
                        "content_length": len(ocr_result["content"]),
                    }

            # Docling (for scanned documents)
            if document_type == "scanned":
                print("\n[3/3] Running Docling+RapidOCR...")
                docling_result = self._run_docling_parser(pdf_path)
                if docling_result:
                    parsed_contents["docling"] = docling_result["content"]
                    results["parsers"]["docling"] = {
                        "success": docling_result["success"],
                        "latency": docling_result["latency"],
                        "content_length": len(docling_result["content"]),
                    }

        # Chunk all parsed contents
        print("\nChunking parsed content...")
        chunked_contents = {}
        for parser_name, content in parsed_contents.items():
            chunks = self._chunk_content(content, f"{test_case.get('id', 'doc')}_{parser_name}")
            chunked_contents[parser_name] = chunks
            results["parsers"][parser_name]["chunk_count"] = len(chunks)

        # Calculate structural metrics if GT available
        if gt_content:
            print("\nCalculating structural metrics...")
            for parser_name, content in parsed_contents.items():
                bs = calculate_boundary_score(content, gt_content)
                results["parsers"][parser_name]["boundary_score"] = bs.to_dict()

        # Run retrieval evaluation if Q&A pairs available
        if qa_pairs and chunked_contents:
            print("\nRunning retrieval evaluation...")
            retrieval_results = self._evaluate_retrieval(
                chunked_contents, qa_pairs
            )
            results["retrieval"] = retrieval_results

            # Compare VLM vs baseline
            if "vlm" in retrieval_results and len(retrieval_results) > 1:
                baseline_parser = "pdfplumber" if "pdfplumber" in retrieval_results else "docling"
                if baseline_parser in retrieval_results:
                    comparison = compare_retrieval_performance(
                        baseline_results=retrieval_results[baseline_parser]["raw_results"],
                        vlm_results=retrieval_results["vlm"]["raw_results"],
                        k_values=self.config["retrieval"]["top_k"],
                    )
                    results["comparison"] = comparison

        # Save individual results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f"results_{self.timestamp}.json", 'w', encoding='utf-8') as f:
                # Remove raw results for cleaner output
                clean_results = self._clean_results(results)
                json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)

        return results

    def _run_vlm_parser(self, pdf_path: str) -> Optional[dict]:
        """Run VLM parser on document."""
        try:
            vlm_config = self.config["parsers"]["vlm"]
            parser = VLMParser(
                api_url=vlm_config.get("api_url", "http://localhost:8000/v1/chat/completions"),
                model=vlm_config.get("model", "Qwen3-VL-2B-Instruct"),
            )

            start = time.time()
            result = parser.parse(pdf_path)
            latency = time.time() - start

            return {
                "success": result.success,
                "content": result.content or "",
                "latency": latency,
                "error": result.error,
            }
        except Exception as e:
            print(f"VLM parser error: {e}")
            return None

    def _run_ocr_parser(self, pdf_path: str) -> Optional[dict]:
        """Run pdfplumber parser on document."""
        try:
            parser = OCRParser()

            start = time.time()
            result = parser.parse(pdf_path)
            latency = time.time() - start

            return {
                "success": result.success,
                "content": result.content or "",
                "latency": latency,
                "error": result.error,
            }
        except Exception as e:
            print(f"OCR parser error: {e}")
            return None

    def _run_docling_parser(self, pdf_path: str) -> Optional[dict]:
        """Run Docling parser on document."""
        try:
            parser = DoclingParser()

            start = time.time()
            result = parser.parse(pdf_path)
            latency = time.time() - start

            return {
                "success": result.success,
                "content": result.content or "",
                "latency": latency,
                "error": result.error,
            }
        except Exception as e:
            print(f"Docling parser error: {e}")
            return None

    def _chunk_content(self, content: str, document_id: str) -> list:
        """Chunk content using configured strategy."""
        chunking_config = self.config["chunking"]
        config = ChunkerConfig(
            strategy=ChunkingStrategy(chunking_config.get("strategy", "recursive_character")),
            chunk_size=chunking_config.get("chunk_size", 500),
            chunk_overlap=chunking_config.get("chunk_overlap", 50),
        )
        chunker = create_chunker(config)
        return chunker.chunk(content, document_id)

    def _evaluate_retrieval(
        self,
        chunked_contents: dict,
        qa_pairs: list
    ) -> dict:
        """Evaluate retrieval for all parser outputs."""
        embedding_config = self.config["embedding"]
        config = EmbeddingConfig(
            model=embedding_config.get("model", "jhgan/ko-sroberta-multitask"),
            device=embedding_config.get("device", "cpu"),
        )

        retrieval_config = RetrievalConfig(
            top_k=self.config["retrieval"]["top_k"],
            embedding_config=config,
        )

        evaluator = RetrievalEvaluator(k_values=retrieval_config.top_k)
        results = {}

        for parser_name, chunks in chunked_contents.items():
            print(f"  Evaluating {parser_name}...")

            try:
                retriever = ChunkRetriever(config=retrieval_config)
                retriever.index_chunks(chunks)

                # Prepare queries with expected chunks
                queries = []
                for qa in qa_pairs:
                    # Find which chunk contains the answer
                    expected_chunk = retriever.find_relevant_chunk(
                        qa.get("answer_span", qa.get("answer", ""))
                    )
                    queries.append({
                        "query": qa["question"],
                        "query_id": qa.get("id", str(len(queries))),
                        "expected_chunk_id": expected_chunk,
                    })

                # Run retrieval
                retrieval_results = retriever.retrieve_batch(
                    queries,
                    top_k=max(retrieval_config.top_k)
                )

                # Evaluate
                metrics = evaluator.evaluate(retrieval_results)
                results[parser_name] = {
                    "metrics": metrics.to_dict(),
                    "raw_results": retrieval_results,  # Keep for comparison
                }
            except Exception as e:
                print(f"    Error: {e}")
                results[parser_name] = {"error": str(e)}

        return results

    def _load_qa_pairs(self, qa_path: str) -> Optional[list]:
        """Load Q&A pairs from JSON file."""
        if not os.path.exists(qa_path):
            return None

        with open(qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("qa_pairs", data)

    def _filter_qa_pairs(
        self,
        qa_pairs: Optional[list],
        document_id: Optional[str]
    ) -> Optional[list]:
        """Filter Q&A pairs for specific document."""
        if not qa_pairs or not document_id:
            return qa_pairs

        return [
            qa for qa in qa_pairs
            if qa.get("document_id") == document_id
        ]

    def _aggregate_results(self, case_results: list) -> dict:
        """Aggregate results across test cases."""
        summary = {
            "total_cases": len(case_results),
            "parsers": {},
        }

        # Aggregate by parser
        for case in case_results:
            for parser_name, parser_data in case.get("parsers", {}).items():
                if parser_name not in summary["parsers"]:
                    summary["parsers"][parser_name] = {
                        "total_runs": 0,
                        "avg_latency": [],
                        "avg_boundary_score": [],
                    }

                summary["parsers"][parser_name]["total_runs"] += 1

                if "latency" in parser_data:
                    summary["parsers"][parser_name]["avg_latency"].append(parser_data["latency"])

                if "boundary_score" in parser_data:
                    summary["parsers"][parser_name]["avg_boundary_score"].append(
                        parser_data["boundary_score"]["score"]
                    )

        # Calculate averages
        for parser_name, data in summary["parsers"].items():
            if data["avg_latency"]:
                data["avg_latency"] = sum(data["avg_latency"]) / len(data["avg_latency"])
            else:
                data["avg_latency"] = None

            if data["avg_boundary_score"]:
                data["avg_boundary_score"] = sum(data["avg_boundary_score"]) / len(data["avg_boundary_score"])
            else:
                data["avg_boundary_score"] = None

        return summary

    def _clean_results(self, results: dict) -> dict:
        """Remove large internal data for cleaner output."""
        clean = results.copy()

        # Remove raw retrieval results
        if "retrieval" in clean:
            for parser_name in clean["retrieval"]:
                if isinstance(clean["retrieval"][parser_name], dict):
                    clean["retrieval"][parser_name].pop("raw_results", None)

        return clean


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM document parsing benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Single PDF to process"
    )
    parser.add_argument(
        "--gt",
        type=str,
        help="Ground truth file for single PDF"
    )
    parser.add_argument(
        "--qa",
        type=str,
        help="Path to Q&A pairs JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["vlm", "pdfplumber", "docling", "all"],
        default="all",
        help="Parser to run"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(config_path=args.config)

    if args.pdf:
        # Single file mode
        test_case = {
            "id": Path(args.pdf).stem,
            "name": Path(args.pdf).name,
            "pdf_path": args.pdf,
            "gt_path": args.gt,
            "document_type": "digital",  # TODO: Auto-detect
        }
        results = runner.run_single_benchmark(
            test_case=test_case,
            qa_pairs=runner._load_qa_pairs(args.qa) if args.qa else None,
            output_dir=args.output,
        )
    else:
        # Load test cases from config
        if os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                test_cases = config.get("datasets", {}).get("test_cases", [])
        else:
            print(f"Config not found: {args.config}")
            print("Use --pdf to specify a single file, or provide valid config.")
            return

        results = runner.run_full_benchmark(
            test_cases=test_cases,
            qa_path=args.qa,
            output_dir=args.output,
        )

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    if "summary" in results:
        print(f"\nProcessed {results['summary']['total_cases']} test cases")
        print("\nParser Results:")
        for parser_name, data in results["summary"]["parsers"].items():
            print(f"\n  {parser_name}:")
            print(f"    Runs: {data['total_runs']}")
            if data['avg_latency']:
                print(f"    Avg Latency: {data['avg_latency']:.2f}s")
            if data['avg_boundary_score']:
                print(f"    Avg Boundary Score: {data['avg_boundary_score']:.3f}")


if __name__ == "__main__":
    main()
