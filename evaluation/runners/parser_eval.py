"""Parser evaluation runner.

Runs 4-parser comparison experiments with CER, Structure F1, and TEDS metrics.
"""

import json
import time
from pathlib import Path
from typing import Optional

from evaluation.metrics.cer import calculate_cer, calculate_wer
from evaluation.metrics.structure import calculate_structure_eval


def evaluate_single_document(
    pdf_path: Path,
    gt_path: Path,
    vlm_api_url: str = "http://localhost:8010/v1/chat/completions",
    vlm_model: str = "qwen3-vl-2b-instruct",
    skip_advanced: bool = False,
) -> dict:
    """Evaluate all parsers on a single document.

    Args:
        pdf_path: Path to PDF file
        gt_path: Path to ground truth markdown
        vlm_api_url: VLM API endpoint
        vlm_model: VLM model name
        skip_advanced: Skip VLM-based advanced parsers

    Returns:
        Evaluation results dict
    """
    from wigtnocr.parsers.pymupdf import OCRParser
    from wigtnocr.pipeline.two_stage import TwoStageParser

    pdf_bytes = pdf_path.read_bytes()
    gt_text = gt_path.read_text(encoding="utf-8")

    results = {}

    # Text-Baseline
    ocr = OCRParser()
    start = time.time()
    ocr_result = ocr.parse_pdf(pdf_bytes)
    elapsed = time.time() - start

    cer = calculate_cer(ocr_result.content, gt_text)
    wer = calculate_wer(ocr_result.content, gt_text)
    struct = calculate_structure_eval(ocr_result.content, gt_text)

    results["text_baseline"] = {
        "cer": cer, "wer": wer,
        "element_ned": struct.element_ned,
        "reading_order_ned": struct.reading_order_ned,
        "detection_f1": struct.detection_f1,
        "per_type_ned": struct.per_type_ned,
        "elapsed_time": elapsed,
    }

    if not skip_advanced:
        # Text-Advanced
        two_stage = TwoStageParser(
            structurer_api_url=vlm_api_url,
            structurer_model=vlm_model,
        )
        ts_result = two_stage.parse_text_pdf(pdf_bytes)

        cer = calculate_cer(ts_result.content, gt_text)
        wer = calculate_wer(ts_result.content, gt_text)
        struct = calculate_structure_eval(ts_result.content, gt_text)

        results["text_advanced"] = {
            "cer": cer, "wer": wer,
            "element_ned": struct.element_ned,
            "reading_order_ned": struct.reading_order_ned,
            "detection_f1": struct.detection_f1,
            "per_type_ned": struct.per_type_ned,
            "elapsed_time": ts_result.elapsed_time,
            "stage1_time": ts_result.stage1_time,
            "stage2_time": ts_result.stage2_time,
        }

    return results


def run_batch_evaluation(
    dataset_dir: Path,
    output_dir: Path,
    vlm_api_url: str = "http://localhost:8010/v1/chat/completions",
    vlm_model: str = "qwen3-vl-2b-instruct",
    skip_advanced: bool = False,
) -> dict:
    """Run evaluation on all documents in a dataset directory.

    Args:
        dataset_dir: Directory containing document folders (each with PDF + gt.md)
        output_dir: Directory to save results
        vlm_api_url: VLM API endpoint
        vlm_model: VLM model name
        skip_advanced: Skip VLM-based advanced parsers

    Returns:
        Aggregated results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for doc_dir in sorted(dataset_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        pdf_files = list(doc_dir.glob("*.pdf"))
        gt_files = list(doc_dir.glob("gt*.md"))

        if not pdf_files or not gt_files:
            continue

        doc_id = doc_dir.name
        print(f"Evaluating: {doc_id}")

        result = evaluate_single_document(
            pdf_path=pdf_files[0],
            gt_path=gt_files[0],
            vlm_api_url=vlm_api_url,
            vlm_model=vlm_model,
            skip_advanced=skip_advanced,
        )

        all_results[doc_id] = result

        # Save per-document result
        result_file = output_dir / doc_id / "evaluation.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return all_results
