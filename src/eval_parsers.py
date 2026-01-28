#!/usr/bin/env python3
"""
CLI 기반 Parser 테스트 스크립트

4가지 파서 비교 테스트:
1. Text-Baseline: PyMuPDF 기반 디지털 PDF 텍스트 추출
2. Image-Baseline: RapidOCR 기반 스캔 PDF OCR
3. Text-Advanced: Text-Baseline + VLM 구조화
4. Image-Advanced: Image-Baseline + VLM 구조화

지원 포맷:
- PDF: 디지털/스캔 PDF

Usage:
    # 전체 테스트 (data/ 폴더의 모든 test_* 스캔)
    python -m src.eval_parsers --all

    # 단일 PDF 테스트
    python -m src.eval_parsers --input data/sample.pdf --gt data/ground_truth.md

    # Baseline만 테스트 (Advanced 스킵)
    python -m src.eval_parsers --all --skip-advanced
"""

import argparse
import sys
import time
import re
from pathlib import Path
from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from difflib import SequenceMatcher

import jiwer

# =============================================================================
# Import Compatibility Layer
# =============================================================================

def _import_parsers():
    """파서 모듈을 동적으로 임포트 (경로 호환성 처리)"""
    global OCRParser, RapidOCRParser, check_rapidocr_available
    global TwoStageParser, TwoStageResult

    try:
        # 방법 1: src.parsers (프로젝트 루트에서 실행)
        from src.parsers.ocr_parser import OCRParser, RapidOCRParser, check_rapidocr_available
        from src.parsers.two_stage_parser import TwoStageParser, TwoStageResult
    except ImportError:
        try:
            # 방법 2: parsers (src/ 디렉토리에서 실행)
            from parsers.ocr_parser import OCRParser, RapidOCRParser, check_rapidocr_available
            from parsers.two_stage_parser import TwoStageParser, TwoStageResult
        except ImportError as e:
            print(f"파서 모듈을 찾을 수 없습니다: {e}")
            print("   프로젝트 루트에서 실행하세요: python -m src.eval_parsers")
            sys.exit(1)

    return OCRParser, RapidOCRParser, check_rapidocr_available, TwoStageParser, TwoStageResult

# 지연 임포트를 위한 플레이스홀더
OCRParser = None
RapidOCRParser = None
check_rapidocr_available = None
TwoStageParser = None
TwoStageResult = None


# =============================================================================
# File Format Detection
# =============================================================================

class FileFormat(Enum):
    """지원하는 파일 포맷"""
    PDF = "pdf"
    UNKNOWN = "unknown"


def detect_file_format(file_path: Path) -> FileFormat:
    """파일 확장자로 포맷 감지"""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return FileFormat.PDF
    else:
        return FileFormat.UNKNOWN


# =============================================================================
# Text Normalization (for fair comparison)
# =============================================================================

def normalize_text(text: str) -> str:
    """텍스트 정규화 - CER/WER 비교를 위한 전처리

    - 마크다운 문법 제거
    - 다중 공백/줄바꿈 정규화
    - 앞뒤 공백 제거
    """
    if not text:
        return ""

    result = text

    # 마크다운 헤더 제거
    result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)

    # Bold/Italic 제거
    result = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', result)
    result = re.sub(r'\*\*(.+?)\*\*', r'\1', result)
    result = re.sub(r'\*(.+?)\*', r'\1', result)
    result = re.sub(r'___(.+?)___', r'\1', result)
    result = re.sub(r'__(.+?)__', r'\1', result)
    result = re.sub(r'_(.+?)_', r'\1', result)

    # 링크 제거
    result = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', result)

    # 리스트 마커 제거
    result = re.sub(r'^[\s]*[-*+]\s+', '', result, flags=re.MULTILINE)
    result = re.sub(r'^[\s]*\d+\.\s+', '', result, flags=re.MULTILINE)

    # 테이블 파이프 제거
    result = re.sub(r'\|', ' ', result)

    # 다중 공백 정규화
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n{2,}', '\n', result)

    # 각 줄 앞뒤 공백 제거
    lines = [line.strip() for line in result.split('\n')]
    lines = [line for line in lines if line]

    return '\n'.join(lines).strip()


# =============================================================================
# Tokenizer
# =============================================================================

def get_tokenizer(tokenizer_type: str = "whitespace"):
    """토크나이저 반환

    Args:
        tokenizer_type: "whitespace", "mecab", "okt"

    Returns:
        tokenize 함수
    """
    if tokenizer_type == "mecab":
        try:
            from konlpy.tag import Mecab
            mecab = Mecab()
            print("Mecab 토크나이저 사용")
            return mecab.morphs
        except ImportError:
            print("konlpy가 설치되지 않았습니다. whitespace로 fallback")
            return lambda x: x.split()
        except Exception as e:
            print(f"Mecab 초기화 실패: {e}. whitespace로 fallback")
            return lambda x: x.split()

    elif tokenizer_type == "okt":
        try:
            from konlpy.tag import Okt
            okt = Okt()
            print("Okt 토크나이저 사용")
            return okt.morphs
        except ImportError:
            print("konlpy가 설치되지 않았습니다. whitespace로 fallback")
            return lambda x: x.split()
        except Exception as e:
            print(f"Okt 초기화 실패: {e}. whitespace로 fallback")
            return lambda x: x.split()

    else:
        return lambda x: x.split()


# =============================================================================
# CER/WER Calculation (using jiwer library)
# =============================================================================


def calculate_cer(hypothesis: str, reference: str) -> dict:
    """Character Error Rate 계산 (jiwer 사용)

    Returns:
        dict with cer, substitutions, deletions, insertions
    """
    if not reference:
        return {"cer": 0.0 if not hypothesis else float('inf')}

    if not hypothesis:
        return {
            "cer": 1.0,
            "substitutions": 0,
            "deletions": len(reference),
            "insertions": 0
        }

    # jiwer.cer: reference first, hypothesis second
    cer_value = jiwer.cer(reference, hypothesis)

    # 상세 정보
    output = jiwer.process_characters(reference, hypothesis)

    return {
        "cer": cer_value,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits
    }


def calculate_wer(hypothesis: str, reference: str, tokenizer=None) -> dict:
    """Word Error Rate 계산 (jiwer 사용)

    Args:
        hypothesis: 파서 출력 텍스트
        reference: Ground Truth 텍스트
        tokenizer: 토크나이저 함수 (None이면 공백 분리)

    Returns:
        dict with wer, substitutions, deletions, insertions
    """
    def default_tokenizer(x: str) -> list[str]:
        return x.split()

    if tokenizer is None:
        tokenizer = default_tokenizer

    if not reference:
        return {"wer": 0.0 if not hypothesis else float('inf')}

    # 토큰화
    ref_tokens = tokenizer(reference)
    hyp_tokens = tokenizer(hypothesis) if hypothesis else []

    if not hyp_tokens:
        return {
            "wer": 1.0,
            "substitutions": 0,
            "deletions": len(ref_tokens),
            "insertions": 0,
            "ref_tokens": len(ref_tokens),
            "hyp_tokens": 0
        }

    # 토큰을 공백으로 연결하여 jiwer에 전달
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)

    # jiwer.wer: reference first, hypothesis second
    wer_value = jiwer.wer(ref_str, hyp_str)

    # 상세 정보
    output = jiwer.process_words(ref_str, hyp_str)

    return {
        "wer": wer_value,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "ref_tokens": len(ref_tokens),
        "hyp_tokens": len(hyp_tokens)
    }


# =============================================================================
# Structure F1 Calculation
# =============================================================================

@dataclass
class StructureElement:
    """구조 요소 데이터 클래스"""
    type: str          # "heading", "list_item", "table_row", "code_block"
    level: int         # 헤딩 레벨 (1-6) 또는 리스트 깊이
    content: str       # 요소 내용 (정규화된 텍스트)
    line_number: int   # 원본 라인 번호


def extract_structure_elements(text: str) -> List[StructureElement]:
    """마크다운 텍스트에서 구조 요소 추출

    Args:
        text: 마크다운 텍스트

    Returns:
        구조 요소 리스트
    """
    elements = []
    lines = text.split('\n')
    in_code_block = False

    for i, line in enumerate(lines):
        # Code Block 시작/종료 검출
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            if not in_code_block:
                # 코드 블록 종료 시에는 추가하지 않음
                continue
            # 코드 블록 시작
            elements.append(StructureElement(
                type="code_block",
                level=0,
                content=line.strip(),
                line_number=i
            ))
            continue

        # 코드 블록 내부는 스킵
        if in_code_block:
            continue

        # Heading 검출
        if match := re.match(r'^(#{1,6})\s+(.+)$', line):
            elements.append(StructureElement(
                type="heading",
                level=len(match.group(1)),
                content=match.group(2).strip(),
                line_number=i
            ))

        # Unordered List 검출
        elif match := re.match(r'^(\s*)[-*+]\s+(.+)$', line):
            indent = len(match.group(1))
            elements.append(StructureElement(
                type="list_item",
                level=indent // 2,
                content=match.group(2).strip(),
                line_number=i
            ))

        # Ordered List 검출
        elif match := re.match(r'^(\s*)\d+\.\s+(.+)$', line):
            indent = len(match.group(1))
            elements.append(StructureElement(
                type="list_item",
                level=indent // 2,
                content=match.group(2).strip(),
                line_number=i
            ))

        # Table Row 검출
        elif re.match(r'^\|.+\|$', line.strip()):
            # 구분선 제외 (|---|---| 또는 |:---|:---:| 등)
            if not re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                elements.append(StructureElement(
                    type="table_row",
                    level=0,
                    content=line.strip(),
                    line_number=i
                ))

    return elements


def match_structure_elements(
    hypothesis_elements: List[StructureElement],
    reference_elements: List[StructureElement],
    similarity_threshold: float = 0.8
) -> Tuple[int, int, int]:
    """구조 요소 매칭 및 TP/FP/FN 계산

    Args:
        hypothesis_elements: 파서 출력의 구조 요소
        reference_elements: GT의 구조 요소
        similarity_threshold: 내용 유사도 임계값

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    matched_ref = set()
    matched_hyp = set()

    # 유형별로 그룹화
    ref_by_type = defaultdict(list)
    for i, elem in enumerate(reference_elements):
        ref_by_type[elem.type].append((i, elem))

    hyp_by_type = defaultdict(list)
    for i, elem in enumerate(hypothesis_elements):
        hyp_by_type[elem.type].append((i, elem))

    # 유형별 매칭
    for elem_type in ref_by_type:
        if elem_type not in hyp_by_type:
            continue

        ref_items = ref_by_type[elem_type]
        hyp_items = hyp_by_type[elem_type]

        # 유사도 행렬 계산
        similarities = []
        for ri, ref_elem in ref_items:
            for hi, hyp_elem in hyp_items:
                sim = SequenceMatcher(
                    None,
                    ref_elem.content.lower(),
                    hyp_elem.content.lower()
                ).ratio()
                if sim >= similarity_threshold:
                    similarities.append((sim, ri, hi))

        # Greedy 매칭 (높은 유사도부터)
        similarities.sort(reverse=True)
        for sim, ri, hi in similarities:
            if ri not in matched_ref and hi not in matched_hyp:
                matched_ref.add(ri)
                matched_hyp.add(hi)

    tp = len(matched_ref)
    fp = len(hypothesis_elements) - len(matched_hyp)
    fn = len(reference_elements) - len(matched_ref)

    return tp, fp, fn


def calculate_structure_f1(hypothesis: str, reference: str, similarity_threshold: float = 0.8) -> dict:
    """Structure F1 스코어 계산

    Args:
        hypothesis: 파서 출력 (마크다운)
        reference: Ground Truth (마크다운)
        similarity_threshold: 내용 유사도 임계값

    Returns:
        {
            "structure_f1": float,
            "structure_precision": float,
            "structure_recall": float,
            "true_positives": int,
            "false_positives": int,
            "false_negatives": int,
            "hypothesis_elements": int,
            "reference_elements": int
        }
    """
    hyp_elements = extract_structure_elements(hypothesis)
    ref_elements = extract_structure_elements(reference)

    if not ref_elements:
        # GT에 구조 요소가 없으면 평가 불가
        return {
            "structure_f1": None,
            "structure_precision": None,
            "structure_recall": None,
            "true_positives": 0,
            "false_positives": len(hyp_elements),
            "false_negatives": 0,
            "hypothesis_elements": len(hyp_elements),
            "reference_elements": 0,
            "error": "No structure elements in reference"
        }

    tp, fp, fn = match_structure_elements(hyp_elements, ref_elements, similarity_threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "structure_f1": f1,
        "structure_precision": precision,
        "structure_recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "hypothesis_elements": len(hyp_elements),
        "reference_elements": len(ref_elements)
    }


# =============================================================================
# Parser Tests
# =============================================================================

def test_text_baseline(pdf_bytes: bytes, verbose: bool = False) -> dict:
    """Text-Baseline Parser (PyMuPDF) 테스트"""
    global OCRParser
    if OCRParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("Text-Baseline (PyMuPDF)")
    print("=" * 60)

    parser = OCRParser()

    # PDF 타입 확인
    pdf_type = parser.detect_pdf_type(pdf_bytes)
    print(f"PDF 타입: {pdf_type}")

    # 파싱
    result = parser.parse_pdf(pdf_bytes)

    print("\n결과:")
    print(f"   - 성공: {'O' if result.success else 'X'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - 텍스트 존재: {'O' if result.has_text else 'X'}")
    print(f"   - 표 개수: {len(result.tables)}")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - 추출 길이: {len(result.content)} chars")

    if verbose and result.content:
        print("\n추출 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "has_text": result.has_text,
        "pdf_type": pdf_type
    }


def test_image_baseline(pdf_bytes: bytes, verbose: bool = False) -> dict:
    """Image-Baseline Parser (RapidOCR) 테스트"""
    global RapidOCRParser, check_rapidocr_available
    if RapidOCRParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("Image-Baseline (RapidOCR)")
    print("=" * 60)

    if not check_rapidocr_available():
        print("RapidOCR가 설치되지 않았습니다.")
        print("   pip install rapidocr-pdf[onnxruntime]")
        return {"success": False, "error": "RapidOCR not installed"}

    parser = RapidOCRParser()
    result = parser.parse_pdf(pdf_bytes)

    print("\n결과:")
    print(f"   - 성공: {'O' if result.success else 'X'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - 추출 길이: {len(result.content)} chars")

    if result.error:
        print(f"   - 에러: {result.error}")

    if verbose and result.content:
        print("\n추출 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "error": result.error
    }


def test_text_advanced(pdf_bytes: bytes, verbose: bool = False) -> dict:
    """Text-Advanced Parser (PyMuPDF + VLM 구조화) 테스트"""
    global TwoStageParser
    if TwoStageParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("Text-Advanced (PyMuPDF + VLM)")
    print("=" * 60)

    parser = TwoStageParser()
    result = parser.parse_text_pdf(pdf_bytes)

    print("\n결과:")
    print(f"   - 성공: {'O' if result.success else 'X'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - Stage 1 ({result.stage1_parser}): {result.stage1_time:.2f}s")
    print(f"   - Stage 2 (VLM 구조화): {result.stage2_time:.2f}s {'O 적용됨' if result.stage2_applied else 'X 미적용'}")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - Stage 1 추출 길이: {len(result.stage1_content)} chars")
    print(f"   - 최종 출력 길이: {len(result.content)} chars")

    if result.error:
        print(f"   - 에러: {result.error}")

    if verbose and result.content:
        print("\n구조화 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "stage1_time": result.stage1_time,
        "stage2_time": result.stage2_time,
        "stage1_parser": result.stage1_parser,
        "stage2_applied": result.stage2_applied,
        "stage1_content_length": len(result.stage1_content),
        "error": result.error
    }


def test_image_advanced(pdf_bytes: bytes, verbose: bool = False) -> dict:
    """Image-Advanced Parser (RapidOCR + VLM 구조화) 테스트"""
    global TwoStageParser, check_rapidocr_available
    if TwoStageParser is None:
        _import_parsers()

    print("\n" + "=" * 60)
    print("Image-Advanced (RapidOCR + VLM)")
    print("=" * 60)

    if not check_rapidocr_available():
        print("RapidOCR가 설치되지 않았습니다.")
        print("   pip install rapidocr-pdf[onnxruntime]")
        return {"success": False, "error": "RapidOCR not installed"}

    parser = TwoStageParser()
    result = parser.parse_scanned_pdf(pdf_bytes)

    print("\n결과:")
    print(f"   - 성공: {'O' if result.success else 'X'}")
    print(f"   - 페이지 수: {result.page_count}")
    print(f"   - Stage 1 ({result.stage1_parser}): {result.stage1_time:.2f}s")
    print(f"   - Stage 2 (VLM 구조화): {result.stage2_time:.2f}s {'O 적용됨' if result.stage2_applied else 'X 미적용'}")
    print(f"   - 총 시간: {result.elapsed_time:.2f}s")
    print(f"   - Stage 1 추출 길이: {len(result.stage1_content)} chars")
    print(f"   - 최종 출력 길이: {len(result.content)} chars")

    if result.error:
        print(f"   - 에러: {result.error}")

    if verbose and result.content:
        print("\n구조화 결과 (처음 500자):")
        print("-" * 40)
        print(result.content[:500])
        print("-" * 40)

    return {
        "success": result.success,
        "content": result.content,
        "elapsed_time": result.elapsed_time,
        "page_count": result.page_count,
        "stage1_time": result.stage1_time,
        "stage2_time": result.stage2_time,
        "stage1_parser": result.stage1_parser,
        "stage2_applied": result.stage2_applied,
        "stage1_content_length": len(result.stage1_content),
        "error": result.error
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_results(results: dict, ground_truth: str, tokenizer=None, tokenizer_name: str = "whitespace") -> dict:
    """결과 평가 (CER, WER, Structure F1 계산) - jiwer 사용

    Args:
        results: 파서별 결과 딕셔너리
        ground_truth: Ground Truth 텍스트
        tokenizer: WER 계산용 토크나이저 함수
        tokenizer_name: 토크나이저 이름 (출력용)
    """
    print("\n" + "=" * 60)
    print("평가 결과 (Ground Truth 비교)")
    print(f"   WER Tokenizer: {tokenizer_name}")
    print("=" * 60)

    gt_normalized = normalize_text(ground_truth)
    gt_structure_elements = extract_structure_elements(ground_truth)
    print(f"Ground Truth 길이: {len(gt_normalized)} chars (정규화 후)")
    print(f"Ground Truth 구조 요소: {len(gt_structure_elements)} elements")
    print()

    evaluation = {}

    for parser_name, result in results.items():
        if not result.get("success"):
            print(f"{parser_name}: SKIP (파싱 실패)")
            evaluation[parser_name] = {"cer": None, "wer": None, "structure_f1": None}
            continue

        content = result.get("content", "")
        if not content:
            print(f"{parser_name}: SKIP (내용 없음)")
            evaluation[parser_name] = {"cer": None, "wer": None, "structure_f1": None}
            continue

        # 정규화 (CER/WER용)
        content_normalized = normalize_text(content)

        # CER, WER 계산 (jiwer)
        cer_result = calculate_cer(content_normalized, gt_normalized)
        wer_result = calculate_wer(content_normalized, gt_normalized, tokenizer)

        # Structure F1 계산 (원본 마크다운 사용)
        structure_f1_result = calculate_structure_f1(content, ground_truth)

        cer = cer_result["cer"]
        wer = wer_result["wer"]
        struct_f1 = structure_f1_result.get("structure_f1")

        print(f"{parser_name}:")
        print(f"   - CER: {cer:.4f} ({cer*100:.2f}%)")
        print(f"      S:{cer_result.get('substitutions', 0)} D:{cer_result.get('deletions', 0)} I:{cer_result.get('insertions', 0)}")
        print(f"   - WER: {wer:.4f} ({wer*100:.2f}%)")
        print(f"      S:{wer_result.get('substitutions', 0)} D:{wer_result.get('deletions', 0)} I:{wer_result.get('insertions', 0)}")
        print(f"      Tokens: ref={wer_result.get('ref_tokens', 0)} hyp={wer_result.get('hyp_tokens', 0)}")

        # Structure F1 출력
        if struct_f1 is not None:
            print(f"   - Structure F1: {struct_f1:.4f} ({struct_f1*100:.2f}%)")
            print(f"      P:{structure_f1_result.get('structure_precision', 0):.2f} R:{structure_f1_result.get('structure_recall', 0):.2f}")
            print(f"      TP:{structure_f1_result.get('true_positives', 0)} FP:{structure_f1_result.get('false_positives', 0)} FN:{structure_f1_result.get('false_negatives', 0)}")
            print(f"      Elements: hyp={structure_f1_result.get('hypothesis_elements', 0)} ref={structure_f1_result.get('reference_elements', 0)}")
        else:
            print(f"   - Structure F1: N/A ({structure_f1_result.get('error', 'unknown error')})")

        print(f"   - Latency: {result.get('elapsed_time', 0):.2f}s")
        print()

        evaluation[parser_name] = {
            "cer": cer,
            "cer_detail": cer_result,
            "wer": wer,
            "wer_detail": wer_result,
            "structure_f1": struct_f1,
            "structure_f1_detail": structure_f1_result,
            "latency": result.get("elapsed_time", 0),
            "tokenizer": tokenizer_name
        }

    return evaluation


def save_results_to_files(results: dict, output_dir: str, pdf_name: str, evaluation: dict = None, tokenizer_name: str = "whitespace", metadata: dict = None):
    """파싱 결과를 파일로 저장

    Args:
        results: 파서별 결과
        output_dir: 출력 디렉토리
        pdf_name: 입력 파일 경로
        evaluation: 평가 결과 (CER, WER)
        tokenizer_name: 토크나이저 이름
        metadata: 테스트 메타데이터
    """
    import json
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print(f"결과 저장: {output_path}")
    print("=" * 60)

    saved_files = []

    # 1. 각 파서별 출력 저장
    for parser_name, result in results.items():
        if not result.get("success"):
            continue

        content = result.get("content", "")
        if not content:
            continue

        safe_name = parser_name.lower().replace(" ", "-").replace("-", "_")
        filename = f"{safe_name}_output.txt"
        filepath = output_path / filename

        filepath.write_text(content, encoding="utf-8")
        saved_files.append(filepath)
        print(f"   O {filename} ({len(content)} chars)")

    # 2. 평가 결과 JSON 저장
    eval_json = {
        "pdf": pdf_name,
        "timestamp": timestamp,
        "tokenizer": tokenizer_name,
        "parsers": {}
    }

    # 메타데이터 추가
    if metadata:
        eval_json["metadata"] = {
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "doc_type": metadata.get("doc_type", "unknown"),
            "language": metadata.get("language", "unknown"),
        }

    for name, result in results.items():
        eval_json["parsers"][name] = {
            "success": result.get("success"),
            "elapsed_time": result.get("elapsed_time"),
            "content_length": len(result.get("content", ""))
        }

        # Advanced 파서 추가 필드
        if "stage1_time" in result:
            eval_json["parsers"][name]["stage1_time"] = result.get("stage1_time")
            eval_json["parsers"][name]["stage2_time"] = result.get("stage2_time")
            eval_json["parsers"][name]["stage1_parser"] = result.get("stage1_parser")
            eval_json["parsers"][name]["stage2_applied"] = result.get("stage2_applied")
            eval_json["parsers"][name]["stage1_content_length"] = result.get("stage1_content_length")

        if evaluation and name in evaluation:
            eval_data = evaluation[name]
            eval_json["parsers"][name]["cer"] = eval_data.get("cer")
            eval_json["parsers"][name]["wer"] = eval_data.get("wer")
            eval_json["parsers"][name]["structure_f1"] = eval_data.get("structure_f1")
            # Structure F1 상세 정보
            if eval_data.get("structure_f1_detail"):
                sf1_detail = eval_data["structure_f1_detail"]
                eval_json["parsers"][name]["structure_f1_detail"] = {
                    "precision": sf1_detail.get("structure_precision"),
                    "recall": sf1_detail.get("structure_recall"),
                    "true_positives": sf1_detail.get("true_positives"),
                    "false_positives": sf1_detail.get("false_positives"),
                    "false_negatives": sf1_detail.get("false_negatives"),
                    "hypothesis_elements": sf1_detail.get("hypothesis_elements"),
                    "reference_elements": sf1_detail.get("reference_elements")
                }

    meta_path = output_path / "evaluation.json"
    meta_path.write_text(json.dumps(eval_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print("   O evaluation.json")

    # 3. 요약 마크다운 저장
    summary_lines = [
        "# Parsing Test Results",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **PDF**: {pdf_name}",
        f"- **Tokenizer**: {tokenizer_name}",
        "",
        "## Results",
        "",
        "| Parser | Success | Latency | Length |",
        "|--------|---------|---------|--------|",
    ]

    for name, result in results.items():
        success = "O" if result.get("success") else "X"
        latency = f"{result.get('elapsed_time', 0):.2f}s"
        length = f"{len(result.get('content', ''))} chars"
        summary_lines.append(f"| {name} | {success} | {latency} | {length} |")

    if evaluation:
        summary_lines.extend([
            "",
            "## Evaluation (vs Ground Truth)",
            "",
            "| Parser | CER | WER | Struct-F1 |",
            "|--------|-----|-----|-----------|",
        ])
        for name, eval_data in evaluation.items():
            cer = eval_data.get("cer")
            wer = eval_data.get("wer")
            struct_f1 = eval_data.get("structure_f1")
            cer_str = f"{cer*100:.2f}%" if cer is not None else "N/A"
            wer_str = f"{wer*100:.2f}%" if wer is not None else "N/A"
            struct_f1_str = f"{struct_f1*100:.2f}%" if struct_f1 is not None else "N/A"
            summary_lines.append(f"| {name} | {cer_str} | {wer_str} | {struct_f1_str} |")

    summary_path = output_path / "README.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("   O README.md")

    return saved_files


def print_summary(results: dict, evaluation: dict = None):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("테스트 요약")
    print("=" * 60)

    print("\n| Parser | 성공 | 시간 | 추출 길이 |")
    print("|--------|------|------|----------|")

    for name, result in results.items():
        success = "O" if result.get("success") else "X"
        time_str = f"{result.get('elapsed_time', 0):.2f}s"
        length = len(result.get("content", ""))
        print(f"| {name} | {success} | {time_str} | {length} chars |")

    if evaluation:
        print("\n| Parser | CER | WER | Struct-F1 | Latency |")
        print("|--------|-----|-----|-----------|---------|")

        for name, eval_result in evaluation.items():
            cer = eval_result.get("cer")
            wer = eval_result.get("wer")
            struct_f1 = eval_result.get("structure_f1")
            latency = eval_result.get("latency", 0)

            cer_str = f"{cer*100:.2f}%" if cer is not None else "N/A"
            wer_str = f"{wer*100:.2f}%" if wer is not None else "N/A"
            struct_f1_str = f"{struct_f1*100:.2f}%" if struct_f1 is not None else "N/A"

            print(f"| {name} | {cer_str} | {wer_str} | {struct_f1_str} | {latency:.2f}s |")


# =============================================================================
# Data Folder Scanning
# =============================================================================

def extract_file_metadata(file_path: Path) -> dict:
    """파일에서 자동으로 메타데이터 추출"""
    import fitz

    metadata = {
        "filename": file_path.name,
        "file_size_kb": round(file_path.stat().st_size / 1024, 1),
        "doc_type": "unknown",
        "pages": 0,
        "title": file_path.stem,
        "language": "unknown",
        "has_text_layer": False,
    }

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        metadata["doc_type"] = "PDF"
        try:
            doc = fitz.open(file_path)
            metadata["pages"] = len(doc)

            # 텍스트 레이어 확인 (첫 페이지)
            if len(doc) > 0:
                first_page_text = doc[0].get_text("text")
                metadata["has_text_layer"] = len(first_page_text.strip()) > 100

            doc.close()
        except Exception as e:
            print(f"PDF 메타데이터 추출 실패: {e}")

    # 언어 감지 (파일명 기반)
    filename = file_path.name
    if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in filename):
        metadata["language"] = "ko"
    elif any(ord(c) >= 0x4E00 and ord(c) <= 0x9FFF for c in filename):
        metadata["language"] = "zh"
    elif any(ord(c) >= 0x3040 and ord(c) <= 0x30FF for c in filename):
        metadata["language"] = "ja"
    else:
        metadata["language"] = "en"

    return metadata


def load_test_metadata(folder_path: Path, input_file: Optional[Path] = None) -> dict:
    """테스트 폴더의 메타데이터 로드"""
    if input_file and input_file.exists():
        return extract_file_metadata(input_file)

    # 폴더에서 입력 파일 찾기
    for f in folder_path.glob("*.pdf"):
        if not f.name.startswith("gt_"):
            return extract_file_metadata(f)

    return {
        "filename": "unknown",
        "file_size_kb": 0,
        "doc_type": "unknown",
        "pages": 0,
        "title": folder_path.name,
        "language": "unknown",
        "has_text_layer": False,
    }


def scan_data_folders(data_dir: Path = Path("data")) -> List[dict]:
    """data/ 폴더의 모든 test_* 폴더를 스캔"""
    test_folders = []

    if not data_dir.exists():
        print(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")
        return []

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("test_"):
            continue

        test_id = folder.name
        input_file = None
        gt_file = None

        for f in folder.iterdir():
            if f.is_file():
                fmt = detect_file_format(f)
                if fmt != FileFormat.UNKNOWN and not f.name.startswith("gt_"):
                    input_file = f
                elif f.name.startswith("gt_") and f.suffix in [".md", ".txt"]:
                    gt_file = f

        if input_file:
            metadata = load_test_metadata(folder, input_file)

            test_folders.append({
                "test_id": test_id,
                "input_file": input_file,
                "gt_file": gt_file,
                "folder_path": folder,
                "metadata": metadata
            })
        else:
            print(f"{test_id}: 입력 파일을 찾을 수 없습니다")

    return test_folders


def run_single_test(
    input_path: Path,
    gt_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skip_advanced: bool = False,
    skip_image: bool = False,
    verbose: bool = False,
    tokenizer_type: str = "whitespace",
    metadata: dict = None
) -> dict:
    """단일 파일 파싱 테스트 실행"""
    file_format = detect_file_format(input_path)
    input_bytes = input_path.read_bytes()

    print("=" * 60)
    print("VLM Document Parsing Quality Test")
    print("=" * 60)
    print(f"입력 파일: {input_path}")
    print(f"포맷: {file_format.value.upper()}")
    print(f"크기: {len(input_bytes) / 1024:.1f} KB")

    if file_format == FileFormat.UNKNOWN:
        print("지원하지 않는 파일 포맷입니다.")
        return {"results": {}, "evaluation": None}

    # Ground Truth 읽기
    ground_truth = None
    if gt_path and gt_path.exists():
        ground_truth = gt_path.read_text(encoding="utf-8")
        print(f"Ground Truth: {gt_path} ({len(ground_truth)} chars)")

    results = {}

    # 1. Text-Baseline (PyMuPDF)
    try:
        results["Text-Baseline"] = test_text_baseline(input_bytes, verbose)
    except Exception as e:
        print(f"Text-Baseline 오류: {e}")
        results["Text-Baseline"] = {"success": False, "error": str(e)}

    # 2. Image-Baseline (RapidOCR)
    if not skip_image:
        try:
            results["Image-Baseline"] = test_image_baseline(input_bytes, verbose)
        except Exception as e:
            print(f"Image-Baseline 오류: {e}")
            results["Image-Baseline"] = {"success": False, "error": str(e)}

    # 3. Text-Advanced (PyMuPDF + VLM)
    if not skip_advanced:
        try:
            results["Text-Advanced"] = test_text_advanced(input_bytes, verbose)
        except Exception as e:
            print(f"Text-Advanced 오류: {e}")
            results["Text-Advanced"] = {"success": False, "error": str(e)}

    # 4. Image-Advanced (RapidOCR + VLM)
    if not skip_advanced and not skip_image:
        try:
            results["Image-Advanced"] = test_image_advanced(input_bytes, verbose)
        except Exception as e:
            print(f"Image-Advanced 오류: {e}")
            results["Image-Advanced"] = {"success": False, "error": str(e)}

    # 평가
    evaluation = None
    if ground_truth:
        tokenizer = get_tokenizer(tokenizer_type)
        evaluation = evaluate_results(results, ground_truth, tokenizer, tokenizer_type)

    # 결과 저장
    if output_dir:
        save_results_to_files(results, str(output_dir), str(input_path), evaluation, tokenizer_type, metadata)

    print_summary(results, evaluation)

    return {"results": results, "evaluation": evaluation}


def run_all_tests(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    skip_advanced: bool = False,
    skip_image: bool = False,
    verbose: bool = False,
    tokenizer_type: str = "whitespace"
) -> dict:
    """data/ 폴더의 모든 테스트 실행"""
    test_folders = scan_data_folders(data_dir)

    if not test_folders:
        print("테스트할 데이터가 없습니다.")
        return {}

    print("=" * 60)
    print("VLM Document Parsing - Batch Test")
    print("=" * 60)
    print(f"데이터 폴더: {data_dir}")
    print(f"테스트 수: {len(test_folders)}")
    if skip_advanced:
        print("Advanced 파서 스킵됨")
    if skip_image:
        print("Image 파서 스킵됨")
    print()

    for info in test_folders:
        fmt = detect_file_format(info["input_file"])
        gt_status = "O" if info["gt_file"] else "X"
        print(f"  - {info['test_id']}: {info['input_file'].name} ({fmt.value}) [GT: {gt_status}]")

    print()

    all_results = {}

    for i, info in enumerate(test_folders, 1):
        test_id = info["test_id"]
        print()
        print("#" * 60)
        print(f"# [{i}/{len(test_folders)}] {test_id}")
        print("#" * 60)

        output_dir = results_dir / test_id

        result = run_single_test(
            input_path=info["input_file"],
            gt_path=info["gt_file"],
            output_dir=output_dir,
            skip_advanced=skip_advanced,
            skip_image=skip_image,
            verbose=verbose,
            tokenizer_type=tokenizer_type,
            metadata=info.get("metadata")
        )

        all_results[test_id] = result

    # 전체 요약
    print()
    print("=" * 60)
    print("전체 테스트 요약")
    print("=" * 60)

    # CER 요약 테이블
    header_parts = ["Test ID", "Text-Base CER", "Image-Base CER"]
    if not skip_advanced:
        header_parts.extend(["Text-Adv CER", "Image-Adv CER"])

    print("\n### CER (Character Error Rate)")
    print(f"| {' | '.join(header_parts)} |")
    print(f"|{'-' * 10}|{'-' * 15}|{'-' * 16}|" +
          (f"{'-' * 14}|{'-' * 15}|" if not skip_advanced else ""))

    for test_id, result in all_results.items():
        eval_data = result.get("evaluation", {}) or {}

        text_base_cer = eval_data.get("Text-Baseline", {}).get("cer")
        image_base_cer = eval_data.get("Image-Baseline", {}).get("cer")

        text_base_str = f"{text_base_cer*100:.1f}%" if text_base_cer is not None else "N/A"
        image_base_str = f"{image_base_cer*100:.1f}%" if image_base_cer is not None else "N/A"

        row = f"| {test_id:<8} | {text_base_str:<13} | {image_base_str:<14} |"

        if not skip_advanced:
            text_adv_cer = eval_data.get("Text-Advanced", {}).get("cer")
            image_adv_cer = eval_data.get("Image-Advanced", {}).get("cer")
            text_adv_str = f"{text_adv_cer*100:.1f}%" if text_adv_cer is not None else "N/A"
            image_adv_str = f"{image_adv_cer*100:.1f}%" if image_adv_cer is not None else "N/A"
            row += f" {text_adv_str:<12} | {image_adv_str:<13} |"

        print(row)

    # Structure F1 요약 테이블
    struct_header_parts = ["Test ID", "Text-Base F1", "Image-Base F1"]
    if not skip_advanced:
        struct_header_parts.extend(["Text-Adv F1", "Image-Adv F1"])

    print("\n### Structure F1")
    print(f"| {' | '.join(struct_header_parts)} |")
    print(f"|{'-' * 10}|{'-' * 14}|{'-' * 15}|" +
          (f"{'-' * 13}|{'-' * 14}|" if not skip_advanced else ""))

    for test_id, result in all_results.items():
        eval_data = result.get("evaluation", {}) or {}

        text_base_f1 = eval_data.get("Text-Baseline", {}).get("structure_f1")
        image_base_f1 = eval_data.get("Image-Baseline", {}).get("structure_f1")

        text_base_str = f"{text_base_f1*100:.1f}%" if text_base_f1 is not None else "N/A"
        image_base_str = f"{image_base_f1*100:.1f}%" if image_base_f1 is not None else "N/A"

        row = f"| {test_id:<8} | {text_base_str:<12} | {image_base_str:<13} |"

        if not skip_advanced:
            text_adv_f1 = eval_data.get("Text-Advanced", {}).get("structure_f1")
            image_adv_f1 = eval_data.get("Image-Advanced", {}).get("structure_f1")
            text_adv_str = f"{text_adv_f1*100:.1f}%" if text_adv_f1 is not None else "N/A"
            image_adv_str = f"{image_adv_f1*100:.1f}%" if image_adv_f1 is not None else "N/A"
            row += f" {text_adv_str:<11} | {image_adv_str:<12} |"

        print(row)

    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4-Parser 비교 테스트 (Text/Image Baseline/Advanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
파서 구조:
  Text-Baseline  : PyMuPDF (디지털 PDF 텍스트 추출)
  Image-Baseline : RapidOCR (스캔 PDF OCR)
  Text-Advanced  : PyMuPDF + VLM 구조화
  Image-Advanced : RapidOCR + VLM 구조화

예시:
  # 전체 테스트 (data/ 폴더의 모든 test_* 스캔)
  python -m src.eval_parsers --all

  # 단일 파일 테스트
  python -m src.eval_parsers --input data/test_1/test.pdf --gt data/test_1/gt.md

  # Baseline만 테스트
  python -m src.eval_parsers --all --skip-advanced
        """
    )

    # 입력 모드 (--all 또는 --input)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="data/ 폴더의 모든 test_* 폴더 테스트"
    )
    input_group.add_argument(
        "--input", "-i",
        help="테스트할 입력 파일 (PDF)"
    )
    input_group.add_argument(
        "--pdf", "-p",
        help="테스트할 PDF 파일 경로 (레거시 옵션)"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="테스트 데이터 폴더 (기본: data/)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="결과 저장 폴더 (기본: results/)"
    )
    parser.add_argument(
        "--gt", "-g",
        help="Ground Truth 파일 경로 (--input 사용 시)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력 (추출 결과 미리보기)"
    )
    parser.add_argument(
        "--skip-advanced",
        action="store_true",
        help="Advanced 파서 테스트 스킵 (Baseline만 테스트)"
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Image 파서 테스트 스킵 (Text 파서만 테스트)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="파싱 결과를 저장할 디렉토리 (--input 사용 시)"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        choices=["whitespace", "mecab", "okt"],
        default="whitespace",
        help="WER 계산용 토크나이저 (기본: whitespace)"
    )

    args = parser.parse_args()

    # --all 모드: 전체 테스트
    if args.all:
        run_all_tests(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            skip_advanced=args.skip_advanced,
            skip_image=args.skip_image,
            verbose=args.verbose,
            tokenizer_type=args.tokenizer
        )
        return

    # 단일 파일 모드
    input_path = Path(args.input if args.input else args.pdf)
    if not input_path.exists():
        print(f"파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    gt_path = Path(args.gt) if args.gt else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_single_test(
        input_path=input_path,
        gt_path=gt_path,
        output_dir=output_dir,
        skip_advanced=args.skip_advanced,
        skip_image=args.skip_image,
        verbose=args.verbose,
        tokenizer_type=args.tokenizer
    )


if __name__ == "__main__":
    main()
