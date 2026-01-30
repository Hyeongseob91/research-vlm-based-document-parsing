"""
Parsing CLI 단위 테스트

VLM Document Parsing의 파서 평가 테스트
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.eval_parsers import (
    FileFormat,
    detect_file_format,
    normalize_text,
    calculate_cer,
    calculate_wer,
    evaluate_results,
    scan_data_folders,
    extract_markdown_tables,
    markdown_table_to_html,
    compute_teds,
    calculate_teds,
    html_to_tree,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_text():
    """테스트용 샘플 텍스트"""
    return "안녕하세요. 이것은 테스트 문장입니다."


@pytest.fixture
def sample_gt_text():
    """Ground Truth 텍스트"""
    return "안녕하세요. 이것은 테스트 문장입니다."


@pytest.fixture
def sample_parsed_text():
    """파싱된 텍스트 (약간의 오류 포함)"""
    return "안녕하세요. 이것은 테스트 문장입니다"  # 마침표 누락


@pytest.fixture
def temp_data_dir(tmp_path):
    """임시 데이터 디렉토리 생성"""
    test_dir = tmp_path / "test_1"
    test_dir.mkdir()

    # 입력 파일 생성 (빈 PDF 시뮬레이션)
    input_file = test_dir / "test.pdf"
    input_file.write_bytes(b"%PDF-1.4 dummy")

    # GT 파일 생성
    gt_file = test_dir / "gt_data.md"
    gt_file.write_text("# 테스트 문서\n\n이것은 테스트입니다.")

    return tmp_path


# =============================================================================
# FileFormat Detection Tests
# =============================================================================

class TestFileFormatDetection:
    """파일 포맷 감지 테스트"""

    def test_detect_pdf(self, tmp_path):
        """PDF 파일 감지"""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        assert detect_file_format(pdf_file) == FileFormat.PDF

    def test_detect_image_jpg(self, tmp_path):
        """JPG 이미지 감지"""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff")
        assert detect_file_format(img_file) == FileFormat.IMAGE

    def test_detect_image_png(self, tmp_path):
        """PNG 이미지 감지"""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG")
        assert detect_file_format(img_file) == FileFormat.IMAGE

    def test_detect_hwp(self, tmp_path):
        """HWP 파일 감지"""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"HWP Document")
        assert detect_file_format(hwp_file) == FileFormat.HWP

    def test_detect_hwpx(self, tmp_path):
        """HWPX 파일 감지"""
        hwpx_file = tmp_path / "test.hwpx"
        hwpx_file.write_bytes(b"HWPX Document")
        assert detect_file_format(hwpx_file) == FileFormat.HWPX

    def test_detect_unknown(self, tmp_path):
        """알 수 없는 파일 포맷"""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.write_bytes(b"unknown")
        assert detect_file_format(unknown_file) == FileFormat.UNKNOWN


# =============================================================================
# Text Normalization Tests
# =============================================================================

class TestTextNormalization:
    """텍스트 정규화 테스트"""

    def test_normalize_whitespace(self):
        """공백 정규화"""
        text = "hello    world"
        result = normalize_text(text)
        assert "  " not in result

    def test_normalize_newlines(self):
        """줄바꿈 정규화"""
        text = "hello\n\n\nworld"
        result = normalize_text(text)
        assert "\n\n\n" not in result

    def test_normalize_preserves_content(self):
        """내용 보존 확인"""
        text = "안녕하세요. 테스트입니다."
        result = normalize_text(text)
        assert "안녕하세요" in result
        assert "테스트" in result


# =============================================================================
# CER/WER Calculation Tests
# =============================================================================

class TestCERCalculation:
    """Character Error Rate 계산 테스트"""

    def test_cer_identical(self, sample_text):
        """동일 텍스트 CER = 0"""
        result = calculate_cer(sample_text, sample_text)
        assert result["cer"] == 0.0

    def test_cer_completely_different(self):
        """완전히 다른 텍스트"""
        reference = "abcdef"
        hypothesis = "xyz"
        result = calculate_cer(reference, hypothesis)
        assert result["cer"] > 0.5

    def test_cer_one_char_error(self):
        """한 글자 오류"""
        reference = "hello"
        hypothesis = "hallo"
        result = calculate_cer(reference, hypothesis)
        assert 0 < result["cer"] < 0.5


class TestWERCalculation:
    """Word Error Rate 계산 테스트"""

    def test_wer_identical(self, sample_text):
        """동일 텍스트 WER = 0"""
        result = calculate_wer(sample_text, sample_text)
        assert result["wer"] == 0.0

    def test_wer_one_word_error(self):
        """한 단어 오류"""
        reference = "hello world test"
        hypothesis = "hello word test"
        result = calculate_wer(reference, hypothesis)
        assert 0 < result["wer"] < 0.5

    def test_wer_completely_different(self):
        """완전히 다른 텍스트"""
        reference = "hello world"
        hypothesis = "goodbye universe"
        result = calculate_wer(reference, hypothesis)
        assert result["wer"] == 1.0


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestEvaluateResults:
    """결과 평가 테스트"""

    def test_evaluate_identical_content(self, sample_gt_text):
        """동일 내용 평가"""
        results = {
            "VLM": {"success": True, "content": sample_gt_text}
        }
        evaluation = evaluate_results(results, sample_gt_text, tokenizer_name="whitespace")

        assert "VLM" in evaluation
        assert evaluation["VLM"]["cer"] == 0.0
        assert evaluation["VLM"]["wer"] == 0.0

    def test_evaluate_with_errors(self, sample_gt_text, sample_parsed_text):
        """오류 포함 평가"""
        results = {
            "VLM": {"success": True, "content": sample_parsed_text}
        }
        evaluation = evaluate_results(results, sample_gt_text, tokenizer_name="whitespace")

        assert "VLM" in evaluation
        assert evaluation["VLM"]["cer"] > 0

    def test_evaluate_failed_parser(self, sample_gt_text):
        """실패한 파서 평가"""
        results = {
            "VLM": {"success": False, "content": ""}
        }
        evaluation = evaluate_results(results, sample_gt_text, tokenizer_name="whitespace")

        # 실패한 파서는 평가에서 제외되거나 None 값을 가짐
        assert "VLM" not in evaluation or evaluation["VLM"].get("cer") is None


# =============================================================================
# Data Folder Scanning Tests
# =============================================================================

class TestScanDataFolders:
    """데이터 폴더 스캔 테스트"""

    def test_scan_finds_test_folders(self, temp_data_dir):
        """test_ 폴더 발견"""
        folders = scan_data_folders(temp_data_dir)
        assert len(folders) == 1
        assert folders[0]["test_id"] == "test_1"

    def test_scan_finds_input_file(self, temp_data_dir):
        """입력 파일 발견"""
        folders = scan_data_folders(temp_data_dir)
        assert folders[0]["input_file"].exists()

    def test_scan_finds_gt_file(self, temp_data_dir):
        """GT 파일 발견"""
        folders = scan_data_folders(temp_data_dir)
        assert folders[0]["gt_file"].exists()

    def test_scan_empty_directory(self, tmp_path):
        """빈 디렉토리 스캔"""
        folders = scan_data_folders(tmp_path)
        assert len(folders) == 0

    def test_scan_ignores_non_test_folders(self, tmp_path):
        """test_ 접두사 없는 폴더 무시"""
        other_dir = tmp_path / "other_folder"
        other_dir.mkdir()
        folders = scan_data_folders(tmp_path)
        assert len(folders) == 0


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================

class TestParserIntegration:
    """파서 통합 테스트 (모킹)"""

    @patch('src.eval_parsers.VLMParser')
    def test_vlm_parser_mock(self, mock_vlm_class):
        """VLM 파서 모킹 테스트"""
        mock_parser = Mock()
        mock_parser.parse.return_value = "파싱된 텍스트"
        mock_vlm_class.return_value = mock_parser

        parser = mock_vlm_class()
        result = parser.parse(b"dummy pdf bytes")

        assert result == "파싱된 텍스트"
        mock_parser.parse.assert_called_once()

    @patch('src.eval_parsers.ImageOCRParser')
    def test_ocr_parser_mock(self, mock_ocr_class):
        """OCR 파서 모킹 테스트"""
        mock_parser = Mock()
        mock_parser.parse.return_value = "OCR 텍스트"
        mock_ocr_class.return_value = mock_parser

        parser = mock_ocr_class()
        result = parser.parse(b"dummy image bytes")

        assert result == "OCR 텍스트"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_text_cer(self):
        """빈 텍스트 CER"""
        result = calculate_cer("", "")
        assert result["cer"] == 0.0

    def test_empty_reference_cer(self):
        """빈 참조 텍스트"""
        result = calculate_cer("", "some text")
        # 빈 참조에 대한 처리 확인
        assert result["cer"] >= 0

    def test_unicode_text_cer(self):
        """유니코드 텍스트 CER"""
        reference = "한글 테스트"
        hypothesis = "한글 테스트"
        result = calculate_cer(reference, hypothesis)
        assert result["cer"] == 0.0

    def test_mixed_language_wer(self):
        """혼합 언어 WER"""
        reference = "Hello 안녕 World 세계"
        hypothesis = "Hello 안녕 World 세계"
        result = calculate_wer(reference, hypothesis)
        assert result["wer"] == 0.0


# =============================================================================
# TEDS (Tree Edit Distance Similarity) Tests
# =============================================================================

class TestExtractMarkdownTables:
    """마크다운 테이블 추출 테스트"""

    def test_extract_single_table(self):
        """단일 테이블 추출"""
        md = "Some text\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nMore text"
        tables = extract_markdown_tables(md)
        assert len(tables) == 1
        assert "| A | B |" in tables[0]

    def test_extract_multiple_tables(self):
        """복수 테이블 추출"""
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n\nText\n\n| C | D |\n|---|---|\n| 3 | 4 |"
        tables = extract_markdown_tables(md)
        assert len(tables) == 2

    def test_no_tables(self):
        """테이블 없는 텍스트"""
        md = "Just some text\nwith no tables"
        tables = extract_markdown_tables(md)
        assert len(tables) == 0

    def test_separator_row_included(self):
        """구분선 행도 테이블 블록에 포함"""
        md = "| H1 | H2 |\n|---|---|\n| a | b |"
        tables = extract_markdown_tables(md)
        assert len(tables) == 1
        assert "|---|---|" in tables[0]


class TestMarkdownTableToHtml:
    """마크다운 → HTML 테이블 변환 테스트"""

    def test_basic_conversion(self):
        """기본 변환"""
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html = markdown_table_to_html(md)
        assert "<table>" in html or "<table" in html
        assert "<td>" in html or "<td " in html

    def test_header_row(self):
        """헤더 행이 <th>로 변환"""
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
        html = markdown_table_to_html(md)
        assert "<th>" in html or "<th " in html


class TestHtmlToTree:
    """HTML → 트리 변환 테스트"""

    def test_basic_tree(self):
        """기본 트리 생성"""
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        tree = html_to_tree(html)
        assert tree is not None
        assert tree.tag == "table"
        assert len(tree.children) == 1  # 1 tr
        assert len(tree.children[0].children) == 2  # 2 td

    def test_empty_html(self):
        """빈 HTML"""
        tree = html_to_tree("")
        assert tree is None

    def test_cell_text(self):
        """셀 텍스트 추출"""
        html = "<table><tr><td>Hello</td></tr></table>"
        tree = html_to_tree(html)
        assert tree.children[0].children[0].text == "Hello"


class TestComputeTeds:
    """TEDS 계산 테스트"""

    def test_identical_tables(self):
        """동일 테이블 TEDS = 1.0"""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        score = compute_teds(html, html)
        assert score == 1.0

    def test_completely_different_tables(self):
        """완전히 다른 테이블"""
        html1 = "<table><tr><td>A</td></tr></table>"
        html2 = "<table><tr><td>X</td><td>Y</td></tr><tr><td>Z</td><td>W</td></tr></table>"
        score = compute_teds(html1, html2)
        assert 0.0 <= score < 1.0

    def test_one_cell_difference(self):
        """한 셀 차이"""
        html1 = "<table><tr><td>A</td><td>B</td></tr></table>"
        html2 = "<table><tr><td>A</td><td>X</td></tr></table>"
        score = compute_teds(html1, html2)
        assert 0.5 < score < 1.0

    def test_empty_vs_nonempty(self):
        """빈 테이블 vs 비어있지 않은 테이블"""
        score = compute_teds("", "<table><tr><td>A</td></tr></table>")
        assert score == 0.0

    def test_both_empty(self):
        """둘 다 빈 입력"""
        score = compute_teds("", "")
        assert score == 1.0


class TestCalculateTeds:
    """마크다운 기반 TEDS 계산 통합 테스트"""

    def test_identical_markdown_tables(self):
        """동일한 마크다운 테이블"""
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = calculate_teds(md, md)
        assert result["teds"] is not None
        assert result["teds"] == 1.0

    def test_no_tables_in_reference(self):
        """참조에 테이블 없음"""
        result = calculate_teds("| A |\n|---|\n| 1 |", "Just text, no tables")
        assert result["teds"] is None

    def test_missing_hypothesis_table(self):
        """예측에 테이블 누락"""
        ref = "| A | B |\n|---|---|\n| 1 | 2 |"
        hyp = "No tables here"
        result = calculate_teds(hyp, ref)
        assert result["teds"] == 0.0
        assert result["teds_detail"]["matched_tables"] == 0

    def test_teds_detail_structure(self):
        """TEDS 상세 결과 구조 확인"""
        md = "| X | Y |\n|---|---|\n| a | b |"
        result = calculate_teds(md, md)
        detail = result["teds_detail"]
        assert "table_count" in detail
        assert "matched_tables" in detail
        assert "per_table_scores" in detail
        assert detail["table_count"] == 1
        assert detail["matched_tables"] == 1
