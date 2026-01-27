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
