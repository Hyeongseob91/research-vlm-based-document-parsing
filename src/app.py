"""
VLM vs OCR Document Parsing Comparison Test

PDF를 업로드하면 VLM과 OCR 두 방식으로 파싱한 결과를 나란히 비교합니다.

실행 방법:
    streamlit run app.py --server.port 8501
"""

import streamlit as st
import time
from io import BytesIO

# Local imports
from parsers.vlm_parser import VLMParser, VLMResult
from parsers.ocr_parser import OCRParser, ImageOCRParser, OCRResult


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="VLM vs OCR Parser Comparison",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Sidebar Configuration
# ============================================================================
st.sidebar.title("⚙️ Settings")

vlm_api_url = st.sidebar.text_input(
    "VLM API URL",
    value="http://localhost:8004/v1/chat/completions"
)

vlm_model = st.sidebar.text_input(
    "VLM Model",
    value="qwen3-vl-8b-thinking"
)

show_thinking = st.sidebar.checkbox(
    "Show VLM Thinking Process",
    value=False,
    help="Thinking 모델의 추론 과정을 표시합니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Comparison Metrics
- **VLM**: 구조화된 마크다운 출력
- **OCR**: 순수 텍스트 추출
""")

# ============================================================================
# Main Content
# ============================================================================
st.title("📄 VLM vs OCR Document Parsing")
st.markdown("""
PDF 문서를 업로드하면 **Vision-Language Model(VLM)**과 **전통적 OCR** 두 방식으로
텍스트를 추출하여 결과를 비교합니다.
""")

# File Upload
uploaded_file = st.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"],
    help="PDF 파일을 드래그하거나 클릭하여 업로드하세요."
)

if uploaded_file is not None:
    # Read file bytes
    pdf_bytes = uploaded_file.read()
    file_size = len(pdf_bytes) / 1024  # KB

    st.info(f"📁 **{uploaded_file.name}** ({file_size:.1f} KB)")

    # Initialize parsers
    ocr_parser = OCRParser()
    image_parser = ImageOCRParser()

    # Detect PDF type
    pdf_type = ocr_parser.detect_pdf_type(pdf_bytes)
    st.markdown(f"**PDF Type**: `{pdf_type}` {'(텍스트 추출 가능)' if pdf_type == 'digital' else '(이미지 기반 - VLM 권장)'}")

    st.markdown("---")

    # Process button
    if st.button("🚀 파싱 시작", type="primary", use_container_width=True):

        # Create two columns for comparison
        col1, col2 = st.columns(2)

        # ====================================================================
        # Column 1: VLM Parser
        # ====================================================================
        with col1:
            st.subheader("🤖 VLM (Qwen3-VL-8B)")
            st.caption("구조화된 마크다운 출력")

            with st.spinner("VLM 처리 중..."):
                vlm_start = time.time()

                # Convert PDF to images
                images = image_parser.pdf_to_images(pdf_bytes, dpi=150)

                if not images:
                    st.error("PDF를 이미지로 변환할 수 없습니다.")
                else:
                    # Parse first page with VLM
                    vlm_parser = VLMParser(api_url=vlm_api_url, model=vlm_model)

                    # Process each page (Context Manager 사용으로 close() 불필요)
                    vlm_results = []
                    progress = st.progress(0)

                    for i, img_bytes in enumerate(images):
                        progress.progress((i + 1) / len(images))
                        result = vlm_parser.parse(img_bytes)
                        vlm_results.append(result)

                        if not result.success:
                            st.warning(f"Page {i+1} 처리 실패: {result.error}")

                    vlm_total_time = time.time() - vlm_start

                    # Display results
                    st.metric("처리 시간", f"{vlm_total_time:.2f}s")

                    # Show thinking process if enabled
                    if show_thinking and vlm_results and vlm_results[0].thinking:
                        with st.expander("🧠 Thinking Process", expanded=False):
                            st.code(vlm_results[0].thinking[:2000] + "...", language=None)

                    # Combine all pages
                    combined_content = "\n\n---\n\n".join(
                        f"## Page {i+1}\n\n{r.content}"
                        for i, r in enumerate(vlm_results)
                        if r.success and r.content
                    )

                    # Render markdown
                    st.markdown("**결과:**")
                    with st.container(height=500):
                        st.markdown(combined_content)

                    # Raw output & Download
                    with st.expander("📝 Raw Output"):
                        st.code(combined_content, language="markdown")

                    # Download button
                    st.download_button(
                        label="📥 Markdown 다운로드",
                        data=combined_content,
                        file_name=f"{uploaded_file.name.replace('.pdf', '')}_vlm.md",
                        mime="text/markdown"
                    )

        # ====================================================================
        # Column 2: OCR Parser (pdfplumber)
        # ====================================================================
        with col2:
            st.subheader("📖 OCR (pdfplumber)")
            st.caption("순수 텍스트 추출")

            with st.spinner("OCR 처리 중..."):
                ocr_result = ocr_parser.parse_pdf(pdf_bytes)

                if ocr_result.success:
                    st.metric("처리 시간", f"{ocr_result.elapsed_time:.2f}s")

                    # Metadata
                    st.markdown(f"""
                    - **페이지 수**: {ocr_result.page_count}
                    - **표 개수**: {len(ocr_result.tables)}
                    - **텍스트 존재**: {'✅' if ocr_result.has_text else '❌'}
                    """)

                    # Display results
                    st.markdown("**결과:**")
                    with st.container(height=500):
                        st.text(ocr_result.content if ocr_result.content else "(텍스트 없음 - 스캔 문서일 수 있음)")

                    # Tables
                    if ocr_result.tables:
                        with st.expander(f"📊 추출된 표 ({len(ocr_result.tables)}개)"):
                            for i, table in enumerate(ocr_result.tables):
                                st.markdown(f"**Table {i+1}**")
                                st.code(table)

                    # Download button
                    if ocr_result.content:
                        st.download_button(
                            label="📥 텍스트 다운로드",
                            data=ocr_result.content,
                            file_name=f"{uploaded_file.name.replace('.pdf', '')}_ocr.txt",
                            mime="text/plain"
                        )

                else:
                    st.error(f"OCR 처리 실패: {ocr_result.error}")

        # ====================================================================
        # Comparison Summary
        # ====================================================================
        st.markdown("---")
        st.subheader("📊 비교 요약")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.markdown("### VLM")
            if vlm_results:
                success_count = sum(1 for r in vlm_results if r.success)
                st.markdown(f"""
                - 성공: {success_count}/{len(vlm_results)} 페이지
                - 총 시간: {vlm_total_time:.2f}s
                - 평균: {vlm_total_time/len(vlm_results):.2f}s/page
                """)

        with summary_col2:
            st.markdown("### OCR")
            st.markdown(f"""
            - 성공: {'✅' if ocr_result.success else '❌'}
            - 총 시간: {ocr_result.elapsed_time:.2f}s
            - 텍스트: {'있음' if ocr_result.has_text else '없음'}
            """)

        with summary_col3:
            st.markdown("### 권장사항")
            if not ocr_result.has_text:
                st.success("🤖 VLM 사용 권장 (스캔 문서)")
            else:
                st.info("📖 OCR 사용 가능 (디지털 문서)")


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption("SoundMind AI Platform - VLM Document Parsing Test")
