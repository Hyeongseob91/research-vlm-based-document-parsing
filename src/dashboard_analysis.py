"""
VLM Document Parsing Quality Analysis Dashboard

CLI 테스트 결과를 시각화하여 Tech Report 작성을 지원하는 정적 대시보드

Features:
- JSON 결과 파일 로드 (results/parsing_results.json)
- @st.cache_data 캐싱 (1시간 TTL)
- 페이지네이션 (10개 테스트 초과 시)
- 차트 PNG 다운로드
- CSV 내보내기

Usage:
    streamlit run src/dashboard_analysis.py
"""

import sys
from pathlib import Path

_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import numpy as np

from dashboard.data_loader import (
    load_all_results_cached,
    get_test_ids,
    get_parser_names,
    get_parsing_summary_df,
    get_chunking_summary_df,
    get_aggregated_parser_df,
    get_test_evaluation,
    get_test_chunking,
    get_chunking_for_test,
    get_tests_with_chunking,
    get_chart_download_config,
    export_df_to_csv,
    # Backward compatibility
    get_parsing_data,
    get_chunking_data,
    paginate_data,
    get_chunking_parsers,
    get_chunking_data_for_parser,
)
from dashboard.charts import (
    STRATEGY_COLORS,
    create_parser_chunking_comparison,
    create_bc_document_flow,
    create_cs_mean_std_bar,
)
from dashboard.styles import PARSER_COLORS as STYLE_PARSER_COLORS

# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="VLM Document Parsing Quality Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# 스타일 설정
# =============================================================================

st.markdown("""
<style>
    /* Sidebar 완전 숨김 */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }

    /* 전체 배경 */
    .stApp { background-color: #FAFAFA; }

    /* 헤더 */
    h1, h2, h3 { color: #1a1a2e !important; font-weight: 600 !important; }

    /* 메트릭 카드 */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E5E5;
    }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] { color: #666666 !important; }

    /* 탭 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #E5E5E5; }
    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a2e !important;
        border-bottom: 2px solid #4F46E5 !important;
    }

    /* 테이블 */
    .stDataFrame { border-radius: 8px; }

    /* 구분선 */
    hr { border-color: #E5E5E5; margin: 2rem 0; }

    /* 다운로드 버튼 */
    .download-btn {
        background-color: #F3F4F6;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 상수
# =============================================================================

VERSION = "v0.4.0"  # Added parser-specific chunking analysis (MoC-based)
PAGE_SIZE = 10  # 페이지네이션 크기

# 동적 색상 생성용 기본 팔레트 (파서 추가 시 자동 확장)
DEFAULT_COLORS = [
    "#4F46E5",  # VLM - 인디고
    "#059669",  # OCR-Text - 에메랄드
    "#D97706",  # OCR-Image - 앰버
    "#7C3AED",  # TwoStage-Text - 보라
    "#0891B2",  # TwoStage-Image - 청록
    "#DC2626",  # 여유 - 레드
    "#EC4899",  # 여유 - 핑크
]


def get_parser_colors(parsers: List[str]) -> Dict[str, str]:
    """파서별 색상 동적 생성 (styles.py의 PARSER_COLORS를 단일 진실 공급원으로 사용)"""
    colors = {}
    for i, parser in enumerate(parsers):
        # styles.py에서 정의된 색상 사용, 없으면 순환 색상
        colors[parser] = STYLE_PARSER_COLORS.get(parser, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    return colors


# =============================================================================
# 데이터 로드
# =============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load data with caching - scans results/test_*/ folders"""
    data = load_all_results_cached()
    if "error" in data:
        return data, True
    return data, False


# 데이터 로드
raw_data, is_error = load_data()

# 파서 색상
PARSER_NAMES = get_parser_names(raw_data)
PARSER_COLORS = get_parser_colors(PARSER_NAMES)

# 변환된 데이터 (호환성 유지)
PARSING_DATA = get_parsing_data(raw_data)
CHUNKING_DATA = get_chunking_data(raw_data)

# 새로운 형식 데이터
TEST_IDS = get_test_ids(raw_data)


# =============================================================================
# 차트 생성 함수
# =============================================================================

def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Hex 색상을 rgba로 변환"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_thin_bar_chart(data: Dict, metric: str, title: str,
                          lower_is_better: bool = False) -> go.Figure:
    """얇은 가로형 Bar Chart"""
    parsers = list(data["parsers"].keys())
    values = [data["parsers"][p].get(metric) or 0 for p in parsers]  # Handle None
    colors = [PARSER_COLORS.get(p, "#888") for p in parsers]

    fig = go.Figure()
    # Format based on metric type
    if metric == "elapsed_time":
        text_values = [f"{v:.1f}s" for v in values]
    else:
        text_values = [f"{v:.3f}" for v in values]

    fig.add_trace(go.Bar(
        y=parsers,
        x=values,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=text_values,
        textposition="outside",
        textfont=dict(size=12, color="#333"),
    ))

    direction = "← Lower is better" if lower_is_better else "Higher is better →"
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a2e"), x=0),
        height=180,
        margin=dict(l=10, r=80, t=40, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#666"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        showlegend=False,
        annotations=[dict(
            text=direction, x=1, y=-0.12, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#888"), xanchor="right"
        )]
    )
    return fig


def create_radar_chart(all_data: Dict) -> go.Figure:
    """파서별 성능 Radar Chart"""
    metrics = ["WER", "CER", "Struct-F1", "Latency"]
    fig = go.Figure()

    for parser in PARSER_NAMES:
        values = []
        for metric_key in ["wer", "cer", "structure_f1", "elapsed_time"]:
            vals = [
                test["parsers"][parser].get(metric_key, 0)
                for test in all_data.values()
                if parser in test["parsers"]
            ]
            # Filter out None values
            vals = [v for v in vals if v is not None]
            avg = np.mean(vals) if vals else 0

            # 정규화 (낮을수록 좋은 것은 반전, 0=worst, 1=best)
            if metric_key in ["wer", "cer"]:
                # WER/CER: 0% = 1.0 (best), 200%+ = 0.0 (worst)
                normalized = max(0, 1 - avg / 2)
            elif metric_key == "elapsed_time":
                # Latency: 0s = 1.0 (best), 120s+ = 0.0 (worst)
                normalized = max(0, 1 - avg / 120)
            elif metric_key == "structure_f1":
                # Structure F1: 0 = 0.0 (worst), 1 = 1.0 (best)
                normalized = avg
            else:
                normalized = avg
            values.append(normalized)

        values.append(values[0])  # 닫기

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=parser,
            line=dict(color=PARSER_COLORS.get(parser, "#888"), width=3),
            fill='toself',
            fillcolor=hex_to_rgba(PARSER_COLORS.get(parser, "#888"), 0.1),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#E5E5E5"),
            angularaxis=dict(tickfont=dict(size=13, color="#333"), gridcolor="#E5E5E5"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
        height=450,
        margin=dict(l=80, r=80, t=40, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_bc_cs_scatter(chunking_data: Dict) -> go.Figure:
    """BC vs CS Scatter Plot"""
    fig = go.Figure()

    for strategy, data in chunking_data.items():
        bc_values = [c.get("bc", 0) for c in data.get("chunks", [])]
        cs_values = [c.get("cs", 0) for c in data.get("chunks", [])]

        if not bc_values:
            continue

        fig.add_trace(go.Scatter(
            x=bc_values, y=cs_values, mode='markers', name=strategy,
            marker=dict(
                size=12,
                color=STRATEGY_COLORS.get(strategy, "#888"),
                line=dict(width=1, color="white"),
                opacity=0.8,
            ),
            hovertemplate=f"<b>{strategy}</b><br>BC: %{{x:.2f}}<br>CS: %{{y:.2f}}<extra></extra>",
        ))

    # Quadrant 영역
    fig.add_shape(type="rect", x0=0.5, x1=1, y0=0, y1=0.5,
                  fillcolor="rgba(16, 185, 129, 0.05)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=0.5, y0=0.5, y1=1,
                  fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)

    fig.add_hline(y=0.5, line_dash="dot", line_color="#ccc", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#ccc", line_width=1)

    annotations = [
        dict(x=0.75, y=0.25, text="이상적<br>(BC↑ CS↓)", showarrow=False,
             font=dict(size=9, color="#059669"), opacity=0.7),
        dict(x=0.25, y=0.75, text="Over-merge<br>(BC↓ CS↑)", showarrow=False,
             font=dict(size=9, color="#DC2626"), opacity=0.7),
        dict(x=0.75, y=0.75, text="Fragmentation<br>(BC↑ CS↑)", showarrow=False,
             font=dict(size=9, color="#D97706"), opacity=0.7),
        dict(x=0.25, y=0.25, text="Structural<br>Failure", showarrow=False,
             font=dict(size=9, color="#6B7280"), opacity=0.7),
    ]

    fig.update_layout(
        title=dict(text="BC–CS Distribution by Strategy", font=dict(size=14, color="#1a1a2e"), x=0),
        xaxis=dict(title="Boundary Clarity (BC) →", range=[0, 1], gridcolor="#E5E5E5", zeroline=False),
        yaxis=dict(title="Chunk Stickiness (CS) ↓", range=[0, 1], gridcolor="#E5E5E5", zeroline=False),
        height=450, margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        annotations=annotations,
    )
    return fig


def create_grouped_bar(all_data: Dict, metric: str, title: str, lower_is_better: bool = False) -> go.Figure:
    """전체 테스트 비교 Grouped Bar Chart"""
    test_ids = [d["id"] for d in all_data.values()]
    fig = go.Figure()

    for parser in PARSER_NAMES:
        color = PARSER_COLORS.get(parser, "#888")
        values = [test["parsers"].get(parser, {}).get(metric) or 0 for test in all_data.values()]  # Handle None
        fig.add_trace(go.Bar(
            name=parser, x=test_ids, y=values,
            marker_color=color, marker_line_width=0,
            text=[f"{v:.2f}" if metric != "elapsed_time" else f"{v:.1f}s" for v in values],
            textposition="outside", textfont=dict(size=11), width=0.3,
        ))

    direction = "↓ Lower is better" if lower_is_better else "↑ Higher is better"
    fig.update_layout(
        title=dict(text=f"{title} ({direction})", font=dict(size=15, color="#1a1a2e"), x=0),
        barmode="group", height=380,
        margin=dict(l=50, r=30, t=60, b=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#666"),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#E5E5E5", gridwidth=0.5, zeroline=False, tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, font=dict(size=11)),
        bargap=0.25, bargroupgap=0.1,
    )
    return fig


def create_metrics_comparison_subplot(all_data: Dict) -> go.Figure:
    """4개 메트릭을 하나의 Subplot으로 통합한 차트

    장점:
    - Legend가 한 번만 표시됨 (중복 제거)
    - 일관된 레이아웃
    - 테스트 간 비교가 용이
    """
    test_ids = [d["id"] for d in all_data.values()]

    # 메트릭 정의: (key, title, lower_is_better, format_func)
    metrics = [
        ("wer", "WER ↓", True, lambda v: f"{v:.2f}"),
        ("cer", "CER ↓", True, lambda v: f"{v:.2f}"),
        ("structure_f1", "Structure F1 ↑", False, lambda v: f"{v:.2f}"),
        ("elapsed_time", "Latency ↓", True, lambda v: f"{v:.1f}s"),
    ]

    # 2x2 서브플롯 생성 (상하 간격 넓게)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    # 각 메트릭별로 바 추가
    for idx, (metric_key, title, lower_is_better, fmt) in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for parser_idx, parser in enumerate(PARSER_NAMES):
            color = PARSER_COLORS.get(parser, "#888")
            values = [
                test["parsers"].get(parser, {}).get(metric_key) or 0
                for test in all_data.values()
            ]

            # 첫 번째 서브플롯에서만 legend 표시
            show_legend = (idx == 0)

            fig.add_trace(
                go.Bar(
                    name=parser,
                    x=test_ids,
                    y=values,
                    marker_color=color,
                    marker_line_width=0,
                    text=[fmt(v) for v in values],
                    textposition="outside",
                    textfont=dict(size=10),
                    showlegend=show_legend,
                    legendgroup=parser,  # legend 그룹핑
                ),
                row=row, col=col
            )

    # 레이아웃 설정
    fig.update_layout(
        height=650,
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#666"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=14),
        ),
        margin=dict(l=50, r=30, t=100, b=40),
        bargap=0.15,
        bargroupgap=0.05,
    )

    # 각 축 설정
    for i in range(1, 5):
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10), row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor="#E5E5E5", gridwidth=0.5, zeroline=False, tickfont=dict(size=10), row=(i-1)//2+1, col=(i-1)%2+1)

    # 서브플롯 타이틀 스타일 (크게, 좌측 정렬)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=15, color="#1a1a2e", weight="bold")
        annotation['xanchor'] = 'left'
        # 좌측 정렬을 위해 x 위치 조정 (각 서브플롯의 시작점)
        if annotation['x'] < 0.5:
            annotation['x'] = 0.0
        else:
            annotation['x'] = 0.55

    return fig


# =============================================================================
# 메인 대시보드
# =============================================================================

# 헤더
st.title("📄 VLM Document Parsing Quality Analysis")
st.caption(f"CLI 테스트 결과 시각화 | Tech Report 작성 지원 | {VERSION}")

# 에러 경고
if is_error:
    st.error(f"⚠️ {raw_data.get('error', '테스트 결과를 찾을 수 없습니다.')}")
    st.info("테스트 실행: `python -m src.eval_parsers --all`")
    st.stop()

# 데이터 정보
data_info_cols = st.columns([1, 1, 1, 2])
with data_info_cols[0]:
    st.metric("Total Tests", raw_data.get("test_count", len(PARSING_DATA)))
with data_info_cols[1]:
    st.metric("Parsers", len(PARSER_NAMES))
with data_info_cols[2]:
    chunking_tests = len(get_tests_with_chunking(raw_data))
    st.metric("Chunking Tests", chunking_tests)
with data_info_cols[3]:
    loaded_at = raw_data.get("loaded_at", "N/A")
    st.caption(f"Data Version: {raw_data.get('version', 'N/A')} | Loaded: {loaded_at}")

st.markdown("---")

# 탭 구성
tab_parsing, tab_chunking, tab_result = st.tabs([
    "🔍 Parsing Test",
    "📦 Chunking Test",
    "📊 종합 분석"
])


# =============================================================================
# TAB 1: Parsing Test
# =============================================================================

with tab_parsing:
    st.markdown("## Parsing Test Results")

    # Metrics 정의
    with st.expander("📐 Metrics 정의", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**WER (Word Error Rate)** · :green[↓ 낮을수록 좋음]")
            st.markdown("단어 단위 오류율. 삽입/삭제/대체 오류 종합.")
            st.markdown("**CER (Character Error Rate)** · :green[↓ 낮을수록 좋음]")
            st.markdown("문자 단위 오류율. 누락/추가/변경 문자 추적.")
        with col2:
            st.markdown("**Structure F1** · :orange[↑ 높을수록 좋음]")
            st.markdown("마크다운 구조 요소(헤딩, 리스트, 테이블) 검출 F1 스코어.")
            st.markdown("**Latency** · :green[↓ 낮을수록 좋음]")
            st.markdown("문서 1건 Parsing 처리 시간 (초).")

    st.markdown("---")

    # Global Performance Summary
    st.markdown("### 📈 Global Performance Summary")

    col_table, col_radar = st.columns([1, 1])

    with col_table:
        # DataFrame 생성
        summary_df = get_parsing_summary_df(raw_data)
        # Use available columns from new format (including Structure F1)
        display_df = summary_df[["Test ID", "Parser", "CER %", "WER %", "Struct-F1 %", "Latency (s)", "Success"]].copy()
        display_df = display_df.rename(columns={
            "Test ID": "Test",
            "Struct-F1 %": "Struct-F1",
            "Latency (s)": "Latency",
        })
        display_df["Latency"] = display_df["Latency"].apply(lambda x: f"{x:.1f}s")

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

        # CSV 다운로드
        csv_data = export_df_to_csv(summary_df)
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv_data,
            file_name="parsing_summary.csv",
            mime="text/csv",
        )

    with col_radar:
        radar_fig = create_radar_chart(PARSING_DATA)
        st.plotly_chart(
            radar_fig,
            use_container_width=True,
            config=get_chart_download_config("radar_chart")
        )

    # Metrics Comparison - 통합 Subplot 차트
    st.markdown("#### Metrics Comparison")

    metrics_fig = create_metrics_comparison_subplot(PARSING_DATA)
    st.plotly_chart(
        metrics_fig,
        use_container_width=True,
        config=get_chart_download_config("metrics_comparison")
    )

    st.markdown("---")

    # Detailed Test Analysis with Pagination
    st.markdown("### 🔬 Detailed Test Analysis")

    # 페이지네이션 (10개 초과 시)
    test_items = list(PARSING_DATA.items())
    total_tests = len(test_items)

    if total_tests > PAGE_SIZE:
        # 페이지 선택
        col_page_info, col_page_nav = st.columns([2, 1])

        with col_page_info:
            st.caption(f"총 {total_tests}개 테스트 (페이지당 {PAGE_SIZE}개)")

        # 페이지 상태
        if "parsing_page" not in st.session_state:
            st.session_state.parsing_page = 1

        total_pages = (total_tests + PAGE_SIZE - 1) // PAGE_SIZE

        with col_page_nav:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.parsing_page,
                key="parsing_page_input"
            )
            st.session_state.parsing_page = page

        # 현재 페이지 데이터
        paginated_items, _, _, _ = paginate_data(test_items, page, PAGE_SIZE)
    else:
        paginated_items = test_items

    # 테스트별 상세 (Lazy Loading via Expander)
    for test_id, test_data in paginated_items:
        # 메타데이터에서 정보 추출 (자동 추출 형식)
        metadata = test_data.get("metadata", {})
        title = metadata.get("title", test_data.get('id', test_id))
        filename = metadata.get("filename", test_data.get('name', ''))
        doc_type = metadata.get("doc_type", test_data.get('doc_type', 'unknown'))
        pages = metadata.get("pages", test_data.get('pages', 0))
        file_size_kb = metadata.get("file_size_kb", test_data.get('file_size_kb', 0))
        language = metadata.get("language", test_data.get('language', ''))
        has_text_layer = metadata.get("has_text_layer", test_data.get('has_text_layer', False))

        # test_id에서 번호 추출 (예: test_1 → 1)
        test_num = test_id.replace("test_", "").replace("_", " ").title()

        # Expander 제목: "📄 Test 1: filename.pdf (PDF, 5p)"
        page_info = f", {pages}p" if pages else ""
        expander_title = f"📄 **Test {test_num}**: {filename} ({doc_type}{page_info})"

        with st.expander(expander_title, expanded=False):
            # 파일 정보 표시
            info_cols = st.columns([2, 1, 1, 1])
            with info_cols[0]:
                st.caption(f"📁 {title}")
            with info_cols[1]:
                if file_size_kb:
                    size_str = f"{file_size_kb:.0f}KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.1f}MB"
                    st.caption(f"💾 {size_str}")
            with info_cols[2]:
                if language:
                    st.caption(f"🌐 {language.upper()}")
            with info_cols[3]:
                text_layer_icon = "✅" if has_text_layer else "❌"
                st.caption(f"📝 Text: {text_layer_icon}")
            st.divider()

            # 테이블
            detail_rows = []
            for parser, metrics in test_data["parsers"].items():
                structure_f1 = metrics.get('structure_f1')
                struct_f1_display = f"{structure_f1:.3f}" if structure_f1 is not None else "N/A"

                detail_rows.append({
                    "Parser": parser,
                    "WER ↓": f"{metrics.get('wer') or 0:.3f}",
                    "CER ↓": f"{metrics.get('cer') or 0:.3f}",
                    "Struct-F1 ↑": struct_f1_display,
                    "Latency ↓": f"{metrics.get('elapsed_time') or 0:.1f}s",
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

            # Bar Charts
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "wer", "WER", lower_is_better=True),
                    use_container_width=True,
                    config=get_chart_download_config(f"{test_id}_wer")
                )
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "cer", "CER", lower_is_better=True),
                    use_container_width=True,
                    config=get_chart_download_config(f"{test_id}_cer")
                )
            with chart_cols[1]:
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "structure_f1", "Structure F1", lower_is_better=False),
                    use_container_width=True,
                    config=get_chart_download_config(f"{test_id}_structure_f1")
                )
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "elapsed_time", "Latency", lower_is_better=True),
                    use_container_width=True,
                    config=get_chart_download_config(f"{test_id}_latency")
                )


# =============================================================================
# TAB 2: Chunking Test
# =============================================================================

def get_chunking_data_dict(raw_data: Dict) -> Dict:
    """Chunking 데이터를 Parsing과 동일한 형식으로 변환"""
    chunking_dict = {}
    for test_id, test_data in raw_data.get("tests", {}).items():
        chunking = test_data.get("chunking", {})
        if not chunking or not chunking.get("results"):
            continue

        # Parsing과 동일한 형식으로 변환
        parsers_data = {}
        for parser, result in chunking.get("results", {}).items():
            bc = result.get("bc", {})
            cs = result.get("cs", {})
            parsers_data[parser] = {
                "bc": bc.get("score"),
                "bc_min": bc.get("min"),
                "bc_max": bc.get("max"),
                "bc_std": bc.get("std"),
                "cs": cs.get("score"),
                "chunk_count": result.get("chunk_count", 0),
            }

        if parsers_data:
            chunking_dict[test_id] = {
                "id": test_id,
                "parsers": parsers_data,
                "config": chunking.get("config", {}),
                "metadata": test_data.get("evaluation", {}).get("metadata", {}),
            }

    return chunking_dict


def create_chunking_metrics_subplot(chunking_data: Dict) -> go.Figure:
    """BC/CS 메트릭을 하나의 Subplot으로 통합한 차트 (Parsing과 동일한 형식)"""
    test_ids = [d["id"] for d in chunking_data.values()]

    # 메트릭 정의: (key, title, lower_is_better, format_func)
    metrics = [
        ("bc", "BC (Boundary Clarity) ↑", False, lambda v: f"{v:.3f}"),
        ("cs", "CS (Chunk Stickiness) ↓", True, lambda v: f"{v:.3f}"),
        ("chunk_count", "Chunk Count", False, lambda v: f"{int(v)}"),
        ("bc_std", "BC Std (Consistency) ↓", True, lambda v: f"{v:.3f}"),
    ]

    # 파서 목록 수집
    all_parsers = set()
    for test in chunking_data.values():
        all_parsers.update(test["parsers"].keys())
    parser_list = sorted(list(all_parsers))

    # 2x2 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    # 각 메트릭별로 바 추가
    for idx, (metric_key, title, lower_is_better, fmt) in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for parser_idx, parser in enumerate(parser_list):
            color = PARSER_COLORS.get(parser, DEFAULT_COLORS[parser_idx % len(DEFAULT_COLORS)])
            values = [
                test["parsers"].get(parser, {}).get(metric_key) or 0
                for test in chunking_data.values()
            ]

            # 첫 번째 서브플롯에서만 legend 표시
            show_legend = (idx == 0)

            fig.add_trace(
                go.Bar(
                    name=parser,
                    x=test_ids,
                    y=values,
                    marker_color=color,
                    marker_line_width=0,
                    text=[fmt(v) if v else "N/A" for v in values],
                    textposition="outside",
                    textfont=dict(size=10),
                    showlegend=show_legend,
                    legendgroup=parser,
                ),
                row=row, col=col
            )

    # 레이아웃 설정
    fig.update_layout(
        height=650,
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#666"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=14),
        ),
        margin=dict(l=50, r=30, t=100, b=40),
        bargap=0.15,
        bargroupgap=0.05,
    )

    # 각 축 설정
    for i in range(1, 5):
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10), row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor="#E5E5E5", gridwidth=0.5, zeroline=False, tickfont=dict(size=10), row=(i-1)//2+1, col=(i-1)%2+1)

    # 서브플롯 타이틀 스타일
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=15, color="#1a1a2e", weight="bold")
        annotation['xanchor'] = 'left'
        if annotation['x'] < 0.5:
            annotation['x'] = 0.0
        else:
            annotation['x'] = 0.55

    return fig


def create_chunking_thin_bar_chart(data: Dict, metric: str, title: str,
                                    lower_is_better: bool = False) -> go.Figure:
    """Chunking용 얇은 가로형 Bar Chart"""
    parsers = list(data["parsers"].keys())
    values = [data["parsers"][p].get(metric) or 0 for p in parsers]
    colors = [PARSER_COLORS.get(p, "#888") for p in parsers]

    fig = go.Figure()

    # Format based on metric type
    if metric == "chunk_count":
        text_values = [f"{int(v)}" for v in values]
    else:
        text_values = [f"{v:.4f}" for v in values]

    fig.add_trace(go.Bar(
        y=parsers,
        x=values,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=text_values,
        textposition="outside",
        textfont=dict(size=12, color="#333"),
    ))

    direction = "← Lower is better" if lower_is_better else "Higher is better →"
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a2e"), x=0),
        height=180,
        margin=dict(l=10, r=80, t=40, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#666"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        showlegend=False,
        annotations=[dict(
            text=direction, x=1, y=-0.12, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#888"), xanchor="right"
        )]
    )
    return fig


with tab_chunking:
    st.markdown("## Chunking Test Results")

    # Metrics 정의
    with st.expander("📐 Metrics 정의", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BC (Boundary Clarity)** · :orange[↑ 높을수록 좋음]")
            st.markdown("인접 청크 간 의미적 독립성. `1 - cosine_similarity`")
            st.markdown("**CS (Chunk Stickiness)** · :green[↓ 낮을수록 좋음]")
            st.markdown("청크 그래프의 구조적 엔트로피. 청크 간 연결성.")
        with col2:
            st.markdown("**Chunk Count** · 문서당 청크 수")
            st.markdown("SemanticChunker 기반 자동 분할 결과.")
            st.markdown("**BC Std** · :green[↓ 낮을수록 좋음]")
            st.markdown("BC 표준편차. 일관된 경계 품질 측정.")

    st.markdown("---")

    # 청킹 데이터 로드
    CHUNKING_DATA_DICT = get_chunking_data_dict(raw_data)
    tests_with_chunking = get_tests_with_chunking(raw_data)

    if not tests_with_chunking:
        st.warning("청킹 테스트 데이터가 없습니다.")
        st.markdown("""
        **실행 방법:**
        ```bash
        python -m src.eval_chunking --parsed-dir results/test_1/ --verbose
        ```
        """)
    else:
        # =====================================================================
        # Global Performance Summary
        # =====================================================================
        st.markdown("### 📈 Global Performance Summary")

        col_table, col_chart = st.columns([2, 3])

        with col_table:
            # Summary DataFrame 생성
            summary_df = get_chunking_summary_df(raw_data)
            if not summary_df.empty:
                display_df = summary_df.copy()
                # 컬럼 포맷팅
                display_df["BC Score"] = display_df["BC Score"].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
                display_df["CS Score"] = display_df["CS Score"].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
                display_df = display_df.rename(columns={
                    "Test ID": "Test",
                    "BC Score": "BC ↑",
                    "CS Score": "CS ↓",
                    "Chunk Count": "Chunks",
                })
                # 필요한 컬럼만 표시
                display_cols = ["Test", "Parser", "BC ↑", "CS ↓", "Chunks"]
                display_df = display_df[[c for c in display_cols if c in display_df.columns]]

                st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

                # CSV 다운로드
                csv_data = export_df_to_csv(summary_df)
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_data,
                    file_name="chunking_summary.csv",
                    mime="text/csv",
                )
            else:
                st.info("표시할 데이터가 없습니다.")

        with col_chart:
            # Bubble Chart - BC vs CS, size = Chunk Count
            if CHUNKING_DATA_DICT:
                # 데이터 수집
                plot_data = []
                for test_id, test_data in CHUNKING_DATA_DICT.items():
                    for parser, metrics in test_data["parsers"].items():
                        if metrics.get("bc") is not None and metrics.get("cs") is not None:
                            plot_data.append({
                                "parser": parser,
                                "test_id": test_id,
                                "bc": metrics.get("bc", 0),
                                "cs": metrics.get("cs", 0),
                                "chunks": metrics.get("chunk_count", 1),
                            })

                if plot_data:
                    # 파서별 색상 매핑
                    unique_parsers = sorted(list(set(d["parser"] for d in plot_data)))
                    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

                    fig = go.Figure()

                    for i, parser in enumerate(unique_parsers):
                        parser_data = [d for d in plot_data if d["parser"] == parser]
                        # 버블 크기 정규화 (min 15, max 50)
                        chunks = [d["chunks"] for d in parser_data]
                        max_chunks = max(d["chunks"] for d in plot_data)
                        min_chunks = min(d["chunks"] for d in plot_data)
                        if max_chunks > min_chunks:
                            sizes = [15 + (c - min_chunks) / (max_chunks - min_chunks) * 35 for c in chunks]
                        else:
                            sizes = [25] * len(chunks)

                        fig.add_trace(go.Scatter(
                            x=[d["bc"] for d in parser_data],
                            y=[d["cs"] for d in parser_data],
                            mode="markers",
                            name=parser,
                            marker=dict(
                                size=sizes,
                                color=colors[i % len(colors)],
                                opacity=0.7,
                                line=dict(width=2, color="white"),
                            ),
                            text=[f"{d['test_id']}<br>Chunks: {d['chunks']}" for d in parser_data],
                            hovertemplate="<b>%{text}</b><br>BC: %{x:.3f}<br>CS: %{y:.2f}<extra></extra>",
                        ))

                    fig.update_layout(
                        title=dict(text="BC vs CS (size=Chunks)", font=dict(size=14, color="#1a1a2e"), x=0),
                        xaxis=dict(title="BC (Boundary Clarity) ↑", gridcolor="#eee"),
                        yaxis=dict(title="CS (Chunk Stickiness) ↓", gridcolor="#eee"),
                        height=350,
                        margin=dict(l=60, r=20, t=50, b=50),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11, color="#666"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        showlegend=True,
                    )

                    st.plotly_chart(fig, use_container_width=True, config=get_chart_download_config("chunking_bubble"))
                else:
                    st.info("차트 데이터가 없습니다.")
            else:
                st.info("차트 데이터가 없습니다.")

        # =====================================================================
        # Metrics Comparison - 통합 Subplot 차트
        # =====================================================================
        if CHUNKING_DATA_DICT:
            st.markdown("#### Metrics Comparison")

            metrics_fig = create_chunking_metrics_subplot(CHUNKING_DATA_DICT)
            st.plotly_chart(
                metrics_fig,
                use_container_width=True,
                config=get_chart_download_config("chunking_metrics_comparison")
            )

        st.markdown("---")

        # =====================================================================
        # Detailed Test Analysis
        # =====================================================================
        st.markdown("### 🔬 Detailed Test Analysis")

        # 페이지네이션 (10개 초과 시)
        chunking_items = list(CHUNKING_DATA_DICT.items())
        total_chunking_tests = len(chunking_items)

        if total_chunking_tests > PAGE_SIZE:
            col_page_info, col_page_nav = st.columns([2, 1])

            with col_page_info:
                st.caption(f"총 {total_chunking_tests}개 테스트 (페이지당 {PAGE_SIZE}개)")

            if "chunking_page" not in st.session_state:
                st.session_state.chunking_page = 1

            total_pages = (total_chunking_tests + PAGE_SIZE - 1) // PAGE_SIZE

            with col_page_nav:
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.chunking_page,
                    key="chunking_page_input"
                )
                st.session_state.chunking_page = page

            paginated_chunking_items, _, _, _ = paginate_data(chunking_items, page, PAGE_SIZE)
        else:
            paginated_chunking_items = chunking_items

        # 테스트별 상세 (Expander)
        for test_id, test_data in paginated_chunking_items:
            metadata = test_data.get("metadata", {})
            title = metadata.get("title", test_id)
            config = test_data.get("config", {})

            test_num = test_id.replace("test_", "").replace("_", " ").title()
            total_chunks = sum(p.get("chunk_count", 0) for p in test_data["parsers"].values())

            expander_title = f"📦 **Test {test_num}**: {title} ({total_chunks} chunks)"

            with st.expander(expander_title, expanded=False):
                # 설정 정보
                info_cols = st.columns([2, 1, 1])
                with info_cols[0]:
                    st.caption(f"📁 {title}")
                with info_cols[1]:
                    strategy = config.get("breakpoint_type", "semantic")
                    st.caption(f"⚙️ Strategy: {strategy}")
                with info_cols[2]:
                    threshold = config.get("breakpoint_threshold", "N/A")
                    st.caption(f"🎯 Threshold: {threshold}")
                st.divider()

                # 테이블
                detail_rows = []
                for parser, metrics in test_data["parsers"].items():
                    bc_val = metrics.get("bc")
                    cs_val = metrics.get("cs")
                    bc_std = metrics.get("bc_std")

                    detail_rows.append({
                        "Parser": parser,
                        "BC ↑": f"{bc_val:.4f}" if bc_val is not None else "N/A",
                        "CS ↓": f"{cs_val:.4f}" if cs_val is not None else "N/A",
                        "BC Std": f"±{bc_std:.4f}" if bc_std is not None else "-",
                        "Chunks": metrics.get("chunk_count", 0),
                    })
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

                # Bar Charts
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    st.plotly_chart(
                        create_chunking_thin_bar_chart(test_data, "bc", "BC (Boundary Clarity)", lower_is_better=False),
                        use_container_width=True,
                        config=get_chart_download_config(f"{test_id}_bc")
                    )
                    st.plotly_chart(
                        create_chunking_thin_bar_chart(test_data, "cs", "CS (Chunk Stickiness)", lower_is_better=True),
                        use_container_width=True,
                        config=get_chart_download_config(f"{test_id}_cs")
                    )
                with chart_cols[1]:
                    st.plotly_chart(
                        create_chunking_thin_bar_chart(test_data, "chunk_count", "Chunk Count", lower_is_better=False),
                        use_container_width=True,
                        config=get_chart_download_config(f"{test_id}_chunks")
                    )
                    if any(test_data["parsers"][p].get("bc_std") for p in test_data["parsers"]):
                        st.plotly_chart(
                            create_chunking_thin_bar_chart(test_data, "bc_std", "BC Std (Consistency)", lower_is_better=True),
                            use_container_width=True,
                            config=get_chart_download_config(f"{test_id}_bc_std")
                        )



# =============================================================================
# TAB 3: 종합 분석
# =============================================================================

with tab_result:
    st.markdown("## 📊 종합 분석 결과")
    st.markdown("> Parsing과 Chunking 결과를 종합하여 파이프라인 품질을 진단합니다.")

    st.markdown("---")

    st.markdown("### 🎯 핵심 발견사항")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Parsing 관점

        1. **VLM이 전반적으로 우수**
           - 대부분의 테스트에서 최저 WER 달성
           - 특히 이미지 기반 문서에서 압도적

        2. **Trade-off 존재**
           - 정확도 ↔ 처리 시간
           - 실시간 서비스에는 pdfplumber 고려

        3. **문서 유형별 차이 큼**
           - 스캔 이미지: VLM 필수
           - 디지털 PDF: pdfplumber도 충분
        """)

    with col2:
        st.markdown("""
        #### Chunking 관점

        1. **Semantic Chunking 권장**
           - BC가 가장 높은 경계 명확도
           - CS가 낮은 내부 의존성

        2. **Fixed Chunking 주의**
           - 의미 경계 무시로 BC 낮음
           - RAG 성능 저하 우려

        3. **최적 파라미터**
           - Chunk Size: 400-600
           - Overlap: 50-100
        """)

    st.markdown("---")

    st.markdown("### 🚀 다음 단계")
    st.markdown("""
    | 우선순위 | 작업 | 목적 |
    |---------|------|------|
    | 1 | Golden Dataset 구축 | 평가 신뢰도 향상 |
    | 2 | VLM SFT 학습 | 구조화 성능 개선 |
    | 3 | Semantic Chunking 적용 | RAG 품질 향상 |
    | 4 | 추가 문서 유형 테스트 | 일반화 검증 |
    """)

    st.markdown("---")
    st.caption(f"VLM Document Parsing Quality Analysis | {VERSION}")
