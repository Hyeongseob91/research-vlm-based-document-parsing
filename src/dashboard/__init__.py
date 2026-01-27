"""
VLM Document Parsing Analysis Dashboard

Streamlit-based dashboard for analyzing parser performance metrics.

v2.0: Now reads from results/test_*/ folders instead of single JSON file.
"""

from .styles import COLORS, apply_dark_theme
from .data_loader import (
    # New v2.0 functions
    load_all_results,
    load_all_results_cached,
    scan_test_folders,
    load_test_result,
    TestResult,
    get_test_ids,
    get_parser_names,
    get_test_evaluation,
    get_test_chunking,
    get_parsing_summary_df,
    get_chunking_summary_df,
    get_aggregated_parser_df,
    get_test_detail_df,
    get_chunking_for_test,
    get_tests_with_chunking,
    export_df_to_csv,
    get_chart_download_config,
    # Backward compatibility (deprecated)
    load_results,
    load_results_cached,
    get_parsing_data,
    get_chunking_data,
    paginate_data,
    get_chunking_parsers,
    get_chunking_data_for_parser,
)
from .charts import (
    create_grouped_bar_chart,
    create_box_plot,
    create_scatter_plot,
    create_heatmap,
    # New chunking charts (MoC-based)
    STRATEGY_COLORS,
    get_strategy_color,
    create_parser_chunking_comparison,
    create_bc_document_flow,
    create_cs_mean_std_bar,
)

__all__ = [
    # Styles
    "COLORS",
    "apply_dark_theme",
    # Data loading (v2.0)
    "load_all_results",
    "load_all_results_cached",
    "scan_test_folders",
    "load_test_result",
    "TestResult",
    "get_test_ids",
    "get_parser_names",
    "get_test_evaluation",
    "get_test_chunking",
    "get_parsing_summary_df",
    "get_chunking_summary_df",
    "get_aggregated_parser_df",
    "get_test_detail_df",
    "get_chunking_for_test",
    "get_tests_with_chunking",
    "export_df_to_csv",
    "get_chart_download_config",
    # Backward compatibility (deprecated)
    "load_results",
    "load_results_cached",
    "get_parsing_data",
    "get_chunking_data",
    "paginate_data",
    "get_chunking_parsers",
    "get_chunking_data_for_parser",
    # Charts
    "create_grouped_bar_chart",
    "create_box_plot",
    "create_scatter_plot",
    "create_heatmap",
    # Chunking charts (MoC-based)
    "STRATEGY_COLORS",
    "get_strategy_color",
    "create_parser_chunking_comparison",
    "create_bc_document_flow",
    "create_cs_mean_std_bar",
]
