"""
Retrieval Evaluation Module

Provides metrics and comparison tools for evaluating
retrieval performance across different parsers.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .retriever import RetrievalResult, RetrievalConfig


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics."""
    total_queries: int = 0
    hit_rate: dict = field(default_factory=dict)  # k -> hit rate
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: dict = field(default_factory=dict)  # k -> NDCG (optional)
    recall: dict = field(default_factory=dict)  # k -> recall (optional)

    # Per-query breakdown
    query_results: list = field(default_factory=list)

    # Confidence intervals (bootstrap)
    hit_rate_ci: dict = field(default_factory=dict)
    mrr_ci: tuple = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "recall": self.recall,
            "hit_rate_ci": self.hit_rate_ci,
            "mrr_ci": list(self.mrr_ci),
        }


class RetrievalEvaluator:
    """
    Evaluates retrieval performance with standard IR metrics.

    Metrics:
    - Hit Rate@k: Proportion of queries with relevant doc in top-k
    - MRR: Mean Reciprocal Rank
    - NDCG@k: Normalized Discounted Cumulative Gain (optional)
    """

    def __init__(
        self,
        k_values: list[int] = None,
        bootstrap_samples: int = 1000
    ):
        self.k_values = k_values or [1, 3, 5, 10]
        self.bootstrap_samples = bootstrap_samples

    def evaluate(self, results: list[RetrievalResult]) -> RetrievalMetrics:
        """
        Evaluate retrieval results.

        Args:
            results: List of RetrievalResult from retriever

        Returns:
            RetrievalMetrics with aggregated scores
        """
        if not results:
            return RetrievalMetrics()

        # Calculate hit rates for each k
        hit_rate = {}
        for k in self.k_values:
            hits = sum(1 for r in results if self._is_hit_at_k(r, k))
            hit_rate[k] = hits / len(results)

        # Calculate MRR
        reciprocal_ranks = []
        for r in results:
            if r.hit_rank:
                reciprocal_ranks.append(1.0 / r.hit_rank)
            else:
                reciprocal_ranks.append(0.0)
        mrr = np.mean(reciprocal_ranks)

        # Bootstrap confidence intervals
        hit_rate_ci = {}
        for k in self.k_values:
            ci = self._bootstrap_ci(
                [1 if self._is_hit_at_k(r, k) else 0 for r in results]
            )
            hit_rate_ci[k] = ci

        mrr_ci = self._bootstrap_ci(reciprocal_ranks)

        return RetrievalMetrics(
            total_queries=len(results),
            hit_rate=hit_rate,
            mrr=float(mrr),
            hit_rate_ci=hit_rate_ci,
            mrr_ci=mrr_ci,
            query_results=[r.to_dict() for r in results],
        )

    def _is_hit_at_k(self, result: RetrievalResult, k: int) -> bool:
        """Check if expected chunk is in top-k results."""
        if result.hit_rank is None:
            return False
        return result.hit_rank <= k

    def _bootstrap_ci(
        self,
        values: list,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not values:
            return (0.0, 0.0)

        values = np.array(values)
        n = len(values)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return (float(lower), float(upper))

    def evaluate_by_type(
        self,
        results: list[RetrievalResult],
        type_mapping: dict
    ) -> dict[str, RetrievalMetrics]:
        """
        Evaluate results grouped by query type.

        Args:
            results: List of RetrievalResult
            type_mapping: Dict mapping query_id to query type

        Returns:
            Dict mapping query type to RetrievalMetrics
        """
        # Group results by type
        grouped = {}
        for r in results:
            q_type = type_mapping.get(r.query_id, "unknown")
            if q_type not in grouped:
                grouped[q_type] = []
            grouped[q_type].append(r)

        # Evaluate each group
        metrics_by_type = {}
        for q_type, type_results in grouped.items():
            metrics_by_type[q_type] = self.evaluate(type_results)

        return metrics_by_type


def compare_retrieval_performance(
    baseline_results: list[RetrievalResult],
    vlm_results: list[RetrievalResult],
    k_values: list[int] = None,
    output_path: Optional[str] = None
) -> dict:
    """
    Compare retrieval performance between baseline and VLM parsers.

    Args:
        baseline_results: Results from baseline parser (pdfplumber/OCR)
        vlm_results: Results from VLM parser
        k_values: List of k values for Hit Rate calculation
        output_path: Optional path to save comparison results

    Returns:
        Dictionary with comparison metrics
    """
    k_values = k_values or [1, 3, 5, 10]
    evaluator = RetrievalEvaluator(k_values=k_values)

    # Evaluate both
    baseline_metrics = evaluator.evaluate(baseline_results)
    vlm_metrics = evaluator.evaluate(vlm_results)

    # Calculate improvements
    improvement = {
        "hit_rate": {},
        "mrr": 0.0,
    }

    for k in k_values:
        baseline_hr = baseline_metrics.hit_rate.get(k, 0)
        vlm_hr = vlm_metrics.hit_rate.get(k, 0)
        improvement["hit_rate"][k] = {
            "absolute": vlm_hr - baseline_hr,
            "relative_pct": (
                (vlm_hr - baseline_hr) / baseline_hr * 100
                if baseline_hr > 0 else 0
            ),
        }

    improvement["mrr"] = {
        "absolute": vlm_metrics.mrr - baseline_metrics.mrr,
        "relative_pct": (
            (vlm_metrics.mrr - baseline_metrics.mrr) / baseline_metrics.mrr * 100
            if baseline_metrics.mrr > 0 else 0
        ),
    }

    # Statistical significance (paired t-test approximation)
    significance = _calculate_significance(baseline_results, vlm_results, k_values)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "total_queries": baseline_metrics.total_queries,
            "hit_rate": baseline_metrics.hit_rate,
            "mrr": baseline_metrics.mrr,
            "hit_rate_ci": baseline_metrics.hit_rate_ci,
            "mrr_ci": baseline_metrics.mrr_ci,
        },
        "vlm": {
            "total_queries": vlm_metrics.total_queries,
            "hit_rate": vlm_metrics.hit_rate,
            "mrr": vlm_metrics.mrr,
            "hit_rate_ci": vlm_metrics.hit_rate_ci,
            "mrr_ci": vlm_metrics.mrr_ci,
        },
        "improvement": improvement,
        "significance": significance,
        "conclusion": _generate_conclusion(improvement, significance),
    }

    # Save if path provided
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    return comparison


def _calculate_significance(
    baseline_results: list[RetrievalResult],
    vlm_results: list[RetrievalResult],
    k_values: list[int]
) -> dict:
    """Calculate statistical significance of improvement."""
    try:
        from scipy import stats
    except ImportError:
        return {"error": "scipy not available for significance testing"}

    significance = {}

    # Ensure aligned results
    baseline_dict = {r.query_id: r for r in baseline_results if r.query_id}
    vlm_dict = {r.query_id: r for r in vlm_results if r.query_id}
    common_ids = set(baseline_dict.keys()) & set(vlm_dict.keys())

    if len(common_ids) < 3:
        return {"error": "Not enough paired samples for significance testing"}

    for k in k_values:
        # Paired samples
        baseline_hits = [
            1 if baseline_dict[qid].hit_rank and baseline_dict[qid].hit_rank <= k else 0
            for qid in common_ids
        ]
        vlm_hits = [
            1 if vlm_dict[qid].hit_rank and vlm_dict[qid].hit_rank <= k else 0
            for qid in common_ids
        ]

        # McNemar's test for paired binary data
        # contingency: [both_hit, vlm_only, baseline_only, neither]
        both_hit = sum(1 for b, v in zip(baseline_hits, vlm_hits) if b and v)
        vlm_only = sum(1 for b, v in zip(baseline_hits, vlm_hits) if not b and v)
        baseline_only = sum(1 for b, v in zip(baseline_hits, vlm_hits) if b and not v)

        # McNemar's chi-squared
        if vlm_only + baseline_only > 0:
            chi2 = (abs(vlm_only - baseline_only) - 1) ** 2 / (vlm_only + baseline_only)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2 = 0
            p_value = 1.0

        significance[f"hit_rate@{k}"] = {
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "vlm_wins": vlm_only,
            "baseline_wins": baseline_only,
        }

    # MRR significance (paired t-test)
    baseline_rr = [
        1.0 / baseline_dict[qid].hit_rank if baseline_dict[qid].hit_rank else 0
        for qid in common_ids
    ]
    vlm_rr = [
        1.0 / vlm_dict[qid].hit_rank if vlm_dict[qid].hit_rank else 0
        for qid in common_ids
    ]

    t_stat, p_value = stats.ttest_rel(vlm_rr, baseline_rr)
    significance["mrr"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }

    return significance


def _generate_conclusion(improvement: dict, significance: dict) -> str:
    """Generate human-readable conclusion from comparison."""
    conclusions = []

    # Hit Rate improvement
    hr_improvements = improvement.get("hit_rate", {})
    for k, stats in hr_improvements.items():
        abs_imp = stats.get("absolute", 0)
        rel_imp = stats.get("relative_pct", 0)

        if abs_imp > 0:
            sig = significance.get(f"hit_rate@{k}", {})
            sig_str = " (statistically significant)" if sig.get("significant") else ""
            conclusions.append(
                f"Hit Rate@{k}: VLM improved by {abs_imp:.2%} ({rel_imp:.1f}% relative){sig_str}"
            )
        elif abs_imp < 0:
            conclusions.append(
                f"Hit Rate@{k}: Baseline better by {-abs_imp:.2%}"
            )
        else:
            conclusions.append(f"Hit Rate@{k}: No difference")

    # MRR improvement
    mrr_imp = improvement.get("mrr", {})
    abs_mrr = mrr_imp.get("absolute", 0)
    rel_mrr = mrr_imp.get("relative_pct", 0)
    mrr_sig = significance.get("mrr", {})
    mrr_sig_str = " (statistically significant)" if mrr_sig.get("significant") else ""

    if abs_mrr > 0:
        conclusions.append(
            f"MRR: VLM improved by {abs_mrr:.3f} ({rel_mrr:.1f}% relative){mrr_sig_str}"
        )
    elif abs_mrr < 0:
        conclusions.append(f"MRR: Baseline better by {-abs_mrr:.3f}")
    else:
        conclusions.append("MRR: No difference")

    return "\n".join(conclusions)
