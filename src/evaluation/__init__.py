"""
Unified Evaluation Module for VLM Document Parsing

This module provides a unified interface to all evaluation functionality:
- Lexical metrics (CER, WER)
- Structural metrics (Boundary Score, Chunk Score)
- Retrieval metrics (Hit Rate, MRR)
- Error analysis and case studies

It integrates the _drafts evaluation modules with the new chunking
and retrieval modules.
"""

import sys
from pathlib import Path

# Add _drafts to path for backward compatibility
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "_drafts"))

# Import from _drafts modules
try:
    from _drafts.normalizer import TextNormalizer, NormalizeMode, create_normalizer
    from _drafts.cer_calculator import CERCalculator, CERResult, create_cer_calculator
    from _drafts.wer_calculator import WERCalculator, WERResult, create_wer_calculator
    from _drafts.latency_tracker import LatencyTracker, LatencyResult, create_latency_tracker
    from _drafts.structure_validator import StructureValidator, StructureResult, create_structure_validator
    from _drafts.evaluation_runner import (
        EvaluationRunner,
        EvaluationResult,
        ComparisonResult,
        ParserOutput,
        ResultStore,
    )
except ImportError:
    # Fallback: modules may not be available
    TextNormalizer = None
    CERCalculator = None
    WERCalculator = None
    LatencyTracker = None
    StructureValidator = None
    EvaluationRunner = None

# Import from new modules
from ..chunking import (
    ChunkerConfig,
    ChunkingStrategy,
    Chunk,
    create_chunker,
    calculate_boundary_score,
    calculate_chunk_score,
    BoundaryScore,
    ChunkScore,
)

from ..retrieval import (
    EmbeddingConfig,
    RetrievalConfig,
    ChunkRetriever,
    RetrievalEvaluator,
    RetrievalMetrics,
    compare_retrieval_performance,
)

from ..error_analysis import (
    ErrorCategory,
    ErrorSeverity,
    ParsingError,
    ErrorAnalyzer,
    CaseStudy,
    CaseStudyGenerator,
    create_html_diff,
)

__all__ = [
    # Legacy _drafts exports
    "TextNormalizer",
    "NormalizeMode",
    "create_normalizer",
    "CERCalculator",
    "CERResult",
    "create_cer_calculator",
    "WERCalculator",
    "WERResult",
    "create_wer_calculator",
    "LatencyTracker",
    "LatencyResult",
    "create_latency_tracker",
    "StructureValidator",
    "StructureResult",
    "create_structure_validator",
    "EvaluationRunner",
    "EvaluationResult",
    "ComparisonResult",
    "ParserOutput",
    "ResultStore",

    # Chunking exports
    "ChunkerConfig",
    "ChunkingStrategy",
    "Chunk",
    "create_chunker",
    "calculate_boundary_score",
    "calculate_chunk_score",
    "BoundaryScore",
    "ChunkScore",

    # Retrieval exports
    "EmbeddingConfig",
    "RetrievalConfig",
    "ChunkRetriever",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "compare_retrieval_performance",

    # Error analysis exports
    "ErrorCategory",
    "ErrorSeverity",
    "ParsingError",
    "ErrorAnalyzer",
    "CaseStudy",
    "CaseStudyGenerator",
    "create_html_diff",

    # Unified runner
    "UnifiedEvaluator",
]


class UnifiedEvaluator:
    """
    Unified evaluation orchestrator.

    Combines all evaluation metrics:
    - Lexical: CER, WER
    - Structural: Boundary Score, Chunk Score
    - Retrieval: Hit Rate, MRR
    - Error Analysis: Error detection and case studies
    """

    def __init__(self, config: dict = None):
        """
        Initialize unified evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        if CERCalculator:
            self.cer_calculator = create_cer_calculator(config)
        else:
            self.cer_calculator = None

        if WERCalculator:
            self.wer_calculator = create_wer_calculator(config)
        else:
            self.wer_calculator = None

        if TextNormalizer:
            self.normalizer = create_normalizer(config.get("normalizer") if config else None)
        else:
            self.normalizer = None

        self.error_analyzer = ErrorAnalyzer()

        # Chunking config
        chunking_config = config.get("chunking", {}) if config else {}
        self.chunker_config = ChunkerConfig(
            strategy=ChunkingStrategy(chunking_config.get("strategy", "recursive_character")),
            chunk_size=chunking_config.get("chunk_size", 500),
            chunk_overlap=chunking_config.get("chunk_overlap", 50),
        )

        # Retrieval config
        retrieval_config = config.get("retrieval", {}) if config else {}
        embedding_config = config.get("embedding", {}) if config else {}
        self.retrieval_config = RetrievalConfig(
            top_k=retrieval_config.get("top_k", [1, 3, 5, 10]),
            embedding_config=EmbeddingConfig(
                model=embedding_config.get("model", "jhgan/ko-sroberta-multitask"),
                device=embedding_config.get("device", "cpu"),
            ),
        )

    def evaluate_lexical(
        self,
        parsed_text: str,
        ground_truth: str,
        normalize: bool = True
    ) -> dict:
        """
        Calculate lexical metrics (CER, WER).

        Args:
            parsed_text: Parser output
            ground_truth: Ground truth text
            normalize: Whether to normalize text before comparison

        Returns:
            Dictionary with CER and WER results
        """
        results = {}

        # Normalize if requested
        if normalize and self.normalizer:
            parsed_text = self.normalizer.normalize(parsed_text)
            ground_truth = self.normalizer.normalize(ground_truth)

        # CER
        if self.cer_calculator:
            cer_result = self.cer_calculator.calculate(parsed_text, ground_truth)
            results["cer"] = {
                "value": cer_result.cer,
                "percentage": cer_result.cer_percentage,
                "accuracy": cer_result.accuracy_percentage,
                "substitutions": cer_result.substitutions,
                "deletions": cer_result.deletions,
                "insertions": cer_result.insertions,
            }

        # WER
        if self.wer_calculator:
            wer_result = self.wer_calculator.calculate(parsed_text, ground_truth)
            results["wer"] = {
                "value": wer_result.wer,
                "percentage": wer_result.wer_percentage,
                "accuracy": wer_result.accuracy_percentage,
                "tokenizer": wer_result.tokenizer_used.value if wer_result.tokenizer_used else None,
            }

        return results

    def evaluate_structural(
        self,
        parsed_text: str,
        ground_truth: str,
        document_id: str = "doc"
    ) -> dict:
        """
        Calculate structural metrics (Boundary Score, Chunk Score).

        Args:
            parsed_text: Parser output
            ground_truth: Ground truth text
            document_id: Document identifier for chunk IDs

        Returns:
            Dictionary with structural metrics
        """
        # Chunk both texts
        chunker = create_chunker(self.chunker_config)
        parsed_chunks = chunker.chunk(parsed_text, f"{document_id}_parsed")
        gt_chunks = chunker.chunk(ground_truth, f"{document_id}_gt")

        # Boundary score
        bs = calculate_boundary_score(parsed_text, ground_truth)

        # Chunk score
        cs = calculate_chunk_score(parsed_chunks)

        return {
            "boundary_score": bs.to_dict(),
            "chunk_score": cs.to_dict(),
            "parsed_chunk_count": len(parsed_chunks),
            "gt_chunk_count": len(gt_chunks),
        }

    def evaluate_retrieval(
        self,
        parsed_text: str,
        qa_pairs: list,
        document_id: str = "doc"
    ) -> dict:
        """
        Calculate retrieval metrics (Hit Rate, MRR).

        Args:
            parsed_text: Parser output
            qa_pairs: Q&A pairs with questions and expected answers
            document_id: Document identifier

        Returns:
            Dictionary with retrieval metrics
        """
        # Chunk parsed text
        chunker = create_chunker(self.chunker_config)
        chunks = chunker.chunk(parsed_text, document_id)

        if not chunks:
            return {"error": "No chunks generated"}

        # Create retriever
        retriever = ChunkRetriever(config=self.retrieval_config)
        retriever.index_chunks(chunks)

        # Prepare queries
        queries = []
        for qa in qa_pairs:
            expected_chunk = retriever.find_relevant_chunk(
                qa.get("answer_span", qa.get("answer", ""))
            )
            queries.append({
                "query": qa["question"],
                "query_id": qa.get("id", str(len(queries))),
                "expected_chunk_id": expected_chunk,
            })

        # Run retrieval
        results = retriever.retrieve_batch(queries, top_k=max(self.retrieval_config.top_k))

        # Evaluate
        evaluator = RetrievalEvaluator(k_values=self.retrieval_config.top_k)
        metrics = evaluator.evaluate(results)

        return metrics.to_dict()

    def analyze_errors(
        self,
        parsed_text: str,
        ground_truth: str,
        parser_name: str = "unknown"
    ) -> dict:
        """
        Analyze parsing errors.

        Args:
            parsed_text: Parser output
            ground_truth: Ground truth text
            parser_name: Name of parser

        Returns:
            Dictionary with error analysis results
        """
        errors = self.error_analyzer.analyze(parsed_text, ground_truth, parser_name)
        summary = self.error_analyzer.summarize_errors(errors)

        return {
            "summary": summary,
            "errors": [e.to_dict() for e in errors[:20]],  # Limit to top 20
        }

    def evaluate_full(
        self,
        parsed_text: str,
        ground_truth: str,
        qa_pairs: list = None,
        document_id: str = "doc",
        parser_name: str = "unknown"
    ) -> dict:
        """
        Run full evaluation suite.

        Args:
            parsed_text: Parser output
            ground_truth: Ground truth text
            qa_pairs: Optional Q&A pairs for retrieval evaluation
            document_id: Document identifier
            parser_name: Name of parser

        Returns:
            Dictionary with all evaluation results
        """
        results = {
            "document_id": document_id,
            "parser_name": parser_name,
        }

        # Lexical metrics
        results["lexical"] = self.evaluate_lexical(parsed_text, ground_truth)

        # Structural metrics
        results["structural"] = self.evaluate_structural(
            parsed_text, ground_truth, document_id
        )

        # Retrieval metrics (if Q&A pairs provided)
        if qa_pairs:
            results["retrieval"] = self.evaluate_retrieval(
                parsed_text, qa_pairs, document_id
            )

        # Error analysis
        results["errors"] = self.analyze_errors(parsed_text, ground_truth, parser_name)

        return results
