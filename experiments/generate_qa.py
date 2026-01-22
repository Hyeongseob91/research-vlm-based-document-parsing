#!/usr/bin/env python3
"""
Q&A Dataset Generator for VLM Document Parsing Evaluation

This script generates question-answer pairs from ground truth documents
for evaluating retrieval performance in RAG systems.

Usage:
    python -m experiments.generate_qa --config experiments/config.yaml
    python -m experiments.generate_qa --gt data/test_1/gt_data_1.md --output data/qa_pairs.json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class QAPair:
    """Represents a single question-answer pair."""
    id: str
    document_id: str
    question: str
    answer: str
    answer_span: str  # Exact text span from document
    question_type: str  # factual, table_lookup, multi_hop, inferential
    difficulty: str  # easy, medium, hard
    metadata: dict


@dataclass
class QADataset:
    """Collection of Q&A pairs for evaluation."""
    version: str
    created_at: str
    document_count: int
    total_questions: int
    questions_by_type: dict
    qa_pairs: list


class QAGenerator:
    """Generates Q&A pairs from ground truth documents using LLM."""

    SYSTEM_PROMPT = """You are a Q&A dataset generator for document retrieval evaluation.
Your task is to create question-answer pairs that test a retrieval system's ability
to find relevant information in documents.

Requirements:
1. Questions must be answerable from the provided document content
2. Answers must be exact quotes or close paraphrases from the document
3. Include the exact text span that contains the answer
4. Vary question difficulty (easy/medium/hard)
5. Follow the specified question type distribution

Output format (JSON):
{
    "questions": [
        {
            "question": "What is X?",
            "answer": "X is Y",
            "answer_span": "exact text from document containing the answer",
            "question_type": "factual|table_lookup|multi_hop|inferential",
            "difficulty": "easy|medium|hard"
        }
    ]
}"""

    QUESTION_TYPE_PROMPTS = {
        "factual": "Create a factual question that can be answered with a single fact from the document.",
        "table_lookup": "Create a question that requires looking up information from a table in the document.",
        "multi_hop": "Create a question that requires combining information from multiple parts of the document.",
        "inferential": "Create a question that requires reasoning or inference based on the document content."
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv(
            "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        )

        if not self.api_key:
            print(f"Warning: No API key found for {provider}. Using mock generation.")
            self.use_mock = True
        else:
            self.use_mock = False

    def generate_for_document(
        self,
        document_id: str,
        content: str,
        questions_per_type: dict,
        language: str = "auto"
    ) -> list[QAPair]:
        """Generate Q&A pairs for a single document."""
        qa_pairs = []

        # Detect language if auto
        if language == "auto":
            language = self._detect_language(content)

        for q_type, count in questions_per_type.items():
            type_prompt = self.QUESTION_TYPE_PROMPTS.get(q_type, "")

            prompt = f"""Document content:
---
{content[:8000]}  # Truncate for token limits
---

Generate exactly {count} questions of type: {q_type}
{type_prompt}

Language: Generate questions in {language} if the document is in {language}.

Respond with valid JSON only."""

            if self.use_mock:
                generated = self._mock_generate(document_id, q_type, count, content)
            else:
                generated = self._llm_generate(prompt, q_type, count)

            for i, qa in enumerate(generated):
                qa_pair = QAPair(
                    id=f"{document_id}_{q_type}_{i+1}",
                    document_id=document_id,
                    question=qa["question"],
                    answer=qa["answer"],
                    answer_span=qa.get("answer_span", qa["answer"]),
                    question_type=q_type,
                    difficulty=qa.get("difficulty", "medium"),
                    metadata={
                        "language": language,
                        "generated_at": datetime.now().isoformat()
                    }
                )
                qa_pairs.append(qa_pair)

        return qa_pairs

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character frequency."""
        korean_pattern = re.compile(r'[\uac00-\ud7af]')
        korean_chars = len(korean_pattern.findall(text))
        total_chars = len(text)

        if total_chars > 0 and korean_chars / total_chars > 0.1:
            return "korean"
        return "english"

    def _llm_generate(self, prompt: str, q_type: str, count: int) -> list[dict]:
        """Generate Q&A pairs using LLM API."""
        try:
            if self.provider == "openai":
                return self._openai_generate(prompt)
            elif self.provider == "anthropic":
                return self._anthropic_generate(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            print(f"LLM generation failed: {e}. Using mock generation.")
            return self._mock_generate("unknown", q_type, count, "")

    def _openai_generate(self, prompt: str) -> list[dict]:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("questions", [])
        except ImportError:
            print("OpenAI package not installed. Run: pip install openai")
            return []

    def _anthropic_generate(self, prompt: str) -> list[dict]:
        """Generate using Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            content = response.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("questions", [])
            return []
        except ImportError:
            print("Anthropic package not installed. Run: pip install anthropic")
            return []

    def _mock_generate(
        self,
        document_id: str,
        q_type: str,
        count: int,
        content: str
    ) -> list[dict]:
        """Generate mock Q&A pairs for testing without API."""
        mock_questions = {
            "factual": [
                {"question": "이 문서의 주요 목적은 무엇인가요?",
                 "answer": "[문서 내용에 따라 답변]",
                 "difficulty": "easy"},
                {"question": "지원 기간은 언제까지인가요?",
                 "answer": "[문서 내용에 따라 답변]",
                 "difficulty": "easy"},
            ],
            "table_lookup": [
                {"question": "지원 금액은 최대 얼마인가요?",
                 "answer": "[테이블 내용에 따라 답변]",
                 "difficulty": "easy"},
                {"question": "지원 대상 기업 유형은 무엇인가요?",
                 "answer": "[테이블 내용에 따라 답변]",
                 "difficulty": "medium"},
            ],
            "multi_hop": [
                {"question": "기본 지원과 추가 지원의 차이점은 무엇인가요?",
                 "answer": "[여러 섹션 내용 종합]",
                 "difficulty": "medium"},
            ],
            "inferential": [
                {"question": "이 프로그램이 중소기업에 어떤 도움이 될 수 있을까요?",
                 "answer": "[추론 기반 답변]",
                 "difficulty": "hard"},
            ]
        }

        # Return mock questions for the type
        type_questions = mock_questions.get(q_type, [])
        result = []
        for i in range(count):
            if i < len(type_questions):
                q = type_questions[i].copy()
                q["answer_span"] = q["answer"]
                result.append(q)
            else:
                # Generate placeholder
                result.append({
                    "question": f"[{q_type} question {i+1} for {document_id}]",
                    "answer": "[Answer placeholder]",
                    "answer_span": "[Answer span placeholder]",
                    "difficulty": "medium"
                })

        return result


def load_ground_truth(path: str) -> str:
    """Load ground truth markdown file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_qa_dataset(dataset: QADataset, output_path: str):
    """Save Q&A dataset to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    data = {
        "version": dataset.version,
        "created_at": dataset.created_at,
        "document_count": dataset.document_count,
        "total_questions": dataset.total_questions,
        "questions_by_type": dataset.questions_by_type,
        "qa_pairs": [asdict(qa) for qa in dataset.qa_pairs]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset.qa_pairs)} Q&A pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs for document retrieval evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--gt",
        type=str,
        help="Single ground truth file to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qa_pairs.json",
        help="Output path for Q&A dataset"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "mock"],
        default="mock",
        help="LLM provider for generation"
    )
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=15,
        help="Number of questions per document"
    )

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}

    # Get Q&A generation settings
    qa_config = config.get("qa_generation", {})
    provider = args.provider if args.provider != "mock" else qa_config.get("llm_provider", "mock")
    model = qa_config.get("llm_model", "gpt-4")
    temperature = qa_config.get("temperature", 0.3)

    # Calculate questions per type
    total_questions = args.questions_per_doc
    questions_per_type = {
        "factual": max(1, total_questions // 3),
        "table_lookup": max(1, total_questions // 4),
        "multi_hop": max(1, total_questions // 5),
        "inferential": max(1, total_questions // 5)
    }

    # Initialize generator
    generator = QAGenerator(
        provider=provider if provider != "mock" else "openai",
        model=model,
        temperature=temperature
    )

    all_qa_pairs = []
    questions_by_type = {"factual": 0, "table_lookup": 0, "multi_hop": 0, "inferential": 0}

    if args.gt:
        # Process single file
        content = load_ground_truth(args.gt)
        doc_id = Path(args.gt).stem
        qa_pairs = generator.generate_for_document(
            document_id=doc_id,
            content=content,
            questions_per_type=questions_per_type
        )
        all_qa_pairs.extend(qa_pairs)
        for qa in qa_pairs:
            questions_by_type[qa.question_type] += 1
        documents_processed = 1
    else:
        # Process all documents from config
        datasets = config.get("datasets", {}).get("test_cases", [])
        for test_case in datasets:
            gt_path = test_case.get("gt_path")
            if gt_path and os.path.exists(gt_path):
                print(f"Processing: {test_case.get('name', gt_path)}")
                content = load_ground_truth(gt_path)
                qa_pairs = generator.generate_for_document(
                    document_id=test_case.get("id", Path(gt_path).stem),
                    content=content,
                    questions_per_type=questions_per_type,
                    language=test_case.get("language", "auto")
                )
                all_qa_pairs.extend(qa_pairs)
                for qa in qa_pairs:
                    questions_by_type[qa.question_type] += 1
            else:
                print(f"Skipping {gt_path}: file not found")
        documents_processed = len(datasets)

    # Create dataset
    dataset = QADataset(
        version="1.0",
        created_at=datetime.now().isoformat(),
        document_count=documents_processed,
        total_questions=len(all_qa_pairs),
        questions_by_type=questions_by_type,
        qa_pairs=all_qa_pairs
    )

    # Save dataset
    save_qa_dataset(dataset, args.output)

    # Print summary
    print(f"\n{'='*50}")
    print("Q&A Generation Summary")
    print(f"{'='*50}")
    print(f"Documents processed: {documents_processed}")
    print(f"Total questions: {len(all_qa_pairs)}")
    print("Questions by type:")
    for q_type, count in questions_by_type.items():
        print(f"  - {q_type}: {count}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
