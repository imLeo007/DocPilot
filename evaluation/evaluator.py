import json

from pathlib import Path

from typing import Any

import httpx

from deepeval import evaluate

from deepeval.evaluate import AsyncConfig

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric

from deepeval.test_case import LLMTestCase

from evaluation.judge import judge


API_ENDPOINT = "http://localhost:8000/question/search"


def load_dataset(path: str = "evaluation/dataset.json") -> list[dict[str, Any]]:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}.")

    with dataset_path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def evaluate_groundforge(
    top_k: int = 5,
    document_id: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset()

    test_cases: list[LLMTestCase] = []

    question_results: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0) as client:
        for item in dataset:
            question = item["question"]

            expected_answer = item["reference_answer"]

            payload: dict[str, Any] = {"query": question, "top_k": top_k}
            
            if document_id is not None:
                payload["document_id"] = document_id

            response = client.post(API_ENDPOINT, json=payload)
            
            response.raise_for_status()
            
            data = response.json()

            retrieved_chunks = data.get("results", [])
            
            retrieved_texts = [str(chunk["text"]) for chunk in retrieved_chunks]            

            retrieved_chunk_indexes = [int(chunk["chunk_index"]) for chunk in retrieved_chunks]

            test_case = LLMTestCase(
                input=question,
                actual_output=expected_answer,
                expected_output=expected_answer,
                retrieval_context=retrieved_texts,
            )

            test_cases.append(test_case)

            question_results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "chunk_indexes": retrieved_chunk_indexes,
                    "top_rerank_score": retrieved_chunks[0]["rerank_score"] if retrieved_chunks else 0.0,
                }
            )

    context_recall_metric = ContextualRecallMetric(
        threshold=0.7,
        model=judge,
        include_reason=False,
        async_mode=False,
    )

    context_precision_metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=judge,
        include_reason=False,
        async_mode=False,
    )

    eval_results = evaluate(
        test_cases=test_cases,
        metrics=[
            context_recall_metric,
            context_precision_metric,
        ],
        async_config=AsyncConfig(run_async=False)
    )

    deepeval_metrics = []
    
    for test_result in eval_results.test_results:
        metrics_dict = {test.name: test.score for test in test_result.metrics_data}
    
        deepeval_metrics.append(metrics_dict)

    total_evals = len(deepeval_metrics) if deepeval_metrics else 1
    
    avg_context_recall = sum(
        metric.get("Contextual Recall", 0.0) for metric in deepeval_metrics
    ) / total_evals

    avg_context_precision = sum(
        metric.get("Contextual Precision", 0.0) for metric in deepeval_metrics
    ) / total_evals

    return {
        "questions": len(dataset),
        "deepeval_metrics": {
            "average_context_recall": round(avg_context_recall, 4),
            "average_context_precision": round(avg_context_precision, 4),
        },
        "details": question_results,
    }